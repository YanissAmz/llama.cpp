// M5-C ESCHAM CUDA — tuile 16x16 bande-major, miroir exact ggml-escham.c
// Pas de Hadamard. Decode une bande une fois et reutilise pour 16 sorties et toutes colonnes.
// Reference: ggml/src/ggml-escham.c (fige) et ggml/src/ggml-cpu/ggml-cpu.c branche ESCHAM
#include "escham.cuh"
#include "common.cuh"
#include "ggml.h"
#include <cstdint>
#include <cstring>

#define ESCHAM_MASK  0x8FFF8FFFu
#define ESCHAM_MAGIC 0x3B603B60u
#define ESCHAM_MUL1  0xCBAC1FEDu

static inline float escham_fp16_to_fp32_host(uint16_t h){
    uint32_t sign=(uint32_t)(h & 0x8000u)<<16;
    uint32_t e=(h>>10)&0x1fu;
    uint32_t m=h & 0x03ffu;
    uint32_t bits;
    if(e==0){
        if(m==0) bits=sign;
        else{
            e=113;
            while((m & 0x0400u)==0){ m<<=1; e--; }
            m &=0x03ffu;
            bits=sign | (e<<23) | (m<<13);
        }
    } else if(e==31) bits=sign | 0x7f800000u | (m<<13);
    else bits=sign | ((e+112u)<<23) | (m<<13);
    float f; memcpy(&f,&bits,sizeof(f)); return f;
}
static inline uint16_t escham_fp32_to_fp16_host(float f){
    uint32_t x; memcpy(&x,&f,sizeof(x));
    uint32_t s=(x>>16)&0x8000u;
    uint32_t e=(x>>23)&0xffu;
    uint32_t m=x & 0x007fffffu;
    if(e==0xff) return (uint16_t)(s | 0x7c00u | (m?0x0200u | (m>>13):0));
    int ee=(int)e-112;
    if(ee>=31) return (uint16_t)(s|0x7c00u);
    if(ee<=0){
        if(ee<-10) return (uint16_t)s;
        m|=0x00800000u;
        int shift=14-ee;
        uint32_t r=m>>shift;
        uint32_t rem=m & ((1u<<shift)-1u);
        uint32_t half=1u<<(shift-1);
        if(rem>half || (rem==half && (r&1u))) r++;
        return (uint16_t)(s|r);
    }
    uint32_t r=((uint32_t)ee<<10)|(m>>13);
    uint32_t rem=m & 0x1fffu;
    if(rem>0x1000u || (rem==0x1000u && (r&1u))) r++;
    return (uint16_t)(s|r);
}

// ---------------------------------------------------------------------------
// Decodage inline du codec (cb==1). Remplace la LUT de 65536 entrees fp32.
//
// Pourquoi : la LUT coutait 4 octets de lecture L2 par poids decode, contre
// 0,31 octet de code. Elle deplacait ~13x plus de donnees que les poids
// eux-memes, et les acces sont disperses sur 256 Kio -> aucune localite. A
// ~24 G poids codes par token, cela plafonnait la generation vers 3,5 t/s.
// Le codec tient en cinq instructions et zero acces memoire.
//
// Bit-exact avec la table construite par escham_cuda_init_tables : meme
// somme en fp32, meme arrondi au plus proche pair vers fp16, meme retour
// en fp32. __half2float est exact, __float2half_rn est RNE comme l'hote.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float escham_decode(uint16_t win){
    const uint32_t t = ((uint32_t)win * ESCHAM_MUL1 & ESCHAM_MASK) ^ ESCHAM_MAGIC;
    // t EST un half2 : ses deux moities sont les deux fp16 a additionner.
    // Les extraire par (t & 0xffff) et (t >> 16) coutait un LOP3 et un SHF,
    // soit deux des ~4 instructions ALU par poids — et le pipe ALU est le
    // goulot (69,8 % au ncu). HADD2 sait lire ses operandes en .H0_H0/.H1_H1
    // sans aucune instruction entiere. Le resultat reste l'addition fp16,
    // donc bit-identique a la table d'origine.
    const __half2 h = *reinterpret_cast<const __half2 *>(&t);
    return __half2float(__hadd(__low2half(h), __high2half(h)));
}

// device constant tables (small)
// HIP ne mappe pas cudaMemcpyToSymbol dans vendors/hip.h (seuls cudaMemcpy et
// ses variantes y figurent). Meme signature des deux cotes.
#if defined(GGML_USE_HIP)
#define ESCHAM_MEMCPY_TO_SYMBOL hipMemcpyToSymbol
#else
#define ESCHAM_MEMCPY_TO_SYMBOL cudaMemcpyToSymbol
#endif

__constant__ uint8_t c_tile_row[256];
__constant__ uint8_t c_tile_col[256];
__constant__ uint8_t c_cur2[32];
__constant__ uint8_t c_prev2[32];
__constant__ uint8_t c_sh2[32];
__constant__ uint8_t c_cur3[32];
__constant__ uint8_t c_prev3[32];
__constant__ uint8_t c_sh3[32];

static float * d_lut = nullptr;
static bool g_inited = false;

static void escham_cuda_init_tables(){
    if(g_inited) return;
    // LUT fp32
    float h_lut[65536];
    for(uint32_t w=0; w<65536u; ++w){
        uint32_t t=((w * ESCHAM_MUL1) & ESCHAM_MASK) ^ ESCHAM_MAGIC;
        float lo=escham_fp16_to_fp32_host((uint16_t)(t & 0xffffu));
        float hi=escham_fp16_to_fp32_host((uint16_t)(t>>16));
        uint16_t h=escham_fp32_to_fp16_host(lo+hi);
        h_lut[w]=escham_fp16_to_fp32_host(h);
    }
    CUDA_CHECK(cudaMalloc(&d_lut, sizeof(h_lut)));
    CUDA_CHECK(cudaMemcpy(d_lut, h_lut, sizeof(h_lut), cudaMemcpyHostToDevice));
    // tile row/col
    uint8_t h_row[256], h_col[256];
    const int off[8][2]={{9,8},{8,8},{1,8},{0,8},{9,0},{8,0},{1,0},{0,0}};
    for(int m=0;m<32;++m){
        int r0=2*(m & 3);
        int c0=2*((m>>3)&1)+4*((m>>4)&1);
        int d=(m>>2)&1;
        for(int s=0;s<8;++s){
            h_row[m*8+s]=(uint8_t)(r0+off[s][0]);
            h_col[m*8+s]=(uint8_t)(c0+off[s][1]+d);
        }
    }
    CUDA_CHECK(ESCHAM_MEMCPY_TO_SYMBOL(c_tile_row, h_row, sizeof(h_row)));
    CUDA_CHECK(ESCHAM_MEMCPY_TO_SYMBOL(c_tile_col, h_col, sizeof(h_col)));
    uint8_t cur2[32], prev2[32], sh2[32];
    uint8_t cur3[32], prev3[32], sh3[32];
    for(int l=0;l<32;++l){
        int c2=(l>>1)&15;
        cur2[l]=(uint8_t)c2;
        prev2[l]=(uint8_t)((c2-1)&15);
        sh2[l]=(l&1)==0?16:0;
        int b=l*24;
        int r86=b+791;
        int r87=r86 & 2016;
        cur3[l]=(uint8_t)((((r86>>3)&252)-96)/4);
        prev3[l]=(uint8_t)(l==0?23:(((b+755)>>5)-24));
        sh3[l]=(uint8_t)(r87-b-760);
    }
    CUDA_CHECK(ESCHAM_MEMCPY_TO_SYMBOL(c_cur2, cur2, sizeof(cur2)));
    CUDA_CHECK(ESCHAM_MEMCPY_TO_SYMBOL(c_prev2, prev2, sizeof(prev2)));
    CUDA_CHECK(ESCHAM_MEMCPY_TO_SYMBOL(c_sh2, sh2, sizeof(sh2)));
    CUDA_CHECK(ESCHAM_MEMCPY_TO_SYMBOL(c_cur3, cur3, sizeof(cur3)));
    CUDA_CHECK(ESCHAM_MEMCPY_TO_SYMBOL(c_prev3, prev3, sizeof(prev3)));
    CUDA_CHECK(ESCHAM_MEMCPY_TO_SYMBOL(c_sh3, sh3, sizeof(sh3)));
    g_inited=true;
}

template<bool K3>
__global__ void escham_kernel(const uint16_t * codes, const float * x, float * y,
                              int ic, int oc, int nc, const float * lut){
    const int TILE_U16 = K3 ? 48 : 32;
    const int NW = K3 ? 24 : 16;
    int64_t ntiles = ic / 16;
    int b = blockIdx.x; // band
    int col_base = blockIdx.y * 4;
    int lane = threadIdx.x; // 0..15
    int col_sub = threadIdx.y; //0..3
    int j = col_base + col_sub;
    bool valid = j < nc;
    int tid = col_sub*16 + lane; // 0..63
    __shared__ uint32_t s_words[24];
    __shared__ float s_tile[256];
    const uint16_t * band = codes + (int64_t)b * ntiles * TILE_U16;
    float acc = 0.f;
    for(int64_t t=0; t<ntiles; ++t){
        if(tid < NW){
            int idx = (int)(t * TILE_U16 + tid*2);
            uint32_t w = (uint32_t)band[idx] | ((uint32_t)band[idx+1] << 16);
            s_words[tid]=w;
        }
        __syncthreads();
        if(tid < 32){
            int m = tid;
            uint8_t cur = K3 ? c_cur3[m] : c_cur2[m];
            uint8_t prv = K3 ? c_prev3[m] : c_prev2[m];
            uint8_t sh  = K3 ? c_sh3[m] : c_sh2[m];
            uint64_t pair = ((uint64_t)s_words[prv] << 32) | s_words[cur];
            #pragma unroll
            for(int s=0;s<8;++s){
                int shift = (int)sh + (K3?3:2)*s;
                uint16_t win = (uint16_t)((pair >> shift) & 0xffffu);
                float w = escham_decode(win);
                int r = c_tile_row[m*8+s];
                int c = c_tile_col[m*8+s];
                s_tile[r*16 + c]=w;
            }
        }
        __syncthreads();
        if(valid){
            #pragma unroll
            for(int r=0;r<16;++r){
                float w = s_tile[r*16 + lane];
                float xv = x[(int64_t)j * ic + t*16 + r];
                acc += w * xv;
            }
        }
        __syncthreads();
    }
    if(valid){
        y[(int64_t)j * oc + b*16 + lane] = acc;
    }
}


// ---------------------------------------------------------------------------
// Chemin rapide : une tuile par warp, zero shared, zero barriere.
//
// Ce que le kernel d'origine faisait mal, mesure a 1,05 t/s en generation :
// 16 threads utiles sur 64, 320 blocs sur 82 SM, deux __syncthreads() par
// tuile de 16x16, et les 16 lanes relisant les memes 16 valeurs de x depuis
// la memoire globale a chaque tuile.
//
// La geometrie de la tuile rend tout ca inutile. Le thread m (0..31) decode 8
// poids aux positions (r0 + {9,8,1,0}, c0 + {8,8,8,8,0,0,0,0} + d) avec
// r0 = 2*(m&3), c0 = 2*((m>>3)&1) + 4*((m>>4)&1), d = (m>>2)&1. Donc :
//   - un thread ne touche que DEUX colonnes de sortie, cA = c0+d et cB = cA+8 ;
//   - ce couple ne depend pas de la tuile, donc les accumulateurs vivent en
//     registres sur toute la bande et la reduction n'a lieu qu'une fois ;
//   - les 4 threads d'un meme groupe (m>>2) visent le meme couple, la reduction
//     est un __shfl_down_sync de largeur 4.
// Les mots de code et les 16 valeurs de x circulent par __shfl_sync : lane l
// charge le mot l (resp. x[t*16+l]) et les autres se servent. Une seule lecture
// coalescee par tuile pour chacun.
// ---------------------------------------------------------------------------

template<bool K3, int NJ>
__global__ void __launch_bounds__(256)
escham_kernel_fast(const uint16_t * __restrict__ codes,
                   const float * __restrict__ x,
                   float * __restrict__ y,
                   int ic, int oc, int nc,
                   const float * __restrict__ lut,
                   int tiles_per_chunk){
    const int TILE_U16 = K3 ? 48 : 32;
    const int NW       = K3 ? 24 : 16;
    const int ntiles   = ic / 16;

    const int band = blockIdx.x;
    const int j0   = blockIdx.y * NJ;
    const int t_begin = (int)blockIdx.z * tiles_per_chunk;
    const int t_end   = min(ntiles, t_begin + tiles_per_chunk);

    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;

    const int m  = lane;
    const int r0 = 2*(m & 3);
    const int c0 = 2*((m>>3)&1) + 4*((m>>4)&1);
    const int d  = (m>>2)&1;
    const int cA = c0 + d;        // s = 4..7
    const int cB = c0 + 8 + d;    // s = 0..3

    int cur, prv, sh;
    if (K3) {
        const int b = m*24, r86 = b + 791, r87 = r86 & 2016;
        cur = (((r86>>3)&252)-96)/4;
        prv = (m==0) ? 23 : (((b+755)>>5)-24);
        sh  = r87 - b - 760;
    } else {
        cur = (m>>1)&15;
        prv = (cur-1)&15;
        sh  = (m&1)==0 ? 16 : 0;
    }

    // TILE_U16 vaut 32 ou 48, donc l'offset de bande est un multiple de 32
    // uint16 : la vue 32 bits est alignee. nvcc ne peut pas le prouver a travers
    // un parametre pointeur et emettait deux LDG.U16 par mot.
    const uint32_t * band32 = reinterpret_cast<const uint32_t *>(codes)
                            + (int64_t)band * ntiles * NW;

    float accA[NJ], accB[NJ];
#pragma unroll
    for (int u = 0; u < NJ; ++u) { accA[u] = 0.f; accB[u] = 0.f; }

    // UNR tuiles par iteration. Les chargements des UNR tuiles sont emis avant
    // toute utilisation : la chaine charge -> shuffle -> decode -> FMA est
    // entierement dependante, un seul flux en vol laisserait le warp bloque sur
    // la latence memoire. Les bornes ne dependent que de (t, q, nwarps), donc
    // elles sont uniformes sur le warp et les __shfl_sync restent legitimes.
    const int UNR = 2;

    for (int t = t_begin + warp; t < t_end; t += UNR*nwarps) {
        uint32_t wrd[UNR];
        float2   xv [UNR][NJ];   // x[r0+0], x[r0+1]
        float2   xv8[UNR][NJ];   // x[r0+8], x[r0+9]
#pragma unroll
        for (int q = 0; q < UNR; ++q) {
            const int tq = t + q*nwarps;
            const bool live = tq < t_end;
            wrd[q] = 0;
            if (live && lane < NW) {
                wrd[q] = band32[tq*NW + lane];   // NW == TILE_U16/2
            }
#pragma unroll
            for (int u = 0; u < NJ; ++u) {
                const int j = j0 + u;
                // Les 4 lignes dont ce lane a besoin sont r0+{0,1,8,9} : deux
                // couples adjacents. Deux lectures 64 bits remplacent une
                // lecture 32 bits suivie de quatre __shfl_sync. r0 est pair et
                // tq*16 est aligne sur 64 octets, donc float2 est aligne.
                if (live && j < nc) {
                    const float * xb = x + (int64_t)j*ic + tq*16 + r0;
                    xv[q][u]   = *reinterpret_cast<const float2 *>(xb);
                    xv8[q][u]  = *reinterpret_cast<const float2 *>(xb + 8);
                } else {
                    xv[q][u]  = make_float2(0.f, 0.f);
                    xv8[q][u] = make_float2(0.f, 0.f);
                }
            }
        }

#pragma unroll
        for (int q = 0; q < UNR; ++q) {
            // La largeur est passee explicitement : vendors/hip.h definit
            // __shfl_sync comme une macro a QUATRE arguments, l'omettre ne
            // compile pas en HIP. Sous CUDA c'est la valeur par defaut.
            const uint64_t pair = ((uint64_t)__shfl_sync(0xffffffffu, wrd[q], prv, WARP_SIZE) << 32)
                                |  (uint64_t)__shfl_sync(0xffffffffu, wrd[q], cur, WARP_SIZE);
            // Extraction 32 bits. Un decalage sur 64 bits coute deux
            // instructions ALU (SHF.R.U64 + SHF.R.U32.HI) et le pipe ALU est
            // le goulot (69,8 % au ncu). Les 8 fenetres d'une tuile tiennent
            // dans une vue 32 bits pour K=2 (sh in {0,16}, bit max sh+29) et
            // dans deux vues pour K=3 (sh in {0,8,24}, bit max sh+36) : un ou
            // deux decalages 64 bits par tuile au lieu de huit.
            const uint32_t v0 = (uint32_t)(pair >> sh);
            const uint32_t v1 = K3 ? (uint32_t)(pair >> (sh + 18)) : 0u;
            float wv[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                uint32_t src; int off;
                if (K3) { const bool hi = s >= 6; src = hi ? v1 : v0; off = 3*s - (hi ? 18 : 0); }
                else    { src = v0; off = 2*s; }
                wv[s] = escham_decode((uint16_t)((src >> off) & 0xffffu));
            }
            // rows[] = { r0+9, r0+8, r0+1, r0+0 } dans cet ordre
#pragma unroll
            for (int u = 0; u < NJ; ++u) {
                const float xr[4] = { xv8[q][u].y, xv8[q][u].x, xv[q][u].y, xv[q][u].x };
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    accB[u] += wv[s]   * xr[s];
                    accA[u] += wv[s+4] * xr[s];
                }
            }
        }
    }

    // les 4 lanes d'un groupe (m>>2) visent le meme couple de colonnes
#pragma unroll
    for (int u = 0; u < NJ; ++u) {
        // Reduction papillon plutot que __shfl_down : HIP ne fournit pas d'alias
        // a quatre arguments pour __shfl_down_sync, alors qu'il en a un pour
        // __shfl_xor_sync. Sur une largeur de 4, xor donne la somme complete a
        // TOUTES les lanes du groupe au lieu de la seule lane 0 ; le seul
        // lecteur teste (lane & 3) == 0, donc le resultat est inchange.
        accA[u] += __shfl_xor_sync(0xffffffffu, accA[u], 2, 4);
        accB[u] += __shfl_xor_sync(0xffffffffu, accB[u], 2, 4);
        accA[u] += __shfl_xor_sync(0xffffffffu, accA[u], 1, 4);
        accB[u] += __shfl_xor_sync(0xffffffffu, accB[u], 1, 4);
    }

    // un seul atomicAdd global par (colonne, sortie) et par bloc
    __shared__ float s_acc[NJ][16];
    for (int i = threadIdx.x; i < NJ*16; i += blockDim.x) {
        s_acc[i>>4][i&15] = 0.f;
    }
    __syncthreads();
    if ((lane & 3) == 0) {
#pragma unroll
        for (int u = 0; u < NJ; ++u) {
            atomicAdd(&s_acc[u][cA], accA[u]);
            atomicAdd(&s_acc[u][cB], accB[u]);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NJ*16; i += blockDim.x) {
        const int u = i >> 4, c = i & 15;
        const int j = j0 + u;
        if (j < nc) {
            atomicAdd(&y[(int64_t)j*oc + band*16 + c], s_acc[u][c]);
        }
    }
}

// ---------------------------------------------------------------------------
// Noyau prefill M7 : identique a escham_kernel_fast mais l'etage x vit en
// memoire partagee au lieu des registres. Chaque lane gardait xv[UNR][NJ] +
// xv8[UNR][NJ] = NJ*8*UNR flottants ; a NJ=16 et UNR=2 cela fait 256 registres
// de trop. La tuile x (UNR * NJ * 16 valeurs) est chargee cooperativement une
// fois par iteration et relue par toutes les lanes : les memes valeurs de x
// servent toutes les bandes. Seuls accA[NJ]/accB[NJ] restent en registres.
// ---------------------------------------------------------------------------

template<bool K3, int NJ, int NWARPS>
__global__ void __launch_bounds__(NWARPS*32)
escham_kernel_prefill(const uint16_t * __restrict__ codes,
                      const float * __restrict__ x,
                      float * __restrict__ y,
                      int ic, int oc, int nc,
                      const float * __restrict__ lut,
                      int tiles_per_chunk){
    const int NW       = K3 ? 24 : 16;
    const int ntiles   = ic / 16;

    const int band = blockIdx.x;
    const int j0   = blockIdx.y * NJ;
    const int t_begin = (int)blockIdx.z * tiles_per_chunk;
    const int t_end   = min(ntiles, t_begin + tiles_per_chunk);

    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;

    const int m  = lane;
    const int r0 = 2*(m & 3);
    const int c0 = 2*((m>>3)&1) + 4*((m>>4)&1);
    const int d  = (m>>2)&1;
    const int cA = c0 + d;
    const int cB = c0 + 8 + d;

    int cur, prv, sh;
    if (K3) {
        const int b = m*24, r86 = b + 791, r87 = r86 & 2016;
        cur = (((r86>>3)&252)-96)/4;
        prv = (m==0) ? 23 : (((b+755)>>5)-24);
        sh  = r87 - b - 760;
    } else {
        cur = (m>>1)&15;
        prv = (cur-1)&15;
        sh  = (m&1)==0 ? 16 : 0;
    }

    const uint32_t * band32 = reinterpret_cast<const uint32_t *>(codes)
                            + (int64_t)band * ntiles * NW;

    float accA[NJ], accB[NJ];
#pragma unroll
    for (int u = 0; u < NJ; ++u) { accA[u] = 0.f; accB[u] = 0.f; }

    const int UNR = 2;
    // tuile x partagee, UNE TRANCHE PAR WARP : chaque warp itere sur un t
    // different, ils ne doivent pas se marcher dessus. La taille derive de
    // NWARPS (parametre de template) : le partage et le bloc ne peuvent plus
    // diverger. V5 : NJ=32, UNR=2, NWARPS=4 -> 4*2*32*16*4 o = 16 Kio.
    __shared__ float s_xt[NWARPS*UNR*NJ*16];
    float * xt_w = &s_xt[warp*UNR*NJ*16];

    for (int t = t_begin + warp; t < t_end; t += UNR*nwarps) {
        uint32_t wrd[UNR];
#pragma unroll
        for (int q = 0; q < UNR; ++q) {
            const int tq = t + q*nwarps;
            const bool live = tq < t_end;
            wrd[q] = 0;
            if (live && lane < NW) {
                wrd[q] = band32[tq*NW + lane];
            }
        }
        // chargement cooperatif de la tuile x, intra-warp, VECTORISE float4.
        // tq*16 est aligne 64 o et j*ic l'est aussi (ic%16==0), donc float4 ok.
        {
#pragma unroll
            for (int q = 0; q < UNR; ++q) {
                const int tq = t + q*nwarps;
                const bool live = tq < t_end;
                // NJ*16 flottants = NJ*4 float4 ; NJ=32 -> 128 float4 sur 32 lanes
                for (int li = lane; li < NJ*4; li += WARP_SIZE) {
                    const int u  = li >> 2;
                    const int r  = (li & 3) * 4;
                    const int j  = j0 + u;
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (live && j < nc) {
                        v = *reinterpret_cast<const float4 *>(x + (int64_t)j*ic + tq*16 + r);
                    }
                    *reinterpret_cast<float4 *>(&xt_w[q*NJ*16 + li*4]) = v;
                }
            }
        }
        __syncwarp();

#pragma unroll
        for (int q = 0; q < UNR; ++q) {
            const uint64_t pair = ((uint64_t)__shfl_sync(0xffffffffu, wrd[q], prv, WARP_SIZE) << 32)
                                |  (uint64_t)__shfl_sync(0xffffffffu, wrd[q], cur, WARP_SIZE);
            const uint32_t v0 = (uint32_t)(pair >> sh);
            const uint32_t v1 = K3 ? (uint32_t)(pair >> (sh + 18)) : 0u;
            float wv[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                uint32_t src; int off;
                if (K3) { const bool hi = s >= 6; src = hi ? v1 : v0; off = 3*s - (hi ? 18 : 0); }
                else    { src = v0; off = 2*s; }
                wv[s] = escham_decode((uint16_t)((src >> off) & 0xffffu));
            }
            const float * xu = &xt_w[q*NJ*16];
#pragma unroll
            for (int u = 0; u < NJ; ++u) {
                const float xr[4] = { xu[u*16 + r0 + 9], xu[u*16 + r0 + 8],
                                      xu[u*16 + r0 + 1], xu[u*16 + r0 + 0] };
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    accB[u] += wv[s]   * xr[s];
                    accA[u] += wv[s+4] * xr[s];
                }
            }
        }
        __syncwarp();
    }

#pragma unroll
    for (int u = 0; u < NJ; ++u) {
        accA[u] += __shfl_xor_sync(0xffffffffu, accA[u], 2, 4);
        accB[u] += __shfl_xor_sync(0xffffffffu, accB[u], 2, 4);
        accA[u] += __shfl_xor_sync(0xffffffffu, accA[u], 1, 4);
        accB[u] += __shfl_xor_sync(0xffffffffu, accB[u], 1, 4);
    }

    __shared__ float s_acc[NJ][16];
    for (int i = threadIdx.x; i < NJ*16; i += blockDim.x) {
        s_acc[i>>4][i&15] = 0.f;
    }
    __syncthreads();
    if ((lane & 3) == 0) {
#pragma unroll
        for (int u = 0; u < NJ; ++u) {
            atomicAdd(&s_acc[u][cA], accA[u]);
            atomicAdd(&s_acc[u][cB], accB[u]);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NJ*16; i += blockDim.x) {
        const int u = i >> 4, c = i & 15;
        const int j = j0 + u;
        if (j < nc) {
            atomicAdd(&y[(int64_t)j*oc + band*16 + c], s_acc[u][c]);
        }
    }
}

template<bool K3, int NJ>
static void escham_launch_fast(const uint16_t * codes, const float * x, float * y,
                               int64_t ic, int64_t oc, int64_t nc, cudaStream_t st){
    const int ntiles = (int)(ic/16);
    const int nbands = (int)(oc/16);
    const int njb    = (int)((nc + NJ - 1)/NJ);
    // on vise ~2048 blocs pour remplir la carte ; en generation de token njb == 1
    // et sans decoupage de ic on ne lancerait que `nbands` blocs.
    int chunks = 2048 / (nbands*njb);
    if (chunks < 1) chunks = 1;
    if (chunks > ntiles/8 + 1) chunks = ntiles/8 + 1;
    const int tiles_per_chunk = (ntiles + chunks - 1)/chunks;
    chunks = (ntiles + tiles_per_chunk - 1)/tiles_per_chunk;
    dim3 grid((unsigned)nbands, (unsigned)njb, (unsigned)chunks);
    escham_kernel_fast<K3, NJ><<<grid, 256, 0, st>>>(codes, x, y, (int)ic, (int)oc, (int)nc,
                                                     d_lut, tiles_per_chunk);
}

// M7 : launcher du noyau prefill. Meme decoupage de ic que escham_launch_fast.
template<bool K3, int NJ, int NWARPS>
static void escham_launch_prefill(const uint16_t * codes, const float * x, float * y,
                                  int64_t ic, int64_t oc, int64_t nc, cudaStream_t st){
    const int ntiles = (int)(ic/16);
    const int nbands = (int)(oc/16);
    const int njb    = (int)((nc + NJ - 1)/NJ);
    int chunks = 2048 / (nbands*njb);
    if (chunks < 1) chunks = 1;
    if (chunks > ntiles/8 + 1) chunks = ntiles/8 + 1;
    const int tiles_per_chunk = (ntiles + chunks - 1)/chunks;
    chunks = (ntiles + tiles_per_chunk - 1)/tiles_per_chunk;
    dim3 grid((unsigned)nbands, (unsigned)njb, (unsigned)chunks);
    escham_kernel_prefill<K3, NJ, NWARPS><<<grid, NWARPS*32, 0, st>>>(codes, x, y, (int)ic, (int)oc, (int)nc,
                                                      d_lut, tiles_per_chunk);
}

static __global__ void escham_zero_kernel(float * y, int64_t n){
    for (int64_t i = (int64_t)blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x*blockDim.x) {
        y[i] = 0.f;
    }
}

// M7 : le bloc prefill decode NJ colonnes qu'elles soient vivantes ou non. Le
// cout suit donc njb*NJ et non nc, et un NJ fixe gaspille tout ce qui depasse.
// On maximise eff(NJ) * nc / (njb*NJ), ou eff est le debit a bloc plein mesure
// sur 3090 (NJ = 4 -> 99, 8 -> 144, 16 -> 169, 32 -> 201 t/s) et interpole
// lineairement entre ces points.
//
// ATTENTION : eff est un ajustement CUDA/3090. Sur gfx1151 la courbe a une
// autre forme (LDS, wave64, banc de registres) et le classement peut differer.
// Le choix y reste correct — test-escha-nj passe — mais pas forcement optimal.
int ggml_cuda_escham_prefill_nj(int64_t nc){
    static const int   NJC[] = {   4,   6,   8,  10,  12,  14,  16,  20,  24,  28,  32 };
    static const float EFF[] = { 99.f,122.f,144.f,150.f,156.f,163.f,169.f,177.f,185.f,193.f,201.f };
    const int NCAND = (int)(sizeof(NJC)/sizeof(NJC[0]));
    int   bestNJ = 32;
    float bestSc = -1.f;
    for (int i = 0; i < NCAND; ++i) {
        const int nj    = NJC[i];
        const int slots = (int)((nc + nj - 1)/nj)*nj;
        const float sc  = EFF[i] * (float)nc / (float)slots;
        if (sc > bestSc) { bestSc = sc; bestNJ = nj; }
    }
    return bestNJ;
}

void ggml_cuda_escham_mul_mat_raw(const void * src0_data, const void * src1_data, void * dst_data,
                                  int64_t ic, int64_t oc, int64_t nc, int is_k3, void * stream){
    escham_cuda_init_tables();
    cudaStream_t st = stream ? (cudaStream_t)stream : nullptr;
    // choose stream: if nullptr use default via ggml cuda pool, but raw uses default
    const uint16_t * codes = (const uint16_t*)src0_data;
    const float * x = (const float*)src1_data;
    float * y = (float*)dst_data;
    int64_t nbands = oc/16;
    dim3 block(16,4);
    dim3 grid((unsigned)nbands, (unsigned)((nc+3)/4));
    // le chemin rapide accumule : il faut partir de zero.
    // Mise a zero par noyau, pas par cudaMemsetAsync.
    //
    // Pourquoi : hipMemsetAsync refuse un pointeur qui n'est pas de la memoire
    // peripherique — « invalid argument » — la ou un noyau ecrit sans probleme
    // des que la memoire est mappee (constate sur gfx1151, hipPointerGetAttributes
    // rendant type=0, non enregistre). CUDA tolere le meme cas par l'acces hote
    // transparent du pilote NVIDIA. Un noyau marche des deux cotes ; le cout est
    // negligeable devant le produit matriciel qui suit.
    {
        const int64_t n = nc*oc;
        const int nb = (int)((n + 255)/256);
        escham_zero_kernel<<<nb > 65535 ? 65535 : nb, 256, 0, st>>>(y, n);
        CUDA_CHECK(cudaGetLastError());
    }
    if(nc == 1){
        if(is_k3) escham_launch_fast<true , 1>(codes,x,y,ic,oc,nc,st);
        else      escham_launch_fast<false, 1>(codes,x,y,ic,oc,nc,st);
    } else {
        const int bestNJ = ggml_cuda_escham_prefill_nj(nc);
        switch (bestNJ) {
            case  4: if(is_k3) escham_launch_prefill<true ,  4, 4>(codes,x,y,ic,oc,nc,st);
                     else      escham_launch_prefill<false,  4, 4>(codes,x,y,ic,oc,nc,st); break;
            case  6: if(is_k3) escham_launch_prefill<true ,  6, 4>(codes,x,y,ic,oc,nc,st);
                     else      escham_launch_prefill<false,  6, 4>(codes,x,y,ic,oc,nc,st); break;
            case  8: if(is_k3) escham_launch_prefill<true ,  8, 4>(codes,x,y,ic,oc,nc,st);
                     else      escham_launch_prefill<false,  8, 4>(codes,x,y,ic,oc,nc,st); break;
            case 10: if(is_k3) escham_launch_prefill<true , 10, 4>(codes,x,y,ic,oc,nc,st);
                     else      escham_launch_prefill<false, 10, 4>(codes,x,y,ic,oc,nc,st); break;
            case 12: if(is_k3) escham_launch_prefill<true , 12, 4>(codes,x,y,ic,oc,nc,st);
                     else      escham_launch_prefill<false, 12, 4>(codes,x,y,ic,oc,nc,st); break;
            case 14: if(is_k3) escham_launch_prefill<true , 14, 4>(codes,x,y,ic,oc,nc,st);
                     else      escham_launch_prefill<false, 14, 4>(codes,x,y,ic,oc,nc,st); break;
            case 16: if(is_k3) escham_launch_prefill<true , 16, 4>(codes,x,y,ic,oc,nc,st);
                     else      escham_launch_prefill<false, 16, 4>(codes,x,y,ic,oc,nc,st); break;
            case 20: if(is_k3) escham_launch_prefill<true , 20, 4>(codes,x,y,ic,oc,nc,st);
                     else      escham_launch_prefill<false, 20, 4>(codes,x,y,ic,oc,nc,st); break;
            case 24: if(is_k3) escham_launch_prefill<true , 24, 4>(codes,x,y,ic,oc,nc,st);
                     else      escham_launch_prefill<false, 24, 4>(codes,x,y,ic,oc,nc,st); break;
            case 28: if(is_k3) escham_launch_prefill<true , 28, 4>(codes,x,y,ic,oc,nc,st);
                     else      escham_launch_prefill<false, 28, 4>(codes,x,y,ic,oc,nc,st); break;
            case 32: if(is_k3) escham_launch_prefill<true , 32, 4>(codes,x,y,ic,oc,nc,st);
                     else      escham_launch_prefill<false, 32, 4>(codes,x,y,ic,oc,nc,st); break;
            default: if(is_k3) escham_launch_prefill<true , 32, 4>(codes,x,y,ic,oc,nc,st);
                     else      escham_launch_prefill<false, 32, 4>(codes,x,y,ic,oc,nc,st); break;
        }
    }
    (void)grid; (void)block; (void)nbands;
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_escham_supports_type(int ggml_type){
    return ggml_type==GGML_TYPE_ESCHAM_2 || ggml_type==GGML_TYPE_ESCHAM_3;
}

void ggml_cuda_escham_mul_mat(ggml_backend_cuda_context & ctx,
                              const ggml_tensor * src0,
                              const ggml_tensor * src1,
                              ggml_tensor * dst){
    GGML_ASSERT(src1->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst));
    int64_t ic = src0->ne[0];
    int64_t oc = src0->ne[1];
    int64_t nc = src1->ne[1];
    GGML_ASSERT(src1->ne[0]==ic && dst->ne[0]==oc && dst->ne[1]==nc);
    GGML_ASSERT(ic%16==0 && oc%16==0);
    int is_k3 = src0->type==GGML_TYPE_ESCHAM_3;
    cudaStream_t stream = ctx.stream();
    ggml_cuda_escham_mul_mat_raw(src0->data, src1->data, dst->data, ic, oc, nc, is_k3, (void*)stream);
}
