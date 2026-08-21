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

// device constant tables (small)
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
    CUDA_CHECK(cudaMemcpyToSymbol(c_tile_row, h_row, sizeof(h_row)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tile_col, h_col, sizeof(h_col)));
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
    CUDA_CHECK(cudaMemcpyToSymbol(c_cur2, cur2, sizeof(cur2)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_prev2, prev2, sizeof(prev2)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_sh2, sh2, sizeof(sh2)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_cur3, cur3, sizeof(cur3)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_prev3, prev3, sizeof(prev3)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_sh3, sh3, sizeof(sh3)));
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
                float w = lut[win];
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
    if(is_k3){
        escham_kernel<true><<<grid, block, 0, st>>>(codes,x,y,(int)ic,(int)oc,(int)nc,d_lut);
    } else {
        escham_kernel<false><<<grid, block, 0, st>>>(codes,x,y,(int)ic,(int)oc,(int)nc,d_lut);
    }
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
