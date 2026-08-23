// Verifie le dispatch NJ du noyau prefill : pour chaque nc de 2 a 40, la
// colonne j du resultat batche doit egaler le resultat du chemin nc==1 sur
// cette meme colonne. Les codes sont aleatoires : seul compte que les deux
// noyaux voient les memes poids.
#include "../ggml/src/ggml-cuda/escham.cuh"
#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#define cudaMalloc            hipMalloc
#define cudaFree              hipFree
#define cudaMemcpy            hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#else
#include <cuda_runtime.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static float rel_rms(const float *a, const float *b, size_t n){
    double num=0, den=0;
    for(size_t i=0;i<n;++i){ double d=(double)a[i]-b[i]; num+=d*d; den+=(double)b[i]*b[i]; }
    return den>0 ? (float)sqrt(num/den) : (float)sqrt(num);
}

// Reference independante du selecteur : argmax a la main sur les memes
// candidats, pour prouver que le dispatch choisit bien ce qu'on croit.
static int expected_nj(int nc){
    static const int   NJC[] = {   4,   6,   8,  10,  12,  14,  16,  20,  24,  28,  32 };
    static const double EFF[] = { 99, 122, 144, 150, 156, 163, 169, 177, 185, 193, 201 };
    int best=32; double bs=-1;
    for(int i=0;i<11;++i){
        int slots = ((nc + NJC[i] - 1)/NJC[i])*NJC[i];
        double sc = EFF[i]*(double)nc/(double)slots;
        if(sc>bs){ bs=sc; best=NJC[i]; }
    }
    return best;
}

int main(){
    const int IC=1024, OC=512;
    int fails=0;

    // 1. le selecteur choisit-il l'argmax attendu ?
    for(int nc=2; nc<=512; ++nc){
        const int got = ggml_cuda_escham_prefill_nj(nc);
        const int exp = expected_nj(nc);
        if(got != exp){ printf("SELECT nc=%d got NJ=%d expected %d  FAIL\n", nc, got, exp); ++fails; }
    }
    printf("select: nc 2..512 %s\n", fails?"FAIL":"OK");
    // quelques points nommes, pour que le choix soit lisible dans la sortie
    for(int nc : {2,4,5,8,9,16,17,24,32,33,512})
        printf("  nc=%3d -> NJ=%d\n", nc, ggml_cuda_escham_prefill_nj(nc));

    for(int is_k3=0; is_k3<2; ++is_k3){
        const int K = is_k3 ? 3 : 2;
        const size_t ncode = (size_t)(IC/16)*(OC/16)*16*K;
        std::vector<uint16_t> h_codes(ncode);
        for(size_t i=0;i<ncode;++i) h_codes[i]=(uint16_t)(rand()&0xFFFF);
        uint16_t *d_codes; cudaMalloc(&d_codes, ncode*sizeof(uint16_t));
        cudaMemcpy(d_codes, h_codes.data(), ncode*sizeof(uint16_t), cudaMemcpyHostToDevice);

        const int NCMAX=40;
        std::vector<float> h_x((size_t)IC*NCMAX);
        for(size_t i=0;i<h_x.size();++i) h_x[i]=(float)((rand()/(double)RAND_MAX)*2.0-1.0);
        float *d_x, *d_y, *d_y1;
        cudaMalloc(&d_x, h_x.size()*sizeof(float));
        cudaMalloc(&d_y, (size_t)OC*NCMAX*sizeof(float));
        cudaMalloc(&d_y1, (size_t)OC*sizeof(float));
        cudaMemcpy(d_x, h_x.data(), h_x.size()*sizeof(float), cudaMemcpyHostToDevice);

        for(int nc=2; nc<=NCMAX; ++nc){
            ggml_cuda_escham_mul_mat_raw(d_codes, d_x, d_y, IC, OC, nc, is_k3, nullptr);
            std::vector<float> y((size_t)OC*nc);
            cudaMemcpy(y.data(), d_y, y.size()*sizeof(float), cudaMemcpyDeviceToHost);
            float worst=0;
            for(int j=0;j<nc;++j){
                ggml_cuda_escham_mul_mat_raw(d_codes, d_x+(size_t)j*IC, d_y1, IC, OC, 1, is_k3, nullptr);
                std::vector<float> y1(OC);
                cudaMemcpy(y1.data(), d_y1, OC*sizeof(float), cudaMemcpyDeviceToHost);
                float r = rel_rms(y.data()+(size_t)j*OC, y1.data(), OC);
                if(r>worst) worst=r;
            }
            // M8 : a partir de 16 colonnes le prefill passe sur les tensor cores
            // en f16. L'ecart attendu est celui de l'arrondi f16, ~2e-04, et il
            // est plat en nc. Une erreur de disposition de fragment donnerait un
            // ecart d'ordre 1, pas 2e-04 : le seuil separe bien les deux cas.
#if defined(GGML_USE_HIP)
            const float tol = 1e-5f;                 // HIP reste sur le chemin V5
#else
            const float tol = (nc >= 16) ? 1e-3f : 1e-5f;
#endif
            const bool ok = worst < tol;
            if(!ok) ++fails;
            printf("K=%d nc=%2d  worst_col_vs_nc1 %.3e  (tol %.0e)  %s\n", K, nc, worst, tol, ok?"OK":"FAIL");
        }
        cudaFree(d_codes); cudaFree(d_x); cudaFree(d_y); cudaFree(d_y1);
    }
    printf("\nverdict: %s (%d fails)\n", fails?"FAIL":"PASS", fails);
    return fails?1:0;
}
