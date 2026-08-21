// M5-C test CPU vs CUDA — 8 formes reelles (IC {5120,6144,17408}, OC dans {1024,5120,6144,10240,12288,17408})
// Passe par ggml backend CPU et CUDA si disponible. Affiche rms relatif brut.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-escham.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define FIXTURE_DEFAULT "/home/yaniss/hermes-work/escha/port/fixture_m4.bin"
#define FIXTURE_MAGIC 0x45534D34u
#define CRIT 1e-3f

static const uint8_t *cur;
static const uint8_t *take(size_t n){ const uint8_t *p=cur; cur+=n; return p; }
static uint32_t rd_u32(void){ uint32_t v; memcpy(&v,take(4),4); return v; }
static float rel_rms(const float *a, const float *b, size_t n){
    double se=0,sb=0;
    for(size_t i=0;i<n;++i){ double d=(double)a[i]-(double)b[i]; se+=d*d; sb+=(double)b[i]*(double)b[i]; }
    return (float)(sqrt(se/n)/(sqrt(sb/n)+1e-12));
}
static void *read_all(const char *path, size_t *n){
    FILE *f=fopen(path,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",path); return NULL;}
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    void *buf=malloc((size_t)sz); if(!buf){fclose(f);return NULL;}
    if(fread(buf,1,(size_t)sz,f)!=(size_t)sz){fclose(f);free(buf);return NULL;}
    fclose(f); *n=(size_t)sz; return buf;
}
struct Case { const char *name; int K, IC, OC, NR; const uint16_t *packed; const float *x,*rin,*rout,*bias,*ref_y; };

static int test_one(struct Case c){
    int64_t itc=c.IC/16, otc=c.OC/16;
    size_t tile_u16=(size_t)16*c.K;
    uint16_t *band=(uint16_t*)malloc((size_t)itc*otc*tile_u16*sizeof(uint16_t));
    if(!band) return 1;
    for(int64_t ot=0;ot<otc;++ot) for(int64_t it=0;it<itc;++it)
        memcpy(band+(ot*itc+it)*tile_u16, c.packed+(it*otc+ot)*tile_u16, tile_u16*sizeof(uint16_t));
    float *xp=(float*)malloc((size_t)c.IC*c.NR*sizeof(float));
    if(!xp){ free(band); return 1; }
    for(int64_t j=0;j<c.NR;++j){ float *col=xp+j*c.IC; for(int64_t i=0;i<c.IC;++i) col[i]=c.x[i*c.NR+j]*c.rin[i]; escham_vec_had128_f32(col,c.IC); }
    float *y_cpu=(float*)malloc((size_t)c.OC*c.NR*sizeof(float));
    float *y_cuda=(float*)malloc((size_t)c.OC*c.NR*sizeof(float));
    if(!y_cpu||!y_cuda){ free(band); free(xp); free(y_cpu); free(y_cuda); return 1; }

    auto run_backend=[&](ggml_backend_t be, float *out)->int{
        size_t pool=(size_t)itc*otc*tile_u16*2 + (size_t)(c.IC+c.OC)*c.NR*4*4 + 64ull*1024*1024;
        struct ggml_init_params ip={pool,NULL,false};
        struct ggml_context *ctx=ggml_init(ip);
        struct ggml_tensor *tw=ggml_new_tensor_2d(ctx, (c.K==3)?GGML_TYPE_ESCHAM_3:GGML_TYPE_ESCHAM_2, c.IC, c.OC);
        if(ggml_nbytes(tw)!=(size_t)itc*otc*tile_u16*sizeof(uint16_t)){ fprintf(stderr,"bytes mismatch\n"); ggml_free(ctx); return 1; }
        memcpy(tw->data, band, ggml_nbytes(tw));
        struct ggml_tensor *txp=ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.IC, c.NR);
        memcpy(txp->data, xp, (size_t)c.IC*c.NR*sizeof(float));
        struct ggml_tensor *z=ggml_mul_mat(ctx, tw, txp);
        struct ggml_cgraph *gf=ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, z);
        enum ggml_status st=ggml_backend_graph_compute(be, gf);
        if(st!=GGML_STATUS_SUCCESS){ fprintf(stderr,"graph compute failed %d\n",(int)st); ggml_free(ctx); return 1; }
        for(int64_t j=0;j<c.NR;++j){ float *col=(float*)z->data+j*c.OC; escham_vec_had128_f32(col,c.OC); for(int64_t o=0;o<c.OC;++o) out[o*c.NR+j]=col[o]*c.rout[o]+c.bias[o]; }
        ggml_free(ctx);
        return 0;
    };

    ggml_backend_t cpu=ggml_backend_cpu_init();
    int rc=run_backend(cpu,y_cpu);
    ggml_backend_free(cpu);
    if(rc){ free(band); free(xp); free(y_cpu); free(y_cuda); return 1; }
    float rms_ref=rel_rms(y_cpu, c.ref_y, (size_t)c.OC*c.NR);

#if defined(GGML_USE_CUDA)
#ifndef GGML_CUDA
#define GGML_CUDA 1
#endif
#else
#ifndef GGML_CUDA
#define GGML_CUDA 0
#endif
#endif
#if GGML_CUDA
    ggml_backend_t cuda_be=nullptr;
    bool have_cuda=false;
    // ggml-cuda is linked (see ldd) — call directly; weak check was dropping the symbol due to --as-needed elision
    cuda_be=ggml_backend_cuda_init(0); have_cuda=cuda_be!=nullptr;
    if(have_cuda){
        rc=run_backend(cuda_be,y_cuda);
        ggml_backend_free(cuda_be);
        if(rc){ printf("%s: K=%d IC=%d OC=%d NR=%d  cpu_vs_ref %.3e  cuda FAIL\n", c.name,c.K,c.IC,c.OC,c.NR,rms_ref); free(band); free(xp); free(y_cpu); free(y_cuda); return 1; }
        float rms_cuda=rel_rms(y_cuda, c.ref_y, (size_t)c.OC*c.NR);
        float rms_cc=rel_rms(y_cuda, y_cpu, (size_t)c.OC*c.NR);
        printf("%s: K=%d IC=%d OC=%d NR=%d  cpu_vs_ref %.3e  cuda_vs_ref %.3e  cpu_vs_cuda %.3e  %s\n",
               c.name,c.K,c.IC,c.OC,c.NR,rms_ref,rms_cuda,rms_cc,(rms_cc<CRIT && rms_cuda<CRIT)?"OK":"FAIL");
        free(band); free(xp); free(y_cpu); free(y_cuda);
        return (rms_cc<CRIT && rms_cuda<CRIT)?0:1;
    } else {
        printf("%s: K=%d IC=%d OC=%d NR=%d  cpu_vs_ref %.3e  cuda N/A  %s\n", c.name,c.K,c.IC,c.OC,c.NR,rms_ref,(rms_ref<CRIT)?"OK":"FAIL");
        free(band); free(xp); free(y_cpu); free(y_cuda);
        return (rms_ref<CRIT)?0:1;
    }
#else
    printf("%s: K=%d IC=%d OC=%d NR=%d  cpu_vs_ref %.3e  cuda N/A (GGML_CUDA=OFF)  %s\n", c.name,c.K,c.IC,c.OC,c.NR,rms_ref,(rms_ref<CRIT)?"OK":"FAIL");
    free(band); free(xp); free(y_cpu); free(y_cuda);
    return (rms_ref<CRIT)?0:1;
#endif
}
int main(int argc,char**argv){
    const char *path= argc>1?argv[1]:FIXTURE_DEFAULT;
    size_t n=0; uint8_t *buf=(uint8_t*)read_all(path,&n); if(!buf) return 1; cur=buf;
    if(rd_u32()!=FIXTURE_MAGIC){ fprintf(stderr,"bad magic\n"); return 1; }
    (void)rd_u32(); uint32_t nrec=rd_u32();
    int fails=0;
    for(uint32_t r=0;r<nrec;++r){
        uint32_t nl=rd_u32();
        char name[512]; uint32_t cl=nl<sizeof(name)-1?nl:(uint32_t)sizeof(name)-1; memcpy(name,take(nl),nl); name[cl]=0;
        int32_t hdr[5]; memcpy(hdr,take(sizeof(hdr)),sizeof(hdr));
        int K=hdr[0],IC=hdr[1],OC=hdr[2],NT=hdr[3],NR=hdr[4];
        take((size_t)NT*16*K*sizeof(uint16_t));
        take((size_t)NT*256*sizeof(uint16_t));
        const uint16_t *packed=(const uint16_t*)take((size_t)IC*OC*K/16*sizeof(uint16_t));
        const float *x=(const float*)take((size_t)IC*NR*sizeof(float));
        const float *rin=(const float*)take((size_t)IC*sizeof(float));
        const float *rout=(const float*)take((size_t)OC*sizeof(float));
        const float *bias=(const float*)take((size_t)OC*sizeof(float));
        const float *ref=(const float*)take((size_t)OC*NR*sizeof(float));
        printf("-- %s: K=%d IC=%d OC=%d NR=%d\n",name,K,IC,OC,NR);
        struct Case c={(const char*)strdup(name),K,IC,OC,NR,packed,x,rin,rout,bias,ref};
        fails+=test_one(c);
        free((void*)c.name);
    }
    free(buf);
    printf("\nverdict: %s (%d fails)\n", fails?"FAIL":"PASS",fails);
    return fails?1:0;
}
