#include "hc-combine.cuh"

#define HC_COMBINE_MAX_HC 8

static __global__ void hc_combine_kernel(
        const float * __restrict__ x, const float * __restrict__ residual,
        const float * __restrict__ post, const float * __restrict__ comb,
        float * __restrict__ dst,
        const int64_t n_embd, const int64_t nt, const int64_t hc) {

    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n_embd*nt) {
        return;
    }

    const int64_t e = idx % n_embd;
    const int64_t t = idx / n_embd;

    const float x_v = x[e + t*n_embd];

    const float * post_row = post + t*hc;
    const float * comb_row = comb + t*hc*hc;
    const float * res_row  = residual + t*n_embd*hc;
    float       * dst_row  = dst + t*n_embd*hc;

    float res_v[HC_COMBINE_MAX_HC];
    for (int64_t src = 0; src < hc; ++src) {
        res_v[src] = res_row[e + src*n_embd];
    }

    for (int64_t d = 0; d < hc; ++d) {
        float out_v = x_v*post_row[d];
        for (int64_t src = 0; src < hc; ++src) {
            out_v += res_v[src]*comb_row[d + src*hc];
        }
        dst_row[e + d*n_embd] = out_v;
    }
}

void ggml_cuda_hc_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(residual));
    GGML_ASSERT(ggml_is_contiguous(post));
    GGML_ASSERT(ggml_is_contiguous(comb));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd = x->ne[0];
    const int64_t nt     = x->ne[1];
    const int64_t hc     = residual->ne[1];

    GGML_ASSERT(hc <= HC_COMBINE_MAX_HC);

    const float * x_d   = (const float *) x->data;
    const float * res_d = (const float *) residual->data;
    const float * post_d = (const float *) post->data;
    const float * comb_d = (const float *) comb->data;
    float       * dst_d = (      float *) dst->data;

    const int64_t n_elem     = n_embd*nt;
    const int     block_size = 256;
    const int     n_blocks   = (n_elem + block_size - 1)/block_size;

    hc_combine_kernel<<<n_blocks, block_size, 0, ctx.stream()>>>(
            x_d, res_d, post_d, comb_d, dst_d, n_embd, nt, hc);
}
