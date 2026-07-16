#include "hc-wsum.cuh"

#define HC_WSUM_MAX_HC 8

static __global__ void hc_wsum_kernel(
        const float * __restrict__ x, const float * __restrict__ weights,
        float * __restrict__ dst,
        const int64_t n_embd, const int64_t nt, const int64_t hc) {

    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n_embd*nt) {
        return;
    }

    const int64_t e = idx % n_embd;
    const int64_t t = idx / n_embd;

    const float * x_row = x + t*n_embd*hc;
    const float * w_row = weights + t*hc;

    float sum_v = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        sum_v += x_row[e + ih*n_embd]*w_row[ih];
    }

    dst[e + t*n_embd] = sum_v;
}

void ggml_cuda_hc_wsum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd = x->ne[0];
    const int64_t hc     = x->ne[1];
    const int64_t nt     = x->ne[2];

    GGML_ASSERT(hc <= HC_WSUM_MAX_HC);

    const float * x_d = (const float *) x->data;
    const float * w_d = (const float *) weights->data;
    float       * dst_d = (      float *) dst->data;

    const int64_t n_elem     = n_embd*nt;
    const int     block_size = 256;
    const int     n_blocks   = (n_elem + block_size - 1)/block_size;

    hc_wsum_kernel<<<n_blocks, block_size, 0, ctx.stream()>>>(x_d, w_d, dst_d, n_embd, nt, hc);
}
