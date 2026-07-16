#include "hc-sinkhorn.cuh"

#define HC_SINKHORN_MAX_HC 8

static __global__ void hc_sinkhorn_kernel(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t hc, const int64_t nt, const int n_iters, const float eps) {

    const int64_t t = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= nt) {
        return;
    }

    const float * row_src = src + t*hc*hc;
    float         c[HC_SINKHORN_MAX_HC][HC_SINKHORN_MAX_HC];

    for (int64_t s = 0; s < hc; ++s) {
        float max_v = -INFINITY;
        for (int64_t d = 0; d < hc; ++d) {
            max_v = fmaxf(max_v, row_src[d + s*hc]);
        }
        float sum_v = 0.0f;
        for (int64_t d = 0; d < hc; ++d) {
            const float v = expf(row_src[d + s*hc] - max_v);
            c[d][s] = v;
            sum_v += v;
        }
        for (int64_t d = 0; d < hc; ++d) {
            c[d][s] = c[d][s]/sum_v + eps;
        }
    }

    for (int it = 0; it < n_iters; ++it) {
        if (it > 0) {
            for (int64_t s = 0; s < hc; ++s) {
                float sum_v = 0.0f;
                for (int64_t d = 0; d < hc; ++d) {
                    sum_v += c[d][s];
                }
                sum_v += eps;
                for (int64_t d = 0; d < hc; ++d) {
                    c[d][s] /= sum_v;
                }
            }
        }

        for (int64_t d = 0; d < hc; ++d) {
            float sum_v = 0.0f;
            for (int64_t s = 0; s < hc; ++s) {
                sum_v += c[d][s];
            }
            sum_v += eps;
            for (int64_t s = 0; s < hc; ++s) {
                c[d][s] /= sum_v;
            }
        }
    }

    float * row_dst = dst + t*hc*hc;
    for (int64_t s = 0; s < hc; ++s) {
        for (int64_t d = 0; d < hc; ++d) {
            row_dst[d + s*hc] = c[d][s];
        }
    }
}

void ggml_cuda_hc_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(src0->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t hc = src0->ne[0];
    const int64_t nt = src0->ne[2]*src0->ne[3];

    GGML_ASSERT(src0->ne[1] == hc);
    GGML_ASSERT(hc <= HC_SINKHORN_MAX_HC);

    const int   n_iters = ggml_get_op_params_i32(dst, 0);
    const float eps     = ggml_get_op_params_f32(dst, 1);

    const float * src_d = (const float *) src0->data;
    float       * dst_d = (      float *) dst->data;

    const int block_size = 256;
    const int n_blocks   = (nt + block_size - 1)/block_size;

    hc_sinkhorn_kernel<<<n_blocks, block_size, 0, ctx.stream()>>>(src_d, dst_d, hc, nt, n_iters, eps);
}
