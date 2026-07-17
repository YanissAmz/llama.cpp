#include "hc-sinkhorn.cuh"

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>

#define HC_SINKHORN_MAX_HC 8

// [TAG_HC_SINKHORN_HC4] Two variants of the same kernel, same math in the same order.
//
// The generic variant takes hc at runtime, so the per-thread working array c[MAX][MAX] is indexed
// by runtime loop variables and LLVM cannot promote it to registers: it spills to scratch (private)
// memory. At decode this kernel runs with nt == 1 -- ONE active thread, one wavefront, zero
// occupancy to hide the spill latency -- and the serial spill chain was profiled at ~90 us/call,
// 7.8 ms/tok over 87 calls/tok (~7% of decode @176k). The templated variant makes HC a
// compile-time constant (the model asserts hc == 4, src/models/deepseek4.cpp), so c[4][4] is 16
// floats in registers and the loops fully unroll.
//
// The generic variant is kept as the dispatch fallback for hc != 4 and as the same-binary A/B
// reference: DSV4_HC_SINKHORN_GENERIC=1 forces it, so the specialization can be gated without a
// rebuild.
//
// The two variants do NOT produce bit-identical output, and cannot be expected to. This TU is
// compiled -ffast-math -funsafe-math-optimizations (ggml/src/ggml-hip/CMakeLists.txt), which
// licenses reassociation, reciprocal-for-divide and FMA contraction. Only the unrolled variant,
// whose loop bounds and array indices are compile-time constants, gives the compiler the
// opportunity to apply them: it reassociates the HC-term sums, hoists one reciprocal in place of
// HC divides, and contracts. That is the same transformation that produces the speedup, so the
// divergence is inseparable from the win. Keeping the bodies textually in lockstep is still
// worthwhile for review, but it buys agreement to within fast-math rounding, NOT bit-identity:
// gate this kernel numerically (DSV4_HC_SINKHORN_XCHECK=1 below), never on a hash.
//
// The rounding is safe to accept: -fno-finite-math-only keeps the -INFINITY init and the +eps
// divide-guard intact, and Sinkhorn renormalizes every iteration, so per-iteration rounding
// cannot amplify across the n_iters steps.

template <int HC>
static __global__ void hc_sinkhorn_kernel_hc(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t nt, const int n_iters, const float eps) {

    const int64_t t = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= nt) {
        return;
    }

    const float * row_src = src + t*HC*HC;
    float         c[HC][HC];

#pragma unroll
    for (int s = 0; s < HC; ++s) {
        float max_v = -INFINITY;
#pragma unroll
        for (int d = 0; d < HC; ++d) {
            max_v = fmaxf(max_v, row_src[d + s*HC]);
        }
        float sum_v = 0.0f;
#pragma unroll
        for (int d = 0; d < HC; ++d) {
            const float v = expf(row_src[d + s*HC] - max_v);
            c[d][s] = v;
            sum_v += v;
        }
#pragma unroll
        for (int d = 0; d < HC; ++d) {
            c[d][s] = c[d][s]/sum_v + eps;
        }
    }

    for (int it = 0; it < n_iters; ++it) {
        if (it > 0) {
#pragma unroll
            for (int s = 0; s < HC; ++s) {
                float sum_v = 0.0f;
#pragma unroll
                for (int d = 0; d < HC; ++d) {
                    sum_v += c[d][s];
                }
                sum_v += eps;
#pragma unroll
                for (int d = 0; d < HC; ++d) {
                    c[d][s] /= sum_v;
                }
            }
        }

#pragma unroll
        for (int d = 0; d < HC; ++d) {
            float sum_v = 0.0f;
#pragma unroll
            for (int s = 0; s < HC; ++s) {
                sum_v += c[d][s];
            }
            sum_v += eps;
#pragma unroll
            for (int s = 0; s < HC; ++s) {
                c[d][s] /= sum_v;
            }
        }
    }

    float * row_dst = dst + t*HC*HC;
#pragma unroll
    for (int s = 0; s < HC; ++s) {
#pragma unroll
        for (int d = 0; d < HC; ++d) {
            row_dst[d + s*HC] = c[d][s];
        }
    }
}

static __global__ void hc_sinkhorn_kernel_generic(
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

// [TAG_HC_SINKHORN_XCHECK] Diagnostic gate for the specialization: run BOTH variants on the same
// live decode inputs and accumulate the same statistic test-backend-ops scores the backend with,
// NMSE = sum((a - b)^2) / sum(b^2), against the generic variant as reference b. Pre-committed
// threshold = test_hc_sinkhorn::max_nmse_err() == 1e-6, i.e. the specialization is accepted only
// if it sits at least as close to the generic kernel as the generic kernel sits to the CPU
// ground truth. Diagnostic only: adds a kernel launch, two copies and a host reduction per call.
static void hc_sinkhorn_xcheck(
        ggml_backend_cuda_context & ctx, const float * src_d, const float * dst_d,
        const int64_t hc, const int64_t nt, const int n_iters, const float eps) {

    const int64_t ne = hc*hc*nt;

    ggml_cuda_pool_alloc<float> ref_d(ctx.pool(), ne);

    const int block_size = nt <= 64 ? 64 : 256;
    const int n_blocks   = (nt + block_size - 1)/block_size;
    hc_sinkhorn_kernel_generic<<<n_blocks, block_size, 0, ctx.stream()>>>(src_d, ref_d.get(), hc, nt, n_iters, eps);

    std::vector<float> h_new(ne), h_ref(ne);
    CUDA_CHECK(cudaMemcpyAsync(h_new.data(), dst_d,     ne*sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(h_ref.data(), ref_d.get(), ne*sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));

    static double sum_sq_diff = 0.0;
    static double sum_sq_ref  = 0.0;
    static double max_abs     = 0.0;
    static double max_rel     = 0.0;
    static int64_t n_elem     = 0;
    static int64_t n_diff     = 0;
    static int64_t n_calls    = 0;

    for (int64_t i = 0; i < ne; ++i) {
        const double a = h_new[i];
        const double b = h_ref[i];
        const double d = a - b;
        sum_sq_diff += d*d;
        sum_sq_ref  += b*b;
        if (d != 0.0) {
            n_diff++;
        }
        const double ad = fabs(d);
        if (ad > max_abs) {
            max_abs = ad;
        }
        if (b != 0.0) {
            const double rd = ad/fabs(b);
            if (rd > max_rel) {
                max_rel = rd;
            }
        }
    }
    n_elem += ne;
    n_calls++;

    if (n_calls % 2000 == 0) {
        const double nmse = sum_sq_ref > 0.0 ? sum_sq_diff/sum_sq_ref : 0.0;
        fprintf(stderr, "[HC_SINKHORN_XCHECK] calls=%lld elems=%lld nt=%lld | NMSE=%.3e max_abs=%.3e max_rel=%.3e differing=%.2f%%\n",
                (long long) n_calls, (long long) n_elem, (long long) nt, nmse, max_abs, max_rel,
                100.0*(double) n_diff/(double) n_elem);
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

    // Decode runs at nt == 1: one wave64 is the honest launch shape for tiny nt instead of a
    // 256-thread block whose upper 224 lanes are dead on arrival.
    const int block_size = nt <= 64 ? 64 : 256;
    const int n_blocks   = (nt + block_size - 1)/block_size;

    static const bool force_generic = getenv("DSV4_HC_SINKHORN_GENERIC") != nullptr;

    if (hc == 4 && !force_generic) {
        hc_sinkhorn_kernel_hc<4><<<n_blocks, block_size, 0, ctx.stream()>>>(src_d, dst_d, nt, n_iters, eps);
    } else {
        hc_sinkhorn_kernel_generic<<<n_blocks, block_size, 0, ctx.stream()>>>(src_d, dst_d, hc, nt, n_iters, eps);
    }

    static const bool xcheck = getenv("DSV4_HC_SINKHORN_XCHECK") != nullptr;
    if (xcheck && !force_generic) {
        hc_sinkhorn_xcheck(ctx, src_d, dst_d, hc, nt, n_iters, eps);
    }
}
