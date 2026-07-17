#include <set>
#include <tuple>
#include "models.h"

#include "llama-kv-cache-dsv4.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

static bool dsv4_hc_fused_enabled() {
    static const bool no_fused = getenv("DSV4_NO_FUSED_SINKHORN") != nullptr;
    return !no_fused;
}
// Per-op bisection gates (each also disabled by the global DSV4_NO_FUSED_SINKHORN).
static bool dsv4_fuse_sinkhorn() {
    static const bool off = getenv("DSV4_NO_FUSE_SINKHORN") != nullptr;
    return dsv4_hc_fused_enabled() && !off;
}
static bool dsv4_fuse_combine() {
    static const bool off = getenv("DSV4_NO_FUSE_COMBINE") != nullptr;
    return dsv4_hc_fused_enabled() && !off;
}
static bool dsv4_fuse_wsum() {
    static const bool off = getenv("DSV4_NO_FUSE_WSUM") != nullptr;
    return dsv4_hc_fused_enabled() && !off;
}

static float dsv4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }

    return 1.0f / (1.0f + 0.1f*logf(1.0f/freq_scale));
}

void llama_model_deepseek4::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,     hparams.swiglu_clamp_exp,   hparams.n_layer());
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,   hparams.swiglu_clamp_shexp, hparams.n_layer(), 0)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    hparams.dsv4_compress_rope_base);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    ml.get_key(LLM_KV_HASH_LAYER_COUNT,                     hparams.dsv4_hash_layer_count);

    uint32_t n_compress_ratios = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_COMPRESS_RATIOS, n_compress_ratios);
    if (n_compress_ratios < hparams.n_layer()) {
        throw std::runtime_error("DeepSeek-V4 compress_ratios is shorter than block_count");
    }
    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios);

    ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);
    if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
        throw std::runtime_error("DeepSeek-V4 loader currently expects sqrtsoftplus MoE scoring");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(0);

    switch (hparams.n_layer()) {
        case 43: type = LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_deepseek4::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t o_groups    = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult     = hparams.dsv4_hc_mult;
    const int64_t hc_dim      = hc_mult * n_embd;
    const int64_t hc_mix_dim  = (2 + hc_mult) * hc_mult;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN, "weight"),    {hc_dim, hc_mult}, 0);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE, "weight"),  {hc_mult}, 0);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank * o_groups}, 0);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        const int64_t ratio = hparams.dsv4_compress_ratios[i];
        if (ratio != 0) {
            const int64_t coff = ratio == 4 ? 2 : 1;

            layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, coff * n_embd_head}, 0);
            layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, coff * n_embd_head}, 0);
            layer.attn_comp_ape   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_APE,   "weight", i), {coff * n_embd_head, ratio}, 0);
            layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, 0);

            if (ratio == 4) {
                const int64_t n_embd_indexer = hparams.indexer_head_size;

                layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, 0);
                layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, 0);

                layer.indexer_comp_wkv   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WKV,   "weight", i), {n_embd, 2 * n_embd_indexer}, 0);
                layer.indexer_comp_wgate = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WGATE, "weight", i), {n_embd, 2 * n_embd_indexer}, 0);
                layer.indexer_comp_ape   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_APE,   "weight", i), {2 * n_embd_indexer, ratio}, 0);
                layer.indexer_comp_norm  = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_NORM,  "weight", i), {n_embd_indexer}, 0);
            } else if (ratio != 128) {
                throw std::runtime_error("DeepSeek-V4 loader only supports compression ratios 0, 4, and 128");
            }
        }

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
        if ((uint32_t) i < hparams.dsv4_hash_layer_count) {
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {n_expert_used, n_vocab}, 0);
        } else {
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
        }
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek4::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

static size_t dsv4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * dsv4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_view_2d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        ne0,
        int64_t        ne1,
        int64_t        i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_append_zero_row(ggml_context * ctx, ggml_tensor * t, bool neg_inf) {
    ggml_tensor * row = ggml_view_1d(ctx, t, t->ne[0], 0);
    row = neg_inf ? ggml_scale_bias(ctx, row, 0.0f, -INFINITY) : ggml_scale(ctx, row, 0.0f);
    row = ggml_reshape_2d(ctx, row, t->ne[0], 1);

    return ggml_concat(ctx, t, row, 1);
}

static ggml_tensor * dsv4_with_zero_dep(ggml_context * ctx, ggml_tensor * t, ggml_tensor * dep) {
    if (dep == nullptr) {
        return t;
    }

    ggml_tensor * zero = ggml_scale(ctx, ggml_sum(ctx, dep), 0.0f);
    return ggml_add(ctx, t, zero);
}

static constexpr int64_t DSV4_CSA_RATIO  = 4;
static constexpr int64_t DSV4_HCA_RATIO  = 128;

static ggml_tensor * dsv4_hc_affine(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * scale,
        ggml_tensor  * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_weighted_sum(
        ggml_tensor * x,
        ggml_tensor * weights) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (dsv4_fuse_wsum()) {
        return ggml_hc_wsum(ctx0, x, weights);
    }

    ggml_tensor * acc = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);

        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        acc = acc ? ggml_add(ctx0, acc, cur) : cur;
    }

    return acc;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_sinkhorn(
        ggml_tensor * comb,
        int           il) const {
    GGML_UNUSED(il);

    if (dsv4_fuse_sinkhorn()) {
        return ggml_hc_sinkhorn(ctx0, comb, (int) hparams.dsv4_hc_sinkhorn_iters, hparams.dsv4_hc_eps);
    }

    // comb is [dst_hc, src_hc, n_tokens]. Sinkhorn follows the reference:
    // row softmax over dst, one column normalization, then repeated row/column normalization.
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);

    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = dsv4_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = dsv4_view_1d(ctx0, hc_scale, 1, 1);
    ggml_tensor * scale_comb = dsv4_view_1d(ctx0, hc_scale, 1, 2);

    ggml_tensor * base_pre  = dsv4_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = dsv4_view_1d(ctx0, hc_base, hc, hc);
    ggml_tensor * base_comb = dsv4_view_1d(ctx0, hc_base, hc*hc, 2*hc);

    ggml_tensor * pre = dsv4_view_2d(ctx0, mixes, hc, nt, 0);
    pre = dsv4_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    *post = dsv4_view_2d(ctx0, mixes, hc, nt, hc);
    *post = dsv4_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    *comb = dsv4_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
    *comb = dsv4_hc_affine(ctx0, *comb, scale_comb, base_comb);
    *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
    *comb = build_hc_sinkhorn(*comb, il);
    cb(*comb, "hc_comb", il);

    return build_hc_weighted_sum(x, pre);
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_UNUSED(il);

    if (dsv4_fuse_combine()) {
        return ggml_hc_combine(ctx0, x, residual, post, comb);
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2], dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_head(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = dsv4_hc_affine(ctx0, mixes, hc_scale, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_head_pre", -1);

    return build_hc_weighted_sum(x, pre);
}

ggml_tensor * llama_model_deepseek4::graph::build_hca_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == DSV4_HCA_RATIO*n_blocks);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    ggml_tensor * kv = ggml_get_rows(ctx0, kv_state, state_read_idxs);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, DSV4_HCA_RATIO, n_blocks);
    cb(kv, name, il);

    ggml_tensor * score = ggml_get_rows(ctx0, score_state, state_read_idxs);
    score = ggml_reshape_3d(ctx0, score, n_embd_head, DSV4_HCA_RATIO, n_blocks);
    cb(score, name, il);

    ggml_tensor * values = ggml_cont(ctx0, ggml_permute(ctx0, kv, 1, 0, 2, 3));
    ggml_tensor * scores = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    ggml_tensor * comp_nope = ggml_view_3d(ctx0, comp, n_embd_head_nope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            0);
    ggml_tensor * comp_pe = ggml_view_3d(ctx0, comp, n_embd_head_rope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head_nope));

    comp_pe = ggml_rope_ext(ctx0, comp_pe, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    cb(comp_pe, name, il);

    comp = ggml_concat(ctx0, comp_nope, comp_pe, 0);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llama_model_deepseek4::graph::build_overlap_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t ratio,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == 2*ratio*n_blocks);
    GGML_ASSERT(kv_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(score_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    kv_state    = dsv4_append_zero_row(ctx0, kv_state,    false);
    score_state = dsv4_append_zero_row(ctx0, score_state, true);

    ggml_tensor * prev_idxs = dsv4_view_1d(ctx0, state_read_idxs, ratio*n_blocks, 0);
    ggml_tensor * cur_idxs  = dsv4_view_1d(ctx0, state_read_idxs, ratio*n_blocks, ratio*n_blocks);

    ggml_tensor * kv_prev = ggml_get_rows(ctx0, kv_state, prev_idxs);
    kv_prev = ggml_cont(ctx0, ggml_view_2d(ctx0, kv_prev, n_embd_head, ratio*n_blocks, kv_prev->nb[1], 0));
    kv_prev = ggml_reshape_3d(ctx0, kv_prev, n_embd_head, ratio, n_blocks);
    cb(kv_prev, name, il);

    ggml_tensor * score_prev = ggml_get_rows(ctx0, score_state, prev_idxs);
    score_prev = ggml_cont(ctx0, ggml_view_2d(ctx0, score_prev, n_embd_head, ratio*n_blocks, score_prev->nb[1], 0));
    score_prev = ggml_reshape_3d(ctx0, score_prev, n_embd_head, ratio, n_blocks);
    cb(score_prev, name, il);

    ggml_tensor * kv_cur = ggml_get_rows(ctx0, kv_state, cur_idxs);
    kv_cur = ggml_cont(ctx0, ggml_view_2d(ctx0, kv_cur, n_embd_head, ratio*n_blocks, kv_cur->nb[1],
            ggml_row_size(kv_cur->type, n_embd_head)));
    kv_cur = ggml_reshape_3d(ctx0, kv_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * score_cur = ggml_get_rows(ctx0, score_state, cur_idxs);
    score_cur = ggml_cont(ctx0, ggml_view_2d(ctx0, score_cur, n_embd_head, ratio*n_blocks, score_cur->nb[1],
            ggml_row_size(score_cur->type, n_embd_head)));
    score_cur = ggml_reshape_3d(ctx0, score_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * values = ggml_concat(ctx0, kv_prev, kv_cur, 1);
    ggml_tensor * scores = ggml_concat(ctx0, score_prev, score_cur, 1);

    values = ggml_cont(ctx0, ggml_permute(ctx0, values, 1, 0, 2, 3));
    scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    ggml_tensor * comp_nope = ggml_view_3d(ctx0, comp, n_embd_head_nope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            0);
    ggml_tensor * comp_pe = ggml_view_3d(ctx0, comp, n_embd_head_rope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head_nope));

    comp_pe = ggml_rope_ext(ctx0, comp_pe, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    cb(comp_pe, name, il);

    comp = ggml_concat(ctx0, comp_nope, comp_pe, 0);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llama_model_deepseek4::graph::build_lid_top_k(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];
    const auto & inp_lid = inp_dsv4->get_lid();
    const int64_t n_embd_indexer_head      = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot();
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const int64_t n_indexer_head           = hparams.indexer_n_head;
    const int64_t nt                       = cur->ne[1];

    GGML_ASSERT(inp_lid.kq_mask);
    GGML_ASSERT(inp_lid.k_rot);
    GGML_ASSERT(n_embd_indexer_head >= n_embd_indexer_head_rope);

    ggml_tensor * indexer_q = build_lora_mm(layer.indexer_attn_q_b, qr);
    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, nt);
    cb(indexer_q, "lid_q", il);

    ggml_tensor * indexer_q_nope = ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_nope, n_indexer_head, nt,
            ggml_row_size(indexer_q->type, n_embd_indexer_head),
            ggml_row_size(indexer_q->type, n_embd_indexer_head)*n_indexer_head,
            0);
    ggml_tensor * indexer_q_pe = ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_rope, n_indexer_head, nt,
            ggml_row_size(indexer_q->type, n_embd_indexer_head),
            ggml_row_size(indexer_q->type, n_embd_indexer_head)*n_indexer_head,
            ggml_row_size(indexer_q->type, n_embd_indexer_head_nope));

    indexer_q_pe = ggml_rope_ext(ctx0, indexer_q_pe, inp_pos, nullptr, n_embd_indexer_head_rope,
            rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale,
            ext_factor, dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    cb(indexer_q_pe, "lid_q_pe", il);

    indexer_q = ggml_concat(ctx0, indexer_q_nope, indexer_q_pe, 0);
    indexer_q = llama_mul_mat_hadamard(ctx0, indexer_q, inp_lid.k_rot);
    cb(indexer_q, "lid_q_rot", il);

    ggml_tensor * indexer_weights = build_lora_mm(layer.indexer_proj, cur);
    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f/sqrtf(float(n_embd_indexer_head*n_indexer_head)));
    cb(indexer_weights, "lid_weights", il);

    ggml_tensor * indexer_k = inp_dsv4->mctx->get_lid()->get_k(ctx0, il);
    const int64_t n_lid = inp_lid.kq_mask->ne[0];
    GGML_ASSERT(n_lid > 0);
    GGML_ASSERT(n_lid <= indexer_k->ne[2]);

    indexer_k = ggml_view_4d(ctx0, indexer_k,
            indexer_k->ne[0], indexer_k->ne[1], n_lid, indexer_k->ne[3],
            indexer_k->nb[1], indexer_k->nb[2], indexer_k->nb[3], 0);
    cb(indexer_k, "lid_k", il);

    const int64_t n_stream = indexer_k->ne[3];
    indexer_q = ggml_view_4d(ctx0, indexer_q,
            indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream,
            indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
    indexer_weights = ggml_view_4d(ctx0, indexer_weights,
            indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream,
            indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

    ggml_tensor * indexer_score = nullptr;
    if (cparams.fused_lid) {
        indexer_score = ggml_lightning_indexer(ctx0, indexer_q, indexer_k, indexer_weights, inp_lid.kq_mask);
        cb(indexer_score, "lid_score_masked", il);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, indexer_score, il});
    } else {
        indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
        cb(indexer_q, "lid_q", il);
        indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
        cb(indexer_k, "lid_k", il);

        ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
        cb(indexer_kq, "lid_kq", il);

        indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
        cb(indexer_kq, "lid_kq", il);

        indexer_score = ggml_relu(ctx0, indexer_kq);
        indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
        indexer_score = ggml_sum_rows(ctx0, indexer_score);
        indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
        cb(indexer_score, "lid_score", il);

        indexer_score = ggml_add(ctx0, indexer_score, inp_lid.kq_mask);
        cb(indexer_score, "lid_score_masked", il);
    }

    const uint32_t n_top_k = indexer_score->ne[0] < hparams.indexer_top_k ? indexer_score->ne[0] : hparams.indexer_top_k;
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
    cb(top_k, "lid_top_k", il);

    return top_k;
}

ggml_tensor * llama_model_deepseek4::graph::build_top_k_mask(
        ggml_tensor * kq_mask,
        ggml_tensor * top_k,
        const char * name,
        int il) const {
    GGML_ASSERT(kq_mask);
    GGML_ASSERT(top_k);

    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k,
            kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    cb(kq_mask_top_k, name, il);

    return kq_mask_top_k;
}

ggml_tensor * llama_model_deepseek4::graph::build_csa_lid_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const auto & inp_csa = inp_dsv4->get_csa();
    GGML_ASSERT(inp_csa.kq_mask);

    ggml_tensor * top_k = build_lid_top_k(model, inp_dsv4, qr, cur, inp_pos, il);

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "csa_raw_k", il);

    ggml_tensor * csa_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);
    const int64_t n_csa = inp_csa.kq_mask->ne[0];
    GGML_ASSERT(n_csa > 0);
    GGML_ASSERT(n_csa <= csa_k->ne[2]);

    csa_k = ggml_view_4d(ctx0, csa_k,
            csa_k->ne[0], csa_k->ne[1], n_csa, csa_k->ne[3],
            csa_k->nb[1], csa_k->nb[2], csa_k->nb[3], 0);
    cb(csa_k, "csa_comp_k", il);

    ggml_tensor * raw_mask = inp_attn->get_kq_mask();

    // [DSV4_FA_COMPACT] FA has no mask-aware row skip: it pays for all n_csa+n_raw rows even
    // though only n_top_k+n_raw of them are unmasked. At decode (nt == 1) we can gather just the
    // selected rows into a small dense operand instead of masking, so FA runs over ~n_top_k+n_raw
    // rows instead of n_csa+n_raw. Building the top-k mask (build_top_k_mask) is skipped entirely
    // on this path -- it exists to make invisible rows in the full-width operand -inf, which is
    // moot once the operand only contains the rows we kept.
    static const bool fa_compact          = getenv("DSV4_FA_COMPACT") != nullptr;
    // [DSV4_FA_COMPACT_IDENTITY] control leg for byte-exactness: same gather machinery, but with
    // identity indices 0..n_csa-1 instead of the real top-k, so the operand is [csa; raw] in
    // natural row order -- the same content/order as DSV4_CONCAT_REORDER. Lets the box owner
    // sha-compare COMPACT_IDENTITY against CONCAT_REORDER to prove the gather itself is correct,
    // independent of whether the top-k selection is correct.
    static const bool fa_compact_identity = getenv("DSV4_FA_COMPACT_IDENTITY") != nullptr;

    // [DSV4_ABLATE_CONCAT] see below (measurement-only leg): declared here so it can also gate the
    // compact path -- ablate takes priority over compact, compact takes priority over mirror/concat.
    static const bool ablate_concat = getenv("DSV4_ABLATE_CONCAT") != nullptr;

    const int64_t nt = cur->ne[1];
    const int64_t n_stream_csa = csa_k->ne[3];

    // Compact path requires single-token decode (nt == 1) and single stream: the gather machinery
    // below assumes both, and this box only ever runs single-stream. Anything else falls back to
    // the existing mirror/concat legs below rather than guessing at multi-stream indexing.
    const bool compact_eligible = (fa_compact || fa_compact_identity) && nt == 1 && n_stream_csa == 1;
    const bool use_compact      = compact_eligible && !ablate_concat;

    ggml_tensor * csa_mask = nullptr;
    int64_t n_csa_compact_rows = 0;

    if (use_compact) {
        ggml_tensor * idx;
        int64_t n_rows;

        if (fa_compact_identity) {
            // Identity indices 0..n_csa-1: gathers every csa row in natural order, so the operand
            // matches DSV4_CONCAT_REORDER's [csa; raw] byte-for-byte -- this leg is a correctness
            // control, not a speed path.
            ggml_tensor * ar = ggml_arange(ctx0, 0.0f, (float) n_csa, 1.0f);
            idx = ggml_cast(ctx0, ar, GGML_TYPE_I32);
            idx = ggml_reshape_4d(ctx0, idx, n_csa, 1, 1, 1);
            n_rows = n_csa;
        } else {
            // top_k is [n_top_k, nt, 1, n_stream] (ggml_cont'd in build_lid_top_k); since nt == 1
            // and n_stream == 1 here, reshaping to [n_top_k, 1, n_stream, 1] is just a relabeling
            // of the same contiguous buffer -- this is the row-index operand ggml_get_rows wants.
            idx = ggml_reshape_4d(ctx0, top_k, top_k->ne[0], 1, top_k->ne[3], 1);
            n_rows = top_k->ne[0];
        }
        cb(idx, "csa_compact_idx", il);

        // csa_k is [n_embd, n_head_kv(=1), n_csa, n_stream] (rows on ne[2]). ggml_get_rows gathers
        // along ne[1], so reinterpret with n_csa moved to ne[1] and the size-1 head dim to ne[2].
        // This is a strided VIEW, not a copy: ggml_get_rows only requires nb[0] == type_size (the
        // CUDA kernel indexes rows via nb01/nb02/nb03 directly), so the gather reads straight out
        // of the cache's strided storage. Do not ggml_cont() this -- a full-width cont here is
        // exactly the copy this whole change exists to eliminate.
        GGML_ASSERT(csa_k->nb[0] == ggml_type_size(csa_k->type));
        ggml_tensor * csa_k_rows = ggml_view_4d(ctx0, csa_k,
                csa_k->ne[0], n_csa, csa_k->ne[1], csa_k->ne[3],
                csa_k->nb[2], csa_k->nb[1], csa_k->nb[3], 0);

        // get_rows has no F16 output path, so the small (n_rows-row) result comes back F32; cast
        // it back to F16 to concat with raw_k. The f16->f32->f16 roundtrip is exact.
        ggml_tensor * csa_sel = ggml_get_rows(ctx0, csa_k_rows, idx);
        csa_sel = ggml_reshape_4d(ctx0, csa_sel, csa_k->ne[0], csa_k->ne[1], n_rows, n_stream_csa);
        csa_sel = ggml_cast(ctx0, csa_sel, csa_k->type);
        cb(csa_sel, "csa_compact_k", il);

        if (fa_compact_identity) {
            // Identity leg: the operand is ALL n_csa rows in natural order, so the mask must be the
            // SAME full-width top-k mask CONCAT_REORDER uses -- operand and mask both identical,
            // leaving the gather machinery as the only variable. Gathering the base mask here would
            // attend every visible row (top-k selection off) and sha(GI) could never match sha(R).
            csa_mask = build_top_k_mask(inp_csa.kq_mask, top_k, "csa_top_k_mask", il);
        } else {
            // Gather the BASE mask (not the top-k mask) at the same indices: any top_k index that
            // points at an invisible row (n_visible < n_top_k padding) still carries its -inf, so
            // the compact operand does not need its own top-k mask built.
            ggml_tensor * mask_rows = ggml_view_4d(ctx0, inp_csa.kq_mask,
                    1, inp_csa.kq_mask->ne[0], inp_csa.kq_mask->ne[1], inp_csa.kq_mask->ne[3],
                    inp_csa.kq_mask->nb[0], inp_csa.kq_mask->nb[1], inp_csa.kq_mask->nb[2], 0);
            ggml_tensor * mask_sel = ggml_get_rows(ctx0, mask_rows, idx);
            mask_sel = ggml_reshape_4d(ctx0, mask_sel, n_rows, 1, 1, n_stream_csa);
            csa_mask = ggml_cast(ctx0, mask_sel, cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32);
            cb(csa_mask, "csa_compact_mask", il);
        }

        csa_k  = csa_sel;
        n_csa_compact_rows = n_rows;
    } else {
        // Every non-compact leg (ablate/mirror/concat/concat_reorder) still needs the full-width
        // top-k mask; only the compact operand skips it.
        csa_mask = build_top_k_mask(inp_csa.kq_mask, top_k, "csa_top_k_mask", il);
    }

    // [DSV4_LOG_SHAPES] geometry dump: the concat's operand sizes decide whether co-allocating
    // raw+comp into one buffer is worth its memory, so measure them instead of inferring from
    // bandwidth. Fires once per layer per graph build; graph builds are rare (cgraph reuse).
    if (getenv("DSV4_LOG_SHAPES")) {
        fprintf(stderr, "[dsv4-shapes] il=%2d CSA  raw_k=[%lld,%lld,%lld] cap=%lld  csa_k=[%lld,%lld,%lld] cap_used=%lld  row=%zu B  concat_MB=%.1f\n",
                il,
                (long long) raw_k->ne[0], (long long) raw_k->ne[1], (long long) raw_k->ne[2],
                (long long) raw_k->ne[2],
                (long long) csa_k->ne[0], (long long) csa_k->ne[1], (long long) n_csa,
                (long long) n_csa,
                ggml_row_size(raw_k->type, raw_k->ne[0]*raw_k->ne[1]),
                (raw_k->ne[2] + n_csa) * ggml_row_size(raw_k->type, raw_k->ne[0]*raw_k->ne[1]) / 1048576.0);
    }

    ggml_tensor * k_all;
    ggml_tensor * kq_mask;

    // [DSV4_ABLATE_CONCAT] MEASUREMENT ONLY -- PRODUCES WRONG OUTPUT ON PURPOSE.
    // Drops the raw window and attends over the compressed cache alone, which removes the
    // concat_cont<f16,2> copy (profiled at 11.4 ms/token = ~10% of decode @176k, 41 calls/token).
    // This is a SPEED CEILING for "make k_all a view instead of a copy", not a working config:
    // it also removes the raw window's attention work, so the true prize is <= what this shows.
    // Never leave this set; never quote a number from it as a result.
    // (declared earlier, alongside the compact gates, since it also has to gate use_compact)

    // [TAG_DSV4_RAW_MIRROR] Preferred path: mirror the raw window into the CSA cache's over-allocated
    // tail and view both halves as one operand, instead of ggml_concat copying all 54528 rows every
    // token. Copies 2304 rows instead -- ~4% of the traffic for the same operand.
    // Requires the compressed operand to be exactly the cache's logical size, since that is where the
    // tail (and hence the raw half) begins; anything else means the view would not line up with the
    // mask, so fall back to the concat rather than guess.
    auto * mctx_csa = inp_dsv4->mctx->get_csa();

    const uint32_t n_raw    = (uint32_t) raw_k->ne[2];
    const uint32_t n_mirror = mctx_csa->get_n_mirror(il);

    // Escape hatch + the A leg of the A/B: forces the concat path with the mirror compiled in, so the
    // two legs differ by exactly one variable.
    static const bool no_mirror = getenv("DSV4_NO_RAW_MIRROR") != nullptr;
    // [CONTROL C] Pure ggml_concat, but in the MIRROR's row order [csa; raw] instead of [raw; csa].
    // Keeps the full copy (no perf win) and changes ONLY the row order. This is the correctness gate:
    // sha(C) == sha(B) proves the mirror wrote byte-identical data and that the permutation is the sole
    // variable. sha equality against A is unachievable by construction here -- FA's tile-order reduction
    // is not permutation-invariant in floating point -- so "fluent output that reconverges" proves
    // nothing: a single wrong row produces that same signature.
    static const bool concat_reorder = getenv("DSV4_CONCAT_REORDER") != nullptr;

    // The mirror lands at row n_csa, which is where the compressed operand ends in THIS graph, so it
    // works at any depth. An earlier version demanded n_csa == get_n_kv() (i.e. a full-capacity
    // compressed block, since the tail was pinned to the cache's logical end); that holds only in the
    // RESERVE graph, so every decode graph silently fell back to concat and the whole thing measured
    // as +0.16% -- concat vs concat. Room is the only real precondition now.
    const bool use_mirror = !ablate_concat && !use_compact && !concat_reorder && !no_mirror && n_mirror >= n_raw &&
                            (uint64_t) n_csa + n_raw <= (uint64_t) mctx_csa->get_n_kv() + n_mirror;

    // One-shot, straight to stderr: LLAMA_LOG_* does not reach the server log on this build, and an
    // identical sha between the two legs is NOT evidence the mirror ran -- if use_mirror is false, B
    // silently falls back to the same concat as A and the legs are trivially identical. This print is
    // what distinguishes "the mirror is correct" from "we compared concat against concat".
    // PER-LAYER, once each: a single global print only proves ONE layer took the branch. The kernel
    // trace showed concat_cont still at 41 calls/token (= every layer) while the global print said
    // MIRROR -- exactly the ambiguity a one-shot print cannot resolve.
    // Count EVERY decision and print periodically. A first-time-per-layer print only ever reports the
    // RESERVE graph (built once at context creation, worst-case shapes) and is then silent for every
    // decode graph -- which is exactly how "path=MIRROR on all 21 layers" coexisted with concat_cont
    // still firing 41x/token and the mirror's cpy kernel absent from the trace.
    // Report every DISTINCT (n_csa, n_raw, path) once. Counting decisions was the wrong instrument
    // twice over: a decision is a graph BUILD, and the decode graph is built once then reused, so a
    // modulo-N print reports only the earliest builds (reserve, at capacity) and never names the shape
    // decode actually executes. A distinct-shape set covers every shape exactly once, spam-free.
    {
        static std::set<std::tuple<long long, unsigned, int>> seen;
        const int path = ablate_concat ? 2 :
                         use_compact   ? (fa_compact_identity ? 5 : 4) :
                         use_mirror    ? 1 :
                         concat_reorder ? 3 : 0;
        if (seen.insert(std::make_tuple((long long) n_csa, n_raw, path)).second) {
            if (path == 4 || path == 5) {
                fprintf(stderr, "[DSV4_MIRROR] shape n_csa=%lld n_raw=%u n_top_k=%lld -> path=%s  (comp_n_kv=%u n_mirror=%u kv_size=%u il=%d)\n",
                        (long long) n_csa, n_raw, (long long) n_csa_compact_rows,
                        path == 4 ? "COMPACT" : "COMPACT_IDENTITY",
                        mctx_csa->get_n_kv(), n_mirror, mctx_csa->get_n_kv(), il);
            } else {
                fprintf(stderr, "[DSV4_MIRROR] shape n_csa=%lld n_raw=%u -> path=%s  (comp_n_kv=%u n_mirror=%u kv_size=%u il=%d)\n",
                        (long long) n_csa, n_raw,
                        path == 1 ? "MIRROR" : path == 2 ? "ABLATE" : path == 3 ? "CONCAT_REORDER" : "CONCAT",
                        mctx_csa->get_n_kv(), n_mirror, mctx_csa->get_n_kv(), il);
            }
        }
    }

    if (ablate_concat) {
        k_all   = csa_k;
        kq_mask = csa_mask;
    } else if (use_compact) {
        // csa_k/csa_mask are already the small gathered (n_top_k- or n_csa-row) operands here; the
        // concat is cheap because the compressed side no longer has 44032 rows to copy.
        k_all   = ggml_concat(ctx0, csa_k, raw_k, 2);
        kq_mask = ggml_concat(ctx0, csa_mask, raw_mask, 0);
    } else if (use_mirror) {
        // Ordering is by graph insertion: the copy must be expanded before the attention op reads it.
        ggml_build_forward_expand(gf, mctx_csa->mirror_raw_k(ctx0, raw_k, il, (uint32_t) n_csa));

        k_all = mctx_csa->get_k_all(ctx0, n_raw, il, (uint32_t) n_csa);

        // A non-contiguous K makes the backend insert a full-height cont, i.e. exactly the copy we came
        // here to delete. Catch that at graph-build time rather than reading it back out of a profile.
        GGML_ASSERT(ggml_is_contiguous(k_all) && "mirror k_all must be contiguous or FA will re-copy it");

        // Mask order follows the operand: compressed rows first, raw window second.
        kq_mask = ggml_concat(ctx0, csa_mask, raw_mask, 0);
    } else if (concat_reorder) {
        k_all   = ggml_concat(ctx0, csa_k, raw_k, 2);
        kq_mask = ggml_concat(ctx0, csa_mask, raw_mask, 0);
    } else {
        k_all   = ggml_concat(ctx0, raw_k, csa_k, 2);
        kq_mask = ggml_concat(ctx0, raw_mask, csa_mask, 0);
    }
    cb(k_all, "csa_k_all", il);
    cb(kq_mask, "csa_lid_kq_mask", il);

    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_csa_lid", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_hca_attention(
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const auto & inp_hca = inp_dsv4->get_hca();
    GGML_ASSERT(inp_hca.kq_mask);

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "hca_raw_k", il);

    ggml_tensor * hca_k = inp_dsv4->mctx->get_hca()->get_k(ctx0, il);
    const int64_t n_hca = inp_hca.kq_mask->ne[0];
    GGML_ASSERT(n_hca > 0);
    GGML_ASSERT(n_hca <= hca_k->ne[2]);

    hca_k = ggml_view_4d(ctx0, hca_k,
            hca_k->ne[0], hca_k->ne[1], n_hca, hca_k->ne[3],
            hca_k->nb[1], hca_k->nb[2], hca_k->nb[3], 0);
    cb(hca_k, "hca_comp_k", il);

    ggml_tensor * raw_mask = inp_attn->get_kq_mask();
    ggml_tensor * hca_mask = inp_hca.kq_mask;

    // [DSV4_LOG_SHAPES] see build_csa_lid_attention.
    if (getenv("DSV4_LOG_SHAPES")) {
        fprintf(stderr, "[dsv4-shapes] il=%2d HCA  raw_k=[%lld,%lld,%lld] cap=%lld  hca_k=[%lld,%lld,%lld] cap_used=%lld  row=%zu B  concat_MB=%.1f\n",
                il,
                (long long) raw_k->ne[0], (long long) raw_k->ne[1], (long long) raw_k->ne[2],
                (long long) raw_k->ne[2],
                (long long) hca_k->ne[0], (long long) hca_k->ne[1], (long long) n_hca,
                (long long) n_hca,
                ggml_row_size(raw_k->type, raw_k->ne[0]*raw_k->ne[1]),
                (raw_k->ne[2] + n_hca) * ggml_row_size(raw_k->type, raw_k->ne[0]*raw_k->ne[1]) / 1048576.0);
    }

    ggml_tensor * k_all;
    ggml_tensor * kq_mask;

    // [DSV4_ABLATE_CONCAT] see build_csa_lid_attention -- measurement only, wrong output on purpose.
    static const bool ablate_concat = getenv("DSV4_ABLATE_CONCAT") != nullptr;
    if (ablate_concat) {
        k_all   = hca_k;
        kq_mask = hca_mask;
    } else {
        k_all   = ggml_concat(ctx0, raw_k, hca_k, 2);
        kq_mask = ggml_concat(ctx0, raw_mask, hca_mask, 0);
    }
    cb(k_all, "hca_k_all", il);
    cb(kq_mask, "hca_kq_mask", il);

    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_hca", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_raw_attention(
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    GGML_ASSERT(hparams.is_swa(il));

    ggml_tensor * k_rot = inp_attn->self_k_rot;

    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_cur = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * kq_mask = inp_attn->get_kq_mask();

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);

    ggml_tensor * out = build_attn_mha(q, k, k, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_raw", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const int64_t nt               = cur->ne[1];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);

    const bool use_compress_rope = hparams.dsv4_compress_ratios[il] != 0;
    const float freq_base_l      = use_compress_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float freq_scale_l     = use_compress_rope ? freq_scale : 1.0f;
    const float ext_factor_l     = use_compress_rope ? ext_factor : 0.0f;
    const float attn_factor_l    = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float beta_fast_l      = use_compress_rope ? beta_fast : 0.0f;
    const float beta_slow_l      = use_compress_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l   = use_compress_rope ? n_ctx_orig : 0;

    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    cb(qr, "qr", il);

    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr_norm", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rms_norm(ctx0, q, norm_rms_eps);
    cb(q, "q_norm", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_nope, n_head, nt,
            ggml_row_size(q->type, n_embd_head),
            ggml_row_size(q->type, n_embd_head)*n_head,
            0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_rope, n_head, nt,
            ggml_row_size(q->type, n_embd_head),
            ggml_row_size(q->type, n_embd_head)*n_head,
            ggml_row_size(q->type, n_embd_head_nope));
    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    cb(q_pe, "q_pe", il);
    q = ggml_concat(ctx0, q_nope, q_pe, 0);
    cb(q, "q", il);

    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    cb(kv, "kv_norm", il);

    ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head),
            0);
    ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head_nope));
    kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    cb(kv_pe, "kv_pe", il);
    kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
    cb(kv, "kv", il);

    const int64_t ratio = hparams.dsv4_compress_ratios[il];

    ggml_tensor * hca_state_kv    = nullptr;
    ggml_tensor * hca_state_score = nullptr;
    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_pos) {
        hca_state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(hca_state_kv, "hca_state_kv", il);

        hca_state_score = build_lora_mm(layer.attn_comp_wgate, cur);
        cb(hca_state_score, "hca_state_score", il);

        ggml_tensor * ape = layer.attn_comp_ape;

        ggml_tensor * ape_rows = ggml_get_rows(ctx0, ape, inp_dsv4->get_hca().state_pos);
        hca_state_score = ggml_add(ctx0, hca_state_score, ape_rows);
        cb(hca_state_score, "hca_state_score_ape", il);

    }

    if (ratio == DSV4_CSA_RATIO && inp_dsv4->get_csa().state_pos) {
        ggml_tensor * csa_state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(csa_state_kv, "csa_state_kv", il);

        ggml_tensor * csa_state_score = build_lora_mm(layer.attn_comp_wgate, cur);
        cb(csa_state_score, "csa_state_score", il);

        ggml_tensor * csa_ape = layer.attn_comp_ape;

        ggml_tensor * csa_ape_rows = ggml_get_rows(ctx0, csa_ape, inp_dsv4->get_csa().state_pos);
        csa_state_score = ggml_add(ctx0, csa_state_score, csa_ape_rows);
        cb(csa_state_score, "csa_state_score_ape", il);

        GGML_ASSERT(inp_dsv4->get_csa().state_write_idxs);

        ggml_tensor * csa_source_kv = ggml_concat(ctx0,
                inp_dsv4->mctx->get_csa_state()->get_kv(ctx0, il), csa_state_kv, 1);
        ggml_tensor * csa_source_score = ggml_concat(ctx0,
                inp_dsv4->mctx->get_csa_state()->get_score(ctx0, il), csa_state_score, 1);

        ggml_tensor * kv_comp_csa_state = build_overlap_compressed_kv_from_state(
                csa_source_kv,
                csa_source_score,
                inp_dsv4->get_csa().state_read_idxs,
                inp_dsv4->get_csa().state_write_pos,
                layer.attn_comp_norm,
                DSV4_CSA_RATIO,
                n_embd_head,
                "csa_state_compress",
                il);

        if (inp_dsv4->get_csa().k_rot) {
            kv_comp_csa_state = llama_mul_mat_hadamard(ctx0, kv_comp_csa_state, inp_dsv4->get_csa().k_rot);
            cb(kv_comp_csa_state, "csa_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_csa()->cpy_k(ctx0,
                    kv_comp_csa_state, inp_dsv4->get_csa().state_write_idxs, il));

        csa_state_kv    = dsv4_with_zero_dep(ctx0, csa_state_kv,    kv_comp_csa_state);
        csa_state_score = dsv4_with_zero_dep(ctx0, csa_state_score, kv_comp_csa_state);

        ggml_tensor * csa_persist_kv = ggml_get_rows(ctx0, csa_state_kv, inp_dsv4->get_csa().state_persist_src_idxs);
        ggml_tensor * csa_persist_score = ggml_get_rows(ctx0, csa_state_score, inp_dsv4->get_csa().state_persist_src_idxs);

        csa_state_kv = inp_dsv4->mctx->get_csa_state()->cpy_kv(ctx0,
                csa_persist_kv, inp_dsv4->get_csa().state_persist_dst_idxs, il);
        csa_state_score = inp_dsv4->mctx->get_csa_state()->cpy_score(ctx0,
                csa_persist_score, inp_dsv4->get_csa().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, csa_state_kv);
        ggml_build_forward_expand(gf, csa_state_score);

        ggml_tensor * lid_state_kv = build_lora_mm(layer.indexer_comp_wkv, cur);
        cb(lid_state_kv, "lid_state_kv", il);

        ggml_tensor * lid_state_score = build_lora_mm(layer.indexer_comp_wgate, cur);
        cb(lid_state_score, "lid_state_score", il);

        ggml_tensor * lid_ape = layer.indexer_comp_ape;

        ggml_tensor * lid_ape_rows = ggml_get_rows(ctx0, lid_ape, inp_dsv4->get_lid().state_pos);
        lid_state_score = ggml_add(ctx0, lid_state_score, lid_ape_rows);
        cb(lid_state_score, "lid_state_score_ape", il);

        GGML_ASSERT(inp_dsv4->get_lid().state_write_idxs);

        ggml_tensor * lid_source_kv = ggml_concat(ctx0,
                inp_dsv4->mctx->get_lid_state()->get_kv(ctx0, il), lid_state_kv, 1);
        ggml_tensor * lid_source_score = ggml_concat(ctx0,
                inp_dsv4->mctx->get_lid_state()->get_score(ctx0, il), lid_state_score, 1);

        ggml_tensor * kv_comp_lid_state = build_overlap_compressed_kv_from_state(
                lid_source_kv,
                lid_source_score,
                inp_dsv4->get_lid().state_read_idxs,
                inp_dsv4->get_lid().state_write_pos,
                layer.indexer_comp_norm,
                DSV4_CSA_RATIO,
                hparams.indexer_head_size,
                "lid_state_compress",
                il);

        if (inp_dsv4->get_lid().k_rot) {
            kv_comp_lid_state = llama_mul_mat_hadamard(ctx0, kv_comp_lid_state, inp_dsv4->get_lid().k_rot);
            cb(kv_comp_lid_state, "lid_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_lid()->cpy_k(ctx0,
                    kv_comp_lid_state, inp_dsv4->get_lid().state_write_idxs, il));

        lid_state_kv    = dsv4_with_zero_dep(ctx0, lid_state_kv,    kv_comp_lid_state);
        lid_state_score = dsv4_with_zero_dep(ctx0, lid_state_score, kv_comp_lid_state);

        ggml_tensor * lid_persist_kv = ggml_get_rows(ctx0, lid_state_kv, inp_dsv4->get_lid().state_persist_src_idxs);
        ggml_tensor * lid_persist_score = ggml_get_rows(ctx0, lid_state_score, inp_dsv4->get_lid().state_persist_src_idxs);

        lid_state_kv = inp_dsv4->mctx->get_lid_state()->cpy_kv(ctx0,
                lid_persist_kv, inp_dsv4->get_lid().state_persist_dst_idxs, il);
        lid_state_score = inp_dsv4->mctx->get_lid_state()->cpy_score(ctx0,
                lid_persist_score, inp_dsv4->get_lid().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, lid_state_kv);
        ggml_build_forward_expand(gf, lid_state_score);
    }

    ggml_tensor * hca_state_dep = nullptr;
    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_write_idxs) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        ggml_tensor * hca_source_kv = ggml_concat(ctx0,
                inp_dsv4->mctx->get_hca_state()->get_kv(ctx0, il), hca_state_kv, 1);
        ggml_tensor * hca_source_score = ggml_concat(ctx0,
                inp_dsv4->mctx->get_hca_state()->get_score(ctx0, il), hca_state_score, 1);

        ggml_tensor * kv_comp_hca = build_hca_compressed_kv_from_state(
                hca_source_kv,
                hca_source_score,
                inp_dsv4->get_hca().state_read_idxs,
                inp_dsv4->get_hca().state_write_pos,
                layer.attn_comp_norm,
                n_embd_head,
                "hca_state_compress",
                il);

        if (inp_dsv4->get_hca().k_rot) {
            kv_comp_hca = llama_mul_mat_hadamard(ctx0, kv_comp_hca, inp_dsv4->get_hca().k_rot);
            cb(kv_comp_hca, "hca_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_hca()->cpy_k(ctx0,
                    kv_comp_hca, inp_dsv4->get_hca().state_write_idxs, il));
        hca_state_dep = kv_comp_hca;
    }

    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_pos) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        hca_state_kv    = dsv4_with_zero_dep(ctx0, hca_state_kv,    hca_state_dep);
        hca_state_score = dsv4_with_zero_dep(ctx0, hca_state_score, hca_state_dep);

        ggml_tensor * hca_persist_kv = ggml_get_rows(ctx0, hca_state_kv, inp_dsv4->get_hca().state_persist_src_idxs);
        ggml_tensor * hca_persist_score = ggml_get_rows(ctx0, hca_state_score, inp_dsv4->get_hca().state_persist_src_idxs);

        hca_state_kv = inp_dsv4->mctx->get_hca_state()->cpy_kv(ctx0,
                hca_persist_kv, inp_dsv4->get_hca().state_persist_dst_idxs, il);
        hca_state_score = inp_dsv4->mctx->get_hca_state()->cpy_score(ctx0,
                hca_persist_score, inp_dsv4->get_hca().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, hca_state_kv);
        ggml_build_forward_expand(gf, hca_state_score);
    }

    ggml_tensor * out = nullptr;
    if (ratio == DSV4_CSA_RATIO &&
            inp_dsv4->get_csa().kq_mask &&
            inp_dsv4->get_lid().kq_mask &&
            inp_dsv4->get_lid().k_rot) {
        out = build_csa_lid_attention(model, inp_dsv4, inp_attn, q, kv, qr, cur, inp_pos, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    } else if (ratio == DSV4_HCA_RATIO &&
            inp_dsv4->get_hca().kq_mask) {
        out = build_hca_attention(inp_dsv4, inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    } else {
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    }

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    ggml_tensor * out_nope = ggml_view_3d(ctx0, out, n_embd_head_nope, n_head, nt,
            ggml_row_size(out->type, n_embd_head),
            ggml_row_size(out->type, n_embd_head)*n_head,
            0);
    ggml_tensor * out_pe = ggml_view_3d(ctx0, out, n_embd_head_rope, n_head, nt,
            ggml_row_size(out->type, n_embd_head),
            ggml_row_size(out->type, n_embd_head)*n_head,
            ggml_row_size(out->type, n_embd_head_nope));
    out_pe = ggml_rope_ext_back(ctx0, out_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    out = ggml_concat(ctx0, out_nope, out_pe, 0);
    cb(out, "attn_derope", il);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, layer.wo_a, layer.wo_a->ne[0], o_lora_rank, n_groups), out);
    cb(oa, "attn_wo_a", il);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

llama_model_deepseek4::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();
    ggml_build_forward_expand(gf, inp_attn->self_kq_mask);

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_hc_pre(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention(model, inp_dsv4, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        ggml_tensor * selected_experts = nullptr;
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;
        if ((uint32_t) il < hparams.dsv4_hash_layer_count) {
            selected_experts = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, res->t_inp_tokens);
            exp_probs_b = nullptr;
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                selected_experts);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
