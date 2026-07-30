#include "models.h"

#include <cmath>
#include <stdexcept>

// fixed epsilon used by the reference PolyNorm and MHC norms (independent of rms_norm_eps)
static constexpr float MOTIF3_POLY_EPS = 1e-6f;

void llama_model_motif3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_NOISE_HEAD_COUNT,  hparams.n_noise_heads);

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    // see build_poly_norm: these three are in config.json but not in modeling_motif.py
    ml.get_key(LLM_KV_POLYNORM_OUTPUT_SCALE,   hparams.polynorm_output_scale,   false);
    ml.get_key(LLM_KV_POLYNORM_BIAS_CLAMP,     hparams.polynorm_bias_clamp,     false);
    ml.get_key(LLM_KV_POLYNORM_SIGMOID_WEIGHT, hparams.polynorm_sigmoid_weight, false);

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);

    if (ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false)) {
        // cancel the factor from the convert script (see deepseek2 [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX])
        hparams.rope_yarn_log_mul /= 0.1f;
    }

    // Interleaved SWA: full attention every `pattern`-th layer (il % period == 0), windowed
    // elsewhere. The HF *sdpa* reference silently drops the window (_update_causal_mask
    // builds one full-causal mask for every layer and the sdpa kernel ignores the
    // `sliding_window` kwarg), so an sdpa-based oracle cannot validate this path -- but the
    // released weights were trained with the window, and running full attention everywhere
    // produces gibberish on the real model.
    //
    // NOTE: set_swa_pattern(0) does NOT mean "no SWA" -- the `n_pattern == 0 ||`
    // short-circuit marks EVERY layer as SWA. The all-dense idiom is set_swa_pattern(1).
    if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        uint32_t swa_period = 4;
        ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false);

        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hparams.set_swa_pattern(swa_period, /* dense_first = */ true);

        // windowed layers use plain RoPE on their own freq base, with no YaRN at all
        ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
        hparams.rope_freq_scale_train_swa = 1.0f;
    } else {
        // The graph always builds an iSWA attention input, so an all-dense hparams would hand
        // it a plain unified KV cache and crash inside build_attn. Every released motif3
        // config declares the window; refuse loudly rather than segfault at graph reserve.
        throw std::runtime_error("motif3: missing or zero motif3.attention.sliding_window");
    }

    switch (hparams.n_layer()) {
        case 53: type = LLM_TYPE_314B_A13B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_motif3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t kv_lora_rank    = hparams.n_lora_kv;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const int64_t n_noise_heads = hparams.n_noise_heads;
    const int64_t n_signal_heads = n_head - n_noise_heads;

    const int64_t hc_mult    = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc_mult * n_embd;
    const int64_t hc_mix_dim = (2 + hc_mult) * hc_mult;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];

        int flags = 0;
        if (i >= n_layer) {
            // NextN/MTP block: loaded (for --spec-type draft-mtp) but optional,
            // so trunk-only GGUFs keep working
            flags |= TENSOR_NOT_REQUIRED;
        }

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, flags);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, flags);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head_k}, flags);
        layer.wqkv_gate     = create_tensor(tn(LLM_TENSOR_ATTN_GATE,     "weight", i), {q_lora_rank, n_signal_heads * n_embd_head_v}, flags);
        layer.wkv_a_mqa     = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, flags);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);
        layer.wkv_b         = create_tensor(tn(LLM_TENSOR_ATTN_KV_B,     "weight", i), {kv_lora_rank, n_head_kv * (n_embd_head_qk_nope + n_embd_head_v)}, flags);
        layer.wk_b          = create_tensor(tn(LLM_TENSOR_ATTN_K_B,      "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head_kv}, flags | TENSOR_NOT_REQUIRED);
        layer.wv_b          = create_tensor(tn(LLM_TENSOR_ATTN_V_B,      "weight", i), {kv_lora_rank, n_embd_head_v, n_head_kv}, flags | TENSOR_NOT_REQUIRED);
        layer.attn_lambda   = create_tensor(tn(LLM_TENSOR_ATTN_LAMBDA,   "weight", i), {n_embd, n_signal_heads}, flags);
        layer.wo            = create_tensor(tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {n_signal_heads * n_embd_head_v, n_embd}, flags);

        layer.hc_attn_norm  = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,  "weight", i), {hc_dim}, flags | TENSOR_NOT_REQUIRED);
        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, flags | TENSOR_NOT_REQUIRED);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, flags | TENSOR_NOT_REQUIRED);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, flags | TENSOR_NOT_REQUIRED);
        layer.hc_ffn_norm   = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,   "weight", i), {hc_dim}, flags | TENSOR_NOT_REQUIRED);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, flags | TENSOR_NOT_REQUIRED);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, flags | TENSOR_NOT_REQUIRED);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, flags | TENSOR_NOT_REQUIRED);

        if (i < n_layer && layer.hc_attn_fn == nullptr) {
            throw std::runtime_error("missing hyper-connection tensors in non-MTP layer");
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        // the NextN/MTP block (i >= n_layer) uses a dense PolyNorm MLP, no experts
        const bool is_dense = (uint32_t) i < hparams.n_layer_dense_lead || i >= n_layer;
        const int flags_dense = is_dense ? flags : flags | TENSOR_NOT_REQUIRED;
        const int flags_moe   = is_dense ? flags | TENSOR_NOT_REQUIRED : flags;

        // dense FFN (leading layers)
        layer.ffn_gate   = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, flags_dense);
        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, flags_dense);
        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, flags_dense);
        layer.ffn_poly   = create_tensor(tn(LLM_TENSOR_FFN_POLY, "weight", i), {3}, flags_dense);
        layer.ffn_poly_b = create_tensor(tn(LLM_TENSOR_FFN_POLY, "bias",   i), {1}, flags_dense);

        // MoE FFN
        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, flags_moe);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, flags_moe);
        layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, flags_moe);
        layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,   "weight", i), {n_ff_exp, n_embd,   n_expert}, flags_moe);
        layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,     "weight", i), {n_embd,   n_ff_exp, n_expert}, flags_moe);
        layer.ffn_poly_exps   = create_tensor(tn(LLM_TENSOR_FFN_POLY_EXPS,   "weight", i), {3, n_expert}, flags_moe);
        layer.ffn_poly_exps_b = create_tensor(tn(LLM_TENSOR_FFN_POLY_EXPS,   "bias",   i), {n_expert}, flags_moe);

        layer.ffn_gate_shexp   = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags_moe);
        layer.ffn_down_shexp   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd}, flags_moe);
        layer.ffn_up_shexp     = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags_moe);
        layer.ffn_poly_shexp   = create_tensor(tn(LLM_TENSOR_FFN_POLY_SHEXP, "weight", i), {3}, flags_moe);
        layer.ffn_poly_shexp_b = create_tensor(tn(LLM_TENSOR_FFN_POLY_SHEXP, "bias",   i), {1}, flags_moe);

        // NextN/MTP tensors (preserved but unused)
        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, flags);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_motif3::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

static ggml_tensor * motif3_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, ggml_row_size(t->type, i0));
}

static ggml_tensor * motif3_view_2d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        ne0,
        int64_t        ne1,
        int64_t        i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], ggml_row_size(t->type, i0));
}

static ggml_tensor * motif3_hc_affine(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * scale,
        ggml_tensor  * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

ggml_tensor * llama_model_motif3::graph::build_hc_sinkhorn(
        ggml_tensor * comb,
        int           il) const {
    GGML_UNUSED(il);

    const float eps = hparams.dsv4_hc_eps;

    // comb is [src_hc, dst_hc, n_tokens]; the reference clamps the logits,
    // exponentiates, then alternates row (src) and column (dst) normalization
    comb = ggml_clamp(ctx0, comb, -20.0f, 20.0f);
    comb = ggml_exp(ctx0, comb);

    for (uint32_t i = 0; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        // normalize over src (torch dim=-1)
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_clamp(ctx0, row_sum, eps, INFINITY);
        comb = ggml_div(ctx0, comb, row_sum);

        // normalize over dst (torch dim=-2)
        ggml_tensor * comb_dst_src = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_dst_src);
        col_sum = ggml_clamp(ctx0, col_sum, eps, INFINITY);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    }

    return comb;
}

ggml_tensor * llama_model_motif3::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_norm,
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

    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    // weighted rms norm over the full expanded dim (fixed eps in the reference)
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, MOTIF3_POLY_EPS);
    flat_norm = ggml_mul(ctx0, flat_norm, hc_norm);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = motif3_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = motif3_view_1d(ctx0, hc_scale, 1, 1);
    ggml_tensor * scale_comb = motif3_view_1d(ctx0, hc_scale, 1, 2);

    ggml_tensor * base_pre  = motif3_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = motif3_view_1d(ctx0, hc_base, hc, hc);
    ggml_tensor * base_comb = motif3_view_1d(ctx0, hc_base, hc*hc, 2*hc);

    ggml_tensor * pre = motif3_view_2d(ctx0, mixes, hc, nt, 0);
    pre = motif3_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_clamp(ctx0, pre, -10.0f, 10.0f);
    pre = ggml_sigmoid(ctx0, pre);
    cb(pre, "hc_pre", il);

    *post = motif3_view_2d(ctx0, mixes, hc, nt, hc);
    *post = motif3_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_clamp(ctx0, *post, -10.0f, 10.0f);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    *comb = motif3_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
    *comb = motif3_hc_affine(ctx0, *comb, scale_comb, base_comb);
    *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
    *comb = build_hc_sinkhorn(*comb, il);
    cb(*comb, "hc_comb", il);

    // weighted sum of the residual streams with the pre weights
    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, pre, 1, nt, pre->nb[1], ih*pre->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }

    return result;
}

ggml_tensor * llama_model_motif3::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_UNUSED(il);
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == hparams.dsv4_hc_mult);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            // comb is [src, dst, n_tokens]
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2],
                    src*comb->nb[0] + dst*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

ggml_tensor * llama_model_motif3::graph::build_hc_mean(
        ggml_tensor * x) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    ggml_tensor * sum = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        sum = sum ? ggml_add(ctx0, sum, xh) : xh;
    }

    return ggml_scale(ctx0, sum, 1.0f/float(hc));
}

// `polynorm_output_scale` (0.5 on the released model) multiplies the gated activation.
// Like the sigmoid and the bias clamp, it is declared in config.json and ignored by
// modeling_motif.py, so an HF-based oracle cannot see it.
ggml_tensor * llama_model_motif3::graph::build_poly_scale(ggml_tensor * act) const {
    if (hparams.polynorm_output_scale == 1.0f) {
        return act;
    }
    return ggml_scale(ctx0, act, hparams.polynorm_output_scale);
}

ggml_tensor * llama_model_motif3::graph::build_poly_norm(
        ggml_tensor * x,
        ggml_tensor * w,
        ggml_tensor * b,
        bool clamp_bias,
        int il) const {
    GGML_ASSERT(w->ne[0] == 3);

    // PolyNorm: w0*rmsnorm(x^3) + w1*rmsnorm(x^2) + w2*rmsnorm(x) + b
    // w is either {3} (shared) or {3, n_expert_used, n_tokens} (gathered per expert)
    //
    // The coefficients are stored *pre-sigmoid* and the per-expert biases were trained under
    // a clamp -- neither is visible in modeling_motif.py, whose PolyNormTorch is an incomplete
    // re-implementation that reads neither `polynorm_sigmoid_weight` nor `polynorm_bias_clamp`
    // from its own config. The checkpoint settles it: the per-expert biases sit flush against
    // +-0.5 = polynorm_bias_clamp, and the coefficients are signed.
    if (hparams.polynorm_sigmoid_weight) {
        w = ggml_sigmoid(ctx0, w);
    }
    if (clamp_bias && hparams.polynorm_bias_clamp > 0.0f) {
        b = ggml_clamp(ctx0, b, -hparams.polynorm_bias_clamp, hparams.polynorm_bias_clamp);
    }

    auto coef = [&](int64_t j) {
        return ggml_view_3d(ctx0, w, 1, w->ne[1], w->ne[2], w->nb[1], w->nb[2], j*w->nb[0]);
    };

    ggml_tensor * x2 = ggml_sqr(ctx0, x);
    ggml_tensor * x3 = ggml_mul(ctx0, x2, x);

    ggml_tensor * cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, x3, MOTIF3_POLY_EPS), coef(0));
    cur = ggml_add(ctx0, cur, ggml_mul(ctx0, ggml_rms_norm(ctx0, x2, MOTIF3_POLY_EPS), coef(1)));
    cur = ggml_add(ctx0, cur, ggml_mul(ctx0, ggml_rms_norm(ctx0, x,  MOTIF3_POLY_EPS), coef(2)));
    cur = ggml_add(ctx0, cur, b);
    cb(cur, "poly_norm", il);

    return cur;
}

ggml_tensor * llama_model_motif3::graph::build_attention(
        const llama_model & model,
        llm_graph_input_attn_kv_iswa * inp_attn,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        float kq_scale_full,
        float kq_scale_swa,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head_k       = hparams.n_embd_head_k();
    const int64_t n_embd_head_v       = hparams.n_embd_head_v();
    const int64_t n_head_kv           = hparams.n_head_kv(il);
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
    const int64_t kv_lora_rank        = hparams.n_lora_kv;

    const int64_t n_slots       = n_head / n_head_kv;                    // query heads per kv group
    const int64_t n_noise_slots = hparams.n_noise_heads / n_head_kv;     // noise heads per kv group
    const int64_t n_sig_slots   = n_slots - n_noise_slots;
    const int64_t n_sig_heads   = n_head - hparams.n_noise_heads;
    const int64_t nt            = cur->ne[1];

    // Per-layer RoPE. Full-attention layers get the YaRN treatment the reference builds
    // from `rope_scaling` (ramped inv_freq + position scaling by `factor`); the windowed
    // layers use plain RoPE on their own freq base with no YaRN at all.
    //
    // attn_factor cancels the mscale ggml's YaRN would fold into cos/sin, because the
    // equivalent mscale^2 correction is applied once in kq_scale instead (deepseek2 idiom).
    const bool  is_swa        = hparams.is_swa(il);
    const float freq_base_l   = is_swa ? hparams.rope_freq_base_train_swa : freq_base;
    const float freq_scale_l  = is_swa ? 1.0f : freq_scale;
    const float ext_factor_l  = is_swa ? 0.0f : ext_factor;
    const float attn_factor_l = (is_swa || ext_factor == 0.0f)
        ? 1.0f
        : 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float kq_scale      = is_swa ? kq_scale_swa : kq_scale_full;

    GGML_ASSERT(hparams.n_noise_heads % n_head_kv == 0);
    GGML_ASSERT(n_noise_slots == 1 && "only one noise head per kv group is supported");

    // q/gate shared low-rank projection
    ggml_tensor * q_latent = build_lora_mm(layer.wq_a, cur);
    q_latent = build_norm(q_latent, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(q_latent, "q_latent", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, q_latent);
    q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, nt);
    cb(q, "q", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, nt,
            ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k)*n_head,
            0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, nt,
            ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k)*n_head,
            ggml_row_size(q->type, n_embd_head_qk_nope));
    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_qk_rope, rope_type, n_ctx_orig,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast, beta_slow);
    cb(q_pe, "q_pe", il);

    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
    cb(Qcur, "Qcur", il);

    // shared kv latent + shared rope key
    ggml_tensor * kv_cmpr_pe = build_lora_mm(layer.wkv_a_mqa, cur);
    cb(kv_cmpr_pe, "kv_cmpr_pe", il);

    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, nt,
            kv_cmpr_pe->nb[1], 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, nt,
            kv_cmpr_pe->nb[1], kv_cmpr_pe->nb[1],
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_cmpr, "kv_cmpr", il);

    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_embd_head_qk_rope, rope_type, n_ctx_orig,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast, beta_slow);
    cb(k_pe, "k_pe", il);

    // decompress k_nope and v for all kv heads
    ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_b, kv_cmpr);
    cb(kv, "kv", il);

    ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head_kv, nt,
            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v)*n_head_kv,
            0);
    ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v, n_head_kv, nt,
            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v)*n_head_kv,
            ggml_row_size(kv->type, n_embd_head_qk_nope));
    Vcur = ggml_cont(ctx0, Vcur);
    cb(Vcur, "Vcur", il);

    ggml_tensor * k_pe_rep = ggml_repeat_4d(ctx0, ggml_cont(ctx0, k_pe),
            n_embd_head_qk_rope, n_head_kv, nt, 1);
    ggml_tensor * Kcur = ggml_concat(ctx0, k_nope, k_pe_rep, 0);
    cb(Kcur, "Kcur", il);

    // grouped attention (each kv group serves n_sig_slots signal heads and one noise head)
    ggml_tensor * attn = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(attn, "attn_heads", il);

    // split into signal and noise heads: [v, slot, group, n_tokens]
    attn = ggml_reshape_4d(ctx0, attn, n_embd_head_v, n_slots, n_head_kv, nt);

    ggml_tensor * signal = ggml_view_4d(ctx0, attn, n_embd_head_v, n_sig_slots, n_head_kv, nt,
            attn->nb[1], attn->nb[2], attn->nb[3], 0);
    ggml_tensor * noise = ggml_view_4d(ctx0, attn, n_embd_head_v, 1, n_head_kv, nt,
            attn->nb[1], attn->nb[2], attn->nb[3], n_sig_slots*attn->nb[1]);
    noise = ggml_cont(ctx0, noise);

    // differential attention: signal - sigmoid(lambda(x)) * noise
    ggml_tensor * lambda = build_lora_mm(layer.attn_lambda, cur);
    lambda = ggml_sigmoid(ctx0, lambda);
    lambda = ggml_reshape_4d(ctx0, lambda, 1, n_sig_slots, n_head_kv, nt);
    cb(lambda, "attn_lambda", il);

    ggml_tensor * noise_rep = ggml_repeat_4d(ctx0, noise, n_embd_head_v, n_sig_slots, n_head_kv, nt);
    ggml_tensor * out = ggml_sub(ctx0, signal, ggml_mul(ctx0, noise_rep, lambda));
    cb(out, "attn_diff", il);

    // output gate from the shared q latent
    ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, q_latent);
    gate = ggml_sigmoid(ctx0, gate);
    gate = ggml_reshape_4d(ctx0, gate, n_embd_head_v, n_sig_slots, n_head_kv, nt);
    out = ggml_mul(ctx0, out, gate);
    cb(out, "attn_gated", il);

    out = ggml_cont_2d(ctx0, out, n_embd_head_v*n_sig_heads, nt);
    out = build_lora_mm(layer.wo, out);
    cb(out, "attn_out", il);

    return out;
}

llama_model_motif3::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head_k = hparams.n_embd_head_k();
    const int64_t n_expert_used = hparams.n_expert_used;
    const int64_t hc            = hparams.dsv4_hc_mult;

    // The mscale^2 softmax correction applies only to the full-attention layers;
    // the windowed layers keep the plain 1/sqrt(d) scale (mscale = 0.1*ln(64)+1).
    const float mscale        = 0.1f * logf(1.0f / hparams.rope_freq_scale_train) + 1.0f;
    const float kq_scale_full = mscale * mscale / sqrtf(float(n_embd_head_k));
    const float kq_scale_swa  = 1.0f / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    auto * inp_attn = build_attn_inp_kv_iswa();

    // expand the embeddings into hc residual streams
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_hc_pre(inpL, layer.hc_attn_norm, layer.hc_attn_fn,
                layer.hc_attn_scale, layer.hc_attn_base, &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention(model, inp_attn, cur, inp_pos, kq_scale_full, kq_scale_swa, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL, layer.hc_ffn_norm, layer.hc_ffn_fn,
                layer.hc_ffn_scale, layer.hc_ffn_base, &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            ggml_tensor * gate = build_lora_mm(layer.ffn_gate, cur);
            ggml_tensor * up   = build_lora_mm(layer.ffn_up,   cur);
            gate = build_poly_norm(gate, layer.ffn_poly, layer.ffn_poly_b, /* clamp_bias = */ false, il);
            cur = build_poly_scale(ggml_mul(ctx0, gate, up));
            cur = build_lora_mm(layer.ffn_down, cur);
            cb(cur, "ffn_out", il);
        } else {
            // MoE with per-expert PolyNorm activation
            ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur); // [n_expert, nt]
            cb(logits, "ffn_moe_logits", il);

            ggml_tensor * probs = ggml_sigmoid(ctx0, logits);
            cb(probs, "ffn_moe_probs", il);

            // expert bias is used for selection only
            ggml_tensor * selection_probs = ggml_add(ctx0, probs, layer.ffn_exp_probs_b);
            cb(selection_probs, "ffn_moe_probs_biased", il);

            ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
            cb(selected_experts, "ffn_moe_topk", il);

            ggml_tensor * weights = ggml_get_rows(ctx0,
                    ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens), selected_experts); // [1, n_expert_used, nt]
            cb(weights, "ffn_moe_weights", il);

            if (hparams.expert_weights_norm) {
                weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
                ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
                weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
                weights = ggml_div(ctx0, weights, weights_sum);
                weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
                cb(weights, "ffn_moe_weights_norm", il);
            }
            if (hparams.expert_weights_scale != 1.0f) {
                weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
                cb(weights, "ffn_moe_weights_scaled", il);
            }

            ggml_build_forward_expand(gf, weights);

            ggml_tensor * cur3 = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

            ggml_tensor * gate = build_lora_mm_id(layer.ffn_gate_exps, cur3, selected_experts); // [n_ff_exp, n_expert_used, nt]
            cb(gate, "ffn_moe_gate", il);
            ggml_tensor * up = build_lora_mm_id(layer.ffn_up_exps, cur3, selected_experts);
            cb(up, "ffn_moe_up", il);

            // Per-expert PolyNorm: gather each routed expert's own coefficients
            // (ffn_poly_exps [3, n_expert], ffn_poly_exps_b [n_expert]) by the
            // selected expert ids, matching HF GroupedPolyNorm.forward_single(gate,
            // expert_idx). The trained coeffs differ substantially across experts
            // (std ~0.1-0.5, max |expert_i - expert_0| up to ~2.9), so the old
            // expert-0-for-all shortcut produced wrong activations => gibberish.
            // No-op on the untrained tiny (all coeffs 1/3). build_poly_norm handles
            // the gathered {3, n_expert_used, n_tokens} weight layout.
            // flatten the [n_expert_used, n_tokens] ids so get_rows uses the
            // non-batched form against the token-shared poly tables, then restore
            // the [., n_expert_used, n_tokens] layout build_poly_norm expects.
            ggml_tensor * sel_flat = ggml_reshape_1d(ctx0,
                    ggml_cont(ctx0, selected_experts), n_expert_used * n_tokens);
            ggml_tensor * poly_w = ggml_get_rows(ctx0, layer.ffn_poly_exps, sel_flat); // [3, neu*nt]
            poly_w = ggml_reshape_3d(ctx0, poly_w, 3, n_expert_used, n_tokens);
            ggml_tensor * poly_b = ggml_get_rows(ctx0,
                    ggml_reshape_2d(ctx0, layer.ffn_poly_exps_b, 1, n_expert), sel_flat); // [1, neu*nt]
            poly_b = ggml_reshape_3d(ctx0, poly_b, 1, n_expert_used, n_tokens);

            gate = build_poly_norm(gate, poly_w, poly_b, /* clamp_bias = */ true, il);
            cur = build_poly_scale(ggml_mul(ctx0, gate, up));

            ggml_tensor * experts = build_lora_mm_id(layer.ffn_down_exps, cur, selected_experts); // [n_embd, n_expert_used, nt]
            experts = ggml_mul(ctx0, experts, weights);
            cb(experts, "ffn_moe_weighted", il);

            ggml_build_forward_expand(gf, experts);

            ggml_tensor * moe_out = nullptr;
            for (int64_t i = 0; i < n_expert_used; ++i) {
                ggml_tensor * exp_i = ggml_view_2d(ctx0, experts, n_embd, n_tokens,
                        experts->nb[2], i*experts->nb[1]);
                moe_out = moe_out ? ggml_add(ctx0, moe_out, exp_i) : exp_i;
            }
            if (n_expert_used == 1) {
                moe_out = ggml_cont(ctx0, moe_out);
            }
            cb(moe_out, "ffn_moe_out", il);

            // shared expert (same PolyNorm MLP shape), applied to the ffn_norm output
            ggml_tensor * ffn_inp = ggml_reshape_2d(ctx0, cur3, n_embd, n_tokens);
            ggml_tensor * sh_gate = build_lora_mm(layer.ffn_gate_shexp, ffn_inp);
            ggml_tensor * sh_up   = build_lora_mm(layer.ffn_up_shexp,   ffn_inp);
            sh_gate = build_poly_norm(sh_gate, layer.ffn_poly_shexp, layer.ffn_poly_shexp_b, /* clamp_bias = */ false, il);
            ggml_tensor * ffn_shexp = build_poly_scale(ggml_mul(ctx0, sh_gate, sh_up));
            ffn_shexp = build_lora_mm(layer.ffn_down_shexp, ffn_shexp);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    if (cparams.embeddings_nextn_masked && inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    // collapse the residual streams
    cur = build_hc_mean(inpL);
    cb(cur, "hc_mean", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    // post-final-norm hidden state: the MTP draft head consumes this directly
    // (the Motif-3 MTP block has no hnorm, so the chained state is post-norm)
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// PolyNorm with shared {3}/{1} coefficients (MTP dense MLP)
static ggml_tensor * motif3_poly_norm_flat(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * w,
        ggml_tensor  * b) {
    GGML_ASSERT(w->ne[0] == 3);

    auto coef = [&](int64_t j) {
        return ggml_view_3d(ctx, w, 1, w->ne[1], w->ne[2], w->nb[1], w->nb[2], j*w->nb[0]);
    };

    ggml_tensor * x2 = ggml_sqr(ctx, x);
    ggml_tensor * x3 = ggml_mul(ctx, x2, x);

    ggml_tensor * cur = ggml_mul(ctx, ggml_rms_norm(ctx, x3, MOTIF3_POLY_EPS), coef(0));
    cur = ggml_add(ctx, cur, ggml_mul(ctx, ggml_rms_norm(ctx, x2, MOTIF3_POLY_EPS), coef(1)));
    cur = ggml_add(ctx, cur, ggml_mul(ctx, ggml_rms_norm(ctx, x,  MOTIF3_POLY_EPS), coef(2)));
    cur = ggml_add(ctx, cur, b);

    return cur;
}

// same differential MLA attention as the trunk. The MTP block is never a windowed
// layer (set_swa_pattern leaves the nextn layers dense), but it shares the model's
// iswa KV cache, so it must take the iswa input to hit the right cache half.
ggml_tensor * llama_model_motif3::graph_mtp::build_attention(
        const llama_model & model,
        llm_graph_input_attn_kv_iswa * inp_attn,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        float kq_scale,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head_k       = hparams.n_embd_head_k();
    const int64_t n_embd_head_v       = hparams.n_embd_head_v();
    // per-layer hparams arrays are only filled for the trunk: the MTP block
    // shares the uniform head layout of layer 0
    const int64_t n_head_kv           = hparams.n_head_kv(0);
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
    const int64_t kv_lora_rank        = hparams.n_lora_kv;

    const int64_t n_slots       = n_head / n_head_kv;
    const int64_t n_noise_slots = hparams.n_noise_heads / n_head_kv;
    const int64_t n_sig_slots   = n_slots - n_noise_slots;
    const int64_t n_sig_heads   = n_head - hparams.n_noise_heads;
    const int64_t nt            = cur->ne[1];

    // see the trunk build_attention for the per-layer RoPE / kq_scale rationale
    const bool  is_swa        = hparams.is_swa(il);
    const float freq_base_l   = is_swa ? hparams.rope_freq_base_train_swa : freq_base;
    const float freq_scale_l  = is_swa ? 1.0f : freq_scale;
    const float ext_factor_l  = is_swa ? 0.0f : ext_factor;
    const float attn_factor_l = (is_swa || ext_factor == 0.0f)
        ? 1.0f
        : 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float kq_scale_l    = is_swa ? 1.0f / sqrtf(float(n_embd_head_k)) : kq_scale;

    GGML_ASSERT(hparams.n_noise_heads % n_head_kv == 0);
    GGML_ASSERT(n_noise_slots == 1 && "only one noise head per kv group is supported");

    // q/gate shared low-rank projection
    ggml_tensor * q_latent = build_lora_mm(layer.wq_a, cur);
    q_latent = build_norm(q_latent, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(q_latent, "mtp_q_latent", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, q_latent);
    q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, nt);
    cb(q, "mtp_q", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, nt,
            ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k)*n_head,
            0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, nt,
            ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k)*n_head,
            ggml_row_size(q->type, n_embd_head_qk_nope));
    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_qk_rope, rope_type, n_ctx_orig,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast, beta_slow);
    cb(q_pe, "mtp_q_pe", il);

    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
    cb(Qcur, "mtp_Qcur", il);

    // shared kv latent + shared rope key
    ggml_tensor * kv_cmpr_pe = build_lora_mm(layer.wkv_a_mqa, cur);
    cb(kv_cmpr_pe, "mtp_kv_cmpr_pe", il);

    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, nt,
            kv_cmpr_pe->nb[1], 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, nt,
            kv_cmpr_pe->nb[1], kv_cmpr_pe->nb[1],
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_cmpr, "mtp_kv_cmpr", il);

    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_embd_head_qk_rope, rope_type, n_ctx_orig,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast, beta_slow);
    cb(k_pe, "mtp_k_pe", il);

    // decompress k_nope and v for all kv heads
    ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_b, kv_cmpr);
    cb(kv, "mtp_kv", il);

    ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head_kv, nt,
            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v)*n_head_kv,
            0);
    ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v, n_head_kv, nt,
            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v)*n_head_kv,
            ggml_row_size(kv->type, n_embd_head_qk_nope));
    Vcur = ggml_cont(ctx0, Vcur);
    cb(Vcur, "mtp_Vcur", il);

    ggml_tensor * k_pe_rep = ggml_repeat_4d(ctx0, ggml_cont(ctx0, k_pe),
            n_embd_head_qk_rope, n_head_kv, nt, 1);
    ggml_tensor * Kcur = ggml_concat(ctx0, k_nope, k_pe_rep, 0);
    cb(Kcur, "mtp_Kcur", il);

    ggml_tensor * attn = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale_l, il);
    cb(attn, "mtp_attn_heads", il);

    attn = ggml_reshape_4d(ctx0, attn, n_embd_head_v, n_slots, n_head_kv, nt);

    ggml_tensor * signal = ggml_view_4d(ctx0, attn, n_embd_head_v, n_sig_slots, n_head_kv, nt,
            attn->nb[1], attn->nb[2], attn->nb[3], 0);
    ggml_tensor * noise = ggml_view_4d(ctx0, attn, n_embd_head_v, 1, n_head_kv, nt,
            attn->nb[1], attn->nb[2], attn->nb[3], n_sig_slots*attn->nb[1]);
    noise = ggml_cont(ctx0, noise);

    // differential attention: signal - sigmoid(lambda(x)) * noise
    ggml_tensor * lambda = build_lora_mm(layer.attn_lambda, cur);
    lambda = ggml_sigmoid(ctx0, lambda);
    lambda = ggml_reshape_4d(ctx0, lambda, 1, n_sig_slots, n_head_kv, nt);
    cb(lambda, "mtp_attn_lambda", il);

    ggml_tensor * noise_rep = ggml_repeat_4d(ctx0, noise, n_embd_head_v, n_sig_slots, n_head_kv, nt);
    ggml_tensor * out = ggml_sub(ctx0, signal, ggml_mul(ctx0, noise_rep, lambda));
    cb(out, "mtp_attn_diff", il);

    // output gate from the shared q latent
    ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, q_latent);
    gate = ggml_sigmoid(ctx0, gate);
    gate = ggml_reshape_4d(ctx0, gate, n_embd_head_v, n_sig_slots, n_head_kv, nt);
    out = ggml_mul(ctx0, out, gate);
    cb(out, "mtp_attn_gated", il);

    out = ggml_cont_2d(ctx0, out, n_embd_head_v*n_sig_heads, nt);
    out = build_lora_mm(layer.wo, out);
    cb(out, "mtp_attn_out", il);

    return out;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Motif-3.
// No public reference forward exists for the MTP block; semantics follow the
// DeepSeek-V3 NextN convention adapted to the tensors actually present:
//   input_proj(concat(embed_norm(emb), h)) -> full Motif-3 decoder block
//   (differential MLA attention + dense PolyNorm MLP, plain residual, no MHC)
//   -> final_layernorm (stored as nextn.shared_head_norm) -> shared LM head.
// The checkpoint has no hnorm: the chained hidden state is the trunk's
// post-final-norm output, consumed as-is.
llama_model_motif3::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "MOTIF3 MTP requires n_layer_nextn > 0");

    const int64_t n_embd_head_k = hparams.n_embd_head_k();

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
                "nextn_layer_offset out of range [0, n_layer_nextn)");
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.shared_head_norm && "MTP block missing nextn.shared_head_norm");
    GGML_ASSERT(layer.ffn_gate && "MTP block missing dense ffn");

    // same kq_scale as the trunk (mscale^2 folded, see graph ctor)
    const float mscale   = 0.1f * logf(1.0f / hparams.rope_freq_scale_train) + 1.0f;
    const float kq_scale = mscale * mscale / sqrtf(float(n_embd_head_k));

    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->embd);
    ggml_set_name(inp->embd, "mtp_h_input");

    ggml_tensor * h_input  = inp->embd;
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    auto * inp_attn           = build_attn_inp_kv_iswa();

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_input, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    cur = build_attention(model, inp_attn, cur, inp_pos, kq_scale, il);

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    {
        ggml_tensor * gate = build_lora_mm(layer.ffn_gate, cur);
        ggml_tensor * up   = build_lora_mm(layer.ffn_up,   cur);
        gate = motif3_poly_norm_flat(ctx0, gate, layer.ffn_poly, layer.ffn_poly_b);
        cur = ggml_mul(ctx0, gate, up);
        cur = build_lora_mm(layer.ffn_down, cur);
        cb(cur, "mtp_ffn_out", il);
    }

    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_out", il);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    // the MTP block's trailing final_layernorm (stored as shared_head_norm)
    cur = build_norm(cur, layer.nextn.shared_head_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
