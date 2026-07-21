from __future__ import annotations

import re

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("MotifForCausalLM")
class Motif3Model(TextModel):
    model_arch = gguf.MODEL_ARCH.MOTIF3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MTP layers are stored as extra blocks after the main ones
        self.n_main_layers = self.hparams["num_hidden_layers"]
        self.block_count = self.n_main_layers + self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        # buffers for fusing the MHC hyper-connection tensors
        self._hc_parts: dict[tuple[int, str], dict[str, Tensor]] = {}

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_leading_dense_block_count(hparams["n_dense_first_layers"])

        # GDLA (MLA-style low-rank attention with grouped differential heads)
        self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])
        # the base class sets both key/value length to head_dim; v_head_dim differs
        self.gguf_writer.add_value_length(hparams["v_head_dim"])
        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])
        self.gguf_writer.add_attention_noise_head_count(hparams["num_noise_heads"])

        # sliding window attention (interleaved: (il + 1) % period != 0 -> SWA)
        if hparams.get("use_sliding_window") and hparams.get("sliding_window") is not None:
            self.gguf_writer.add_sliding_window(hparams["sliding_window"])
            self.gguf_writer.add_sliding_window_pattern(hparams.get("sliding_window_period", 4))

        # MoE (expert count and sigmoid gating are set by the base class)
        self.gguf_writer.add_expert_used_count(hparams["experts_top_k"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(hparams["num_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hparams["route_scale"])
        self.gguf_writer.add_expert_weights_norm(hparams["route_norm"])

        # MHC hyper-connections
        self.gguf_writer.add_hyper_connection_count(hparams["mhc_expansion_rate"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(hparams["mhc_sinkhorn_iters"])
        self.gguf_writer.add_hyper_connection_epsilon(1e-8)

        # MTP
        self.gguf_writer.add_nextn_predict_layers(hparams.get("num_nextn_predict_layers", 0))

        # mscale for attention scaling (DeepSeek-style YaRN correction, applied
        # even when rope frequency scaling is disabled)
        rope_factor = hparams.get("rope_factor", 1.0)
        mscale = hparams.get("mscale", 1.0)
        if hparams["max_position_embeddings"] > hparams.get("original_seq_len", 32768) and rope_factor > 1.0:
            self.gguf_writer.add_rope_scaling_yarn_log_mul(0.1 * mscale)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # keep every small side tensor exact: hyper-connections, polynorm
        # coefficients, lambda projections, expert bias
        for marker in ("hc_attn_", "hc_ffn_", "ffn_poly", "attn_lambda", "exp_probs_b"):
            if marker in new_name:
                return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def _fuse_hc(self, bid: int, kind: str) -> Iterable[tuple[str, Tensor]]:
        parts = self._hc_parts[(bid, kind)]
        if len(parts) < 10:
            return

        e = self.hparams["mhc_expansion_rate"]
        # fn rows: [proj_pre (E); proj_post (E); proj_res (E*E)] to match the
        # deepseek4 hyper-connection layout (pre at 0, post at E, res at 2*E)
        fn = torch.cat([parts["proj_pre"], parts["proj_post"], parts["proj_res"]], dim=0).float()
        base = torch.cat([parts["bias_pre"], parts["bias_post"], parts["bias_res"].reshape(e * e)], dim=0).float()
        scale = torch.cat([parts["alpha_pre"], parts["alpha_post"], parts["alpha_res"]], dim=0).float()

        prefix = f"blk.{bid}.hc_{kind}"
        yield f"{prefix}_fn.weight", fn
        yield f"{prefix}_base.weight", base
        yield f"{prefix}_scale.weight", scale
        yield f"{prefix}_norm.weight", parts["rms_norm"]
        del self._hc_parts[(bid, kind)]

    _attn_map = {
        "wq_a": "attn_q_a",
        "q_norm": "attn_q_a_norm",
        "wq_b": "attn_q_b",
        "wq_b_gate": "attn_gate",
        "wkv_a": "attn_kv_a_mqa",
        "kv_norm": "attn_kv_a_norm",
        "lambda_proj": "attn_lambda",
        "wo": "attn_output",
    }

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "model.embed_tokens.weight":
            yield "token_embd.weight", data_torch
            return
        if name == "model.norm.weight":
            yield "output_norm.weight", data_torch
            return
        if name == "lm_head.weight":
            yield "output.weight", data_torch
            return

        # MTP layers: model.mtp_layers.<i>.* -> blk.<n_main_layers + i>.*
        if (match := re.match(r"model\.mtp_layers\.(\d+)\.(.+)", name)) is not None:
            bid = self.n_main_layers + int(match.group(1))
            sub = match.group(2)
            mtp_direct = {
                "embed_norm.weight":       f"blk.{bid}.nextn.enorm.weight",
                "input_proj.weight":       f"blk.{bid}.nextn.eh_proj.weight",
                "final_layernorm.weight":  f"blk.{bid}.nextn.shared_head_norm.weight",
            }
            if sub in mtp_direct:
                yield mtp_direct[sub], data_torch
                return
            name = f"model.layers.{bid}.{sub}"

        match = re.match(r"model\.layers\.(\d+)\.(.+)", name)
        assert match is not None, f"unexpected tensor name: {name}"
        bid = int(match.group(1))
        sub = match.group(2)

        # hyper-connections: buffer the 10 raw parts, emit fused fn/base/scale + norm
        if (hc := re.match(r"mhc_(attn|ffn)\.(\w+)(?:\.weight)?$", sub)) is not None:
            key = (bid, hc.group(1))
            self._hc_parts.setdefault(key, {})[hc.group(2)] = data_torch
            yield from self._fuse_hc(bid, hc.group(1))
            return

        if sub == "input_layernorm.weight":
            yield f"blk.{bid}.attn_norm.weight", data_torch
            return
        if sub == "post_attention_layernorm.weight":
            yield f"blk.{bid}.ffn_norm.weight", data_torch
            return

        if (attn := re.match(r"self_attn\.(\w+)\.weight$", sub)) is not None:
            part = attn.group(1)
            if part == "wkv_b":
                # split for MLA absorption: k_b transposed per head, v_b as is
                n_head_kv = self.hparams["num_key_value_heads"]
                v_head_dim = self.hparams["v_head_dim"]
                qk_nope_head_dim = self.hparams["head_dim"] - self.hparams["qk_rope_head_dim"]
                assert data_torch.shape[0] == n_head_kv * (qk_nope_head_dim + v_head_dim)
                kv_b = data_torch.view(n_head_kv, qk_nope_head_dim + v_head_dim, data_torch.shape[-1])
                k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
                yield f"blk.{bid}.attn_k_b.weight", k_b.transpose(1, 2).contiguous()
                yield f"blk.{bid}.attn_v_b.weight", v_b.contiguous()
                return
            yield f"blk.{bid}.{self._attn_map[part]}.weight", data_torch
            return

        # dense FFN (first k layers and MTP layer)
        if (mlp := re.match(r"mlp\.(gate_proj|up_proj|down_proj)\.weight$", sub)) is not None:
            yield f"blk.{bid}.ffn_{mlp.group(1).removesuffix('_proj')}.weight", data_torch
            return
        if sub == "mlp.act_fn.weight":
            yield f"blk.{bid}.ffn_poly.weight", data_torch
            return
        if sub == "mlp.act_fn.bias":
            yield f"blk.{bid}.ffn_poly.bias", data_torch
            return

        # MoE
        if sub == "moe.router.gate.weight":
            yield f"blk.{bid}.ffn_gate_inp.weight", data_torch
            return
        if sub == "moe.expert_bias":
            yield f"blk.{bid}.exp_probs_b.bias", data_torch
            return
        if sub == "moe.experts.gate_up_proj":
            # fused (n_expert, 2*n_ff_exp, n_embd) -> split into gate and up
            gate, up = data_torch.chunk(2, dim=1)
            yield f"blk.{bid}.ffn_gate_exps.weight", gate.contiguous()
            yield f"blk.{bid}.ffn_up_exps.weight", up.contiguous()
            return
        if sub == "moe.experts.down_proj":
            yield f"blk.{bid}.ffn_down_exps.weight", data_torch
            return
        if sub == "moe.experts.act_fn.weight":
            yield f"blk.{bid}.ffn_poly_exps.weight", data_torch
            return
        if sub == "moe.experts.act_fn.bias":
            yield f"blk.{bid}.ffn_poly_exps.bias", data_torch.reshape(-1)
            return
        if (shexp := re.match(r"moe\.shared_experts\.(gate_proj|up_proj|down_proj)\.weight$", sub)) is not None:
            yield f"blk.{bid}.ffn_{shexp.group(1).removesuffix('_proj')}_shexp.weight", data_torch
            return
        if sub == "moe.shared_experts.act_fn.weight":
            yield f"blk.{bid}.ffn_poly_shexp.weight", data_torch
            return
        if sub == "moe.shared_experts.act_fn.bias":
            yield f"blk.{bid}.ffn_poly_shexp.bias", data_torch
            return

        raise ValueError(f"unhandled tensor: {name}")

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._hc_parts:
            raise ValueError(f"unprocessed hyper-connection parts: {list(self._hc_parts.keys())}")
