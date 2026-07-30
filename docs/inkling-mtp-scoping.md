# Inkling MTP speculation — scoping (2026-07-31)

MTP is the only credible route from ~20 t/s to the 30 t/s target (roofline is
~45, no flag closes a 50% gap). This is the scoping pass, done read-only while
the r13 bench holds the box. **It is much cheaper than I first wrote.**

## Correction to the findings file

I wrote that MTP "needs the upstream checkpoint" and implied a 532 GB download
plus a full re-convert. That is wrong, and the error mattered — it is what made
MTP look like a multi-day project.

`model.safetensors.index.json` puts **every MTP tensor in a single shard**:

    mtp.safetensors    4,463,845,392 bytes   (4.46 GB)

versus `total_size` 531,912,898,740 (~532 GB) across 32 model shards + that one.
The MTP head is also **dense, not MoE** — the tensors are
`transformer_block.mlp.w13_dn.weight` / `mlp.w2_md.weight`, singular, with no
per-expert dimension. That is why 8 layers fit in 4.5 GB, and it is also what
makes the draft step cheap: a draft token does not pay the 256-expert gather.

## What Inkling's MTP actually is

From `config.json`:

    "mtp_config": {
      "num_nextn_predict_layers": 8,
      "chain_hidden_post_norm": false,
      "local_layer_ids": [0, 2, 4, 5, 6, 7]
    }

**Eight chained MTP layers**, with their own SWA pattern (6 of the 8 are local).
This is the one place Inkling does not fit the existing llama.cpp template:
`qwen35moe.cpp:556` and `cohere2moe.cpp:297` both assert `n_layer_nextn == 1`.
Stage the port around that — see below.

## Tensor naming maps cleanly onto llama.cpp's nextn convention

    model.mtp.layers.N.embed_norm.weight        -> nextn.enorm
    model.mtp.layers.N.hidden_norm.weight       -> nextn.hnorm
    model.mtp.layers.N.input_proj.weight        -> nextn.eh_proj
    model.mtp.layers.N.transformer_block.*      -> the standard blk.N.* tensors
        attn.{wq_du,wk_dv,wv_dv,wo_ud,wr_du}.weight, attn.{q,k}_norm.weight,
        attn.{k,v}_sconv.weight, attn.rel_logits_proj.proj,
        attn_norm, attn_sconv, mlp_norm, mlp_sconv,
        mlp.{w13_dn,w2_md}.weight, mlp.global_scale

`_Qwen35MtpMixin` (`conversion/qwen.py:539-620`) already does exactly this remap
for Qwen (`fc`->`eh_proj`, `pre_fc_norm_embedding`->`enorm`,
`pre_fc_norm_hidden`->`hnorm`), extends `block_count`, and emits
`add_nextn_predict_layers`. Inkling needs the same shape with its own three
names — a subclass, not a rewrite.

## Runtime: the draft can be a SEPARATE small GGUF

`llama-context.cpp:3578` refuses `LLAMA_CONTEXT_TYPE_MTP` when
`model->hparams.n_layer_nextn == 0`, and `server-context.cpp:1080` notes "MTP
draft context lives on the target model" — but it reads `params_dft.model.path`,
so the check applies to the **draft** model when `-md` is given. Qwen's
`mtp_only` path (`conversion/qwen.py:604-620`) exists precisely to emit a
standalone `mtp-*.gguf` for that. So we do **not** have to re-convert and
re-quantise the 92 GB target — the existing Unsloth UD-IQ3_XXS stays as-is.

The standalone draft does need the embedding/norm/head, which are not in
`mtp.safetensors`:

    model.llm.embed.weight      -> model-00030-of-00032.safetensors
    model.llm.norm.weight       -> model-00025-of-00032.safetensors
    model.llm.unembed.weight    -> model-00019-of-00032.safetensors

Three shards, ~50 GB, against 525 GB free. Total download ~55 GB, not 532.

## Plan

0. **Download** `mtp.safetensors`, `config.json`, tokenizer files, the index,
   and shards 19/25/30. `HF_HUB_DISABLE_XET=1` (the freeze landmine), and do it
   with the box otherwise idle.
1. **Converter** — `InklingMtpMixin` on `conversion/inkling.py`: set
   `supports_mtp_export = True`, drop `model.mtp.` from `_SKIP_PREFIXES`, remap
   the three head names above, emit `add_nextn_predict_layers`, and support
   `--mtp-only`. **Export layer 0 only at first** (`n_layer_nextn = 1`) so the
   existing single-block runtime assertions hold.
2. **Runtime** — `ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn,
   false)` in `src/models/inkling.cpp` load_hparams, the `mtp_only`/`trunk_only`
   tensor-loading branches, and a `graph_mtp` mirroring
   `qwen35moe.cpp:553+`. The wrinkle Inkling adds: the MTP block carries sconv
   and banded rel-attention like a trunk block, so reuse Inkling's own layer
   builder rather than Qwen's attention path, and it must honour
   `mtp_config.local_layer_ids` for the SWA flag.
3. **Measure** — `--spec-type mtp --spec-draft-n-max N`, sweep N, report
   acceptance and NET t/s. Judge on net t/s at agentic depth (14-16 t/s is the
   baseline to beat, not the 20 headline).
4. **Only then** consider the full 8-layer chain, which needs multi-block MTP
   support that no arch in the tree has today.

Expected payoff if stage 3 lands: MTP reports 1.4-2.2x elsewhere; from a 14-16
t/s agentic baseline that is 20-35 t/s, i.e. the target is in reach but not
guaranteed. Verify is bandwidth-bound on Strix Halo, so assume the low end.
