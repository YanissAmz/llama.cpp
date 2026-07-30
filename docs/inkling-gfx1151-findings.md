# Inkling-Small on gfx1151 — backend findings (2026-07-30 night)

## Headline

Two independent bugs found, both previously unknown, both reproducible:

1. **`-fa on` was costing 2.9x on Vulkan.** `FLASH_ATTN_EXT_BANDED` has no Vulkan
   kernel, so `-fa on` silently drops attention to the CPU in all 42 blocks.
   `-fa off` keeps it on the iGPU: **7.7 -> 22.1 t/s decode**. The "CPU-fallback
   baseline" was not a floor, it was a flag you opt into.

2. **HIP graph capture loses `GGML_PREC_F32_PEDANTIC`, and Inkling emits garbage
   because of it.** Root-caused, see below. This is a real upstream bug.

## The precision bug (the important one)

Inkling asks for pedantic F32 in three places (`src/models/inkling.cpp`):

    383  ggml_mul_mat_set_prec(rel, GGML_PREC_F32_PEDANTIC)          // banded rel logits
    535  ...ffn_gate_inp... GGML_PREC_F32_PEDANTIC                   // the EXPERT ROUTER
    655  model.output ... GGML_PREC_F32_PEDANTIC

All three source tensors are genuinely F32 in the GGUF (verified by reading the
tensor table: `ffn_gate_inp` = F32 [4096,258], `attn_rel_proj` = F32 [512,16]), so
the `src0->type == GGML_TYPE_F32` guard on the pedantic path is satisfied and the
hint *should* be honoured.

On ROCm it is not, and the model emits pure garbage
(`'???????...'`) at any context length, in BOTH `-fa on` and `-fa off`.

Bisect, in order:

- gemma-4-31B on the same HIP build -> coherent. **The HIP build is sound**; the bug
  is Inkling-specific, not a bad `AMDGPU_TARGETS` or a broken toolchain.
- `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` -> coherent. So it IS a compute-precision
  problem, not MoE routing logic, not the quant, not the template.
- `GGML_CUDA_CUBLAS_COMPUTE_TYPE=bf16` -> coherent. So it is specifically an f16
  **range** problem (bf16 has f32's exponent), not a mantissa problem.
- **`GGML_CUDA_DISABLE_GRAPHS=1` alone -> coherent, with the DEFAULT compute type.**

That last line is the root cause. With graphs enabled the pedantic request is lost
and the router matmul runs in f16; the router picks 6 experts out of 256 from
overflowed logits, so the wrong experts are selected and the output is garbage that
still has the *shape* of fluent text. It is the same class of defect the PR author
already documented for the banded op in `fattn-banded.cu:221`:

    "the tensor extent (not op_params) is authoritative after graph cloning"

i.e. op_params are known to go stale across graph cloning, and
`ggml_prec(dst->op_params[0])` at `ggml-cuda.cu:1501/1675/1865/1901` reads exactly
that. Note `ggml-cuda.cu:2035` (`dst_slice.op_params[0] = dst->op_params[0];`) is a
site that copies op_params deliberately — evidence the codebase already knows some
clone paths must carry them and suggests others do not.

**Why this matters beyond Inkling:** any model that relies on
`GGML_PREC_F32_PEDANTIC` is silently degraded on the CUDA/HIP graph path. Inkling
just happens to fail loudly because a router is a winner-take-all operation, so a
precision slip becomes a discrete wrong answer instead of a small numeric error.

## The banded FA numerical bug (second, separate)

`test-backend-ops -o FLASH_ATTN_EXT_BANDED` on ROCm0: **8/13 pass**.
All 5 failures share `kv_type=f16, rel_type=f32` — the mixed-type combination that
`fattn-banded.cu:211` routes to the MMA fast path instead of the standalone kernel:

    d=64  ,n_kv=64   ,rel_extent=512  ERR 0.485   <- badly wrong
    d=128 ,n_kv=16403,rel_extent=1024 ERR 0.0236
    d=128 ,n_kv=17024,rel_extent=1024 ERR 0.0280
    d=128 ,n_q=1 ,n_kv=17024          ERR 0.0255  <- THE DECODE SHAPE
    d=128 ,n_kv=32768,rel_extent=1024 ERR 0.0013  (marginal)

n_kv 8192 / 16384 / 16896 pass; 16403 / 17024 / 32768 fail. Consistent with a
split-K boundary rather than a plain indexing error.

Empirically this matches the server: ROCm `-fa on` produces `' is is is is...'` on a
6-token prompt while producing coherent text on a 3128-token prompt.
**Conclusion: `-fa on` on ROCm is not trustworthy.** Do not ship it.

`TOP_K` (the op behind 256-expert top-6 selection) is **not** a suspect, contrary
to my first read. I probed it as `ARGSORT_TOP_K`, which matches nothing, and
mistook "0/0 tests" for "untested op". Under its real name it has cases: the small
ones pass on ROCm, and only the large-ne cases (1035-2059) report
"not supported [ROCm0]" and fall back to CPU. Inkling's router is 256 wide, i.e.
squarely in the passing range. Ruled out.

## Measured matrix (3128-token prompt, 150 decoded, temp 0)

    backend  fa   compute/env                prefill   decode   correct?
    Vulkan   on   -                          158.8      7.7     yes (attn on CPU)
    Vulkan   off  -                          167.7     20.0     YES  <- best decode
    ROCm     off  default                    196.2     16.1     NO  (garbage)
    ROCm     on   default                    246.0     15.2     NO  (garbage)
    ROCm     off  CUBLAS_COMPUTE_TYPE=f32    110.0     16.0     yes
    ROCm     off  CUBLAS_COMPUTE_TYPE=bf16   182.6     16.1     YES  <- best correct prefill
    ROCm     on   CUBLAS_COMPUTE_TYPE=bf16   245.2     15.0     NO  (short-ctx garbage)
    ROCm     off  DISABLE_GRAPHS=1           181.2     16.4     yes

**Do not read the 246 as recoverable.** It appears ONLY in rows whose output is
garbage. 246 t/s is the speed of computing the wrong answer: it is what you get
when the router runs in f16. Any correct implementation does the pedantic matmuls
in f32/bf16 by definition, and that is precisely what costs the ~64 t/s. Both
correct ROCm rows land at ~182 regardless of which workaround gets you there.

Corollary: fixing the op_params/graph staleness is worth ~0 t/s. It is a real
upstream bug and a worthwhile contribution, but it is NOT a speed lever — the
cost is inherent to computing the router correctly, not to the graph path.
Note also that disabling graphs costs nothing measurable here (181.2 vs 182.6
prefill, 16.4 vs 16.1 decode) — both inside noise at n=1.

Best correct config today: **Vulkan, `-fa off`** for decode (20.0 t/s);
**ROCm + bf16, `-fa off`** for prefill (182.6 t/s).

## Decode vs context depth (the measurement that could have inverted the choice)

Every row above is at 3128 tokens. `-fa off` materializes attention, so decode
must decay with context — and the agentic bench lives at 10k-40k, not 3k. If it
collapsed there, benching in this config would degrade exactly where it matters.
Measured on Vulkan `-fa off`, `-c 32768`, cache_prompt off, 150 tok, temp 0:

    ctx      prefill      decode
    3128     159.0        19.86
    18008    154.4        16.14   (-19%)
    30008    130.6        14.36   (-28%)

It decays, it does not collapse. Vulkan `-fa off` stays the shipping config.
Quote **~14-16 t/s** as Inkling's realistic agentic decode, not the 20 headline.
(A >32k prompt returns HTTP 400 rather than degrading — respect the ctx bound.)

## Against the targets

Targets were 30 t/s decode / 300 t/s prefill.

Roofline: ~12B active params at IQ3_XXS ~= 4.6 GB read per token; Strix Halo
sustains ~210 GB/s => **~45 t/s is the hard decode ceiling**. At 20 t/s we are at
~45% of roofline, so 30 is physically possible but needs ~50% more, and **no flag
delivers that**. Both targets came up short and should be reported as short:

    decode   20.0 t/s at 3k, 14-16 t/s at agentic depth   vs target 30
    prefill  182.6 t/s (best CORRECT)                     vs target 300

The 300 is not "nearly there via the 246" — see above, the 246 is garbage-only.

### The one real remaining lever: MTP, and it is a port, not a flag

llama.cpp does have first-class MTP speculation (`--spec-type mtp`,
`--spec-draft-n-max`), and Inkling genuinely ships MTP layers, so this is the
credible route from 20 to 30. But it is NOT a matter of un-skipping tensors:

- `conversion/inkling.py:16` skips `model.mtp.*`, AND `InklingModel` does not set
  `supports_mtp_export` (`conversion/base.py:114` defaults it False). Only glm,
  hunyuan, qwen and step3 set it. So `--mtp` / `--no-mtp` are refused outright
  for this architecture.
- The runtime side is missing too: `LLAMA_CONTEXT_TYPE_MTP` keys off
  `hparams.n_layer_nextn`, which nothing sets for `LLM_ARCH_INKLING`.
- We do not even hold the source weights — `~/models/inkling-small/` contains
  only the UD-IQ3_XXS GGUF. Re-converting needs the upstream checkpoint
  (NVFP4 ~130G would fit in the 525G free; BF16 would not).

So MTP is the next *project* — converter export path + arch wiring + a re-convert
— not tonight's flag sweep. Expected payoff if it lands: the 1.4-2.2x that MTP
reports elsewhere would put decode at 28-44 t/s, i.e. squarely on target.

## Reproduce

    # correct, fastest decode
    bash ~/hermes-work/inkling-launch.sh 8360 32768 f16      # (Vulkan build, -fa off)

    # correct, fastest prefill
    LD_LIBRARY_PATH=build-hip/bin GGML_CUDA_CUBLAS_COMPUTE_TYPE=bf16 \
      ./build-hip/bin/llama-server -ngl 99 -fa off -ctk f16 -ctv f16 ...

HIP build recipe (ROCm 7.2, gfx1151) — the one non-obvious flag: this box has
gcc-14 installed *without* libstdc++ headers (only `/usr/include/c++/13`), and
ROCm's clang picks the newest gcc dir, so HIP compilation fails with
"Could not find standard C++ header 'cmath'". Pin it:

    cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \
      -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF \
      -DCMAKE_HIP_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
