#!/usr/bin/env python3
"""B2 — Escha checkpoint -> GGUF writer (Python only).

Writes the 2-bit escha codes into a GGUF, band-major (see write_linear):
  <name>.weight     : raw escha_code buffer, raw_dtype = 43 (K=2) / 44 (K=3), cf ggml.h
                      ne = (IC, OC), bytes band-major (OC/16, IC/16, 16*K)
  <name>.escha_in   : F32 [IC]   = s_in * rin   (folded in float32, D5)
  <name>.escha_out  : F32 [OC]   = s_out * rout (folded in float32, D5)
  <name>.bias       : F32 [OC]   = the checkpoint's STANDARD bias (NOT an escha
                      extra; the oracle carries it under the same llama.cpp
                      name and the loader already applies it).

The converter's tensor transforms are reproduced here (conversion/qwen.py,
Qwen3_5TextModel <- _LinearAttentionVReorderBase <- Qwen3NextModel), because
the llama.cpp qwen35 loader expects the CONVERTED forms, not the HF ones:
  - RMS norms +1.0 (attn_norm, post_attention_norm, attn_q/k_norm, output_norm;
    NOT linear_attn.norm) — loader subtracts 1;
  - ssm_a = -exp(A_log), v-head reorder on A_log / dt_bias / in_proj_a /
    in_proj_b / conv1d v-channels;
  - linear_attn projections (in_proj_qkv, in_proj_z, out_proj): the 48 v-heads
    go from HF grouped-by-k-head order to tiled order (16 groups of 3 ->
    3 groups of 16), on code rows/cols, escha_in/escha_out and the qkv/z biases.
All verified value-exact against escha-oracle-q8_0.gguf (see m5a_datacheck.py).
Embeddings / lm_head arrive ALREADY rowwise-int8-quantized in the checkpoint
(`*.weight_int8` + per-row `*.weight_scale` F16): re-encoded EXACTLY as Q8_0
(the row scale repeated over the row's 32-value blocks). Everything else
non-quantized: F32.

Change the quant type ids in ONE place: ESCHA_TYPE_ID below.
"""
import argparse
import json
import random
import re
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

GGUF_PY = Path("/home/yaniss/lab/llamacpp-q38-master/gguf-py")
sys.path.insert(0, str(GGUF_PY))
import gguf  # noqa: E402

PORT = Path(__file__).resolve().parent
sys.path.insert(0, str(PORT))
import escham_cpu  # noqa: E402

CKPT_DIR = Path("/home/yaniss/models/qwen3.8-27b-escha-w2")  # brief said w/, but
# w/ holds only the escha+sglang CODE trees; the weights repo is flat, here.
DEFAULT_OUT = PORT / "escha-qwen35.gguf"
ARCH = "qwen35"

# ---------------------------------------------------------------------------
# Quant type ids — THE place to change. MUST mirror ggml/include/ggml.h:
#   GGML_TYPE_ESCHAM_2 = 43, GGML_TYPE_ESCHAM_3 = 44.
# NOTE 43/44 are fork-only: upstream hands out the next free ids and will
# collide. Renumber (or mark the GGUF fork-only) before any publication.
# ---------------------------------------------------------------------------
ESCHA_TYPE_ID = {2: 43, 3: 44}


def escha_type_id(K: int) -> int:
    return ESCHA_TYPE_ID[K]


# Suffixes owned by write_linear. ".bias" is NOT here: "linear_attn.dt_bias"
# also ends in "bias" and a blanket match silently dropped it from the GGUF.
# The coded projections' own biases are excluded by prefix in other_keys().
SUFFIXES = (".escha_code", ".escha_rin", ".escha_rout", ".escha_s_in",
            ".escha_s_out", ".escha_config")
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


# ---------------------------------------------------------------------------
# Checkpoint access (lazy: key index only, never a whole shard in RAM)
# ---------------------------------------------------------------------------
class Checkpoint:
    def __init__(self, root: Path):
        self.shards = sorted(root.rglob("*.safetensors"))
        if not self.shards:
            raise SystemExit(f"no *.safetensors under {root}")
        self.cfg_path = next(iter(root.rglob("config.json")), None)
        self.index = {}  # tensor key -> shard path (str)
        for p in self.shards:
            with safe_open(str(p), framework="numpy") as f:
                for k in f.keys():
                    if k in self.index:
                        raise SystemExit(f"duplicate key across shards: {k}")
                    self.index[k] = str(p)
        self._handles = {}

    def _handle(self, shard: str):
        h = self._handles.get(shard)
        if h is None:
            h = safe_open(shard, framework="numpy")
            self._handles[shard] = h
        return h

    def get(self, key: str) -> np.ndarray:
        return self._handle(self.index[key]).get_tensor(key)

    def slice(self, key: str):
        return self._handle(self.index[key]).get_slice(key)

    def close(self):
        self._handles.clear()

    # ---- grouping -------------------------------------------------------
    def linear_prefixes(self, limit: int | None):
        """prefix -> None, for every <name> carrying an .escha_code."""
        out = {}
        for k in self.index:
            if not k.endswith(".escha_code"):
                continue
            prefix = k[: -len(".escha_code")]
            if limit is not None:
                m = LAYER_RE.search(prefix)
                if m and int(m.group(1)) >= limit:
                    continue
            out[prefix] = None
        return out

    def other_keys(self, limit: int | None):
        """Non-escha tensors (norms, embeddings, lm_head...)."""
        coded = {k[: -len(".escha_code")] for k in self.index
                 if k.endswith(".escha_code")}
        out = []
        for k in self.index:
            if k.endswith(SUFFIXES):
                continue
            if k.endswith(".bias") and k[: -len(".bias")] in coded:
                continue  # write_linear owns it
            if limit is not None:
                m = LAYER_RE.search(k)
                if m and int(m.group(1)) >= limit:
                    continue
            out.append(k)
        return sorted(out)

    def escha_config(self, prefix: str) -> tuple:
        cfg = self.get(prefix + ".escha_config").astype(np.int64).ravel()
        L, K, V, cb, IC, OC = (int(x) for x in cfg[:6])
        return L, K, V, cb, IC, OC


# ---------------------------------------------------------------------------
# HF -> llama.cpp tensor names
#
# TensorNameMap resolves only a handful of these (the SSM half of qwen35 is
# not in it), so the table is explicit. It is not guessed: the produced name
# set was diffed against escha-oracle-q8_0.gguf and matches its 1251 names
# exactly, plus 800 extras (.escha_in / .escha_out, 2 per coded projection).
# ---------------------------------------------------------------------------
LAYER_NAMES = {
    "self_attn.q_proj":        "attn_q",
    "self_attn.k_proj":        "attn_k",
    "self_attn.v_proj":        "attn_v",
    "self_attn.o_proj":        "attn_output",
    "self_attn.q_norm":        "attn_q_norm",
    "self_attn.k_norm":        "attn_k_norm",
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z":   "attn_gate",
    "linear_attn.out_proj":    "ssm_out",
    "linear_attn.in_proj_a":   "ssm_alpha",
    "linear_attn.in_proj_b":   "ssm_beta",
    "linear_attn.conv1d":      "ssm_conv1d",
    "linear_attn.norm":        "ssm_norm",
    "linear_attn.A_log":       "ssm_a",
    "linear_attn.dt_bias":     "ssm_dt.bias",
    "mlp.gate_proj":           "ffn_gate",
    "mlp.up_proj":             "ffn_up",
    "mlp.down_proj":           "ffn_down",
    "input_layernorm":         "attn_norm",
    "post_attention_layernorm": "post_attention_norm",
}

GLOBAL_NAMES = {
    "model.embed_tokens": "token_embd",
    "model.norm":         "output_norm",
    "lm_head":            "output",
}

# ---------------------------------------------------------------------------
# Converter value transforms (conversion/qwen.py :: Qwen3NextModel +
# _LinearAttentionVReorderBase; active here because linear_num_key_heads=16 !=
# linear_num_value_heads=48). The llama.cpp qwen35 loader expects the CONVERTED
# forms, so raw checkpoint values would be silently wrong:
#   - RMS norms stored +1.0 (loader subtracts 1); NOT linear_attn.norm
#   - ssm_a = -exp(A_log)
#   - the 48 v-heads go grouped-by-k-head (16 groups of 3) -> tiled (3 of 16),
#     on A_log/dt_bias/in_proj_a/in_proj_b rows, conv1d v-channels, the OC side
#     of in_proj_qkv/in_proj_z (rows, codes, escha_out, bias) and the IC side
#     of out_proj. Validated against the oracle in m5a_datacheck.py.
# ---------------------------------------------------------------------------
_N_K_HEADS, _N_V_HEADS, _V_HEAD_DIM = 16, 48, 128   # config.json, linear_attn
VPERM48 = np.arange(_N_V_HEADS).reshape(_N_K_HEADS, _N_V_HEADS // _N_K_HEADS) \
                 .transpose(1, 0).ravel()            # grouped -> tiled, heads
R192 = (VPERM48[:, None] * _V_HEAD_DIM
        + np.arange(_V_HEAD_DIM)[None, :]).ravel()   # rows within heads
T48 = (VPERM48[:, None] * (_V_HEAD_DIM // 16)
       + np.arange(_V_HEAD_DIM // 16)[None, :]).ravel()  # 16-row code tiles


def convert_transform(name: str, a: np.ndarray) -> np.ndarray:
    """Mapped-name F32 tensor: checkpoint values -> loader-convention values."""
    if name.endswith(("attn_norm.weight", "post_attention_norm.weight",
                      "attn_q_norm.weight", "attn_k_norm.weight")) \
            or name == "output_norm.weight":
        return (a + np.float32(1.0)).astype(np.float32)
    if re.fullmatch(r"blk\.\d+\.ssm_a", name):
        return np.take((-np.exp(a.astype(np.float32))).astype(np.float32),
                       VPERM48, axis=0)
    if re.fullmatch(r"blk\.\d+\.ssm_dt\.bias", name):
        return np.take(a, VPERM48, axis=0)
    if re.fullmatch(r"blk\.\d+\.ssm_(alpha|beta)\.weight", name):
        # the checkpoint flattens these; llama.cpp wants (n_embd, n_v_heads),
        # i.e. numpy (n_v_heads, n_embd). Raveling back drops the shape and the
        # loader then reads one long row.
        return np.take(a.reshape(_N_V_HEADS, -1), VPERM48, axis=0)
    if re.fullmatch(r"blk\.\d+\.ssm_conv1d\.weight", name):
        a = a.reshape(a.shape[0], -1)                  # (channels, kernel)
        v0 = _N_K_HEADS * 2 * _V_HEAD_DIM              # q|k channels first
        a[v0:] = a[v0:].reshape(_N_V_HEADS, _V_HEAD_DIM, a.shape[1]) \
            [VPERM48].reshape(-1, a.shape[1])
        return a
    m = re.fullmatch(r"blk\.\d+\.(attn_qkv|attn_gate)\.bias", name)
    if m:                                              # v-part rows only
        n_qk = a.shape[0] - _N_V_HEADS * _V_HEAD_DIM
        idx = np.concatenate([np.arange(n_qk), n_qk + R192])
        return np.take(a, idx, axis=0)
    return a


_SUFFIXES = (".escha_in", ".escha_out", ".weight_int8", ".weight_scale",
             ".weight", ".bias")


def gguf_name(key: str) -> str:
    """Checkpoint tensor key -> llama.cpp tensor name. Never guesses: an
    unknown key aborts rather than landing under a name nothing reads."""
    k = key.replace("model.language_model.", "model.")
    base, tail = k, ""
    for sfx in _SUFFIXES:
        if k.endswith(sfx):
            base, tail = k[: -len(sfx)], sfx
            break
    # int8 + its row scale become one Q8_0 tensor
    if tail in (".weight_int8", ".weight_scale"):
        tail = ".weight"
    m = re.match(r"model\.layers\.(\d+)\.(.+)$", base)
    if m:
        n, rest = m.group(1), m.group(2)
        assert rest in LAYER_NAMES, f"unmapped tensor: {key}"
        t = LAYER_NAMES[rest]
        if t.endswith(".bias") or t == "ssm_a":
            return f"blk.{n}.{t}"   # already carries its own suffix
        return f"blk.{n}.{t}{tail or '.weight'}"
    assert base in GLOBAL_NAMES, f"unmapped tensor: {key}"
    return GLOBAL_NAMES[base] + (tail or ".weight")


# ---------------------------------------------------------------------------
# Metadata (mirrors conversion/qwen.py :: Qwen3_5ForConditionalGeneration)
# ---------------------------------------------------------------------------
ORACLE = Path("/home/yaniss/models/qwen3.8-27b-escha-gguf/escha-oracle-q8_0.gguf")


def set_metadata(w: "gguf.GGUFWriter", cp: Checkpoint, codebooks=()):
    """Copy the oracle's KV verbatim.

    The oracle is convert_hf_to_gguf's own output for this checkpoint, so its
    KV is by construction what the qwen35 loader expects -- including the six
    ssm.* keys, full_attention_interval and rope.dimension_sections that a
    hand-written metadata block quietly omits. Rebuilding it from config.json
    means reimplementing the converter and getting to find out which key was
    missed at load time.
    """
    if not ORACLE.exists():
        raise SystemExit(f"[fatal] oracle not found: {ORACLE}")
    r = gguf.GGUFReader(str(ORACLE))
    skipped = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
               "general.file_type"}
    n = 0
    for key, field in r.fields.items():
        if key in skipped:
            continue
        val = field.contents()
        if val is None:
            print(f"[warn] KV {key} unreadable, skipped")
            continue
        vt = field.types[0]
        if vt == gguf.GGUFValueType.ARRAY:
            w.add_array(key, val)
        else:
            w.add_key_value(key, val, vt)
        n += 1
    del r
    # our own file type, and the codebook id the decoder needs
    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q2_K)
    if codebooks:
        assert len(codebooks) == 1, f"mixed codebooks: {codebooks}"
        w.add_uint32("escha.codebook", codebooks.pop())
    print(f"[kv] {n} keys copied from the oracle")



# ---------------------------------------------------------------------------
# Tensor writers
# ---------------------------------------------------------------------------
def copy_f32(cp: Checkpoint, key: str, w: "gguf.GGUFWriter"):
    """Copy a non-quantized tensor as F32, chunked (never a whole shard)."""
    sl = cp.slice(key)
    shape = tuple(int(x) for x in sl.get_shape())
    n0 = shape[0]
    tail = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    step = max(1, min(n0, (1 << 20) // max(1, tail)))
    out = np.empty(shape, dtype=np.float32)
    for i in range(0, n0, step):
        j = min(i + step, n0)
        out[i:j] = np.asarray(sl[i:j], dtype=np.float32)
    # HF keeps the depthwise conv as (channels, 1, kernel); llama.cpp declares
    # it 2-D and check_tensor_dims rejects the singleton. Drop it.
    if out.ndim > 2:
        squeezed = out.reshape([d for d in out.shape if d != 1])
        assert squeezed.ndim == 2, (key, out.shape)
        assert "conv1d" in key, f"unexpected >2D tensor: {key} {out.shape}"
        out = squeezed
    name = gguf_name(key)
    w.add_tensor(name, convert_transform(name, out))


def copy_f16(cp: Checkpoint, key: str, w: "gguf.GGUFWriter"):
    """Verbatim F16 copy (checkpoint-native scales)."""
    sl = cp.slice(key)
    shape = tuple(int(x) for x in sl.get_shape())
    w.add_tensor(gguf_name(key), np.asarray(sl[:], dtype=np.float16))


def write_int8_row(cp: Checkpoint, key: str, w: "gguf.GGUFWriter"):
    """`<x>.weight_int8` + `<x>.weight_scale` -> `<x>.weight` as Q8_0.

    The checkpoint stores a per-ROW fp16 scale and int8 values. Q8_0 stores a
    per-32 fp16 scale and int8 values. Repeating the row scale across the
    row's 32-value blocks is therefore an EXACT re-encoding, not a requant:
    same int8 payload, same fp16 scale, zero error. It also spares us a
    custom type -- every backend already has Q8_0 kernels.
    """
    assert key.endswith(".weight_int8"), key
    name = key[: -len("_int8")]
    sl = cp.slice(key)
    n_row, n_col = (int(x) for x in sl.get_shape())
    assert n_col % 32 == 0, (key, n_col)
    scale = np.asarray(cp.get(name + "_scale"), dtype=np.float16).reshape(-1)
    assert scale.size == n_row, (key, scale.shape)

    nb = n_col // 32
    blob = np.empty((n_row, nb, 34), dtype=np.uint8)
    step = max(1, (1 << 22) // n_col)
    for i in range(0, n_row, step):
        j = min(i + step, n_row)
        q = np.ascontiguousarray(np.asarray(sl[i:j], dtype=np.int8))
        blob[i:j, :, 2:] = q.reshape(j - i, nb, 32).view(np.uint8)
        d = np.repeat(scale[i:j, None], nb, axis=1).view(np.uint8).reshape(j - i, nb, 2)
        blob[i:j, :, :2] = d
    # ssm_alpha/ssm_beta are linear_attn projections with one row per V head:
    # they need the same grouped->tiled reorder as the rest. convert_transform
    # cannot do it here, the rows are already Q8_0 blocks.
    gname = gguf_name(name)
    if re.fullmatch(r"blk\.\d+\.ssm_(alpha|beta)\.weight", gname):
        assert n_row == _N_V_HEADS, (name, n_row)
        blob = np.take(blob, VPERM48, axis=0)

    # data is uint8, so gguf reads raw_shape as a BYTE shape and converts it
    # back to (n_row, n_col) itself.
    w.add_tensor(gname, blob.reshape(-1),
                 raw_shape=(n_row, nb * 34),
                 raw_dtype=gguf.GGMLQuantizationType.Q8_0)


def write_linear(cp: Checkpoint, prefix: str, w: "gguf.GGUFWriter"):
    L, K, V, cb, IC, OC = cp.escha_config(prefix)
    # mapped base name WITHOUT suffix, e.g. blk.8.attn_qkv
    base = gguf_name(prefix + ".weight").rpartition(".")[0]
    kind = base.split(".", 2)[2]
    v_oc = kind in ("attn_qkv", "attn_gate")       # OC side carries v-heads
    v_ic = kind == "ssm_out"                       # IC side carries v-heads

    code = np.asarray(cp.get(prefix + ".escha_code"), dtype=np.int16)
    # Checkpoint order is (IC//16, OC//16, 16*K): IC-tile major. A ggml row is
    # one OC band (D2: ne0 = IC*16, ne1 = OC/16), so the band's IC tiles must be
    # contiguous -> transpose the two tile axes. NOT a verbatim copy.
    assert code.shape == (IC // 16, OC // 16, 16 * K), (prefix, code.shape)
    flat = np.ascontiguousarray(code.transpose(1, 0, 2))
    # v-head reorder on the coded payload too: whole 16-row tiles move
    # together (the 16x16 code tiles never mix rows across a head boundary).
    # attn_qkv/attn_gate OC side = q|k bands first: permute the v bands only.
    if v_oc:
        n_qk_bands = (OC - _N_V_HEADS * _V_HEAD_DIM) // 16
        # each v-head owns DV//16 contiguous bands: permute the head axis
        v = flat[n_qk_bands:].reshape(_N_V_HEADS, _V_HEAD_DIM // 16,
                                      IC // 16, 16 * K)[VPERM48]
        flat = np.concatenate(
            [flat[:n_qk_bands], v.reshape(-1, IC // 16, 16 * K)], axis=0)
    if v_ic:  # ssm_out IC side is all-v (out_proj input = z output)
        flat = flat.reshape(OC // 16, _N_V_HEADS, _V_HEAD_DIM // 16,
                            16 * K)[:, VPERM48]
    flat = np.ascontiguousarray(flat).reshape(-1)
    expect = IC * OC * K // 16  # int16 elements: 16*K words per 16x16 tile
    assert flat.size == expect, (prefix, flat.size, expect)
    # raw_shape is numpy-order (rows=ne1, cols=ne0) -> ne = (IC, OC), the
    # natural ggml shape. The band-major byte order above is what the
    # tile-aware mul_mat branch expects; blck_size 16 / type_size 2*K makes
    # ggml_row_size land exactly on IC*K/8 bytes per output row.
    w.add_tensor(base + ".weight", flat,
                 raw_shape=(OC, IC),
                 raw_dtype=escha_type_id(K))

    rin = cp.get(prefix + ".escha_rin").astype(np.float32).ravel()
    s_in = cp.get(prefix + ".escha_s_in").astype(np.float32).ravel()
    assert rin.size == IC and s_in.size == IC, prefix
    ein = (s_in * rin).astype(np.float32)
    if v_ic:
        ein = np.take(ein, R192, axis=0)
    w.add_tensor(base + ".escha_in", ein)

    rout = cp.get(prefix + ".escha_rout").astype(np.float32).ravel()
    s_out = cp.get(prefix + ".escha_s_out").astype(np.float32).ravel()
    assert rout.size == OC and s_out.size == OC, prefix
    eout = (s_out * rout).astype(np.float32)
    if v_oc:
        eout = np.concatenate([eout[:OC - _N_V_HEADS * _V_HEAD_DIM],
                               np.take(eout[OC - _N_V_HEADS * _V_HEAD_DIM:],
                                       R192, axis=0)])
    w.add_tensor(base + ".escha_out", eout)

    # NOT an escha extra: <proj>.bias IS the model's standard bias. The oracle
    # carries it as blk.N.ffn_down.bias and the llama.cpp loader already reads
    # and applies it, after the matmul, which is the right place. Naming it
    # .escha_bias would hide it from the loader.
    if prefix + ".bias" in cp.index:
        bias = cp.get(prefix + ".bias").astype(np.float32).ravel()
        assert bias.size == OC, prefix
        w.add_tensor(base + ".bias", convert_transform(base + ".bias", bias))
    return K


# ---------------------------------------------------------------------------
# Minimal GGUF reader for --verify (independent of gguf-py internals:
# our raw type ids are unknown to GGMLQuantizationType on purpose).
# ---------------------------------------------------------------------------
_VT_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
           7: "?", 10: "Q", 11: "q", 12: "d"}


def parse_gguf(path: Path):
    f = open(path, "rb")

    def rd(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

    def rstr():
        return f.read(rd("Q")).decode("utf-8", "replace")

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            et, n = rd("I"), rd("Q")
            return [rval(et) for _ in range(n)]
        return rd(_VT_FMT[t])

    assert f.read(4) == b"GGUF", "not a GGUF file"
    rd("I")  # version
    n_tensors, n_kv = rd("Q"), rd("Q")
    kv = {}
    for _ in range(n_kv):
        name = rstr()
        kv[name] = rval(rd("I"))
    infos = {}
    for _ in range(n_tensors):
        name = rstr()
        nd = rd("I")
        ne = [rd("Q") for _ in range(nd)]
        ttype, off = rd("I"), rd("Q")
        infos[name] = (ne, ttype, off)
    align = int(kv.get("general.alignment", 32))
    data_start = (f.tell() + align - 1) // align * align
    return kv, infos, data_start, f


def read_raw_i16(f, data_start: int, off: int, nbytes: int) -> np.ndarray:
    f.seek(data_start + off)
    buf = f.read(nbytes)
    assert len(buf) == nbytes, "short read in GGUF data section"
    return np.frombuffer(buf, dtype=np.int16)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
def pick_targets(cp: Checkpoint, prefixes, want=3):
    rng = random.Random(0xB2)
    ks = {p: cp.escha_config(p)[1] for p in prefixes}
    pools = [
        sorted(p for p in prefixes if ks[p] == 3),
        sorted(p for p in prefixes if ks[p] == 2),
        sorted(p for p in prefixes if "attn" in p),
    ]
    chosen = []
    for pool in pools:
        if pool:
            c = rng.choice(pool)
            if c not in chosen:
                chosen.append(c)
    for p in sorted(prefixes):  # top up if pools overlapped
        if len(chosen) >= want:
            break
        if p not in chosen:
            chosen.append(p)
    return chosen[:want]


def verify(cp: Checkpoint, prefixes, out: Path):
    _, infos, data_start, f = parse_gguf(out)
    ok = True
    for prefix in pick_targets(cp, prefixes):
        L, K, V, cb, IC, OC = cp.escha_config(prefix)
        name = gguf_name(prefix + ".weight")
        assert name in infos, f"{name} missing from GGUF"
        ne, ttype, off = infos[name]
        assert ttype == escha_type_id(K), (name, ttype, escha_type_id(K))
        assert ne == [IC, OC], (name, ne)

        shape = (IC // 16, OC // 16, 16 * K)
        orig = np.ascontiguousarray(
            np.asarray(cp.get(prefix + ".escha_code"), dtype=np.int16)
        ).reshape(shape)
        # mirror write_linear's v-head reorder so the comparison stays exact
        base = gguf_name(prefix + ".weight").rpartition(".")[0]
        kind = base.split(".", 2)[2]
        if kind in ("attn_qkv", "attn_gate"):
            n_qk_bands = (OC - _N_V_HEADS * _V_HEAD_DIM) // 16
            # checkpoint order (IC//16, OC//16, 16K): v-heads live on axis 1
            v = orig[:, n_qk_bands:].reshape(
                IC // 16, _N_V_HEADS, _V_HEAD_DIM // 16,
                16 * K)[:, VPERM48]
            orig = np.ascontiguousarray(
                np.concatenate([orig[:, :n_qk_bands],
                                v.reshape(IC // 16, -1, 16 * K)], axis=1)
            ).reshape(shape)
        elif kind == "ssm_out":
            # all-v IC side: heads group the axis-0 tiles by 8
            orig = np.ascontiguousarray(
                orig.reshape(_N_V_HEADS, _V_HEAD_DIM // 16, OC // 16,
                             16 * K)[VPERM48]
            ).reshape(shape)
        # The GGUF is band-major: (OC//16, IC//16, 16*K). Undo the transpose
        # and require the checkpoint order back, byte for byte.
        got = read_raw_i16(f, data_start, off, orig.size * 2)
        got = np.ascontiguousarray(
            got.reshape(OC // 16, IC // 16, 16 * K).transpose(1, 0, 2))

        cbA, mul1 = (cb == 1), (cb == 2)
        dec_gguf = escham_cpu.reconstruct_code(got, IC, OC, K, cbA, mul1)
        dec_orig = escham_cpu.reconstruct_code(orig, IC, OC, K, cbA, mul1)
        diff = int((dec_gguf != dec_orig).sum())
        verbatim = bool(np.array_equal(got, orig))
        ok &= (diff == 0) and verbatim
        print(f"[verify] {prefix}\n"
              f"         K={K} cb={cb} elements={dec_gguf.size} "
              f"differing={diff} verbatim_bytes={verbatim}")
    f.close()
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N decoder layers")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    cp = Checkpoint(CKPT_DIR)
    try:
        prefixes = cp.linear_prefixes(args.limit)
        others = cp.other_keys(args.limit)
        print(f"[plan] {len(prefixes)} escha linears, {len(others)} other "
              f"tensors (limit={args.limit})")

        w = gguf.GGUFWriter(str(args.out), ARCH)
        cbs = {cp.escha_config(p)[3] for p in prefixes}
        set_metadata(w, cp, cbs)

        written = 0
        for key in others:
            if key.endswith(".weight_int8"):
                write_int8_row(cp, key, w)
                written += 1
            elif key.endswith(".weight_scale"):
                continue  # folded into the Q8_0 block scales by write_int8_row
            else:
                copy_f32(cp, key, w)
                written += 1
        for prefix in sorted(prefixes):
            write_linear(cp, prefix, w)
            written += 4
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        cp.close()

        gib = args.out.stat().st_size / 2**30
        print(f"[done] tensors written: {written}")
        print(f"[done] file size: {gib:.2f} GiB  (~9.5 GiB expected at full run)")

        if args.verify:
            if not verify(cp, prefixes, args.out):
                raise SystemExit("[verify] FAILED")
            print("[verify] OK — all checked tensors bit-exact")
    finally:
        cp.close()


if __name__ == "__main__":
    main()
