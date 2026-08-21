#!/usr/bin/env python3
"""Generate the M4 test fixture from the real checkpoint.

For one K=2 and one K=3 projection (real escha_code / rin / rout / bias):
  - unit part: NT real tiles -> packed codes + expected fp16 output, both
    produced by escham_cpu.reconstruct_code (the normative oracle);
  - graph part: full packed tensor, Wh = reconstruct_code, x ~ N(0,1) fp16,
    and ref_y computed with torch exactly as proof_prepost.py does:
        xs = x*(s_in*rin) ; xp = T128(xs) ; z = xp @ Wh ; y = T128(z)*(s_out*rout)
    plus the deploy bias added at the end.
"""
import json
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/home/yaniss/hermes-work/escha/port")
from escham_cpu import reconstruct_code  # noqa: E402

CKPT = "/home/yaniss/models/qwen3.8-27b-escha-w2"
OUT = "/home/yaniss/hermes-work/escha/port/fixture_m4.bin"
NT = 8   # unit tiles per record
NR = 4   # graph samples per record


def had128():
    H = np.ones((1, 1), dtype=np.float64)
    for _ in range(7):
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(128.0)


T128 = had128()


def t128_vec(v):  # v: (..., n) float32 numpy, n % 128 == 0; transform along last axis
    out = np.asarray(v, dtype=np.float32).copy()
    flat = out.reshape(-1, out.shape[-1])
    for r in range(flat.shape[0]):
        row = flat[r]
        for i in range(0, len(row), 128):
            row[i:i + 128] = T128 @ row[i:i + 128]
    return out.reshape(out.shape)


def main():
    index = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    names = sorted({k.rsplit(".", 1)[0] for k in index if k.endswith(".escha_code")})
    by_kind = {}
    for n in names:
        by_kind.setdefault(n.split(".")[-1], []).append(n)
    sample = [lst[len(lst) // 2] for _, lst in sorted(by_kind.items())]

    opened = {}

    def get(name):
        f = index[name]
        if f not in opened:
            opened[f] = safe_open(f"{CKPT}/{f}", framework="pt")
        return opened[f].get_tensor(name)

    blob = bytearray()
    blob += struct.pack("<II", 0x45534D34, 1)
    blob += struct.pack("<I", len(sample))

    for name in sample:
        code = get(name + ".escha_code")
        rin = get(name + ".escha_rin").float()
        rout = get(name + ".escha_rout").float()
        # The trained scales are NOT ones (max|v-1| ~ 2.7e-2 over the 400
        # projections). The reference below keeps them separate, exactly as
        # linear.py does; the fixture ships them folded, exactly as
        # gguf_escha.py writes .escha_in / .escha_out. The test therefore
        # exercises the fold instead of assuming it.
        s_in = get(name + ".escha_s_in").float()
        s_out = get(name + ".escha_s_out").float()
        rin_folded = rin * s_in
        rout_folded = rout * s_out
        bias = get(name + ".bias").float()
        L, K, V, cb_id, IC, OC = get(name + ".escha_config").tolist()
        assert L == 16 and V == 2 and cb_id == 1, (name, L, K, V, cb_id)
        itc, otc = IC // 16, OC // 16
        packed_np = code.cpu().numpy().reshape(itc, otc, 16 * K)

        # ---- oracle decode of the whole tensor ----
        Wh = reconstruct_code(packed_np, IC, OC, int(K), True, True)  # (IC, OC) f16

        # ---- unit records: first NT tiles, flat order (it*otc+ot) ----
        nt = min(NT, itc * otc)
        codes_unit = np.ascontiguousarray(packed_np.reshape(-1, 16 * K)[:nt]).view(np.uint16)
        # invert reconstruct_code's .transpose(0,2,1,3).reshape(IC,OC)
        tiles = (Wh.reshape(itc, 16, otc, 16).transpose(0, 2, 1, 3)
                 .reshape(-1, 16, 16)[:nt])
        exp_unit = np.ascontiguousarray(tiles.reshape(nt, 256)).view(np.uint16)

        # ---- graph reference, same chain as proof_prepost.py ----
        torch.manual_seed(42)
        x = torch.randn(NR, IC, dtype=torch.float16).float()
        xs = (x * s_in.unsqueeze(0)) * rin.unsqueeze(0)
        xp = torch.from_numpy(t128_vec(xs.numpy()))
        z = xp @ torch.from_numpy(np.ascontiguousarray(Wh)).float()  # (NR, OC)
        # ORDER MATTERS: rout is diagonal, T128 is dense per 128-block, they do
        # NOT commute. Vendor (linear.py:276-277 + transform.py:59) is
        # y = post_scale * H(z), i.e. T128 FIRST then * rout. Proven at 6.0e-07
        # by check_graph_c.py; the reverse lands at 1.41 (uncorrelated).
        y = t128_vec(z.numpy()) * rout.numpy()[None, :] * s_out.numpy()[None, :]
        y_ref = y + bias.numpy()[None, :]

        nb = name.encode()
        blob += struct.pack("<I", len(nb)) + nb
        blob += struct.pack("<iiiii", int(K), IC, OC, nt, NR)
        blob += codes_unit.tobytes()
        blob += exp_unit.tobytes()
        blob += np.ascontiguousarray(packed_np).view(np.uint16).tobytes()
        blob += x.numpy().astype(np.float32).T.copy().tobytes()
        blob += rin_folded.numpy().astype(np.float32).tobytes()
        blob += rout_folded.numpy().astype(np.float32).tobytes()
        blob += bias.numpy().astype(np.float32).tobytes()
        blob += y_ref.astype(np.float32).T.copy().tobytes()

    with open(OUT, "wb") as f:
        f.write(blob)
    print(f"wrote {OUT} ({len(blob)} bytes)")


if __name__ == "__main__":
    main()
