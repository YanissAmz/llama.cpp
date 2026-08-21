"""Compare every tensor of a native escha GGUF against the Q8_0 oracle.

The oracle is convert_hf_to_gguf's own output for this checkpoint, so it is the
ground truth for everything the writer has to reproduce: the norm +1, the A_log
negation, tensor layout. A writer that skips one of those still produces a file
that loads and runs, and answers nonsense.

Escha-coded weights are folded back to dense (reconstruct_deploy_weight) before
comparing, so the escha payload and its rin/rout are checked too.
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from gguf_escha import parse_gguf, read_raw_i16   # noqa: E402
import escham_cpu                                  # noqa: E402

NAT = "/home/yaniss/models/qwen3.8-27b-escha-gguf/escha-native.gguf"
ORA = "/home/yaniss/models/qwen3.8-27b-escha-gguf/escha-oracle-q8_0.gguf"

# Q8_0 round-trip noise. Anything above this is a real difference.
TOL = 0.02


def read_dense(f, ds, info):
    ne, t, off = info
    n = int(np.prod(ne))
    f.seek(ds + off)
    if t == 0:                       # F32
        return np.frombuffer(f.read(n * 4), dtype=np.float32)
    if t == 1:                       # F16
        return np.frombuffer(f.read(n * 2), dtype=np.float16).astype(np.float32)
    if t == 8:                       # Q8_0, blocks of 32 along ne0
        nb = int(ne[0]) // 32
        rows = n // int(ne[0])
        raw = np.frombuffer(f.read(rows * nb * 34), dtype=np.uint8).reshape(rows, nb, 34)
        sc = raw[:, :, :2].copy().view(np.float16).astype(np.float32).reshape(rows, nb, 1)
        qs = raw[:, :, 2:].copy().view(np.int8).astype(np.float32)
        return (qs * sc).reshape(-1)
    return None                      # escha, handled by the caller


def fold_escha(f, ds, info, aux_in, aux_out):
    ne, t, off = info
    IC, OC = int(ne[0]), int(ne[1])
    K = 2 if t == 43 else 3
    itc, otc = IC // 16, OC // 16
    codes = read_raw_i16(f, ds, off, IC * OC * K // 16 * 2)
    packed = np.ascontiguousarray(codes.reshape(otc, itc, 16 * K).transpose(1, 0, 2))
    return escham_cpu.reconstruct_deploy_weight(
        packed, aux_in, aux_out, IC, OC, K, True, True).astype(np.float32).T.ravel()


def main():
    _, ni, nds, nf = parse_gguf(NAT)
    _, oi, ods, of = parse_gguf(ORA)

    missing = [k for k in oi if k not in ni]
    if missing:
        print(f"[fatal] {len(missing)} oracle tensors absent from the native file:")
        for k in missing[:10]:
            print("   ", k)
        return 1

    # folding an escha projection back to dense costs seconds; sample them
    n_escha = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seen_escha = 0

    bad, checked = [], 0
    for name, oinfo in sorted(oi.items()):
        ninfo = ni[name]
        if list(ninfo[0]) != list(oinfo[0]):
            bad.append((name, "shape", ninfo[0], oinfo[0]))
            continue
        b = read_dense(of, ods, oinfo)
        if ninfo[1] in (43, 44):
            seen_escha += 1
            if seen_escha > n_escha:
                continue
            base = name[: -len(".weight")]
            IC, OC = int(ninfo[0][0]), int(ninfo[0][1])
            ain = read_dense(nf, nds, ni[base + ".escha_in"])[:IC]
            aout = read_dense(nf, nds, ni[base + ".escha_out"])[:OC]
            a = fold_escha(nf, nds, ninfo, ain, aout)
        else:
            a = read_dense(nf, nds, ninfo)
        den = float(np.sqrt((b ** 2).mean()))
        rel = float(np.sqrt(((a - b) ** 2).mean()) / den) if den else float(np.abs(a - b).max())
        checked += 1
        if rel > TOL:
            bad.append((name, "value", rel, None))

    print(f"[crosscheck] {checked} tensors compared, tolerance {TOL}")
    for name, kind, x, y in bad[:40]:
        print(f"  MISMATCH {name:44s} {kind} {x} {y if y is not None else ''}")
    print(f"[crosscheck] {len(bad)} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
