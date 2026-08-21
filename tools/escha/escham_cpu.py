"""CPU reference for escha_escham::escham_reconstruct_kernel<CB,K>.

Hand-translated, instruction by instruction, from the PTX recovered out of
escha/_C.cpython-312-x86_64-linux-gnu.so (.nv_fatbin, sm_120, entry
_ZN12escha_escham25escham_reconstruct_kernelILi1ELi2EEEvP6__halfPKti).

Codec: tail-biting trellis, 16-bit (K=2) window sliding K bits per element
over the tile's 512-bit circular code, one integer multiply + one lop3 to
synthesise two fp16 lanes, summed.  CB selects the multiplier:
    CB=0 (neither flag)  no multiply
    CB=1 (cbA, default)  * 0xCBAC_1FED  (3417055213), then (& MASK) ^ MAGIC
    CB=2 (mul1)          * 0x83DC_D12D  (2212286765), no lop3
Les valeurs qui font foi sont celles du dict MUL ci-dessous ; l'hexa de cette
docstring etait faux jusqu'au 21/08/2026 (piege : un portage qui recopie la
docstring au lieu du code produit un decodeur muet et entierement errone).
"""
import numpy as np

MASK = np.uint32(0x8FFF8FFF)
MAGIC = np.uint32(0x3B603B60)
MUL = {0: None, 1: np.uint32(3417055213), 2: np.uint32(2212286765)}


def _codebook(cb: int, nbits: int) -> np.ndarray:
    """LUT: window value -> fp16 weight.  Size 2**nbits."""
    x = np.arange(1 << nbits, dtype=np.uint32)
    if cb == 2:
        t = (x * MUL[2]).astype(np.uint32)
    elif cb == 1:
        t = (x * MUL[1]).astype(np.uint32)
        t = (t & MASK) ^ MAGIC
    else:
        t = (x & MASK) ^ MAGIC
    lo = (t & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    hi = (t >> np.uint32(16)).astype(np.uint16).view(np.float16)
    return (lo.astype(np.float32) + hi.astype(np.float32)).astype(np.float16)


def _tile_map(K: int):
    """(lane, step) -> (row, col) inside the 16x16 tile.  Same for K=2 and K=3.

    r143 = (tid&3)*512 + warp*32 + ((lane>>1)&12); eight st.shared.u32 at byte
    offsets 0,256,2048,2304,16,272,2064,2320, each writing lane l's value then
    lane l+4's (shfl.sync.down delta 4).  Steps run high->low: offset 0 holds
    the step-7 window, offset 2320 the step-0 one.
    """
    rows = np.empty((32, 8), np.int64)
    cols = np.empty((32, 8), np.int64)
    off = {7: (0, 0), 6: (1, 0), 5: (8, 0), 4: (9, 0),
           3: (0, 8), 2: (1, 8), 1: (8, 8), 0: (9, 8)}
    for m in range(32):
        r0 = 2 * (m & 3)
        c0 = 2 * ((m >> 3) & 1) + 4 * ((m >> 4) & 1)
        d = (m >> 2) & 1
        for s in range(8):
            dr, dc = off[s]
            rows[m, s] = r0 + dr
            cols[m, s] = c0 + dc + d
    return rows, cols


def _lane_addr(K: int):
    """(cur_word, prev_word, shift) per lane, straight from the PTX prologue."""
    cur = np.empty(32, np.int64); prev = np.empty(32, np.int64)
    sh = np.empty(32, np.int64)
    for l in range(32):
        if K == 2:
            c = (l >> 1) & 15
            cur[l], prev[l], sh[l] = c, (c - 1) & 15, 16 if (l & 1) == 0 else 0
        elif K == 3:
            b = l * 24
            r86 = b + 791
            r87 = r86 & 2016
            cur[l] = (((r86 >> 3) & 252) - 96) // 4
            p = ((b + 755) >> 5) - 24
            prev[l] = 23 if l == 0 else p
            sh[l] = r87 - b - 760
        else:
            raise ValueError(K)
        assert prev[l] == (cur[l] - 1) % (16 if K == 2 else 24), (l, cur[l], prev[l])
    assert sh.min() >= 0 and sh.max() + 7 * K + 16 <= 64
    return cur, prev, sh


def _windows(words: np.ndarray, K: int) -> np.ndarray:
    """words: (T, 16*K/2) uint32 -> (T,32,8) 16-bit window values.

    Tail-biting: the window that starts near the end of a lane's slice wraps
    into the previous u32, which is why the kernel builds a 64-bit pair.
    """
    cur, prev, sh = _lane_addr(K)
    pair = ((words[:, prev].astype(np.uint64) << np.uint64(32))
            | words[:, cur].astype(np.uint64))
    steps = (sh[None, :, None] + K * np.arange(8)[None, None, :]).astype(np.uint64)
    return ((pair[:, :, None] >> steps) & np.uint64(0xFFFF)).astype(np.uint16)


def reconstruct_code(packed: np.ndarray, in_features: int, out_features: int,
                     K: int, cbA: bool, mul1: bool) -> np.ndarray:
    """packed: (IC//16, OC//16, 16*K) int16  ->  (IC, OC) float16."""
    cb = 1 if cbA else (2 if mul1 else 0)
    itc, otc = in_features // 16, out_features // 16
    assert packed.shape[:2] == (itc, otc), packed.shape
    words = np.ascontiguousarray(packed).view(np.uint32).reshape(itc * otc, 8 * K)
    win = _windows(words, K)                       # (T,32,8)
    lut = _codebook(cb, 16)
    vals = lut[win]                                # (T,32,8) f16
    rows, cols = _tile_map(K)
    tiles = np.empty((words.shape[0], 16, 16), np.float16)
    tiles[:, rows.ravel(), cols.ravel()] = vals.reshape(words.shape[0], -1)
    return (tiles.reshape(itc, otc, 16, 16).transpose(0, 2, 1, 3)
            .reshape(in_features, out_features))


_H128 = None


def _had128() -> np.ndarray:
    global _H128
    if _H128 is None:
        h = np.ones((1, 1), np.float32)
        while h.shape[0] < 128:
            h = np.block([[h, h], [h, -h]])
        _H128 = h / np.sqrt(128.0)
    return _H128


def escha_t128(x: np.ndarray) -> np.ndarray:
    """Blockwise normalised Hadamard-128 on the last axis (transform.py)."""
    s = x.shape
    return (x.reshape(-1, s[-1] // 128, 128) @ _had128()).reshape(s)


def reconstruct_deploy_weight(code, rin, rout, in_features, out_features,
                              K, cbA, mul1) -> np.ndarray:
    """linear.py:reconstruct_deploy_weight -> (IC, OC) float32."""
    w = reconstruct_code(code, in_features, out_features, K, cbA, mul1).astype(np.float32)
    w = escha_t128(w.T.copy()).T.copy()      # Hadamard along IC
    w = w * rin.astype(np.float32)[:, None]
    w = escha_t128(w)                        # Hadamard along OC
    w = w * rout.astype(np.float32)[None, :]
    return w
