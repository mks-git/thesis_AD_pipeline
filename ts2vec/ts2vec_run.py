#!/usr/bin/env python3
# ts2vec_run.py
#
# Usage (from ts2vec repo root OR with PYTHONPATH set to it):
#   python ts2vec_run.py \
#     --manifest /ABS/PATH/pipeline/BRIDGE_ROOT/manifests/manifest_app2.json \
#     --outdir   /ABS/PATH/pipeline/BRIDGE_ROOT/scores \
#     --epochs 40 --emb_dim 128 --k 5
#
# Notes:
# - Expects the official TS2Vec class at pipeline/ts2vec/ts2vec.py
# - Inputs must be z-scored already (exported via Exathlon).
# - Produces per-timestamp scores aligned with val/test lengths.

import argparse, json, os
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Import TS2Vec from the official repo clone (this file should live in the same folder as ts2vec.py)
from ts2vec import TS2Vec

# def sliding_windows(X: np.ndarray, W: int, S: int):
#     """
#     X: (T, D) -> windows: (N, W, D), index_map: (N, 2) with [start, end)
#     """
#     T, D = X.shape
#     if T < W:
#         raise ValueError(f"T={T} < W={W}: increase data or reduce W")
#     starts = np.arange(0, T - W + 1, S, dtype=int)
#     ends = starts + W
#     idx = np.stack([starts, ends], axis=1)
#     # build windows
#     Wnds = np.lib.stride_tricks.sliding_window_view(X, (W, D)).reshape(-1, W, D)[::S]
#     # sliding_window_view steps by 1, so pick every S-th
#     assert Wnds.shape[0] == idx.shape[0]
#     return Wnds, idx

def sliding_windows(X: np.ndarray, W: int, S: int):
    """
    Pure-Python/NumPy slicing version (no sliding_window_view).
    X: (T, D) -> windows: (N, W, D), index_map: (N, 2) with [start, end)
    """
    T, D = X.shape
    if T < W:
        raise ValueError(f"T={T} < W={W}: increase data or reduce W")
    starts = np.arange(0, T - W + 1, S, dtype=int)
    idx = np.stack([starts, starts + W], axis=1)
    # stack slices; slightly slower but version-proof
    Wnds = np.stack([X[s:e] for (s, e) in idx], axis=0)
    return Wnds, idx

def aggregate_overlap_mean(scores_w: np.ndarray, idx_map: np.ndarray, T: int) -> np.ndarray:
    """
    scores_w: (N_windows,)
    idx_map: (N_windows, 2)
    returns per-timestamp scores of length T via mean over overlapping windows.
    """
    ts = np.zeros(T, dtype=np.float64)
    cnt = np.zeros(T, dtype=np.int32)
    for (s, e), sw in zip(idx_map, scores_w):
        ts[s:e] += float(sw)
        cnt[s:e] += 1
    cnt[cnt == 0] = 1
    return (ts / cnt).astype(np.float32)

def median_filter_1d(x: np.ndarray, k: int = 11) -> np.ndarray:
    if k is None or k < 3 or k % 2 == 0:
        return x
    try:
        from scipy.signal import medfilt
        return medfilt(x, kernel_size=k).astype(np.float32)
    except Exception:
        # fallback: simple rolling median (slower)
        half = k // 2
        y = np.empty_like(x)
        for i in range(len(x)):
            a = max(0, i - half)
            b = min(len(x), i + half + 1)
            y[i] = np.median(x[a:b])
        return y.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to BRIDGE_ROOT/manifests/manifest_appX.json")
    ap.add_argument("--outdir", required=True, help="Output dir (e.g., BRIDGE_ROOT/scores)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--k", type=int, default=5, help="k for kNN scoring")
    ap.add_argument("--distance", choices=["cosine", "euclidean"], default="cosine")
    ap.add_argument("--smooth", type=int, default=11, help="median filter window (odd), 0/1 to disable")
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text())
    W = int(man["windowing"]["W"])
    S = int(man["windowing"]["S"])
    assign = man["windowing"].get("assign", "overlap_mean")

    Xtr = np.load(man["processed_arrays"]["train"], mmap_mode="r")
    Xva = np.load(man["processed_arrays"]["val"],   mmap_mode="r")
    Xte = np.load(man["processed_arrays"]["test"],  mmap_mode="r")

    print(f"[info] Loaded arrays: train={Xtr.shape}, val={Xva.shape}, test={Xte.shape}")

    # 1) windowing
    tr_win, _    = sliding_windows(Xtr, W, S)
    va_win, va_i = sliding_windows(Xva, W, S)
    te_win, te_i = sliding_windows(Xte, W, S)
    print(f"[info] Windows: tr={tr_win.shape}, va={va_win.shape}, te={te_win.shape}")

    # 2) build & train TS2Vec
    # Detect device (TS2Vec in the official repo takes device in __init__)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    model = TS2Vec(
        input_dims=Xtr.shape[1],
        output_dims=args.emb_dim,
        hidden_dims=args.hidden_dim,
        depth=args.depth,
        device=device,
        lr=1e-3,                 # adjust if repo expects lr in fit()
        batch_size=args.batch_size
    )

    # fit() signature varies slightly across forks; pass what your version expects
    model.fit(
        tr_win,
        n_epochs=args.epochs,
        verbose=True
        # If your TS2Vec expects lr/batch_size here instead of __init__, add them.
        # lr=1e-3,
        # batch_size=args.batch_size
    )

    # 3) encode windows -> embeddings
    # Many repos support encoding_window='full' to get one embedding per window:
    try:
        Ztr = model.encode(tr_win, encoding_window="full")
        Zva = model.encode(va_win, encoding_window="full")
        Zte = model.encode(te_win, encoding_window="full")
    except TypeError:
        # fallback if encoding_window isn't supported
        Ztr = model.encode(tr_win)
        Zva = model.encode(va_win)
        Zte = model.encode(te_win)

    # If encode returns (N, W, C), pool over time
    if Ztr.ndim == 3:
        Ztr = Ztr.mean(axis=1)
        Zva = Zva.mean(axis=1)
        Zte = Zte.mean(axis=1)

    print(f"[info] Embeddings: Ztr={Ztr.shape}, Zva={Zva.shape}, Zte={Zte.shape}")

    # 4) kNN scoring in embedding space
    nn = NearestNeighbors(n_neighbors=args.k, metric=args.distance).fit(Ztr)
    def knn_scores(Z):
        dists, _ = nn.kneighbors(Z, return_distance=True)
        return dists.mean(1).astype(np.float32)

    s_va_w = knn_scores(Zva)  # (N_va_windows,)
    s_te_w = knn_scores(Zte)  # (N_te_windows,)

    # 5) aggregate window scores to per-timestamp
    if assign != "overlap_mean":
        print(f"[warn] assign={assign} not supported, falling back to overlap_mean")
    s_val = aggregate_overlap_mean(s_va_w, va_i, Xva.shape[0])
    s_tst = aggregate_overlap_mean(s_te_w, te_i, Xte.shape[0])

    # 6) optional smoothing
    if args.smooth and args.smooth >= 3 and args.smooth % 2 == 1:
        s_val = median_filter_1d(s_val, k=args.smooth)
        s_tst = median_filter_1d(s_tst, k=args.smooth)

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "app2_val_scores.npy",  s_val.astype(np.float32))
    np.save(out / "app2_test_scores.npy", s_tst.astype(np.float32))

    meta = dict(
        method="TS2Vec+kNN",
        W=W, S=S,
        emb_dim=int(Ztr.shape[1]),
        k=args.k, metric=args.distance,
        smooth=args.smooth,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        batch_size=args.batch_size,
        device=str(device)
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("[ok] wrote:", out / "app2_val_scores.npy")
    print("[ok] wrote:", out / "app2_test_scores.npy")
    print("[ok] meta :", out / "meta.json")

if __name__ == "__main__":
    main()
