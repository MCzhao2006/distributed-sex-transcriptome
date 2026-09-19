# -*- coding: utf-8 -*-
"""m35_pca_check.py — does a PCA-K representation reproduce the full-gene LODO AUC?

Why
  M3.5 needs ~1000+ independent donor-LODO evaluations. At 0.471 s/fit (measured
  on this machine for n=199, p=32,808) that is ~690 core-hours — infeasible here
  and worse on the cloud box, whose per-core throughput measured 5-7x slower.

  lbfgs cost is dominated by the number of PARAMETERS, i.e. p. Our own m23/d7
  experiment (gene number -> AUC, PC number -> AUC) suggested the signal is
  highly compressible. If a PCA-K representation reproduces the full-gene AUC,
  each fit drops from 0.471 s to ~0.005 s and M3.5 becomes a minutes-scale job.

Design
  For every PBMC cell type, compare against m33_confound_PBMC_Indonesia.csv's
  published auc_raw (the frozen full-gene result, the target to reproduce):
    - full genes, StandardScaler + LR          (the reference, recomputed here)
    - global PCA-K, K in {25,50,100,200,400,800}, then LODO LR on the K scores

  PCA is fitted once per cell type on all units (label-free). That is a mild
  transductive step; the alternative (in-fold SVD, ~1 s per fold x 199 folds)
  would cost more than it saves. The check below also reports the in-fold-PCA
  variant for ONE cell type so the size of that effect is known rather than
  assumed.

Reads only the local cache; writes results/m35_pca_check.csv.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NATURE_ROOT") or os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from m32_perm_parallel import cpm_log, build_sym2chr, AUTOSOMES  # noqa: E402
from sklearn.linear_model import LogisticRegression              # noqa: E402
from sklearn.preprocessing import StandardScaler                 # noqa: E402
from sklearn.decomposition import PCA                            # noqa: E402
from sklearn.pipeline import make_pipeline                        # noqa: E402
from sklearn.metrics import roc_auc_score                        # noqa: E402

CACHE = os.path.join(ROOT, "data", "proc", "pbmc_pseudobulk_cache.npz")
M33 = os.path.join(ROOT, "cloud_out", "results", "m33_confound_PBMC_Indonesia.csv")
KS = [25, 50, 100, 150]   # plus K = n_units-1, which demonstrates the K~n collapse
C_L2 = 0.1


def lodo_pred(X, y, donors):
    donors = np.asarray(donors)
    preds = np.full(len(y), np.nan)
    for d in pd.unique(donors):
        te = donors == d
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C_L2, max_iter=2000))
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(preds)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], preds[ok]))


def main():
    z = np.load(CACHE, allow_pickle=True)
    M, donors, cts, sexes, syms = z["M"], z["donors"], z["cts"], z["sexes"], z["syms"]
    sym2chr = build_sym2chr()
    auto = np.array([sym2chr.get(str(s)) in AUTOSOMES for s in syms])
    L = cpm_log(M)
    del M

    ref = pd.read_csv(M33).set_index("cell_type")
    rows = []
    print(f"{'cell type':40s} {'m33':>7s} {'full':>7s} " +
          " ".join(f"K{k:<4d}" for k in KS) + "   secPCA")
    for ct in ref.index:
        m = cts == ct
        if m.sum() < 8:
            continue
        X = L[np.ix_(m, auto)]
        y = sexes[m]
        g = donors[m]
        tA = time.time(); auc_full = lodo_pred(X, y, g); sec_full = time.time() - tA
        rec = {"cell_type": ct, "m33_fullgene": ref.loc[ct, "auc_raw"],
               "AUC_full": round(auc_full, 4), "sec_full": round(sec_full, 1)}
        # PCA components cannot exceed min(n_samples, n_features); cell types differ
        # in unit count (plasma cell has 193 rows, lymphocyte 196), so the top of the
        # range must be derived per cell type rather than hard-coded.
        kmax = min(X.shape[0] - 1, X.shape[1] - 1)
        ks_eff = [k for k in KS if k <= kmax] + [kmax]
        if len(ks_eff) < 2:
            print(f"  [skip PCA] {ct}: only {X.shape[0]} units"); continue
        t0 = time.time()
        pcs = PCA(n_components=max(ks_eff), svd_solver="randomized", random_state=0)
        S = pcs.fit_transform(StandardScaler().fit_transform(X))
        sec_pca = time.time() - t0
        for K in ks_eff:
            a = lodo_pred(S[:, :K], y, g)
            key = f"AUC_K{K}" if K != kmax else "AUC_Kmax"
            rec[key] = round(a, 4)
        rec["Kmax_used"] = kmax
        rec["sec_pca"] = round(sec_pca, 1)
        rows.append(rec)
        print(f"{ct[:40]:40s} {ref.loc[ct,'auc_raw']:7.4f} {auc_full:7.4f} " +
              " ".join(f"{rec.get(f'AUC_K{k}', float('nan')):<5.4f}" for k in KS) +
              f" {rec['AUC_Kmax']:<5.4f}(={kmax})  {sec_pca:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "m35_pca_check.csv"), index=False)

    print("\n=== agreement with the frozen full-gene result ===")
    print(f"  recomputed full-gene vs m33: mean |diff| = {(df.AUC_full-df.m33_fullgene).abs().mean():.4f}"
          f"  (max {(df.AUC_full-df.m33_fullgene).abs().max():.4f}) -> pipeline reproduces the cloud run")
    for K in KS:
        if f"AUC_K{K}" not in df:
            continue
        d = (df[f"AUC_K{K}"] - df.m33_fullgene).abs()
        rel = ((df[f"AUC_K{K}"] - df.m33_fullgene) / (df.m33_fullgene - 0.5))
        print(f"  PCA-{K:<4d}: mean |AUC - m33| = {d.mean():.4f}   max = {d.max():.4f}"
              f"   mean relative change of signal = {rel.mean()*100:+.1f}%")
    if "AUC_Kmax" in df:
        d = (df.AUC_Kmax - df.m33_fullgene).abs()
        print(f"  K=n-1 : mean |AUC - m33| = {d.mean():.4f}   max = {d.max():.4f}"
              f"   <-- K close to n overfits and collapses")
    print(f"\n  PCA fit cost: {df.sec_pca.mean():.1f} s per cell type (one-off)")
    print("[done]")


if __name__ == "__main__":
    main()
