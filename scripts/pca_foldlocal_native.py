# -*- coding: utf-8 -*-
"""pca_foldlocal_native.py — partial fold-local PCA sensitivity check (PBMC native condition).

Raised in REVIEWER_ATTACK_SURFACE.md (A1) and METHODS.md §7.2: in the single-cell pipeline the
auto-scaling and PCA are fitted on the COMPLETE donor set before leave-one-donor-out, so the
representation is transductive with respect to the held-out donor. This script measures how much
that costs, using the locally cached PBMC native pseudobulk.

WHAT THIS DOES AND DOES NOT COVER
  DOES   : the NATIVE (uncapped) condition of PBMC_Indonesia, all cell types with enough donors.
  DOES NOT: the fixed-N (15/30/50/100) conditions, which require cell-level counts to draw N cells
            per unit; that staging is not cached locally. OneK1K is not local at all.
  So this is a PARTIAL check. If fold-local PCA barely moves the native values, that is reassuring
  but not proof for the fixed-N curves.

Writes results/pca_foldlocal_native.csv (new file; refuses to overwrite).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

ROOT = r"F:\nature"
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from m32_perm_parallel import build_sym2chr, AUTOSOMES, MIN_DONORS_PER_SEX  # noqa: E402

K_PCA_MAX = 100
C_L2 = 0.1
CACHE = os.path.join(ROOT, "data", "proc", "pbmc_pseudobulk_cache.npz")
OUT = os.path.join(RES, "pca_foldlocal_native.csv")


def k_for(n):
    return int(max(3, min(K_PCA_MAX, n // 2)))


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_L2, max_iter=2000, solver="lbfgs"))


def main():
    d = np.load(CACHE, allow_pickle=True)
    M, donors, cts, sexes, syms = d["M"], d["donors"], d["cts"], d["sexes"], d["syms"]
    # M holds RAW counts; cpm_log as in m32_perm_parallel.cpm_log
    lib = M.sum(axis=1, keepdims=True)
    Y = np.log2(M / np.maximum(lib, 1) * 1e6 + 1.0).astype(np.float32)

    sym2chr = build_sym2chr()
    chr_of = np.array([sym2chr.get(s, "NA") for s in syms])
    auto = np.isin(chr_of, list(AUTOSOMES))
    print(f"genes: {len(syms)}  autosomal: {int(auto.sum())}", flush=True)
    Ya = Y[:, auto]

    rows = []
    for ct in pd.unique(cts):
        m = cts == ct
        if int(m.sum()) < 20:
            continue
        y = sexes[m]
        if min((y == 0).sum(), (y == 1).sum()) < MIN_DONORS_PER_SEX:
            print(f"[{ct[:44]:44s}] skipped (too few donors per sex)", flush=True)
            continue
        S = Ya[m]
        dn = donors[m]
        k = k_for(len(y))

        # --- global representation (the executed analysis) ---
        Z = PCA(n_components=k, svd_solver="randomized", random_state=0).fit_transform(
            StandardScaler().fit_transform(S))
        preds = np.full(len(y), np.nan)
        for u in pd.unique(dn):
            te = dn == u
            c = clf(); c.fit(Z[~te], y[~te])
            preds[te] = c.predict_proba(Z[te])[:, 1]
        ok = ~np.isnan(preds)
        auc_g = roc_auc_score(y[ok], preds[ok])

        # --- fold-local representation (the sensitivity) ---
        preds2 = np.full(len(y), np.nan)
        for u in pd.unique(dn):
            te = dn == u
            sc = StandardScaler().fit(S[~te])
            pc = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(sc.transform(S[~te]))
            c = clf()
            c.fit(pc.transform(sc.transform(S[~te])), y[~te])
            preds2[te] = c.predict_proba(pc.transform(sc.transform(S[te])))[:, 1]
        ok2 = ~np.isnan(preds2)
        auc_f = roc_auc_score(y[ok2], preds2[ok2])

        rows.append({"cell_type": ct, "n_donors": len(y), "k_pca": k,
                     "S_global_pca": round(max(auc_g, 1 - auc_g), 4),
                     "S_foldlocal_pca": round(max(auc_f, 1 - auc_f), 4),
                     "delta_S": round(max(auc_f, 1 - auc_f) - max(auc_g, 1 - auc_g), 4)})
        print(f"[{ct[:44]:44s}] k={k:3d}  S_global={max(auc_g,1-auc_g):.4f}  "
              f"S_foldlocal={max(auc_f,1-auc_f):.4f}  delta={max(auc_f,1-auc_f)-max(auc_g,1-auc_g):+.4f}",
              flush=True)

    df = pd.DataFrame(rows)
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING TO OVERWRITE existing {OUT}")
    df.to_csv(OUT, index=False)
    print(f"\nn={len(df)}  mean delta_S = {df.delta_S.mean():+.4f}  "
          f"median = {df.delta_S.median():+.4f}  range {df.delta_S.min():+.4f} .. {df.delta_S.max():+.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
