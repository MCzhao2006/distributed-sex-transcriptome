# -*- coding: utf-8 -*-
"""m35_solver_check.py — can we make the LODO fit ~20x cheaper WITHOUT changing
the estimator?

Why this is the gating question for M3.5
  M3.5 (the measurement-scale controlled experiment requested in chatGPT_10.md)
  needs ~4 cohorts x 4 cells/unit levels x 50-100 seeds x all usable cell types,
  each an independent donor-LODO. At the current cost that is ~1100 core-hours,
  i.e. ~34 h on the whole 32-core box. Infeasible.

  The cost is not the gene count, it is that lbfgs optimises 32,808 parameters.
  With n=199 donors << p=32,808 genes the solution is confined to the span of
  the training samples, so the DUAL problem has only n variables.
  `LogisticRegression(solver='liblinear', dual=True)` solves exactly that.

This script checks, on PBMC_Indonesia (already on this machine):
  A. StandardScaler + lbfgs          <- current pipeline; MUST reproduce m33's
                                        published auc_raw values
  B. StandardScaler + liblinear-dual <- proposed; must agree with A
  C. no scaler + lbfgs               <- secondary check
  D. no scaler + liblinear-dual

  plus wall-clock per fold for each, so the OneK1K cost can be extrapolated
  (kernel cost scales as n^2 * p, so 981 donors is ~24x the 199-donor kernel).

Outputs a table only; writes nothing over the frozen results.
"""
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NATURE_ROOT") or os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from m32_perm_parallel import (  # noqa: E402
    pseudo_bulk, cpm_log, build_sym2chr, AUTOSOMES)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

CACHE = os.path.join(ROOT, "data", "proc", "pbmc_pseudobulk_cache.npz")

# m33_confound_PBMC_Indonesia.csv, full-gene lbfgs pipeline — the target values
KNOWN = {
    "CD4-positive, alpha-beta T cell": 0.8612,
    "natural killer cell": 0.8717,
    "CD8-positive, alpha-beta T cell": 0.8180,
    "CD14-positive monocyte": 0.7800,
    "naive thymus-derived CD8-positive, alpha-beta T cell": 0.7389,
}
C_L2 = 0.1


def _fit_predict(Xtr, ytr, Xte, variant):
    if variant in ("A", "B"):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C_L2, max_iter=2000,
                                               **({"solver": "liblinear", "dual": True}
                                                  if variant == "B" else {})))
    else:
        clf = LogisticRegression(C=C_L2, max_iter=2000,
                                 **({"solver": "liblinear", "dual": True}
                                    if variant == "D" else {}))
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def lodo(X, y, donors, variant):
    donors = np.asarray(donors)
    preds = np.full(len(y), np.nan)
    for d in pd.unique(donors):
        te = donors == d
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        preds[te] = _fit_predict(X[tr], y[tr], X[te], variant)
    ok = ~np.isnan(preds)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return np.nan, 0.0
    return float(roc_auc_score(y[ok], preds[ok])), 0.0


def _one(args):
    ct, Yg, y, g = args
    out = {}
    for v in ("A", "B", "C", "D"):
        t0 = time.time()
        a, _ = lodo(Yg, y, g, v)
        out[v] = (a, time.time() - t0)
    return ct, out


def main():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    if os.path.exists(CACHE):
        print(f"[cache] loading {CACHE}", flush=True)
        z = np.load(CACHE, allow_pickle=True)
        M, donors, cts, sexes, syms = z["M"], z["donors"], z["cts"], z["sexes"], z["syms"]
    else:
        print("[load] pseudobulk from h5ad (one-off, ~15-25 min)", flush=True)
        t0 = time.time()
        M, donors, cts, sexes, syms = pseudo_bulk(
            os.path.join(ROOT, "data", "singlecell", "PBMC_Indonesia.h5ad"))
        print(f"       done in {time.time()-t0:.0f}s  units={M.shape[0]} genes={len(syms)}",
              flush=True)
        np.savez(CACHE, M=M, donors=donors, cts=cts, sexes=sexes, syms=np.asarray(syms))

    sym2chr = build_sym2chr()
    auto = np.array([sym2chr.get(str(s)) in AUTOSOMES for s in syms])
    print(f"autosomal genes: {int(auto.sum()):,}", flush=True)
    L = cpm_log(M)
    del M

    tasks = []
    for ct in KNOWN:
        m = cts == ct
        if m.sum() < 8:
            print(f"  [skip] {ct}"); continue
        tasks.append((ct, L[np.ix_(m, auto)], sexes[m], donors[m]))
    print(f"{len(tasks)} cell types x 4 variants x {len(np.unique(donors))} folds", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=min(6, len(tasks))) as ex:
        futs = [ex.submit(_one, t) for t in tasks]
        for f in as_completed(futs):
            ct, out = f.result()
            rec = {"cell_type": ct, "m33_fullgene_auc": KNOWN[ct]}
            for v in ("A", "B", "C", "D"):
                rec[f"AUC_{v}"] = round(out[v][0], 4)
                rec[f"sec_{v}"] = round(out[v][1], 1)
            rows.append(rec)
            print(f"  {ct[:38]:38s} m33={KNOWN[ct]:.4f} | " +
                  " ".join(f"{v}={out[v][0]:.4f}({out[v][1]:.0f}s)" for v in "ABCD"),
                  flush=True)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, "results", "m35_solver_check.csv")
    df.to_csv(out_csv, index=False)
    print("\n=== summary ===")
    print(df.to_string(index=False))
    for v in "ABCD":
        d = (df[f"AUC_{v}"] - df.m33_fullgene_auc).abs().mean()
        print(f"  variant {v}: mean |AUC - m33| = {d:.4f}   total fit time {df[f'sec_{v}'].sum():.0f}s")
    sp = df.sec_A.sum() / max(1e-9, df.sec_B.sum())
    print(f"\n  observed speedup B vs A: {sp:.1f}x")
    n_pbmc = len(np.unique(donors))
    per_fold_b = df.sec_B.sum() / (len(df) * n_pbmc)
    print(f"  kernel cost scales ~ n^2*p, so OneK1K (981 vs {n_pbmc} donors) = "
          f"{(981/n_pbmc)**2:.1f}x the per-fold kernel of PBMC")
    print(f"  measured PBMC per-fold (B) = {per_fold_b*1000:.0f} ms"
          f"  => projected OneK1K ~ {per_fold_b*(981/n_pbmc)**2:.2f}s/fold")
    print(f"  projected OneK1K LODO eval (981 folds) ~ "
          f"{per_fold_b*(981/n_pbmc)**2*981/60:.1f} min")
    print("[done]")


if __name__ == "__main__":
    main()
