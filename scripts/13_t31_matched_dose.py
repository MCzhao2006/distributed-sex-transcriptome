# -*- coding: utf-8 -*-
"""T3-1: Matched random-gene dose-response (chatGPT third review, item 4).

Objection: plain random sets may be enriched for high-expression genes whose
statistics differ. Controls:
  A  plain random (baseline, from fig5)
  B  mean-expression matched: bin all autosomal genes by median expression
     quartile, sample within bins proportionally
  C  variance matched: same by expression variance quartile
  D  chromosome matched: sample proportional to per-chromosome gene counts
All donor-split Muscle/Thyroid/Blood, N in {200,1000,5000}, 50 seeds each.

Output: results/t3_1_matched_dose.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

ROOT = r"F:\nature"
PROC = os.path.join(ROOT, "data", "proc")
RES = os.path.join(ROOT, "results")

def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"))

def stratified_sample(keys, n, rng):
    """Sample n indices proportional to key strata (keys: array of strata)."""
    uniq, counts = np.unique(keys, return_counts=True)
    prob = counts / counts.sum()
    # assign each stratum its share
    take = np.floor(prob * n).astype(int)
    rem = n - take.sum()
    extra = rng.choice(len(uniq), size=rem, p=prob)
    for e in extra: take[e] += 1
    idx = []
    for u, k in zip(uniq, take):
        pool = np.where(keys == u)[0]
        idx.extend(rng.choice(pool, size=min(k, len(pool)), replace=False).tolist())
    return np.array(idx[:n])

def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]
    SIZES = [200, 1000, 5000]
    SEEDS = 50
    rows = []
    for t in ["Muscle", "Thyroid", "Blood"]:
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        genes = X.columns.to_numpy()
        Y = np.log2(X.to_numpy(dtype=np.float32) + 1.0)
        y = meta.loc[X.index, "sex"].to_numpy(int)
        donors = meta.loc[X.index, "SUBJID"].to_numpy()
        rng = np.random.RandomState(7)
        uniq = pd.unique(donors); rng.shuffle(uniq)
        tr = set(uniq[:int(len(uniq) * 0.6)])
        tr_mask = pd.Series(donors).isin(tr).to_numpy()
        chr_of = gene2chr.reindex(genes).fillna("NA").to_numpy()
        auto_idx = np.where(~np.isin(chr_of, ["chrX", "chrY", "chrM"]))[0]

        # matching strata computed on TRAIN donors only
        Ytr = Y[tr_mask]
        med_expr = np.median(Ytr[:, auto_idx], axis=0)
        var_expr = Ytr[:, auto_idx].var(axis=0)
        strata = {
            "A_plain": np.zeros(len(auto_idx), int),
            "B_expr_matched": np.asarray(pd.qcut(med_expr, 4, labels=False, duplicates="drop")),
            "C_var_matched": np.asarray(pd.qcut(var_expr, 4, labels=False, duplicates="drop")),
            "D_chr_matched": pd.factorize(chr_of[auto_idx])[0],
        }
        for scheme, strat in strata.items():
            for n in SIZES:
                aucs = []
                for s in range(SEEDS):
                    r = np.random.RandomState(20_000 + s)
                    if scheme == "A_plain":
                        idx = r.choice(auto_idx, size=n, replace=False)
                    else:
                        sel_local = stratified_sample(strat, n, r)
                        idx = auto_idx[sel_local]
                    m = np.zeros(len(genes), bool); m[idx] = True
                    clf = make_clf()
                    clf.fit(Y[np.ix_(tr_mask, m)], y[tr_mask])
                    p = clf.predict_proba(Y[np.ix_(~tr_mask, m)])[:, 1]
                    aucs.append(roc_auc_score(y[~tr_mask], p))
                aucs = np.array(aucs)
                rows.append({"tissue": t, "scheme": scheme, "n_genes": n,
                             "auc_median": round(float(np.median(aucs)), 4),
                             "auc_q025": round(float(np.quantile(aucs, .025)), 4),
                             "auc_q975": round(float(np.quantile(aucs, .975)), 4)})
                print(f"[{t}] {scheme} N={n}: {np.median(aucs):.3f} "
                      f"({np.quantile(aucs,.025):.3f},{np.quantile(aucs,.975):.3f})", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "t3_1_matched_dose.csv"), index=False)
    print("[done] T3-1 complete", flush=True)

if __name__ == "__main__":
    main()
