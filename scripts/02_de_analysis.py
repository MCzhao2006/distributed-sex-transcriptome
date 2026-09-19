# -*- coding: utf-8 -*-
"""Step 2: Per-tissue differential expression (male vs female) with covariates.

Model per gene:  log2TPM+1 ~ sex + age + RIN   (OLS via statsmodels)
  age is CONTINUOUS: GTEx AGE decade bins mapped to their midpoint
  (20-29->25 ... 70-79->75), missing filled with 60. It is NOT a categorical
  age_group; an earlier docstring said age_group and was wrong.
- TPM floor: genes expressed (TPM>=1) in <20% of samples per tissue are dropped
- BH-FDR correction per tissue
Outputs:
  results/de_{tissue}.csv          (gene, log2FC, p, q, n_m, n_f)
  results/de_summary.csv           (per-tissue counts at q<0.05)
"""
import os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = r"F:\nature"
PROC = os.path.join(ROOT, "data", "proc")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

TISSUES = ["Blood", "Muscle", "Lung", "Thyroid", "Skin", "Esophagus", "Artery",
           "AdiposeSubq", "Nerve", "Heart", "Colon", "AdiposeVisc"]

meta_all = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
meta_all["age_num"] = meta_all["AGE"].map({
    "20-29": 25, "30-39": 35, "40-49": 45, "50-59": 55, "60-69": 65, "70-79": 75})

summary = []
for t in TISSUES:
    pq = os.path.join(PROC, f"tpm_{t}.parquet")
    print(f"[{t}] loading ...", flush=True)
    X = pd.read_parquet(pq)
    m = meta_all.loc[X.index]
    # expression filter: TPM>=1 in >=20% samples
    keep_genes = (X >= 1.0).mean(axis=0) >= 0.20
    X = X.loc[:, keep_genes]
    print(f"    {X.shape[0]} samples x {X.shape[1]} expressed genes", flush=True)

    Y = np.log2(X.to_numpy(dtype=np.float32) + 1.0)          # log2TPM+1
    genes = X.columns.to_numpy()
    cov = pd.DataFrame({
        "sex": m["sex"].to_numpy(dtype=np.float32),
        "age": m["age_num"].fillna(60).to_numpy(dtype=np.float32),
        "rin": m["SMRIN"].fillna(m["SMRIN"].median()).to_numpy(dtype=np.float32),
    })
    D = np.column_stack([np.ones(len(Y)), cov.to_numpy()])    # [1, sex, age, rin]

    # vectorized OLS per gene: Y (n x G) = D (n x 4) @ B (4 x G)
    XtX_inv = np.linalg.pinv(D.T @ D)                          # 4 x 4
    beta = XtX_inv @ D.T @ Y                                   # 4 x G
    resid = Y - D @ beta                                       # n x G
    dof = Y.shape[0] - D.shape[1]
    sigma2 = (resid ** 2).sum(axis=0) / dof                    # G
    se = np.sqrt(np.outer(sigma2, np.diag(XtX_inv)))           # G x 4
    tvals = beta.T / se
    pvals = 2 * __import__("scipy").stats.t.sf(np.abs(tvals), dof)
    rej, qvals, _, _ = multipletests(pvals[:, 1], method="fdr_bh")

    # log2FC = beta_sex (already on log2 scale, adjusted for covariates)
    de = pd.DataFrame({
        "gene": genes, "chr": pd.read_csv(os.path.join(PROC, "gene2chr.csv"),
                                          index_col=0).loc[genes, "chr"].values,
        "log2FC_adj": beta[1], "p_sex": pvals[:, 1], "q_sex": qvals,
    })
    de.to_csv(os.path.join(RES, f"de_{t}.csv"), index=False)
    n_sig = int((qvals < 0.05).sum())
    up = int(((qvals < 0.05) & (beta[1] > 0)).sum())     # female-high
    dn = int(((qvals < 0.05) & (beta[1] < 0)).sum())     # male-high
    summary.append({"tissue": t, "n_tested": len(genes), "n_sig_q05": n_sig,
                    "female_high": up, "male_high": dn})
    print(f"    DE genes (q<0.05): {n_sig}  (F-high {up} / M-high {dn})", flush=True)
    del X, Y, de

pd.DataFrame(summary).to_csv(os.path.join(RES, "de_summary.csv"), index=False)
print("[done] DE across tissues complete", flush=True)
