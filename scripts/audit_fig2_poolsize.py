# -*- coding: utf-8 -*-
"""AUDIT DIAGNOSTIC ONLY -- Fig.2 provenance audit.

This script RE-RUNS NOTHING that feeds a figure. It writes NO result file.
It only answers provenance questions about ALREADY-EXISTING products:

  Q1  What is the actual size of the train-only p>0.5 non-significant pool?
      `d1_nonsig_dose.csv` / `d7_pc_dose_fine.csv` both request an N=4000
      point via `size=min(n, len(pool))`. If the pool is smaller than 4000,
      that plotted point is silently at the pool size, not at 4000.

  Q2  Are the >=3 independent `train_only_de()` implementations identical?
      17_d1 L35-49 and 19_d3 L43-57 are verbatim copies; 23_d7 L114-123 is
      an inline rewrite. Compare the resulting p-values on autosomal genes.

Run: python scripts/audit_fig2_poolsize.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"F:\nature"
PROC = os.path.join(ROOT, "data", "proc")

TISSUES = ["Muscle", "Thyroid", "Blood"]
SIZES = [50, 100, 200, 500, 1000, 2000, 4000]
AGE_MAP = {"20-29": 25, "30-39": 35, "40-49": 45,
           "50-59": 55, "60-69": 65, "70-79": 75}


def train_only_de_d1(Ytr, genes, m_tr):
    """Verbatim copy of 17_d1_nonsig_dose.py L35-49 (= 19_d3 L43-57)."""
    age = m_tr["AGE"].map(AGE_MAP).fillna(60)
    D = np.column_stack([np.ones(len(m_tr)), m_tr.sex.to_numpy(float),
                         age.to_numpy(float),
                         m_tr.SMRIN.fillna(m_tr.SMRIN.median()).to_numpy(float)]).astype(np.float64)
    Yg = Ytr.astype(np.float64)
    XtX_inv = np.linalg.pinv(D.T @ D)
    B = XtX_inv @ D.T @ Yg
    resid = Yg - D @ B
    dof = Yg.shape[0] - D.shape[1]
    s2 = (resid ** 2).sum(axis=0) / dof
    se = np.sqrt(np.outer(s2, np.diag(XtX_inv)))
    p = 2 * stats.t.sf(np.abs(B[1] / se[:, 1]), dof)
    return p, B[1]


def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    meta["age_num"] = meta["AGE"].map(AGE_MAP)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]

    print("=" * 78)
    print("Q1/Q2  Fig.2c+d  non-significant gene pool size (AUDIT DIAGNOSTIC)")
    print("=" * 78)

    for t in TISSUES:
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        genes = X.columns.to_numpy()
        Y = np.log2(X.to_numpy(dtype=np.float32) + 1.0)
        m = meta.loc[X.index]
        donors = m.SUBJID.to_numpy()

        rng = np.random.RandomState(7)
        uniq = pd.unique(donors)
        rng.shuffle(uniq)
        tr = set(uniq[:int(len(uniq) * 0.6)])
        tr_mask = pd.Series(donors).isin(tr).to_numpy()

        chr_of = gene2chr.reindex(genes).fillna("NA").to_numpy()
        auto = ~np.isin(chr_of, ["chrX", "chrY", "chrM"])

        n_na = int(np.sum(chr_of == "NA"))
        n_sexchr = int(np.sum(np.isin(chr_of, ["chrX", "chrY", "chrM"])))

        # ---- d1 implementation ----
        p_d1, fc_d1 = train_only_de_d1(Y[tr_mask], genes, m.loc[tr_mask])

        # ---- d7 implementation (inline, L114-123); NOTE: rin filled with the
        #      FULL-tissue median, not the train-fold median ----
        Ytr_auto = Y[np.ix_(tr_mask, auto)].astype(np.float64)
        age_all = m.age_num.fillna(60).to_numpy(float)
        rin_all = m.SMRIN.fillna(m.SMRIN.median()).to_numpy(float)
        D7 = np.column_stack([np.ones(int(tr_mask.sum())),
                              m.sex.to_numpy(int)[tr_mask].astype(float),
                              age_all[tr_mask], rin_all[tr_mask]]).astype(np.float64)
        XtXi = np.linalg.pinv(D7.T @ D7)
        B7 = XtXi @ D7.T @ Ytr_auto
        resid7 = Ytr_auto - D7 @ B7
        dof7 = Ytr_auto.shape[0] - D7.shape[1]
        s2_7 = (resid7 ** 2).sum(axis=0) / dof7
        se7 = np.sqrt(np.outer(s2_7, np.diag(XtXi)))
        p_d7 = 2 * stats.t.sf(np.abs(B7[1] / se7[:, 1]), dof7)

        p_d1_auto = p_d1[auto]
        dmax = float(np.max(np.abs(p_d1_auto - p_d7)))
        nonsig_d1 = int(np.sum(auto & (p_d1 > 0.5)))
        nonsig_d7 = int(np.sum(p_d7 > 0.5))
        n_sig = int(np.sum(auto & (p_d1 < 0.05)))

        print(f"\n--- {t} ---")
        print(f"  samples={X.shape[0]}  autosomal={int(auto.sum())}  "
              f"sex-chr/MT={n_sexchr}  chr-annotated-NA={n_na}")
        print(f"  train-only |p<0.05| autosomal = {n_sig}")
        print(f"  train-only NON-SIG pool (p>0.5)  d1-style = {nonsig_d1}   "
              f"d7-style = {nonsig_d7}")
        print(f"  max |p_d1 - p_d7| over autosomes = {dmax:.3e}   "
              f"({'IDENTICAL' if dmax == 0 else 'DIFFER'})")
        print(f"  effective N actually used at each requested point:")
        for n in SIZES:
            eff = min(n, nonsig_d1)
            flag = "" if eff == n else "   <-- CAPPED, point is NOT at this N"
            print(f"      requested N={n:5d} -> draws {eff:5d} genes{flag}")


if __name__ == "__main__":
    main()
