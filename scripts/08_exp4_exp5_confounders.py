# -*- coding: utf-8 -*-
"""Experiments 4+5: confounder stress tests.

Exp4  Cell-composition stress (proxy without xCell):
      Residualize out top-K PCs of the expression matrix that correlate with
      global expression structure, then re-run donor-split autosomal ML.
      If AUC survives removing PC structure associated with cell composition,
      the signal is not purely cell-type-mix.
      Simpler and stricter: regress out top 20 PCs from Y, keep residuals,
      re-run classifier. Also report how much variance PCs capture.

Exp5  sex x age interaction OLS per gene (12 tissues):
      log2TPM ~ sex + age + sex:age + RIN
      Count interaction q<0.05 genes; report top examples.
      -> Is the sex program stable across lifespan or age-modulated?

Output: results/exp4_pc_residualized_ml.csv, results/exp5_sex_age_interaction.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
import scipy.stats as st

ROOT = r"F:\nature"
PROC = os.path.join(ROOT, "data", "proc")
RES = os.path.join(ROOT, "results")

TISSUES = ["Blood", "Muscle", "Lung", "Thyroid", "Skin", "Esophagus", "Artery",
           "AdiposeSubq", "Nerve", "Heart", "Colon", "AdiposeVisc"]

def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"))

def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]
    meta["age_num"] = meta["AGE"].map({"20-29":25,"30-39":35,"40-49":45,
                                       "50-59":55,"60-69":65,"70-79":75})

    # ---------- Exp4 ----------
    rows4 = []
    for t in ["Blood", "Muscle", "Thyroid"]:
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        genes = X.columns.to_numpy()
        Y = np.log2(X.to_numpy(dtype=np.float32) + 1.0)
        y = meta.loc[X.index, "sex"].to_numpy(dtype=int)
        donors = meta.loc[X.index, "SUBJID"].to_numpy()
        rng = np.random.RandomState(7)
        uniq = pd.unique(donors); rng.shuffle(uniq)
        tr = set(uniq[:int(len(uniq)*0.6)])
        tr_mask = pd.Series(donors).isin(tr).to_numpy()
        chr_of = gene2chr.reindex(genes).fillna("NA").to_numpy()
        auto = ~np.isin(chr_of, ["chrX","chrY","chrM"])

        for k_pc in [0, 10, 20]:
            Yw = Y.copy()
            if k_pc > 0:
                # residualize top-k PCs computed on TRAIN donors only (leak-free)
                mu = Y[tr_mask].mean(axis=0)
                Xc = Y[tr_mask] - mu
                pca = PCA(n_components=k_pc, random_state=0).fit(Xc)
                scores = pca.transform(Y - mu)          # n x k
                comp = pca.components_                   # k x G
                # exact residual: remove PC-projected structure
                Yw = Y - scores @ comp
            clf = make_clf()
            clf.fit(Yw[np.ix_(tr_mask, auto)], y[tr_mask])
            p = clf.predict_proba(Yw[np.ix_(~tr_mask, auto)])[:, 1]
            auc = roc_auc_score(y[~tr_mask], p)
            rows4.append({"tissue": t, "pcs_removed": k_pc, "features": "autosome",
                          "auc": round(auc, 4)})
            print(f"[exp4] {t} remove {k_pc:2d} PCs -> autosome AUC={auc:.3f}", flush=True)

    pd.DataFrame(rows4).to_csv(os.path.join(RES, "exp4_pc_residualized_ml.csv"), index=False)

    # ---------- Exp5 ----------
    rows5 = []
    for t in TISSUES:
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        genes = X.columns.to_numpy()
        Y = np.log2(X.to_numpy(dtype=np.float32) + 1.0)
        m = meta.loc[X.index]
        sex = m.sex.to_numpy(dtype=np.float32)
        age = m.age_num.fillna(60).to_numpy(dtype=np.float32)
        rin = m.SMRIN.fillna(m.SMRIN.median()).to_numpy(dtype=np.float32)
        # design: 1, sex, age, sex*age, rin
        D = np.column_stack([np.ones(len(Y)), sex, age, sex*age, rin])
        XtX_inv = np.linalg.pinv(D.T @ D)
        B = XtX_inv @ D.T @ Y
        resid = Y - D @ B
        dof = Y.shape[0] - D.shape[1]
        s2 = (resid**2).sum(axis=0)/dof
        se = np.sqrt(np.outer(s2, np.diag(XtX_inv)))
        t_int = B[3] / se[:, 3]
        p_int = 2*st.t.sf(np.abs(t_int), dof)
        _, q_int, _, _ = multipletests(p_int, method="fdr_bh")
        t_sex = B[1] / se[:, 1]
        p_sex = 2*st.t.sf(np.abs(t_sex), dof)
        _, q_sex, _, _ = multipletests(p_sex, method="fdr_bh")
        rows5.append({"tissue": t, "n_tested": len(genes),
                      "n_int_q05": int((q_int < 0.05).sum()),
                      "n_main_sex_q05": int((q_sex < 0.05).sum()),
                      "median_abs_sex_t": float(np.median(np.abs(t_sex))),
                      "median_abs_int_t": float(np.median(np.abs(t_int)))})
        print(f"[exp5] {t}: sex:age interaction q<0.05 = {(q_int<0.05).sum()} "
              f"(main sex q<0.05 = {(q_sex<0.05).sum()})", flush=True)

    pd.DataFrame(rows5).to_csv(os.path.join(RES, "exp5_sex_age_interaction.csv"), index=False)
    print("[done] exp4+5 complete", flush=True)

if __name__ == "__main__":
    main()
