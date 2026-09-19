# -*- coding: utf-8 -*-
"""d3_a2_a3_matched_space.py — corrected-space diagnostic for D3 batteries A2/A3.

WHY THIS FILE EXISTS (do not merge into d3_mystery_battery.csv)
  `19_d3_mystery_battery.py` L108-111 takes the weight vector from the
  STANDARDISED space (`clf.named_steps["logisticregression"].coef_[0]`) and then
    * applies it to the RAW expression matrix  (L109), and
    * correlates it with the RAW-space delta-mu (L111).
  AUC is invariant to a constant offset, so the centring term is harmless, but the
  per-gene 1/sigma factor reweights the direction across genes.

  The historical executed values are therefore:
      A2 = 0.8892      A3 = 0.8280        (results/d3_mystery_battery.csv)
  They are CORRECT as a record of what was run and are NOT modified here.

  This script emits the same statistics computed in the matched (standardised)
  space, as a separate, explicitly labelled diagnostic product.

NOTE
  In the matched space, A2 reproduces the classifier's own held-out AUC by
  construction (projecting a fitted linear model's decision function and ranking
  is the model's prediction), so it carries no information beyond A0. The
  mismatched form is the only one that says anything independent.

Output: results/d3_a2_a3_matched_space.csv   (new file; nothing is overwritten)
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy import stats

ROOT = r"F:\nature"
PROC = os.path.join(ROOT, "data", "proc")
RES = os.path.join(ROOT, "results")
AGE = {"20-29": 25, "30-39": 35, "40-49": 45, "50-59": 55, "60-69": 65, "70-79": 75}


def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]
    rows = []

    for t in ["Muscle"]:                       # D3 was run on Muscle only
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        genes = X.columns.to_numpy()
        Y = np.log2(X.to_numpy(dtype=np.float32) + 1.0)
        m = meta.loc[X.index]
        y = m.sex.to_numpy(int)
        donors = m.SUBJID.to_numpy()

        rng = np.random.RandomState(7)
        uniq = pd.unique(donors)
        rng.shuffle(uniq)
        tr_mask = pd.Series(donors).isin(set(uniq[:int(len(uniq) * 0.6)])).to_numpy()
        te_mask = ~tr_mask

        chr_of = gene2chr.reindex(genes).fillna("NA").to_numpy()
        auto = ~np.isin(chr_of, ["chrX", "chrY", "chrM"])

        # train-only DE, identical to 19_d3_mystery_battery.py L43-57
        m_tr = m.loc[tr_mask]
        age = m_tr["AGE"].map(AGE).fillna(60)
        D = np.column_stack([np.ones(len(m_tr)), m_tr.sex.to_numpy(float),
                             age.to_numpy(float),
                             m_tr.SMRIN.fillna(m_tr.SMRIN.median()).to_numpy(float)]).astype(np.float64)
        Ytr = Y[tr_mask].astype(np.float64)
        XtX_inv = np.linalg.pinv(D.T @ D)
        B = XtX_inv @ D.T @ Ytr
        resid = Ytr - D @ B
        dof = Ytr.shape[0] - D.shape[1]
        s2 = (resid ** 2).sum(axis=0) / dof
        se = np.sqrt(np.outer(s2, np.diag(XtX_inv)))
        p_arr = 2 * stats.t.sf(np.abs(B[1] / se[:, 1]), dof)

        idx = np.where(auto & (p_arr > 0.5))[0]
        ytr, yte = y[tr_mask], y[te_mask]

        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"))
        clf.fit(Y[np.ix_(tr_mask, idx)], ytr)
        sc = clf.named_steps["standardscaler"]
        w = clf.named_steps["logisticregression"].coef_[0]

        Yte_raw = Y[np.ix_(te_mask, idx)]
        Yte_std = sc.transform(Yte_raw)
        dmu = (Y[np.ix_(tr_mask & (y == 1), idx)].mean(0)
               - Y[np.ix_(tr_mask & (y == 0), idx)].mean(0))

        a2_mis = float(roc_auc_score(yte, Yte_raw @ w))
        a2_mat = float(roc_auc_score(yte, Yte_std @ w))
        a3_mis = float(np.corrcoef(w, dmu)[0, 1])
        a3_mat = float(np.corrcoef(w / sc.scale_, dmu)[0, 1])
        a0 = float(roc_auc_score(yte, clf.predict_proba(Yte_raw)[:, 1]))

        def add(stat, space, val, note):
            rows.append({"tissue": t, "statistic": stat, "feature_space": space,
                         "value": round(val, 4), "note": note})

        add("n_pool", "-", len(idx), "train-only autosomal p>0.5 pool size")
        add("A0_baseline_auc", "standardised", a0, "classifier's own held-out AUC")
        add("A2_weight_projection_auc", "mismatched (raw X @ w)",
            a2_mis, "HISTORICAL EXECUTED VALUE; reproduces d3_mystery_battery.csv A2")
        add("A2_weight_projection_auc", "matched (standardised)",
            a2_mat, "equal to A0 by construction; carries no independent information")
        add("A3_weight_dmu_corr", "mismatched (w scaled, dmu raw)",
            a3_mis, "HISTORICAL EXECUTED VALUE; reproduces d3_mystery_battery.csv A3")
        add("A3_weight_dmu_corr", "matched (w in raw units)",
            a3_mat, "weights expressed back in raw units")
        add("cosine_w_raw_vs_scaled", "-",
            float(w @ (w / sc.scale_) / (np.linalg.norm(w) * np.linalg.norm(w / sc.scale_))),
            "how far the two weight representations deviate in direction")

    out = os.path.join(RES, "d3_a2_a3_matched_space.csv")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING TO OVERWRITE existing {out}")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
