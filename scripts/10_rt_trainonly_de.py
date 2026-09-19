# -*- coding: utf-8 -*-
"""Red-Team A1+A2: Kill the feature-selection leakage.

Previous flaw: de_{t}.csv was computed on ALL donors of the tissue, so the
top-500 / significance masks used in transfer & ablation encodes TEST labels.

Fix (strict protocol, per tissue split):
 1. split donors 60/40 (seed 7)
 2. DE computed on TRAIN donors ONLY:  log2TPM ~ sex + age + RIN (train)
 3. feature sets built from train-only stats:
      transfer: top-500 autosomal DE (train-only)
      L3: remove autosomal q<0.05 (train-only)
      L4: random autosomal matched size (train-only selection)
 4. donor-disjoint transfer: test donors = test-tissue donors MINUS train donors

Outputs:
  results/rt_exp1_transfer_trainonly.csv
  results/rt_exp2_ablation_trainonly.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = r"F:\nature"
PROC = os.path.join(ROOT, "data", "proc")
RES = os.path.join(ROOT, "results")

TISSUES = ["Blood", "Muscle", "Lung", "Thyroid", "Skin", "Esophagus", "Artery",
           "AdiposeSubq", "Nerve", "Heart", "Colon", "AdiposeVisc"]
AUTO = [f"chr{i}" for i in range(1, 23)]

def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"))

def train_only_de(Y, genes, chr_of, meta_sub, tr_mask):
    """OLS per gene on TRAIN donors only: log2TPM ~ sex + age + RIN."""
    m = meta_sub.loc[tr_mask]
    age = m["AGE"].map({"20-29": 25, "30-39": 35, "40-49": 45,
                        "50-59": 55, "60-69": 65, "70-79": 75}).fillna(60)
    D = np.column_stack([np.ones(tr_mask.sum()), m.sex.to_numpy(float),
                         age.to_numpy(float),
                         m.SMRIN.fillna(m.SMRIN.median()).to_numpy(float)]).astype(np.float64)
    Ytr = Y[tr_mask].astype(np.float64)
    XtX_inv = np.linalg.pinv(D.T @ D)
    B = XtX_inv @ D.T @ Ytr
    resid = Ytr - D @ B
    dof = Ytr.shape[0] - D.shape[1]
    s2 = (resid ** 2).sum(axis=0) / dof
    se = np.sqrt(np.outer(s2, np.diag(XtX_inv)))
    p = 2 * stats.t.sf(np.abs(B[1] / se[:, 1]), dof)
    _, q, _, _ = multipletests(p, method="fdr_bh")
    return pd.DataFrame({"gene": genes, "chr": chr_of,
                         "log2FC": B[1], "p": p, "q": q}).set_index("gene")

def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]

    data = {}
    for t in TISSUES:
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        data[t] = {"genes": X.columns.to_numpy(),
                   "Y": np.log2(X.to_numpy(dtype=np.float32) + 1.0),
                   "y": meta.loc[X.index, "sex"].to_numpy(dtype=int),
                   "donors": meta.loc[X.index, "SUBJID"].to_numpy(),
                   "idx": X.index}
        print(f"[load] {t}", flush=True)

    # ---------- A1: train-only DE transfer (132 pairs) ----------
    rows1 = []
    for t in TISSUES:
        d = data[t]
        genes, Y, y, donors = d["genes"], d["Y"], d["y"], d["donors"]
        rng = np.random.RandomState(7)
        uniq = pd.unique(donors); rng.shuffle(uniq)
        tr_donors = set(uniq[:int(len(uniq) * 0.6)])
        tr_mask = pd.Series(donors).isin(tr_donors).to_numpy()
        chr_of = gene2chr.reindex(genes).fillna("NA").to_numpy()

        de_tr = train_only_de(Y, genes, chr_of, meta.loc[d["idx"]], tr_mask)
        de_auto = de_tr[de_tr.chr.isin(AUTO)]
        top = set(de_auto.reindex(
            de_auto.log2FC.abs().sort_values(ascending=False).index
        ).head(500).index)

        for t2 in TISSUES:
            if t2 == t: continue
            d2 = data[t2]
            te_ok = ~pd.Series(d2["donors"]).isin(tr_donors).to_numpy()
            common = sorted(top & set(genes) & set(d2["genes"]))
            if len(common) < 100 or te_ok.sum() < 30: continue
            te_cols = np.array([np.where(d2["genes"] == g)[0][0] for g in common])
            tr_cols = np.array([np.where(genes == g)[0][0] for g in common])
            clf = make_clf(); clf.fit(Y[np.ix_(tr_mask, tr_cols)], y[tr_mask])
            p = clf.predict_proba(d2["Y"][np.ix_(te_ok, te_cols)])[:, 1]
            yte = d2["y"][te_ok]
            rows1.append({"train": t, "test": t2, "n_feat": len(common),
                          "n_test": int(te_ok.sum()),
                          "auc": round(roc_auc_score(yte, p), 4),
                          "auprc": round(average_precision_score(yte, p), 4)})
        print(f"[A1] {t} done", flush=True)
    df1 = pd.DataFrame(rows1)
    df1.to_csv(os.path.join(RES, "rt_exp1_transfer_trainonly.csv"), index=False)
    print(f"[A1] n={len(df1)} mean AUC={df1.auc.mean():.3f} median={df1.auc.median():.3f}", flush=True)

    # ---------- A2: train-only ablation (Muscle/Thyroid/Blood) ----------
    rows2 = []
    for t in ["Muscle", "Thyroid", "Blood"]:
        d = data[t]
        genes, Y, y, donors = d["genes"], d["Y"], d["y"], d["donors"]
        rng = np.random.RandomState(7)
        uniq = pd.unique(donors); rng.shuffle(uniq)
        tr_donors = set(uniq[:int(len(uniq) * 0.6)])
        tr_mask = pd.Series(donors).isin(tr_donors).to_numpy()
        chr_of = gene2chr.reindex(genes).fillna("NA").to_numpy()
        auto = ~np.isin(chr_of, ["chrX", "chrY", "chrM"])

        de_tr = train_only_de(Y, genes, chr_of, meta.loc[d["idx"]], tr_mask)
        sig_auto = set(de_tr.index[(de_tr.chr.isin(AUTO)) & (de_tr.q < 0.05)])
        g_ser = pd.Series(genes)
        ytr, yte = y[tr_mask], y[~tr_mask]

        layers = {
            "L1_autosome": auto,
            "L3_auto_minus_sigTRAINONLY": auto & ~g_ser.isin(sig_auto).to_numpy(),
        }
        # matched random control drawn from the FULL autosomal pool (incl. sig
        # genes), same size as L3 -> tests whether removing sig genes hurts
        rng2 = np.random.RandomState(99)
        full_auto_pool = np.where(auto)[0]
        n_l3 = int((auto & ~g_ser.isin(sig_auto).to_numpy()).sum())
        ridx = rng2.choice(full_auto_pool, size=n_l3, replace=False)
        l4 = np.zeros(len(genes), bool); l4[ridx] = True
        layers["L4_random_fullpool_matchL3"] = l4

        for name, mask in layers.items():
            clf = make_clf(); clf.fit(Y[np.ix_(tr_mask, mask)], ytr)
            p = clf.predict_proba(Y[np.ix_(~tr_mask, mask)])[:, 1]
            auc = roc_auc_score(yte, p)
            rows2.append({"tissue": t, "layer": name, "n_genes": int(mask.sum()),
                          "auc": round(auc, 4)})
            print(f"[A2] {t} {name} n={mask.sum()} AUC={auc:.3f}", flush=True)
    pd.DataFrame(rows2).to_csv(os.path.join(RES, "rt_exp2_ablation_trainonly.csv"), index=False)
    print("[done] RT-A1/A2 complete", flush=True)

if __name__ == "__main__":
    main()
