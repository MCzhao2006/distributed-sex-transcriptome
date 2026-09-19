# -*- coding: utf-8 -*-
"""Experiment 1: TRUE donor-disjoint cross-tissue transfer.

Previous flaw: test tissue used ALL donors -> some donors seen in training
(same person, different tissue) -> inflated transfer AUC.
Fix: test only on donors NOT in the training donor set:
    set(train_donors) ∩ set(test_donors) = ∅

Also adds metrics chatGPT asked for: AUPRC + balanced accuracy (2:1 imbalance
makes raw accuracy misleading).

Output: results/exp1_donor_disjoint_transfer.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = r"F:\nature"
PROC = os.path.join(ROOT, "data", "proc")
RES = os.path.join(ROOT, "results")

TISSUES = ["Blood", "Muscle", "Lung", "Thyroid", "Skin", "Esophagus", "Artery",
           "AdiposeSubq", "Nerve", "Heart", "Colon", "AdiposeVisc"]

def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"))

def balanced_acc(y, p, thr=0.5):
    yhat = (p > thr).astype(int)
    tpr = yhat[y == 1].mean()          # sensitivity
    tnr = (yhat[y == 0] == 0).mean()   # specificity
    return (tpr + tnr) / 2

def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]

    data = {}
    for t in TISSUES:
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        Y = np.log2(X.to_numpy(dtype=np.float32) + 1.0)
        data[t] = {"genes": X.columns.to_numpy(), "Y": Y,
                   "y": meta.loc[X.index, "sex"].to_numpy(dtype=int),
                   "donors": meta.loc[X.index, "SUBJID"].to_numpy(),
                   "idx": X.index}
        print(f"[load] {t}: {Y.shape}", flush=True)

    rows = []
    for t in TISSUES:                                  # train tissue
        d = data[t]
        genes, Y = d["genes"], d["Y"]
        # donor-level 60/40 split inside train tissue
        rng = np.random.RandomState(7)
        uniq = pd.unique(d["donors"]); rng.shuffle(uniq)
        tr_donors = set(uniq[:int(len(uniq) * 0.6)])
        tr_mask = pd.Series(d["donors"]).isin(tr_donors).to_numpy()
        ytr = d["y"][tr_mask]

        # features: top-500 autosomal DE genes of the train tissue
        de = pd.read_csv(os.path.join(RES, f"de_{t}.csv"))
        de_auto = de[de.chr.isin([f"chr{i}" for i in range(1, 23)])]
        top = set(de_auto.reindex(
            de_auto.log2FC_adj.abs().sort_values(ascending=False).index
        ).head(500)["gene"])

        for t2 in TISSUES:                             # test tissue
            if t2 == t:
                continue
            d2 = data[t2]
            # HARD RULE: exclude any donor that contributed to training
            te_ok = ~pd.Series(d2["donors"]).isin(tr_donors).to_numpy()
            common = sorted(top & set(genes) & set(d2["genes"]))
            if len(common) < 100 or te_ok.sum() < 30:
                continue
            g = d2["genes"]; gtr = genes
            te_cols = np.array([np.where(g == gg)[0][0] for gg in common])
            tr_cols = np.array([np.where(gtr == gg)[0][0] for gg in common])
            clf = make_clf(); clf.fit(Y[np.ix_(tr_mask, tr_cols)], ytr)
            p_te = clf.predict_proba(d2["Y"][np.ix_(te_ok, te_cols)])[:, 1]
            yte = d2["y"][te_ok]
            auc = roc_auc_score(yte, p_te)
            auprc = average_precision_score(yte, p_te)
            bacc = balanced_acc(yte, p_te)
            rows.append({"train_tissue": t, "test_tissue": t2,
                         "n_features": len(common),
                         "n_test_donor_disjoint": int(te_ok.sum()),
                         "frac_donors_removed": round(1 - te_ok.mean(), 3),
                         "auc": round(auc, 4), "auprc": round(auprc, 4),
                         "bal_acc": round(bacc, 4)})
            print(f"  {t:12s}->{t2:12s} AUC={auc:.3f} AUPRC={auprc:.3f} bACC={bacc:.3f} "
                  f"(removed {1-te_ok.mean():.0%} donors)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "exp1_donor_disjoint_transfer.csv"), index=False)
    print(f"\n[n={len(df)} donor-disjoint pairs] mean AUC={df.auc.mean():.3f} "
          f"median={df.auc.median():.3f} min={df.auc.min():.3f} max={df.auc.max():.3f}")
    nonblood = df[df.train_tissue != "Blood"]
    print(f"[non-Blood train] mean AUC={nonblood.auc.mean():.3f}")
    print("[done] exp1 complete", flush=True)

if __name__ == "__main__":
    main()
