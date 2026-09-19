# -*- coding: utf-8 -*-
"""Step 3: Machine-learning sex classification with strict no-leakage design.

Design (methodological heart of the project):
- Split by DONOR (SUBJID) at train time -> no same-person train/test leak
- Cross-tissue transfer: train on tissue A (60% donors), test on ALL donors of tissue B
- Feature sets:
    (a) all expressed genes   (b) autosomes only (no X/Y/MT)   (c) X+Y only
    (d) transfer: top-500 autosomal DE genes from the TRAIN tissue
- Model: L2 logistic regression on log2TPM+1, standardized
- Null control: one label-shuffle run per feature set (sanity baseline)
Outputs: results/ml_results.csv, results/ml_results_detail.csv
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

TISSUES = ["Blood", "Muscle", "Lung", "Thyroid", "Skin", "Esophagus", "Artery",
           "AdiposeSubq", "Nerve", "Heart", "Colon", "AdiposeVisc"]

def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"))

def donor_split(index, meta, frac=0.6, seed=7):
    rng = np.random.RandomState(seed)
    donors = meta.loc[index, "SUBJID"].unique()
    rng.shuffle(donors)
    tr = set(donors[:int(len(donors) * frac)])
    return meta.loc[index, "SUBJID"].isin(tr).to_numpy()

def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]

    rows, detail = [], []
    # preload all tissue matrices once (12 x ~800 x 59k float32 ~ 2.7GB worst case;
    # trim to expressed genes per tissue on load to stay within RAM)
    data = {}
    for t in TISSUES:
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]      # expressed-gene filter
        Y = np.log2(X.to_numpy(dtype=np.float32) + 1.0)
        data[t] = {"genes": X.columns.to_numpy(), "Y": Y,
                   "y": meta.loc[X.index, "sex"].to_numpy(dtype=int),
                   "idx": X.index}
        print(f"[load] {t}: {Y.shape}", flush=True)

    for t in TISSUES:
        d = data[t]
        genes, Y, y = d["genes"], d["Y"], d["y"]
        is_train = donor_split(d["idx"], meta)
        chr_of = gene2chr.reindex(genes).fillna("NA").to_numpy()
        ytr, yte = y[is_train], y[~is_train]
        print(f"=== {t}  (train {is_train.sum()} / test {(~is_train).sum()}) ===", flush=True)

        for name, mask in [("all", np.ones(len(genes), bool)),
                           ("autosome", ~np.isin(chr_of, ["chrX", "chrY", "chrM"])),
                           ("sexchr", np.isin(chr_of, ["chrX", "chrY"]))]:
            if mask.sum() < 10:
                continue
            Ytr, Yte = Y[np.ix_(is_train, mask)], Y[np.ix_(~is_train, mask)]
            clf = make_clf(); clf.fit(Ytr, ytr)
            p_te = clf.predict_proba(Yte)[:, 1]
            auc = roc_auc_score(yte, p_te)
            acc = ((p_te > 0.5).astype(int) == yte).mean()
            # null: single label-shuffle run (sanity baseline)
            rng = np.random.RandomState(123)
            clf0 = make_clf(); clf0.fit(Ytr, rng.permutation(ytr))
            auc_null = roc_auc_score(yte, clf0.predict_proba(Yte)[:, 1])
            rows.append({"tissue": t, "features": name, "n_genes": int(mask.sum()),
                         "n_test": int((~is_train).sum()), "auc_test": round(auc, 4),
                         "acc_test": round(acc, 4), "auc_null": round(auc_null, 4)})
            print(f"    {name:9s} n={mask.sum():6d}  AUC={auc:.3f} ACC={acc:.3f} null={auc_null:.3f}", flush=True)

        # (d) cross-tissue transfer with top-500 AUTOSOMAL DE genes from this tissue
        de = pd.read_csv(os.path.join(RES, f"de_{t}.csv"))
        de_auto = de[de.chr.isin([f"chr{i}" for i in range(1, 23)])]
        top = set(de_auto.reindex(
            de_auto.log2FC_adj.abs().sort_values(ascending=False).index
        ).head(500)["gene"])
        # aligned gene list (sorted order) present in BOTH train & test tissues
        common = sorted(top & set(genes))
        tr_cols = np.array([np.where(genes == g)[0][0] for g in common])
        Ytr = Y[np.ix_(is_train, tr_cols)]
        clf = make_clf(); clf.fit(Ytr, ytr)
        for t2 in TISSUES:
            if t2 == t:
                continue
            d2 = data[t2]
            common2 = sorted(set(common) & set(d2["genes"]))
            if len(common2) < 100:
                continue
            g = d2["genes"]
            te_cols = np.array([np.where(g == gg)[0][0] for gg in common2])
            tr_cols2 = np.array([np.where(genes == gg)[0][0] for gg in common2])
            if len(tr_cols2) != Ytr.shape[1]:   # retrain if subset lost genes
                clf2 = make_clf()
                clf2.fit(Y[np.ix_(is_train, tr_cols2)], ytr)
                p_te = clf2.predict_proba(d2["Y"][:, te_cols])[:, 1]
            else:
                p_te = clf.predict_proba(d2["Y"][:, te_cols])[:, 1]
            auc = roc_auc_score(d2["y"], p_te)
            acc = ((p_te > 0.5).astype(int) == d2["y"]).mean()
            detail.append({"train_tissue": t, "test_tissue": t2,
                           "n_genes": len(common2), "n_test": len(d2["y"]),
                           "auc": round(auc, 4), "acc": round(acc, 4)})
            print(f"    -> {t2:12s} AUC={auc:.3f} ACC={acc:.3f}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(RES, "ml_results.csv"), index=False)
    pd.DataFrame(detail).to_csv(os.path.join(RES, "ml_results_detail.csv"), index=False)
    print("[done] ML experiments complete", flush=True)

if __name__ == "__main__":
    main()
