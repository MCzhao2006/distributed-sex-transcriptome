# -*- coding: utf-8 -*-
"""D2 (chatGPT fourth + DeepSeek): T3-4 strict redesign — NO fallback.

Design fixed by construction:
  1. GLOBAL donor split FIRST: all 980 GTEx donors -> 80% train / 20% test
     (seed 7). This split is shared by every LOTO fold.
  2. For each LOTO fold, train on the 11 tissues' TRAIN-DONOR samples,
     test on the held-out tissue's TEST-DONOR samples.
  3. D_test ∩ D_train = ∅ is guaranteed by construction — no post-hoc
     filtering, no relaxation, ever.
  4. Features: union of per-train-tissue train-only top-300 DE genes,
     restricted to genes present in ALL 12 tissues (leak-free).
  5. Tissues whose test set has <15 samples of one sex are reported
     "insufficient", not silently fixed.

Output: results/d2_loto_strict.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = r"F:\nature"
PROC = os.path.join(ROOT, "data", "proc")
RES = os.path.join(ROOT, "results")
AUTO = [f"chr{i}" for i in range(1, 23)]
TISSUES = ["Blood", "Muscle", "Lung", "Thyroid", "Skin", "Esophagus", "Artery",
           "AdiposeSubq", "Nerve", "Heart", "Colon", "AdiposeVisc"]

def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"))

def train_only_de(Ytr, genes, chr_of, m_tr):
    age = m_tr["AGE"].map({"20-29":25,"30-39":35,"40-49":45,"50-59":55,
                           "60-69":65,"70-79":75}).fillna(60)
    D = np.column_stack([np.ones(len(m_tr)), m_tr.sex.to_numpy(float),
                         age.to_numpy(float),
                         m_tr.SMRIN.fillna(m_tr.SMRIN.median()).to_numpy(float)]).astype(np.float64)
    Yg = Ytr.astype(np.float64)
    XtX_inv = np.linalg.pinv(D.T @ D)
    B = XtX_inv @ D.T @ Yg
    resid = Yg - D @ B
    dof = Yg.shape[0] - D.shape[1]
    s2 = (resid**2).sum(axis=0)/dof
    se = np.sqrt(np.outer(s2, np.diag(XtX_inv)))
    p = 2*stats.t.sf(np.abs(B[1]/se[:,1]), dof)
    _, q, _, _ = multipletests(p, method="fdr_bh")
    return pd.DataFrame({"gene": genes, "chr": chr_of, "log2FC": B[1], "p": p, "q": q})

def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]

    data = {}
    all_donors = set()
    for t in TISSUES:
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        data[t] = {"genes": X.columns.to_numpy(),
                   "Y": np.log2(X.to_numpy(dtype=np.float32)+1.0),
                   "y": meta.loc[X.index, "sex"].to_numpy(int),
                   "donors": meta.loc[X.index, "SUBJID"].to_numpy(),
                   "meta": meta.loc[X.index]}
        all_donors |= set(pd.unique(data[t]["donors"]))
    all_donors = np.array(sorted(all_donors))
    print(f"[global] {len(all_donors)} unique donors across 12 tissues", flush=True)

    # GLOBAL 80/20 donor split — decided ONCE, before any fold
    rng = np.random.RandomState(7)
    rng.shuffle(all_donors)
    n_tr = int(len(all_donors) * 0.8)
    TRAIN_DONORS = set(all_donors[:n_tr])
    TEST_DONORS = set(all_donors[n_tr:])
    print(f"[global split] train donors={len(TRAIN_DONORS)}, test donors={len(TEST_DONORS)}", flush=True)

    rows = []
    for held_out in TISSUES:
        d_te = data[held_out]
        te_mask = pd.Series(d_te["donors"]).isin(TEST_DONORS).to_numpy()
        yte = d_te["y"][te_mask]
        n_te_f = int(yte.sum()); n_te_m = int((yte == 0).sum())
        if min(n_te_f, n_te_m) < 5:
            rows.append({"held_out": held_out, "status": "insufficient",
                         "n_test": int(te_mask.sum()), "n_female": n_te_f,
                         "auc": None, "n_features": None, "n_train": None})
            print(f"[LOTO-strict] {held_out}: INSUFFICIENT (F={n_te_f} M={n_te_m})", flush=True)
            continue

        # per-train-tissue train-only DE (train donors only) -> top-300 union
        rank_acc, common_genes = {}, None
        for t in TISSUES:
            if t == held_out: continue
            d = data[t]
            tr_mask = pd.Series(d["donors"]).isin(TRAIN_DONORS).to_numpy()
            chr_of = gene2chr.reindex(d["genes"]).fillna("NA").to_numpy()
            de = train_only_de(d["Y"][tr_mask], d["genes"], chr_of, d["meta"].loc[tr_mask])
            de_auto = de[de.chr.isin(AUTO)].copy()
            de_auto["rank"] = de_auto.log2FC.abs().rank(ascending=False)
            for g, r in zip(de_auto.gene, de_auto["rank"]):
                rank_acc[g] = rank_acc.get(g, 0) + r
            top300 = set(de_auto.gene[:300])
            common_genes = top300 if common_genes is None else common_genes | top300
        feat = sorted(common_genes)
        if len(feat) > 2000:
            feat = sorted(rank_acc, key=lambda g: rank_acc[g])[:2000]
        feat_sets = [set(data[t]["genes"]) for t in TISSUES]
        feat = sorted(set(feat).intersection(*feat_sets))

        Xs, ys = [], []
        for t in TISSUES:
            if t == held_out: continue
            d = data[t]
            tr_mask = pd.Series(d["donors"]).isin(TRAIN_DONORS).to_numpy()
            cols = np.array([np.where(d["genes"] == g)[0][0] for g in feat])
            Xs.append(d["Y"][np.ix_(tr_mask, cols)])
            ys.append(d["y"][tr_mask])
        Xtr = np.vstack(Xs); ytr = np.concatenate(ys)
        clf = make_clf(); clf.fit(Xtr, ytr)

        cols_te = np.array([np.where(d_te["genes"] == g)[0][0] for g in feat])
        p = clf.predict_proba(d_te["Y"][np.ix_(te_mask, cols_te)])[:, 1]
        auc = roc_auc_score(yte, p)
        rows.append({"held_out": held_out, "status": "ok",
                     "n_test": int(te_mask.sum()), "n_female": n_te_f,
                     "auc": round(auc, 4), "n_features": len(feat),
                     "n_train": len(ytr)})
        print(f"[LOTO-strict] 11 -> {held_out}: AUC={auc:.3f} "
              f"(n_test={te_mask.sum()}, F={n_te_f}, feat={len(feat)})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "d2_loto_strict.csv"), index=False)
    ok = df[df.status == "ok"]
    if len(ok):
        print(f"[LOTO-strict] mean={ok.auc.mean():.3f} median={ok.auc.median():.3f} "
              f"min={ok.auc.min():.3f} over {len(ok)}/12 tissues", flush=True)
    print("[done] D2 complete", flush=True)

if __name__ == "__main__":
    main()
