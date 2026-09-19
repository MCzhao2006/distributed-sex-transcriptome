# -*- coding: utf-8 -*-
"""split_stability.py — does the bulk headline depend on the particular donor split?

Raised in REVIEWER_ATTACK_SURFACE.md (A2): every bulk headline AUROC in the manuscript comes from a
SINGLE 60/40 donor split at seed 7, with no uncertainty interval. This answers the reviewer
question directly: **does the result depend materially on the particular donor split?**

Protocol matched to the executed scripts (METHODS_parameters.md §3):
  * donor split: RandomState(seed).shuffle(pd.unique(donors)), first 60% = train
  * transfer features: train-fold autosomal DE, top-500 by |log2FC|
  * transfer test set: target-tissue donors MINUS the source's TRAIN donors
  * classifier: StandardScaler + L2 logistic (C=0.1)

Rows are appended as they are produced, so the run is resumable.
Env: PHASE=tissue|transfer|both   R_TISSUE  R_TRANSFER
Writes results/split_stability.csv
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
AGE = {"20-29": 25, "30-39": 35, "40-49": 45, "50-59": 55, "60-69": 65, "70-79": 75}
AUTO = [f"chr{i}" for i in range(1, 23)]
TISSUES = ["Blood", "Muscle", "Lung", "Thyroid", "Skin", "Esophagus", "Artery",
           "AdiposeSubq", "Nerve", "Heart", "Colon", "AdiposeVisc"]
R_TISSUE = int(os.environ.get("R_TISSUE", "50"))
R_TRANSFER = int(os.environ.get("R_TRANSFER", "10"))
PHASE = os.environ.get("PHASE", "both")
OUT = os.path.join(RES, "split_stability.csv")


def emit(row):
    """Append one row immediately, so a long run survives interruption."""
    header = not os.path.exists(OUT)
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=header, index=False)


def done_units(analysis):
    if not os.path.exists(OUT):
        return set()
    try:
        d = pd.read_csv(OUT)
    except Exception:
        return set()
    if "unit" not in d.columns:
        return set()
    return set(d.loc[d.analysis == analysis, "unit"].astype(str))


def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"))


def split_masks(donors, seed, frac=0.6):
    rng = np.random.RandomState(seed)
    uniq = pd.unique(donors)
    rng.shuffle(uniq)
    tr_donors = set(uniq[:int(len(uniq) * frac)])
    return pd.Series(donors).isin(tr_donors).to_numpy(), tr_donors


def train_only_de(Ytr, genes, chr_of, m_tr):
    age = m_tr["AGE"].map(AGE).fillna(60)
    D = np.column_stack([np.ones(len(m_tr)), m_tr.sex.to_numpy(float), age.to_numpy(float),
                         m_tr.SMRIN.fillna(m_tr.SMRIN.median()).to_numpy(float)]).astype(np.float64)
    Yg = Ytr.astype(np.float64)
    XtXi = np.linalg.pinv(D.T @ D)
    B = XtXi @ D.T @ Yg
    resid = Yg - D @ B
    dof = Yg.shape[0] - D.shape[1]
    s2 = (resid ** 2).sum(axis=0) / dof
    se = np.sqrt(np.outer(s2, np.diag(XtXi)))
    p = 2 * stats.t.sf(np.abs(B[1] / se[:, 1]), dof)
    q = multipletests(p, method="fdr_bh")[1]
    return pd.DataFrame({"gene": genes, "chr": chr_of, "log2FC": B[1], "p": p, "q": q}
                        ).set_index("gene")


def summ(a):
    a = np.asarray(a, float)
    return dict(n=len(a), mean=round(float(a.mean()), 4), median=round(float(np.median(a)), 4),
                p025=round(float(np.quantile(a, .025)), 4),
                p975=round(float(np.quantile(a, .975)), 4),
                min=round(float(a.min()), 4), max=round(float(a.max()), 4),
                sd=round(float(a.std(ddof=1)), 4))


def phase_tissue(meta, gene2chr):
    done = done_units("per_tissue_autosome")
    for t in TISSUES:
        if t in done:
            print(f"[{t:12s}] already done, skipping", flush=True)
            continue
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        genes = X.columns.to_numpy()
        auto = ~np.isin(gene2chr.reindex(genes).fillna("NA").to_numpy(), ["chrX", "chrY", "chrM"])
        Ya = np.log2(X.to_numpy(dtype=np.float32) + 1.0)[:, auto]
        y = meta.loc[X.index, "sex"].to_numpy(int)
        donors = meta.loc[X.index, "SUBJID"].to_numpy()

        aucs, seed7 = [], None
        for r in range(R_TISSUE):
            tr, _ = split_masks(donors, 7 + 1000 * r)
            clf = make_clf()
            clf.fit(Ya[tr], y[tr])
            a = roc_auc_score(y[~tr], clf.predict_proba(Ya[~tr])[:, 1])
            if r == 0:
                seed7 = a
            aucs.append(a)
        s = summ(aucs)
        emit({"analysis": "per_tissue_autosome", "unit": t, "seed7": round(float(seed7), 4), **s})
        print(f"[{t:12s}] seed7={seed7:.4f}  across {s['n']} splits: median={s['median']:.4f} "
              f"[{s['p025']:.4f},{s['p975']:.4f}] sd={s['sd']:.4f}", flush=True)
        del X, Ya


def phase_transfer(meta, gene2chr):
    doneT = done_units("cross_tissue_transfer")
    cache = {}
    for t in TISSUES:
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        cache[t] = {"genes": X.columns.to_numpy(),
                    "Y": np.log2(X.to_numpy(dtype=np.float32) + 1.0),
                    "idx": X.index}
        print(f"[load] {t}", flush=True)

    means = []
    for r in range(R_TRANSFER):
        seed = 7 + 1000 * r
        if f"split_seed{seed}" in doneT:
            print(f"[transfer seed {seed}] already done, skipping", flush=True)
            continue
        per_pair = []
        for t in TISSUES:
            d1 = cache[t]
            y1 = meta.loc[d1["idx"], "sex"].to_numpy(int)
            tr, tr_donors = split_masks(meta.loc[d1["idx"], "SUBJID"].to_numpy(), seed)
            de = train_only_de(d1["Y"][tr], d1["genes"],
                               gene2chr.reindex(d1["genes"]).fillna("NA").to_numpy(),
                               meta.loc[d1["idx"]][tr])
            de_auto = de[de.chr.isin(AUTO)]
            top = set(de_auto.reindex(
                de_auto.log2FC.abs().sort_values(ascending=False).index).head(500).index)
            for t2 in TISSUES:
                if t2 == t:
                    continue
                d2 = cache[t2]
                te_ok = ~pd.Series(meta.loc[d2["idx"], "SUBJID"].to_numpy()).isin(tr_donors).to_numpy()
                common = sorted(top & set(d1["genes"]) & set(d2["genes"]))
                if len(common) < 100 or te_ok.sum() < 30:
                    continue
                c1 = np.array([np.where(d1["genes"] == g)[0][0] for g in common])
                c2 = np.array([np.where(d2["genes"] == g)[0][0] for g in common])
                y2 = meta.loc[d2["idx"], "sex"].to_numpy(int)
                clf = make_clf()
                clf.fit(d1["Y"][np.ix_(tr, c1)], y1[tr])
                pr = clf.predict_proba(d2["Y"][np.ix_(te_ok, c2)])[:, 1]
                per_pair.append(roc_auc_score(y2[te_ok], pr))
        m = float(np.mean(per_pair))
        means.append(m)
        emit({"analysis": "cross_tissue_transfer", "unit": f"split_seed{seed}",
              "seed7": round(m, 4) if r == 0 else np.nan, **summ(per_pair)})
        print(f"[transfer seed {seed}] n_pairs={len(per_pair)} mean={m:.4f}", flush=True)

    s = summ(means)
    emit({"analysis": "cross_tissue_transfer_ACROSS_SPLITS", "unit": "mean per split",
          "seed7": round(means[0], 4), **s})
    print(f"\ntransfer mean across {s['n']} splits: median={s['median']:.4f} "
          f"[{s['p025']:.4f},{s['p975']:.4f}] sd={s['sd']:.4f}")


def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]
    if os.path.exists(OUT):
        print(f"resuming from existing {OUT}", flush=True)
    if PHASE in ("both", "tissue"):
        phase_tissue(meta, gene2chr)
    if PHASE in ("both", "transfer"):
        phase_transfer(meta, gene2chr)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
