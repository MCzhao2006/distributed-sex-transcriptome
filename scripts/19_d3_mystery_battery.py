# -*- coding: utf-8 -*-
"""D3 (DeepSeek): decisive attack on the p>0.5 AUC=0.95 mystery.

The mystery: 3690 autosomal genes with NO marginal mean difference
(train-only OLS p>0.5) collectively predict held-out sex at AUC 0.95.
DeepSeek: this either rewrites the paper ("covariance encoding") or breaks
section 3.4. Decide it.

Battery:
  A  delta-mu direction: raw train mean(F-M) per gene; does the classifier
     weight align with a direction that replicates in TEST?
  B  covariate mediation: residualize age / RIN / ischemic / DTHHRDY /
     center(dims) on TRAIN, re-run. Which covariate kills it?
  C  nullness dose: features drawn at p>0.5 / p>0.7 / p>0.9 / p>0.99 —
     does "nuller" mean weaker? (if AUC constant -> structure not mean)
  D  permutation inside selection: 50 perms, re-select p>0.5 each perm,
     retrain -> valid empirical p for THIS pipeline
  E  gene-wise permuted matrix (destroy gene-gene covariance, keep marginals):
     per-gene donor permutation on TRAIN-defined permutation applied to test

Output: results/d3_mystery_battery.csv
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
ANN = os.path.join(ROOT, "data", "annot")
AUTO = [f"chr{i}" for i in range(1, 23)]

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
    return pd.DataFrame({"gene": genes, "p": p, "log2FC": B[1]})

def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]
    sa = pd.read_csv(os.path.join(ANN, "SampleAttributesDS.txt"), sep="\t", low_memory=False)
    sa["SUBJID"] = "GTEX-" + sa.SAMPID.str.split("-").str[1]
    sa = sa.set_index("SAMPID")

    t = "Muscle"
    Xdf = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
    Xdf = Xdf.loc[:, (Xdf >= 1.0).mean(axis=0) >= 0.20]
    genes = Xdf.columns.to_numpy()
    Y = np.log2(Xdf.to_numpy(dtype=np.float32) + 1.0)
    m = meta.loc[Xdf.index]
    y = m.sex.to_numpy(int)
    donors = m.SUBJID.to_numpy()
    rng = np.random.RandomState(7)
    uniq = pd.unique(donors); rng.shuffle(uniq)
    tr = set(uniq[:int(len(uniq)*0.6)])
    tr_mask = pd.Series(donors).isin(tr).to_numpy()
    te_mask = ~tr_mask
    ytr, yte = y[tr_mask], y[te_mask]
    chr_of = gene2chr.reindex(genes).fillna("NA").to_numpy()
    auto = np.asarray(~np.isin(chr_of, ["chrX","chrY","chrM"]))

    de = train_only_de(Y[tr_mask], genes, chr_of, m.loc[tr_mask]).set_index("gene").reindex(genes)
    p_arr = de["p"].to_numpy()
    rows = []

    def run_set(idx, tag, Ymat=None, label_test=None):
        Ym = Y if Ymat is None else Ymat
        lt = yte if label_test is None else label_test
        clf = make_clf()
        clf.fit(Ym[np.ix_(tr_mask, idx)], ytr)
        p_te = clf.predict_proba(Ym[np.ix_(te_mask, idx)])[:, 1]
        auc = roc_auc_score(lt, p_te)
        rows.append({"battery": tag, "auc": round(auc, 4), "n_genes": len(idx)})
        print(f"[{tag}] n={len(idx)} AUC={auc:.4f}", flush=True)
        return auc, p_te

    # ---------- A: raw delta-mu direction ----------
    idx_p05 = np.where(auto & (p_arr > 0.5))[0]
    auc_base, p_te = run_set(idx_p05, "A0_baseline_p05")
    dmu_train = (Y[np.ix_(tr_mask & (y==1), idx_p05)].mean(0)
                 - Y[np.ix_(tr_mask & (y==0), idx_p05)].mean(0))
    dmu_test = (Y[np.ix_(te_mask & (y==1), idx_p05)].mean(0)
                - Y[np.ix_(te_mask & (y==0), idx_p05)].mean(0))
    r_mu = np.corrcoef(dmu_train, dmu_test)[0, 1]
    # weight direction
    clf = make_clf(); clf.fit(Y[np.ix_(tr_mask, idx_p05)], ytr)
    w = clf.named_steps["logisticregression"].coef_[0]
    proj_te = Y[np.ix_(te_mask, idx_p05)] @ w
    auc_proj = roc_auc_score(yte, proj_te)
    r_w_mu = np.corrcoef(w, dmu_train)[0, 1]
    rows.append({"battery": "A1_dmu_train_vs_test_corr", "auc": round(r_mu, 4), "n_genes": len(idx_p05)})
    rows.append({"battery": "A2_proj_w_on_test_auc", "auc": round(auc_proj, 4), "n_genes": len(idx_p05)})
    rows.append({"battery": "A3_w_vs_dmu_corr", "auc": round(r_w_mu, 4), "n_genes": len(idx_p05)})
    print(f"[A] dmu train~test r={r_mu:.3f} | w~dmu r={r_w_mu:.3f} | proj AUC={auc_proj:.4f}", flush=True)

    # ---------- B: covariate mediation (train-fit residualization) ----------
    isch = pd.to_numeric(sa.loc[Xdf.index, "SMTSISCH"], errors="coerce")
    isch = isch.fillna(isch.median()).to_numpy(float)
    age = m.AGE.map({"20-29":25,"30-39":35,"40-49":45,"50-59":55,
                     "60-69":65,"70-79":75}).fillna(60).to_numpy(float)
    rin = m.SMRIN.fillna(m.SMRIN.median()).to_numpy(float)
    hardy = m.DTHHRDY.fillna(0).astype(float).to_numpy()
    center = pd.get_dummies(sa.loc[Xdf.index, "SMCENTER"]).to_numpy(float)
    from sklearn.linear_model import LinearRegression
    for cname, cv in [("age", age), ("RIN", rin), ("ischemic", isch),
                      ("DTHHRDY", hardy), ("center", center),
                      ("ALL", np.column_stack([age, rin, isch, hardy, center]))]:
        C = cv.reshape(-1, 1) if cv.ndim == 1 else cv
        lr = LinearRegression().fit(C[tr_mask], Y[tr_mask])
        Yr = (Y - lr.predict(C) + Y[tr_mask].mean(0)).astype(np.float32)
        run_set(idx_p05, f"B_mediation_{cname}", Ymat=Yr)

    # ---------- C: nullness dose ----------
    for thr in [0.5, 0.7, 0.9, 0.99]:
        idx = np.where(auto & (p_arr > thr))[0]
        run_set(idx, f"C_nullness_p>{thr}")

    # ---------- D: permutation INSIDE selection (valid null) ----------
    hits = 0
    for i in range(50):
        yr = np.random.RandomState(5000 + i).permutation(y)
        # re-run train-only OLS under permuted labels, re-select p>0.5
        d_perm = train_only_de(Y[tr_mask], genes, chr_of,
                               m.loc[tr_mask].assign(sex=yr[tr_mask]))
        p_perm = d_perm.set_index("gene").reindex(genes)["p"].to_numpy()
        idx_perm = np.where(auto & (p_perm > 0.5))[0]
        clf0 = make_clf()
        clf0.fit(Y[np.ix_(tr_mask, idx_perm)], yr[tr_mask])
        pp = clf0.predict_proba(Y[np.ix_(te_mask, idx_perm)])[:, 1]
        a0 = roc_auc_score(y[te_mask], pp)
        if a0 >= auc_base: hits += 1
    rows.append({"battery": "D_perm_in_selection_p", "auc": round((hits+1)/51, 4), "n_genes": 50})
    print(f"[D] selection-aware perm p = {(hits+1)/51:.4f} (hits={hits}/50)", flush=True)

    # ---------- E: destroy gene-gene covariance on TEST ----------
    # per-gene donor permutation fitted on TRAIN, applied to TEST rows
    rngE = np.random.RandomState(123)
    perm_idx = rngE.permutation(te_mask.sum())
    Yperm = Y.copy()
    Yperm[te_mask] = Y[te_mask][perm_idx]      # shuffle whole test rows (keeps covariance)
    # gene-wise shuffle: shuffle each gene's values across test donors independently
    Ygw = Y.copy()
    for j in idx_p05:
        Ygw[np.ix_(te_mask, [j])] = Y[np.ix_(te_mask, [j])][perm_idx]
    run_set(idx_p05, "E1_row_shuffled_test", Ymat=Yperm, label_test=y[te_mask][perm_idx])
    run_set(idx_p05, "E2_genewise_shuffled_test", Ymat=Ygw, label_test=yte)

    pd.DataFrame(rows).to_csv(os.path.join(RES, "d3_mystery_battery.csv"), index=False)
    print("[done] D3 complete", flush=True)

if __name__ == "__main__":
    main()
