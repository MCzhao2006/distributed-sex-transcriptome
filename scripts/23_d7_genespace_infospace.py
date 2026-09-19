# -*- coding: utf-8 -*-
"""D7 (chatGPT sixth): Sex signal — gene-space vs information-space.

One integrated experiment on Muscle/Thyroid/Blood, strict donor split (seed 7):
  A  fine-grained PC dose: N in {1,2,5,10,20,30,50,75,100,150,200,300,500}
  B  gene dose using ONLY train-only p>0.5 non-significant genes
  C  PC interpretability: corr(PC_i, sex/age/RIN/ischemic/DTHHRDY) top-100
  D  PC residualization dose: remove PC1..k (k=0,1,5,10,20,50,100) then
     re-run autosomal gene ML -> does sex signal live IN the leading axes
     or beyond them?

Output: results/d7_pc_dose_fine.csv, results/d7_pc_metadata.csv,
        results/d7_pc_residual.csv, results/figures/fig8_genespace_infospace.png
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy import stats as st

ROOT = r"F:\nature"
PROC = os.path.join(ROOT, "data", "proc")
RES = os.path.join(ROOT, "results")
ANN = os.path.join(ROOT, "data", "annot")

PC_GRID = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
RESID_GRID = [0, 1, 5, 10, 20, 50, 100]
N_SEEDS_GENE = 30

def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"))

def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    meta["age_num"] = meta["AGE"].map({"20-29":25,"30-39":35,"40-49":45,
                                       "50-59":55,"60-69":65,"70-79":75})
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]
    sa = pd.read_csv(os.path.join(ANN, "SampleAttributesDS.txt"), sep="\t", low_memory=False)
    sa["SUBJID"] = "GTEX-" + sa.SAMPID.str.split("-").str[1]
    sa = sa.set_index("SAMPID")

    rows_a, rows_c, rows_d = [], [], []
    for t in ["Muscle", "Thyroid", "Blood"]:
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        genes = X.columns.to_numpy()
        Y = np.log2(X.to_numpy(dtype=np.float32) + 1.0)
        m = meta.loc[X.index]
        y = m.sex.to_numpy(int)
        donors = m.SUBJID.to_numpy()
        rng = np.random.RandomState(7)
        uniq = pd.unique(donors); rng.shuffle(uniq)
        tr = set(uniq[:int(len(uniq)*0.6)])
        tr_mask = pd.Series(donors).isin(tr).to_numpy()
        chr_of = gene2chr.reindex(genes).fillna("NA").to_numpy()
        auto = ~np.isin(chr_of, ["chrX","chrY","chrM"])
        Yauto = Y[:, auto]
        mu = Yauto[tr_mask].mean(axis=0)
        n_comp = min(500, min(Yauto[tr_mask].shape)-1)
        pca = PCA(n_components=n_comp, random_state=0)
        Ztr = pca.fit_transform(Yauto[tr_mask] - mu)
        Zte = pca.transform(Yauto[~tr_mask] - mu)
        ytr, yte = y[tr_mask], y[~tr_mask]
        print(f"=== {t}: pca done (n_comp={n_comp})", flush=True)

        # ---- A: fine PC dose ----
        for n_pc in PC_GRID:
            if n_pc > n_comp: continue
            clf = make_clf(); clf.fit(Ztr[:, :n_pc], ytr)
            auc = roc_auc_score(yte, clf.predict_proba(Zte[:, :n_pc])[:, 1])
            rows_a.append({"tissue": t, "n_pcs": n_pc, "auc": round(auc, 4),
                           "n_genes_nonsig": np.nan})
        print("  A done", flush=True)

        # ---- C: PC <-> metadata (top 100 PCs) ----
        age = m.age_num.fillna(60).to_numpy(float)
        rin = m.SMRIN.fillna(m.SMRIN.median()).to_numpy(float)
        isch = pd.to_numeric(sa.loc[X.index, "SMTSISCH"], errors="coerce")
        isch = isch.fillna(isch.median()).to_numpy(float)
        hardy = m.DTHHRDY.fillna(0).astype(float).to_numpy()
        Zall = pca.transform(Yauto - mu)
        for i in range(min(100, n_comp)):
            z = Zall[:, i]
            a = roc_auc_score(y, z); a = max(a, 1-a)
            rows_c.append({"tissue": t, "pc": i+1,
                           "r_sex": round(pearson(z, y.astype(float)), 3),
                           "r_age": round(pearson(z, age), 3),
                           "r_rin": round(pearson(z, rin), 3),
                           "r_isch": round(pearson(z, isch), 3),
                           "r_hardy": round(pearson(z, hardy), 3),
                           "auc_sex": round(a, 3),
                           "var_explained": round(float(pca.explained_variance_ratio_[i]), 4)})
        print("  C done", flush=True)

        # ---- D: PC residualization dose ----
        for k in RESID_GRID:
            Yw = Yauto - (Zall[:, :k] @ pca.components_[:k]) if k > 0 else Yauto
            clf = make_clf(); clf.fit(Yw[tr_mask], ytr)
            auc = roc_auc_score(yte, clf.predict_proba(Yw[~tr_mask])[:, 1])
            rows_d.append({"tissue": t, "pcs_removed": k, "auc": round(auc, 4)})
        print("  D done", flush=True)

        # ---- B: gene dose, non-sig pool only ----
        Ytr_auto = Y[np.ix_(tr_mask, auto)].astype(np.float64)
        D = np.column_stack([np.ones(int(tr_mask.sum())), ytr.astype(float), age[tr_mask],
                             rin[tr_mask]]).astype(np.float64)
        XtXi = np.linalg.pinv(D.T @ D)
        B = XtXi @ D.T @ Ytr_auto
        resid = Ytr_auto - D @ B
        dof = Ytr_auto.shape[0] - D.shape[1]
        s2 = (resid**2).sum(axis=0)/dof
        se = np.sqrt(np.outer(s2, np.diag(XtXi)))
        pvals = 2*st.t.sf(np.abs(B[1]/se[:, 1]), dof)
        p_arr = np.full(len(genes), np.nan)
        p_arr[auto] = pvals
        nonsig = np.where(auto & (p_arr > 0.5))[0]
        print(f"  nonsig pool: {len(nonsig)}", flush=True)
        for n in [50, 100, 200, 500, 1000, 2000, 4000]:
            aucs = []
            for s_i in range(N_SEEDS_GENE):
                r_ = np.random.RandomState(70_000 + s_i)
                idx = r_.choice(nonsig, size=min(n, len(nonsig)), replace=False)
                msk = np.zeros(len(genes), bool); msk[idx] = True
                clf = make_clf(); clf.fit(Y[np.ix_(tr_mask, msk)], ytr)
                aucs.append(roc_auc_score(yte, clf.predict_proba(Y[np.ix_(~tr_mask, msk)])[:, 1]))
            rows_a.append({"tissue": t, "n_pcs": np.nan,
                           "auc": round(float(np.median(aucs)), 4),
                           "n_genes_nonsig": n})
        print("  B done", flush=True)

    pd.DataFrame(rows_a).to_csv(os.path.join(RES, "d7_pc_dose_fine.csv"), index=False)
    pd.DataFrame(rows_c).to_csv(os.path.join(RES, "d7_pc_metadata.csv"), index=False)
    pd.DataFrame(rows_d).to_csv(os.path.join(RES, "d7_pc_residual.csv"), index=False)

    # ---------- fig8 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    colors = {"Muscle": "#c9599b", "Thyroid": "#4878a8", "Blood": "#67a26b"}
    da = pd.DataFrame(rows_a)
    for t in ["Muscle", "Thyroid", "Blood"]:
        d = da[(da.tissue == t) & da.n_pcs.notna()]
        axes[0].plot(d.n_pcs, d.auc, "o-", ms=3.5, color=colors[t], label=f"{t}")
        dg = da[(da.tissue == t) & da.n_pcs.isna()]
        axes[0].plot(dg.n_genes_nonsig, dg.auc, "s--", ms=3.5, color=colors[t],
                     alpha=0.6, label=f"{t} (non-DE genes)")
    axes[0].set_xscale("log"); axes[0].axhline(0.5, color="gray", ls=":", lw=0.8)
    axes[0].set_xlabel("N features"); axes[0].set_ylabel("AUROC")
    axes[0].set_title("A: PC dose vs non-DE gene dose")
    axes[0].legend(fontsize=6)
    dc = pd.DataFrame(rows_c)
    for t, mk in [("Muscle","o"), ("Thyroid","s"), ("Blood","^")]:
        d = dc[dc.tissue == t]
        axes[1].plot(d.pc, d.auc_sex, mk+"-", ms=3, color=colors[t], label=f"{t} |AUC_sex|")
        axes[1].plot(d.pc, np.abs(d.r_age), mk+"--", ms=3, color=colors[t], alpha=0.5,
                     label=f"{t} |r_age|")
        axes[1].plot(d.pc, np.abs(d.r_isch), mk+":", ms=3, color=colors[t], alpha=0.35,
                     label=f"{t} |r_isch|")
    axes[1].set_xlabel("PC index"); axes[1].set_ylabel("association strength")
    axes[1].set_title("C: what do the leading PCs encode?")
    axes[1].legend(fontsize=5)
    dd = pd.DataFrame(rows_d)
    for t in ["Muscle", "Thyroid", "Blood"]:
        d = dd[dd.tissue == t]
        axes[2].plot(d.pcs_removed, d.auc, "o-", ms=3.5, color=colors[t], label=t)
    axes[2].set_xlabel("leading PCs removed")
    axes[2].set_ylabel("AUROC (remaining genes)")
    axes[2].set_title("D: residualization dose")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "figures", "fig8_genespace_infospace.png"))
    plt.close(fig)
    print("[done] D7 complete", flush=True)

if __name__ == "__main__":
    main()
