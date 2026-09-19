# -*- coding: utf-8 -*-
"""D4 (chatGPT fifth P0): PC-count vs gene-count dose-response.

Question: is the diffuse autosomal signal a low-dimensional latent factor
(a few PCs) or genuinely distributed across transcriptomic dimensions?

Same donors/split as D1 (Muscle/Thyroid/Blood). PCs computed on TRAIN fold,
test projected. Features = first N PCs (scores). Compare against the
gene-count curve.

Output: results/d4_pc_dose.csv, results/figures/fig7_pc_vs_gene.png
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

ROOT = r"F:\nature"
PROC = os.path.join(ROOT, "data", "proc")
RES = os.path.join(ROOT, "results")

def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"))

PC_COUNTS = [1, 5, 10, 20, 50, 100, 200, 500]

def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]
    rows = []
    for t in ["Muscle", "Thyroid", "Blood"]:
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        genes = X.columns.to_numpy()
        Y = np.log2(X.to_numpy(dtype=np.float32) + 1.0)
        y = meta.loc[X.index, "sex"].to_numpy(int)
        donors = meta.loc[X.index, "SUBJID"].to_numpy()
        rng = np.random.RandomState(7)
        uniq = pd.unique(donors); rng.shuffle(uniq)
        tr = set(uniq[:int(len(uniq)*0.6)])
        tr_mask = pd.Series(donors).isin(tr).to_numpy()
        chr_of = gene2chr.reindex(genes).fillna("NA").to_numpy()
        auto = ~np.isin(chr_of, ["chrX", "chrY", "chrM"])
        Yauto = Y[:, auto]
        # PCA on TRAIN autosomes only
        mu = Yauto[tr_mask].mean(axis=0)
        n_comp = min(500, min(Yauto[tr_mask].shape) - 1)
        pca = PCA(n_components=n_comp, random_state=0)
        Ztr = pca.fit_transform(Yauto[tr_mask] - mu)
        Zte = pca.transform(Yauto[~tr_mask] - mu)
        for n_pc in PC_COUNTS:
            clf = make_clf()
            clf.fit(Ztr[:, :n_pc], y[tr_mask])
            auc = roc_auc_score(y[~tr_mask], clf.predict_proba(Zte[:, :n_pc])[:, 1])
            rows.append({"tissue": t, "n_pcs": n_pc, "auc": round(auc, 4)})
            print(f"[{t}] {n_pc} PCs -> AUC={auc:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "d4_pc_dose.csv"), index=False)

    # fig7
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = {"Muscle": "#c9599b", "Thyroid": "#4878a8", "Blood": "#67a26b"}
    for t in ["Muscle", "Thyroid", "Blood"]:
        d = df[df.tissue == t]
        ax.plot(d.n_pcs, d.auc, "o-", ms=4, color=colors[t], label=f"{t} (PCs)")
    # overlay gene curve (old contaminated pool as reference, dashed)
    try:
        g = pd.read_csv(os.path.join(RES, "rt_dose_response.csv"))
        for t in ["Muscle", "Thyroid", "Blood"]:
            d = g[g.tissue == t]
            ax.plot(d.n_genes, d.auc_median, "--", color=colors[t], alpha=0.5,
                    label=f"{t} (genes, plain pool)")
    except Exception:
        pass
    ax.set_xscale("log")
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("N features (PCs solid, genes dashed)")
    ax.set_ylabel("Sex-prediction AUROC (donor-split)")
    ax.set_title("PC-dimensionality vs gene-count dose-response\n(autosomal, train-fold PCA)")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(RES, "figures", "fig7_pc_vs_gene.png"))
    plt.close(fig)
    print("[done] D4 complete", flush=True)

if __name__ == "__main__":
    main()
