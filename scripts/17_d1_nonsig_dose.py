# -*- coding: utf-8 -*-
"""D1 (DeepSeek audit): Fix the contaminated random pool.

DeepSeek's decisive objection: plain random pools contain 48% significant DE
genes, so "random ~ DE" does NOT imply diffuse. Fix with three curves on the
same donors/split (Muscle/Thyroid/Blood, 50 seeds each point):

  C1  non-sig pool: random genes from train-only p>0.5 autosomes ONLY
  C2  DE-fraction 2D: N fixed, DE fraction in {0,25,50,75,100}%
      (0% = C1; 100% = top-N train-only DE)
  C3  saturation fit: AUC(k)=0.5+alpha*(1-exp(-k/tau)) on the C1 curve
      -> transcriptomic sex-information density per tissue (tau)

Output: results/d1_nonsig_dose.csv, results/d1_de_fraction.csv,
        results/d1_saturation_fit.csv, results/figures/fig6_nonsig_dose.png
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
    SIZES = [50, 100, 200, 500, 1000, 2000, 4000]
    SEEDS = 50
    rows1, rows2, rows3 = [], [], []

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
        auto = ~np.isin(chr_of, ["chrX","chrY","chrM"])

        de = train_only_de(Y[tr_mask], genes, chr_of, meta.loc[X.index].loc[tr_mask])
        de = de.set_index("gene").reindex(genes)
        auto_arr = np.asarray(auto)
        p_arr = de["p"].to_numpy()
        order = np.argsort(np.abs(de["log2FC"].to_numpy()))[::-1]   # |log2FC| desc
        # pools (train-only statistics)
        nonsig_pool = np.where(auto_arr & (p_arr > 0.5))[0]
        sig_rank = [i for i in order if auto_arr[i] and p_arr[i] < 0.05]
        print(f"[{t}] nonsig pool={len(nonsig_pool)}, sig(auto)={len(sig_rank)}, "
              f"auto total={auto_arr.sum()}", flush=True)

        # ---- C1: non-sig pool dose curve ----
        for n in SIZES:
            aucs = []
            for s in range(SEEDS):
                r = np.random.RandomState(30_000 + s)
                idx = r.choice(nonsig_pool, size=min(n, len(nonsig_pool)), replace=False)
                m = np.zeros(len(genes), bool); m[idx] = True
                clf = make_clf()
                clf.fit(Y[np.ix_(tr_mask, m)], y[tr_mask])
                aucs.append(roc_auc_score(y[~tr_mask],
                           clf.predict_proba(Y[np.ix_(~tr_mask, m)])[:, 1]))
            aucs = np.array(aucs)
            rows1.append({"tissue": t, "pool": "nonsig_p>0.5", "n_genes": n,
                          "auc_median": round(float(np.median(aucs)), 4),
                          "auc_q025": round(float(np.quantile(aucs, .025)), 4),
                          "auc_q975": round(float(np.quantile(aucs, .975)), 4)})
            print(f"[C1] {t} N={n}: {np.median(aucs):.3f}", flush=True)

        # ---- C2: DE-fraction 2D at fixed N=1000 ----
        N = 1000
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            n_de = int(N * frac)
            aucs = []
            for s in range(SEEDS):
                r = np.random.RandomState(40_000 + s)
                de_part = np.array(r.choice(sig_rank, size=n_de, replace=False)) if n_de else np.array([], int)
                non_part = r.choice(nonsig_pool, size=N - n_de, replace=False)
                idx = np.concatenate([de_part, non_part]).astype(int)
                m = np.zeros(len(genes), bool); m[idx] = True
                clf = make_clf()
                clf.fit(Y[np.ix_(tr_mask, m)], y[tr_mask])
                aucs.append(roc_auc_score(y[~tr_mask],
                           clf.predict_proba(Y[np.ix_(~tr_mask, m)])[:, 1]))
            rows2.append({"tissue": t, "N": N, "de_fraction": frac,
                          "auc_median": round(float(np.median(aucs)), 4),
                          "auc_q025": round(float(np.quantile(aucs, .025)), 4),
                          "auc_q975": round(float(np.quantile(aucs, .975)), 4)})
            print(f"[C2] {t} de_frac={frac}: {np.median(aucs):.3f}", flush=True)

        # ---- C3: saturation fit on C1 medians ----
        d1 = pd.DataFrame(rows1)
        dd = d1[(d1.tissue == t)]
        kk = dd.n_genes.to_numpy(float)
        aa = dd.auc_median.to_numpy(float) - 0.5
        # grid search alpha, tau
        best = None
        for alpha in np.linspace(0.2, 0.6, 41):
            for tau in np.geomspace(50, 5000, 60):
                pred = alpha * (1 - np.exp(-kk / tau))
                sse = ((aa - pred) ** 2).sum()
                if best is None or sse < best[0]:
                    best = (sse, alpha, tau)
        sse, alpha, tau = best
        rows3.append({"tissue": t, "alpha": round(alpha, 3),
                      "tau_genes": round(tau, 1), "sse": round(sse, 5),
                      "auc_at_inf": round(0.5 + alpha, 3)})
        print(f"[C3] {t}: alpha={alpha:.2f} tau={tau:.0f} genes", flush=True)

    pd.DataFrame(rows1).to_csv(os.path.join(RES, "d1_nonsig_dose.csv"), index=False)
    pd.DataFrame(rows2).to_csv(os.path.join(RES, "d1_de_fraction.csv"), index=False)
    pd.DataFrame(rows3).to_csv(os.path.join(RES, "d1_saturation_fit.csv"), index=False)

    # ---- fig6 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    colors = {"Muscle": "#c9599b", "Thyroid": "#4878a8", "Blood": "#67a26b"}
    # panel A: non-sig dose (this study) vs plain dose (old, contaminated)
    old = pd.read_csv(os.path.join(RES, "rt_dose_response.csv"))
    for t in ["Muscle", "Thyroid", "Blood"]:
        d = pd.DataFrame(rows1); d = d[d.tissue == t]
        axes[0].plot(d.n_genes, d.auc_median, "o-", ms=4, color=colors[t],
                     label=f"{t} non-DE only")
        o = old[old.tissue == t]
        axes[0].plot(o.n_genes, o.auc_median, "--", color=colors[t], alpha=0.55,
                     label=f"{t} plain (48% DE)")
    axes[0].set_xscale("log"); axes[0].axhline(0.5, color="gray", ls=":", lw=0.8)
    axes[0].set_xlabel("N genes"); axes[0].set_ylabel("AUROC")
    axes[0].set_title("Dose curve from NON-significant genes only\n"
                      "(train-only p>0.5 autosomes, 50 seeds)")
    axes[0].legend(fontsize=7)
    # panel B: DE-fraction
    d2 = pd.DataFrame(rows2)
    for t in ["Muscle", "Thyroid", "Blood"]:
        d = d2[d2.tissue == t]
        axes[1].plot(d.de_fraction*100, d.auc_median, "o-", ms=4, color=colors[t], label=t)
        axes[1].fill_between(d.de_fraction*100, d.auc_q025, d.auc_q975,
                             color=colors[t], alpha=0.15)
    axes[1].set_xlabel("DE-gene fraction in feature set (%)  [N=1000]")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Sex signal vs DE concentration\n(0% = non-DE random, 100% = top DE)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "figures", "fig6_nonsig_dose.png"))
    plt.close(fig)
    print("[done] D1 complete", flush=True)

if __name__ == "__main__":
    main()
