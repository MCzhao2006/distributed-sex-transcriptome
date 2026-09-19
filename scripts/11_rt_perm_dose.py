# -*- coding: utf-8 -*-
"""Red-Team A3+A4: 5000 permutations + random-gene dose-response curve.

A3: 5000 label permutations, Muscle autosome model (5 parallel processes).
    Resolution 1/5001 = 0.0002.
A4: N-genes -> AUC dose-response, 7 sizes x 100 seeds x 3 tissues.
    THE key figure for "diffuse autosomal signal".

Outputs:
  results/rt_perm5000.csv
  results/rt_dose_response.csv
  results/figures/fig5_dose_response.png
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

ROOT = r"F:\nature"
PROC = os.path.join(ROOT, "data", "proc")
RES = os.path.join(ROOT, "results")

def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"))

def get_split(meta, idx, seed=7, frac=0.6):
    donors = meta.loc[idx, "SUBJID"].to_numpy()
    rng = np.random.RandomState(seed)
    uniq = pd.unique(donors); rng.shuffle(uniq)
    tr = set(uniq[:int(len(uniq) * frac)])
    return pd.Series(donors).isin(tr).to_numpy()

def a3_perms():
    """5000 permutations, parallel over 5 processes via subprocess."""
    import subprocess
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]
    t = "Muscle"
    X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
    X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
    Y = np.log2(X.to_numpy(dtype=np.float32) + 1.0)
    y = meta.loc[X.index, "sex"].to_numpy(dtype=int)
    tr_mask = get_split(meta, X.index)
    chr_of = gene2chr.reindex(X.columns).fillna("NA").to_numpy()
    auto = ~np.isin(chr_of, ["chrX", "chrY", "chrM"])
    np.save(os.path.join(ROOT, "data", "rt_a3_cache.npy"),
            {"Y": Y, "y": y, "tr": tr_mask, "auto": auto}, allow_pickle=True)

    # observed
    clf = make_clf(); clf.fit(Y[np.ix_(tr_mask, auto)], y[tr_mask])
    auc_obs = roc_auc_score(y[~tr_mask], clf.predict_proba(Y[np.ix_(~tr_mask, auto)])[:, 1])

    # parallel children
    procs = []
    for w in range(5):
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", str(w)],
                             cwd=ROOT)
        procs.append(p)
    for p in procs: p.wait()

    nulls = np.concatenate([np.load(os.path.join(ROOT, "data", f"rt_a3_null_{w}.npy"))
                            for w in range(5)])
    hits = int((nulls >= auc_obs).sum())
    pval = (hits + 1) / (len(nulls) + 1)
    out = pd.DataFrame([{"tissue": t, "features": "autosome", "n_perm": len(nulls),
                         "auc_obs": round(auc_obs, 4),
                         "null_mean": round(float(nulls.mean()), 4),
                         "null_p99": round(float(np.quantile(nulls, 0.99)), 4),
                         "null_max": round(float(nulls.max()), 4),
                         "n_ge_obs": hits, "perm_p": f"{pval:.6f}"}])
    out.to_csv(os.path.join(RES, "rt_perm5000.csv"), index=False)
    print(f"[A3] obs={auc_obs:.4f} null mean={nulls.mean():.3f} max={nulls.max():.3f} "
          f"p={pval:.6f} (n={len(nulls)})", flush=True)

def a3_worker(w):
    cache = np.load(os.path.join(ROOT, "data", "rt_a3_cache.npy"), allow_pickle=True).item()
    Y, y, tr_mask, auto = cache["Y"], cache["y"], cache["tr"], cache["auto"]
    rng_master = np.random.RandomState(9000 + w)
    aucs = []
    for i in range(1000):
        yr = rng_master.permutation(y)
        clf = make_clf()
        clf.fit(Y[np.ix_(tr_mask, auto)], yr[tr_mask])
        aucs.append(roc_auc_score(yr[~tr_mask], clf.predict_proba(Y[np.ix_(~tr_mask, auto)])[:, 1]))
    np.save(os.path.join(ROOT, "data", f"rt_a3_null_{w}.npy"), np.array(aucs))
    print(f"[worker {w}] done", flush=True)

def a4_dose():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]
    sizes = [50, 100, 200, 500, 1000, 3000, 5000]
    n_seeds = 100
    rows = []
    for t in ["Muscle", "Thyroid", "Blood"]:
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        genes = X.columns.to_numpy()
        Y = np.log2(X.to_numpy(dtype=np.float32) + 1.0)
        y = meta.loc[X.index, "sex"].to_numpy(dtype=int)
        tr_mask = get_split(meta, X.index)
        chr_of = gene2chr.reindex(genes).fillna("NA").to_numpy()
        auto_idx = np.where(~np.isin(chr_of, ["chrX", "chrY", "chrM"]))[0]
        for size in sizes:
            aucs = []
            for s in range(n_seeds):
                r = np.random.RandomState(10_000 + s)
                m = np.zeros(len(genes), bool)
                m[r.choice(auto_idx, size=size, replace=False)] = True
                clf = make_clf()
                clf.fit(Y[np.ix_(tr_mask, m)], y[tr_mask])
                p = clf.predict_proba(Y[np.ix_(~tr_mask, m)])[:, 1]
                aucs.append(roc_auc_score(y[~tr_mask], p))
            aucs = np.array(aucs)
            rows.append({"tissue": t, "n_genes": size, "n_seeds": n_seeds,
                         "auc_median": round(float(np.median(aucs)), 4),
                         "auc_q025": round(float(np.quantile(aucs, 0.025)), 4),
                         "auc_q975": round(float(np.quantile(aucs, 0.975)), 4),
                         "auc_min": round(float(aucs.min()), 4)})
            print(f"[A4] {t} N={size}: median={np.median(aucs):.3f} "
                  f"95%CI=({np.quantile(aucs,0.025):.3f},{np.quantile(aucs,0.975):.3f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "rt_dose_response.csv"), index=False)

    # fig5
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    colors = {"Muscle": "#c9599b", "Thyroid": "#4878a8", "Blood": "#67a26b"}
    for t in ["Muscle", "Thyroid", "Blood"]:
        d = df[df.tissue == t]
        ax.plot(d.n_genes, d.auc_median, "o-", color=colors[t], label=t, ms=4)
        ax.fill_between(d.n_genes, d.auc_q025, d.auc_q975, color=colors[t], alpha=0.18)
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("N random autosomal genes")
    ax.set_ylabel("Sex-prediction AUROC (donor-split)")
    ax.set_title("Sex information is distributed across the autosomal transcriptome\n"
                 "median ± 95% CI over 100 random gene sets per point")
    ax.legend()
    fig.savefig(os.path.join(RES, "figures", "fig5_dose_response.png"))
    plt.close(fig)
    print("[A4] fig5 saved", flush=True)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        a3_worker(int(sys.argv[2])); return
    a3_perms()
    a4_dose()
    print("[done] RT-A3/A4 complete", flush=True)

if __name__ == "__main__":
    main()
