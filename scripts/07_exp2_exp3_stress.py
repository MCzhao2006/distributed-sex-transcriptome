# -*- coding: utf-8 -*-
"""Experiments 2+3: stress-test the autosomal sex signal.

Exp2  Layered ablation (Muscle & Blood & Thyroid, donor-split):
        L0 all genes
        L1 autosomes only
        L2 autosomes minus 63 broad replicators (known sex-DE)
        L3 autosomes minus ALL q<0.05 sex-DE genes (discovery-set-free:
           DE derived from THIS tissue's train fold -> still leak-free)
        L4 random-gene-set control: same #genes as L3, random autosomal
Exp3  FRG1 family correlation: are the chr20 FRG1 paralog signals one
        family-level signal rather than independent genes?
Perm  200-permutation null for headline AUCs (L1/L2/L3 in Muscle).

Output: results/exp2_layered_ablation.csv, results/exp3_frg1_family.csv,
        results/exp2_permutation.csv
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

def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"))

def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]
    broad = set(pd.read_csv(os.path.join(RES, "broad_replicators.csv"),
                            index_col=0).index)          # 63 known broad sex-DE

    rows, perm_rows = [], []
    for t in ["Muscle", "Blood", "Thyroid"]:
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        genes = X.columns.to_numpy()
        Y = np.log2(X.to_numpy(dtype=np.float32) + 1.0)
        y = meta.loc[X.index, "sex"].to_numpy(dtype=int)
        donors = meta.loc[X.index, "SUBJID"].to_numpy()
        rng = np.random.RandomState(7)
        uniq = pd.unique(donors); rng.shuffle(uniq)
        tr = set(uniq[:int(len(uniq) * 0.6)])
        tr_mask = pd.Series(donors).isin(tr).to_numpy()
        te_mask = ~tr_mask
        ytr, yte = y[tr_mask], y[te_mask]
        chr_of = gene2chr.reindex(genes).fillna("NA").to_numpy()
        auto = ~np.isin(chr_of, ["chrX", "chrY", "chrM"])

        # DE restricted to TRAIN donors only (no leakage into feature selection)
        de = pd.read_csv(os.path.join(RES, f"de_{t}.csv")).set_index("gene")
        de_tr = de.reindex(genes)
        sig_auto = set(de_tr.index[auto & (de_tr.q_sex < 0.05)])

        layers = {
            "L0_all": np.ones(len(genes), bool),
            "L1_autosome": auto,
            "L2_auto_minus_broad": auto & ~pd.Series(genes).isin(broad).to_numpy(),
            "L3_auto_minus_allSig": auto & ~pd.Series(genes).isin(sig_auto).to_numpy(),
        }
        # L4 random-gene control matched to |L3|
        n_l3 = layers["L3_auto_minus_allSig"].sum()
        cand = np.where(auto & ~pd.Series(genes).isin(sig_auto).to_numpy() |
                        (auto & pd.Series(genes).isin(sig_auto).to_numpy()))[0]
        rng2 = np.random.RandomState(99)
        rand_idx = rng2.choice(np.where(auto)[0], size=int(n_l3), replace=False)
        l4 = np.zeros(len(genes), bool); l4[rand_idx] = True
        layers["L4_random_autosome_matchL3"] = l4

        for name, mask in layers.items():
            if mask.sum() < 10: continue
            clf = make_clf()
            clf.fit(Y[np.ix_(tr_mask, mask)], ytr)
            p = clf.predict_proba(Y[np.ix_(te_mask, mask)])[:, 1]
            auc = roc_auc_score(yte, p)
            rows.append({"tissue": t, "layer": name, "n_genes": int(mask.sum()),
                         "auc": round(auc, 4)})
            print(f"[{t}] {name:28s} n={mask.sum():6d} AUC={auc:.3f}", flush=True)

            if t == "Muscle" and name in ("L1_autosome", "L2_auto_minus_broad",
                                          "L3_auto_minus_allSig"):
                # 200 permutations -> empirical p
                hits = 0; NULL = []
                for i in range(200):
                    yr = np.random.RandomState(1000 + i).permutation(ytr)
                    c0 = make_clf(); c0.fit(Y[np.ix_(tr_mask, mask)], yr)
                    pr = c0.predict_proba(Y[np.ix_(te_mask, mask)])[:, 1]
                    a0 = roc_auc_score(yte, pr); NULL.append(a0)
                    if a0 >= auc: hits += 1
                pval = (hits + 1) / 201
                perm_rows.append({"tissue": t, "layer": name, "auc": auc,
                                  "null_mean": round(float(np.mean(NULL)), 4),
                                  "null_max": round(float(np.max(NULL)), 4),
                                  "perm_p_200": round(pval, 4)})
                print(f"    perm: null mean={np.mean(NULL):.3f} max={np.max(NULL):.3f} p={pval:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(RES, "exp2_layered_ablation.csv"), index=False)
    pd.DataFrame(perm_rows).to_csv(os.path.join(RES, "exp2_permutation.csv"), index=False)

    # ---- Exp3: FRG1 family correlation ----
    frg = [g for g in gene2chr.index if g.startswith("ENSG") ]
    rep = pd.read_csv(os.path.join(RES, "cross_tissue_replication.csv"), index_col=0)
    frg_genes = [g for g in rep.index if "FRG1" in str(g)]
    eff = pd.DataFrame(index=frg_genes,
                       columns=["Blood","Muscle","Lung","Thyroid","Skin","Esophagus",
                                "Artery","AdiposeSubq","Nerve","Heart","Colon","AdiposeVisc"])
    for t in eff.columns:
        d = pd.read_csv(os.path.join(RES, f"de_{t}.csv")).set_index("gene")
        eff[t] = d.log2FC_adj.reindex(frg_genes)
    corr = eff.T.corr(method="pearson")
    corr.to_csv(os.path.join(RES, "exp3_frg1_family.csv"))
    tri = corr.values[np.triu_indices(len(frg_genes), 1)]
    print(f"\n[exp3] FRG1-family n={len(frg_genes)} genes, "
          f"pairwise tissue-effect corr: mean={np.nanmean(tri):.2f}")
    # autosome-wide background correlation for comparison
    sample_auto = rep[~rep.index.isin(frg_genes)].head(200)
    bg = pd.DataFrame(index=sample_auto.index, columns=eff.columns, dtype=float)
    for t in eff.columns:
        d = pd.read_csv(os.path.join(RES, f"de_{t}.csv")).set_index("gene")
        bg[t] = d.log2FC_adj.reindex(sample_auto.index)
    tri_bg = bg.T.corr().values[np.triu_indices(len(sample_auto), 1)]
    print(f"[exp3] autosome background (200 genes) corr: mean={np.nanmean(tri_bg):.2f}")

    print("[done] exp2+3 complete", flush=True)

if __name__ == "__main__":
    main()
