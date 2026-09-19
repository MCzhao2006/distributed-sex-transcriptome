# -*- coding: utf-8 -*-
"""Step 4: Cross-tissue consistency of sex-biased expression.

Questions answered:
 1. Which sex-DE genes replicate across MANY tissues? (core: cross-tissue program)
 2. Meta-analysis: combine effect sizes across tissues (Stouffer's Z)
 3. Direction consistency: fraction of tissues where log2FC has the same sign
 4. Chromosome enrichment: are sex-DE genes enriched on X/Y vs autosomes?
Outputs:
  results/cross_tissue_replication.csv   (per-gene replication stats)
  results/consistency_summary.csv
  results/chr_enrichment.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2_contingency

ROOT = r"F:\nature"
RES = os.path.join(ROOT, "results")
PROC = os.path.join(ROOT, "data", "proc")

TISSUES = ["Blood", "Muscle", "Lung", "Thyroid", "Skin", "Esophagus", "Artery",
           "AdiposeSubq", "Nerve", "Heart", "Colon", "AdiposeVisc"]

de = {}
for t in TISSUES:
    d = pd.read_csv(os.path.join(RES, f"de_{t}.csv")).set_index("gene")
    de[t] = d

# union of genes tested in >= 6 tissues
from functools import reduce
counts = reduce(lambda a, b: a.add(b, fill_value=1), [d.notna()["q_sex"] for d in de.values()])
union = counts[counts >= 6].index

eff = pd.DataFrame({t: de[t].loc[union.intersection(de[t].index), "log2FC_adj"] for t in TISSUES})
p = pd.DataFrame({t: de[t].loc[union.intersection(de[t].index), "p_sex"] for t in TISSUES})

# Stouffer's Z (one-sided, signed by effect direction)
z = pd.DataFrame(index=union, columns=TISSUES, dtype=float)
for t in TISSUES:
    sig = (p[t] < 1).to_numpy()
    zt = norm.isf(np.clip(p[t].to_numpy() / 2, 1e-300, 1)) * np.sign(eff[t].to_numpy())
    z[t] = zt
Z_meta = z.sum(axis=1) / np.sqrt(z.notna().sum(axis=1).clip(lower=1) * 1.0)   # approximate
P_meta = 2 * norm.sf(np.abs(Z_meta))

med = eff.median(axis=1).to_numpy()
rep = pd.DataFrame({
    "n_tissues_tested": eff.notna().sum(axis=1),
    "n_sig_q05": pd.DataFrame({t: de[t]["q_sex"] < 0.05 for t in TISSUES}).loc[eff.index].fillna(False).sum(axis=1),
    "frac_same_sign": (np.sign(eff.to_numpy()) == np.sign(med)[:, None]).sum(axis=1) / eff.notna().sum(axis=1).to_numpy(),
    "median_log2FC": eff.median(axis=1),
    "Z_meta": Z_meta, "p_meta": P_meta,
}).sort_values("n_sig_q05", ascending=False)
rep.to_csv(os.path.join(RES, "cross_tissue_replication.csv"))

# broad replicators: significant (q<0.05) in >= 8 of 12 tissues
broad = rep[(rep.n_sig_q05 >= 8)]
broad.to_csv(os.path.join(RES, "broad_replicators.csv"))
print(f"broad replicators (q<0.05 in >=8/12 tissues): {len(broad)}")
print(broad.head(30).to_string())

# chromosome enrichment among broad replicators
gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]
broad_set = set(broad.index)
all_tested = set(union)
tab = pd.crosstab(
    pd.Series(["sexchr" if gene2chr.get(g, "NA") in ("chrX", "chrY") else "autosome" for g in all_tested]),
    pd.Series(["broad" if g in broad_set else "other" for g in all_tested]))
tab.to_csv(os.path.join(RES, "chr_enrichment.csv"))
try:
    chi2, pval, dof, exp = chi2_contingency(tab)
    print(f"\nchr enrichment chi2={chi2:.1f} p={pval:.2e}")
    print(tab.to_string())
except Exception as e:
    print("chi2 failed:", e)

# summary
cons = pd.DataFrame({
    "tissue": TISSUES,
    "n_sig": [int((de[t]["q_sex"] < 0.05).sum()) for t in TISSUES],
    "n_sig_in_broad": [int(de[t].loc[de[t].index.isin(broad_set), "q_sex"].lt(0.05).sum()) for t in TISSUES],
})
cons.to_csv(os.path.join(RES, "consistency_summary.csv"), index=False)
print("\n[done] cross-tissue analysis complete")
