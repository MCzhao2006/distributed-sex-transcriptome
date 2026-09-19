# -*- coding: utf-8 -*-
"""AUDIT DIAGNOSTIC ONLY -- is sex confounded with site/ancestry/batch in PBMC_Indonesia?

Raised by chat01.md (P0): PBMC_Indonesia is a population cohort (199 donors,
multiple villages/communities), and the original study was about ancestry and
environment. If sex is imbalanced across village / batch / ethnicity, the
within-cell-type sex signal could carry that structure.

Reads donor-level metadata only. Writes nothing.
"""
import h5py
import numpy as np
import pandas as pd
from scipy import stats

P = r"F:\nature\data\singlecell\PBMC_Indonesia.h5ad"
AXES = ["Village", "batch", "pop_eth_env", "self_reported_ethnicity",
        "institute", "assay", "sample_collection_year"]

with h5py.File(P, "r") as hf:
    o = hf["obs"]

    def cats(n):
        c = [x.decode() if isinstance(x, bytes) else x for x in o[n]["categories"][:]]
        return pd.Categorical.from_codes(o[n]["codes"][:], c)

    keep = ["donor_id", "sex"] + [a for a in AXES if a in o]
    df = pd.DataFrame({k: cats(k) for k in keep})

d = df.drop_duplicates("donor_id").copy()
print(f"cells={len(df)}  donors={len(d)}")
print(f"sex: {d.sex.value_counts().to_dict()}\n")

for ax in [a for a in AXES if a in d.columns]:
    tab = pd.crosstab(d[ax], d.sex)
    if tab.shape[0] < 2:
        print(f"--- {ax}: only one level ---")
        continue
    # chi-square on the contingency table (donor level)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    frac = tab.div(tab.sum(axis=1), axis=0)
    print(f"--- {ax}  ({tab.shape[0]} levels)  chi2 p = {p:.3g} ---")
    if tab.shape[0] <= 12:
        print(tab.to_string())
    else:
        print(f"    male fraction per level: min {frac.iloc[:, -1].min():.2f} "
              f"max {frac.iloc[:, -1].max():.2f}")
    print()

# how much of the donor-level sex variance does each axis explain? (Cramer's V)
print("=== Cramer's V of sex vs each axis ===")
for ax in [a for a in AXES if a in d.columns]:
    tab = pd.crosstab(d[ax], d.sex).values
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        continue
    chi2 = stats.chi2_contingency(tab)[0]
    n = tab.sum()
    v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
    print(f"  {ax:28s} V = {v:.3f}")
