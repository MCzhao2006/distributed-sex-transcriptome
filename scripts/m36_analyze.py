# -*- coding: utf-8 -*-
"""m36_analyze.py — summarise the M3.6 fixed-cells experiment.

Three outputs the reviewer's plan asks for:

  1. The attenuation curve at MATCHED cell number, with 95% CI across seeds.
     Only cell types present at every cap level are used, otherwise the cap
     levels are not comparable (N100 covers 6 cell types, native covers 13,
     so their aggregate medians differ for reasons that have nothing to do
     with the cap).

  2. Observed vs the matched-N donor-label permutation null.
     S = max(AUC, 1-AUC). For a null AUC distribution F symmetric about 0.5,
         P(S_null <= s) = F(s) - F(1-s) = 2F(s) - 1
     so S_null's 95th percentile is AUC_null's **97.5th** percentile, NOT its
     95th. (An earlier version of this docstring claimed the two 95th
     percentiles coincide. That was WRONG.)
     The block printed below still compares against null_p95 -- the AUC
     percentile -- and is therefore ANTI-CONSERVATIVE. The manuscript table
     uses null_p975 from m36_summary.py, which is the correct threshold. Do not
     read the 'sig' column printed below as the manuscript's significance call.

  3. Cells-per-unit vs sequencing depth, separated.
     UMI per unit ~ cells per unit x depth per cell, so regressing S on
     log(cells) and log(UMI) is badly collinear. The second regressor is
     therefore log(UMI) - log(cells) = depth per cell, which is close to
     orthogonal to log(cells). Cell-type fixed effects absorb baseline
     differences between cell types.

Writes results/m36_summary.csv + results/m36_pairs.csv and prints a report.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NATURE_ROOT") or os.path.dirname(_HERE)
RES = os.path.join(ROOT, "results")
SRC = os.path.join(RES, "m36_fixed_cells.csv")
ORDER = ["N15", "N30", "N50", "N100", "native"]


def ci(s):
    return s.quantile(.025), s.quantile(.975)


def r2(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    return 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum(), b


def main():
    d = pd.read_csv(SRC)
    print(f"loaded {len(d)} rows from {SRC}")
    print(f"cohorts: {dict(d.cohort.value_counts())}")
    print(f"seeds per group: min={d.groupby(['cohort','cap_label']).seed.nunique().min()}, "
          f"max={d.groupby(['cohort','cap_label']).seed.nunique().max()}\n")

    for cohort in d.cohort.unique():
        c = d[d.cohort == cohort]
        print("=" * 78)
        print(f"COHORT {cohort}   (K_pca={c.k_pca.iloc[0]}, n_donors={c.n_donors.max()})")
        print("=" * 78)

        # ---- 1. matched cell types only ----
        piv = c.pivot_table(index="cell_type", columns="cap_label", values="S", aggfunc="median")
        piv = piv.reindex(columns=[o for o in ORDER if o in piv.columns]).dropna()
        if len(piv) == 0:
            print("  no cell type spans every cap level; skipping the matched curve\n")
            shared = sorted(piv.index) if len(piv) else []
        else:
            shared = sorted(piv.index)
            print(f"\nmatched cell types ({len(shared)}): " + ", ".join(x[:40] for x in shared))
            print("\n-- attenuation curve, matched cell types (median S [95% CI]) --")
            for cap in piv.columns:
                s = c[(c.cap_label == cap) & (c.cell_type.isin(shared))].S
                lo, hi = ci(s)
                cells = c[(c.cap_label == cap) & (c.cell_type.isin(shared))].achieved_median_cells.median()
                print(f"  {cap:7s} cells~{cells:5.0f}   S={s.median():.3f} [{lo:.3f}, {hi:.3f}]  n={len(s)}")
            print("\n  per cell type:")
            print(piv.round(3).to_string())

        # ---- 2. observed vs matched-N permutation null ----
        nn = c[c.null_n > 0]
        if len(nn):
            # WARNING: the 'nh' below is null_p95 (the AUC percentile), NOT the
            # S percentile. Comparing obs S against it is anti-conservative; the
            # manuscript uses null_p975 from m36_summary.py. Kept as-is so this
            # block's output stays reproducible against logs/m36_analyze.log.
            print("\n-- observed vs matched-N permutation null --")
            print(f"  {'cap':7s} {'obs_med':>8s} {'obs_p97.5':>10s} {'null_med':>9s} "
                  f"{'null_p95':>9s} {'excess':>7s} {'sig':>4s}")
            for cap in [o for o in ORDER if o in set(nn.cap_label)]:
                s = nn[nn.cap_label == cap]
                om, oh = s.S.median(), s.S.quantile(.975)
                nm, nh = s.null_median.median(), s.null_p95.median()
                print(f"  {cap:7s} {om:8.3f} {oh:10.3f} {nm:9.3f} {nh:9.3f} "
                      f"{om-nm:+7.3f} {'YES' if om > nh else 'no':>4s}")
            print("\n  per cell type (median S; * = above its own null p95)")
            g = nn.groupby(["cell_type", "cap_label"]).agg(
                obs=("S", "median"), p95=("null_p95", "median")).reset_index()
            for ct in sorted(g.cell_type.unique()):
                s = g[g.cell_type == ct].set_index("cap_label")
                parts = []
                for cap in ORDER:
                    if cap in s.index and pd.notna(s.loc[cap, "obs"]):
                        o, p = s.loc[cap, "obs"], s.loc[cap, "p95"]
                        parts.append(f"{cap}:{o:.2f}{'*' if o > p else ' '}")
                    else:
                        parts.append(f"{cap}: -  ")
                print(f"    {ct[:44]:44s} " + " ".join(parts))

        # ---- 3. cells/unit vs depth per cell ----
        print("\n-- cells per unit vs depth per cell (cell-type fixed effects) --")
        m = c.dropna(subset=["S", "achieved_median_cells", "median_log_lib"]).copy()
        m["log_cells"] = np.log(m.achieved_median_cells.clip(lower=1))
        m["log_umi"] = m.median_log_lib
        m["log_depth"] = m.log_umi - m.log_cells
        dums = pd.get_dummies(m.cell_type, drop_first=True).astype(float).to_numpy()
        ones = np.ones((len(m), 1))
        y = m.S.to_numpy()
        r_c, _ = r2(np.column_stack([ones, m.log_cells, dums]), y)
        r_u, _ = r2(np.column_stack([ones, m.log_umi, dums]), y)
        r_d, _ = r2(np.column_stack([ones, m.log_depth, dums]), y)
        r_b, _ = r2(np.column_stack([ones, m.log_cells, m.log_depth, dums]), y)
        _, b = r2(np.column_stack([ones, m.log_cells, m.log_depth, dums]), y)
        print(f"  n={len(m)}  (cell_type dummies included in every model)")
        print(f"    S ~ log(cells/unit)              R2 = {r_c:.3f}")
        print(f"    S ~ log(UMI/unit)                R2 = {r_u:.3f}")
        print(f"    S ~ log(depth per cell)          R2 = {r_d:.3f}")
        print(f"    S ~ log(cells) + log(depth/cell) R2 = {r_b:.3f}")
        print(f"      => partial R2 of cells/unit  = {r_b - r_d:+.3f}")
        print(f"      => partial R2 of depth/cell  = {r_b - r_c:+.3f}")
        print(f"      coefficients: log_cells={b[1]:+.3f}  log_depth={b[2]:+.3f}")
        print()

    print(f"[done] -> run m36_summary.py for the tidy results/m36_summary.csv table")


if __name__ == "__main__":
    main()
