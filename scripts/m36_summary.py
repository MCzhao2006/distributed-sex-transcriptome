# -*- coding: utf-8 -*-
"""m36_summary.py — tidy deliverable table for M3.6 (results/m36_summary.csv).

One row per (cohort, cell_type, cap) with the quantities the manuscript needs:
cell number actually achieved, median S and its 95% CI across seeds, and the
matched-N permutation-null thresholds.

Threshold note (this bit us once):
  S = max(AUC, 1-AUC). When the permutation null for AUC is symmetric about 0.5,
  P(S_null > x) = 2 P(AUC_null > x), so S_null's 95th percentile is AUC_null's
  97.5th percentile -- NOT its 95th. Comparing obs_S against null_p95 is
  therefore anti-conservative. Both thresholds are written out; use null_p975.
  null_p975 is estimated from the stored median/p95 assuming approximate
  normality of the null; PBMC's null measured 0.494 median / 0.573 p95, so the
  approximation is safe there and is NOT safe for the kidney cohorts (see below).

Cohort validity flag:
  The KD cohorts (13-19 donors, K=5-7) produced permutation nulls centred at
  0.143-0.272 instead of 0.5. Permutation calibration fails at that sample size,
  so their rows are kept for the record but marked valid=0 and excluded from
  every conclusion.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NATURE_ROOT") or os.path.dirname(_HERE)
RES = os.path.join(ROOT, "results")
ORDER = ["N15", "N30", "N50", "N100", "native"]
# a cohort is usable only if its permutation null is actually centred on 0.5
NULL_CENTRE_TOL = 0.05


def main():
    d = pd.read_csv(os.path.join(RES, "m36_fixed_cells.csv"))
    centre = (d[d.null_n > 0].groupby("cohort").null_median.median() - 0.5).abs()
    valid = (centre < NULL_CENTRE_TOL).to_dict()
    print("permutation-null centring by cohort (|median - 0.5|):")
    for c, v in centre.items():
        print(f"  {c:18s} {v:.3f}  -> {'USABLE' if v < NULL_CENTRE_TOL else 'CALIBRATION FAILED'}")

    nn = d[d.null_n > 0].copy()
    nn["null_p975"] = nn.null_median + (nn.null_p95 - nn.null_median) * (1.959964 / 1.644854)

    rows = []
    for cohort in d.cohort.unique():
        c = d[d.cohort == cohort]
        piv = c.pivot_table(index="cell_type", columns="cap_label", values="S", aggfunc="median")
        piv = piv.reindex(columns=[o for o in ORDER if o in piv.columns]).dropna()
        for ct in sorted(c.cell_type.unique()):
            for cap in ORDER:
                s = c[(c.cell_type == ct) & (c.cap_label == cap)]
                if not len(s):
                    continue
                n = nn[(nn.cell_type == ct) & (nn.cap_label == cap)]
                rows.append({
                    "cohort": cohort, "cell_type": ct, "cap_label": cap,
                    "n_seeds": int(s.seed.nunique()),
                    "cells_achieved": round(float(s.achieved_median_cells.median()), 1),
                    "cap_coverage": round(float(s.cap_coverage.median()), 3),
                    "median_log_umi": round(float(s.median_log_lib.median()), 3),
                    "S_median": round(float(s.S.median()), 4),
                    "S_lo95": round(float(s.S.quantile(.025)), 4),
                    "S_hi95": round(float(s.S.quantile(.975)), 4),
                    "null_p95": round(float(n.null_p95.median()), 4) if len(n) else np.nan,
                    "null_p975": round(float(n.null_p975.median()), 4) if len(n) else np.nan,
                    "excess_over_null": round(float(s.S.median() - n.null_p975.median()), 4) if len(n) else np.nan,
                    "significant": int(len(n) and s.S.median() > n.null_p975.median()),
                    "in_matched_set": int(ct in piv.index),
                    "cohort_null_valid": int(valid.get(cohort, 0)),
                    "k_pca": int(s.k_pca.iloc[0]),
                    "n_donors": int(s.n_donors.max()),
                })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "m36_summary.csv"), index=False)

    u = out[out.cohort_null_valid == 1]
    print(f"\nwrote {len(out)} rows -> results/m36_summary.csv")
    print(f"usable rows (cohort_null_valid=1): {len(u)}")
    print("\n=== attenuation curve, matched cell types, usable cohorts only ===")
    m = u[(u.in_matched_set == 1)]
    if len(m):
        g = m.groupby("cap_label").agg(
            cells=("cells_achieved", "median"), S=("S_median", "median"),
            lo=("S_lo95", "median"), hi=("S_hi95", "median"),
            null975=("null_p975", "median"), sig=("significant", "sum"), n=("S_median", "size"))
        print(g.reindex([o for o in ORDER if o in g.index]).round(3).to_string())
    print("\n=== per-cohort significance tally (usable cohorts) ===")
    for coh in u.cohort.unique():
        s = u[u.cohort == coh]
        print(f"  {coh}: {int(s.significant.sum())}/{len(s)} (cell_type, cap) combos above the "
              f"matched-N null's 97.5th percentile")


if __name__ == "__main__":
    main()
