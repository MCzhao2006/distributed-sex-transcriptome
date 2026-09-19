# -*- coding: utf-8 -*-
"""m36_selection_structure.py — the two OneK1K analyses behind Results 8.6.

WHY THIS FILE EXISTS
  Both analyses were previously reported in RESULTS_submission.md 8.6 and
  FIG3_LEGEND.md (d) with NO script and NO result file anywhere in the repo.
  This script makes them traceable: it writes results/m36_selection_structure.csv
  and refuses to overwrite an existing file.

  A) Selection structure. Among cell types meeting the prespecified >=80%
     coverage criterion at N = 15, compare NATIVE sex-prediction strength
     (S_raw, from the m33 confounder audit, which uses all cell types and the
     uncapped pseudobulk) against the remaining cell types, by one-sided
     Mann-Whitney (greater).

  B) Adjacent-level paired change. For each adjacent aggregation step, pair by
     sampling seed and keep only cell types present at BOTH levels; report the
     median DeltaS per cell type and the median of those per-cell-type medians.

  NOTE on (B): the summary statistic is the median of the per-cell-type medians
  (definition A below). A pooled median over all (cell_type, seed) pairs gives a
  different number; both are written out so the choice is explicit.

Output: results/m36_selection_structure.csv   (new file; nothing is overwritten)
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"F:\nature"
RES = os.path.join(ROOT, "results")
CLOUD = os.path.join(ROOT, "cloud_out", "results")
STEPS = [("N15", "N30"), ("N30", "N50"), ("N50", "N100")]


def main():
    rows = []

    # ---------- A: selection structure ----------
    m33 = pd.read_csv(os.path.join(CLOUD, "m33_confound_OneK1K.csv"))
    summ = pd.read_csv(os.path.join(CLOUD, "m36_summary_OneK1K.csv"))
    qualified = set(summ.loc[summ.cap_label == "N15", "cell_type"])

    m33 = m33.assign(covered=m33.cell_type.isin(qualified))
    g1 = m33.loc[m33.covered, "S_raw"]
    g0 = m33.loc[~m33.covered, "S_raw"]
    u, p = stats.mannwhitneyu(g1, g0, alternative="greater")

    rows.append({"analysis": "selection_structure", "group": "coverage_qualified",
                 "n_cell_types": len(g1), "median_native_S_raw": round(float(g1.median()), 4),
                 "source": "m33_confound_OneK1K.csv"})
    rows.append({"analysis": "selection_structure", "group": "remaining",
                 "n_cell_types": len(g0), "median_native_S_raw": round(float(g0.median()), 4),
                 "source": "m33_confound_OneK1K.csv"})
    rows.append({"analysis": "selection_structure", "group": "mannwhitney_greater",
                 "n_cell_types": f"U={u:.0f}", "median_native_S_raw": f"p={p:.3e}",
                 "source": "scipy.stats.mannwhitneyu, alternative=greater"})

    # ---------- B: adjacent-level paired change ----------
    d = pd.read_csv(os.path.join(CLOUD, "m36_fixed_cells_OneK1K.csv"))
    w = d.pivot_table(index=["cell_type", "seed"], columns="cap_label",
                      values="S").reset_index()

    for a, b in STEPS:
        col = f"d_{a}_{b}"
        w[col] = w[b] - w[a]
        x = w.dropna(subset=[col])
        per_ct = x.groupby("cell_type")[col].median()
        for ct, v in per_ct.items():
            rows.append({"analysis": f"paired_dS_{a}_{b}", "group": ct,
                         "n_cell_types": int(x.seed.nunique()),
                         "median_native_S_raw": round(float(v), 4),
                         "source": "m36_fixed_cells_OneK1K.csv, seed-paired"})
        rows.append({"analysis": f"paired_dS_{a}_{b}", "group": "MEDIAN_of_per_cell_type_medians",
                     "n_cell_types": len(per_ct), "median_native_S_raw": round(float(per_ct.median()), 4),
                     "source": "definition A"})
        rows.append({"analysis": f"paired_dS_{a}_{b}", "group": "pooled_median_all_cell_type_seed_pairs",
                     "n_cell_types": len(x), "median_native_S_raw": round(float(x[col].median()), 4),
                     "source": "definition C (written for comparison)"})

    out = os.path.join(RES, "m36_selection_structure.csv")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING TO OVERWRITE existing {out}")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
