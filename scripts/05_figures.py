# -*- coding: utf-8 -*-
"""Step 5: Publication-grade figures (matplotlib only, no seaborn).

 Fig 1: DE landscape per tissue (volcano-style summary + bar of counts)
 Fig 2: Cross-tissue replication heatmap (top broad replicators)
 Fig 3: ML performance — in-tissue (all/autosome/sexchr) + cross-tissue transfer heatmap
 Fig 4: Effect-size concordance scatter (tissue A vs B median log2FC)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"F:\nature"
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

TISSUES = ["Blood", "Muscle", "Lung", "Thyroid", "Skin", "Esophagus", "Artery",
           "AdiposeSubq", "Nerve", "Heart", "Colon", "AdiposeVisc"]
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})

# ---------- Fig 1: DE summary bar ----------
de_sum = pd.read_csv(os.path.join(RES, "de_summary.csv"))
de_sum = de_sum.set_index("tissue").loc[TISSUES]
fig, ax = plt.subplots(figsize=(7, 3.2))
x = np.arange(len(TISSUES))
ax.bar(x, de_sum.female_high, label="female-high", color="#c9599b")
ax.bar(x, -de_sum.male_high, label="male-high", color="#4878a8")
ax.set_xticks(x, TISSUES, rotation=45, ha="right")
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("# sex-DE genes (q<0.05)")
ax.set_title("Sex-biased gene expression across 12 GTEx v10 tissues\n(covariate-adjusted: age + RIN)")
ax.legend()
fig.savefig(os.path.join(FIG, "fig1_de_summary.png")); plt.close(fig)
print("fig1 done")

# ---------- Fig 2: replication heatmap ----------
rep = pd.read_csv(os.path.join(RES, "cross_tissue_replication.csv"), index_col=0)
broad = rep[rep.n_sig_q05 >= 8].head(40)
if len(broad) > 0:
    eff = pd.DataFrame(index=broad.index, columns=TISSUES, dtype=float)
    for t in TISSUES:
        d = pd.read_csv(os.path.join(RES, f"de_{t}.csv")).set_index("gene")
        eff[t] = d.log2FC_adj.reindex(broad.index)
    fig, ax = plt.subplots(figsize=(7, 0.32*len(broad)+1.6))
    vmax = np.nanmax(np.abs(eff.values))
    im = ax.imshow(eff.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(TISSUES)), TISSUES, rotation=45, ha="right")
    ax.set_yticks(range(len(broad)), [g for g in broad.index])
    ax.set_title("Cross-tissue sex-biased genes (q<0.05 in >=8/12 tissues)")
    fig.colorbar(im, ax=ax, label="adj. log2FC (F-M)")
    fig.savefig(os.path.join(FIG, "fig2_replication_heatmap.png")); plt.close(fig)
    print(f"fig2 done ({len(broad)} genes)")

# ---------- Fig 3: ML results ----------
ml = pd.read_csv(os.path.join(RES, "ml_results.csv"))
det = pd.read_csv(os.path.join(RES, "ml_results_detail.csv"))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
# panel A: in-tissue ablation
piv = ml.pivot(index="tissue", columns="features", values="auc_test").loc[TISSUES]
order = ["all", "autosome", "sexchr"]
piv = piv[order]
im = axes[0].imshow(piv.values, cmap="viridis", vmin=0.5, vmax=1.0)
axes[0].set_xticks(range(3), ["all", "auto", "X/Y"])
for lbl in axes[0].get_xticklabels():
    lbl.set_rotation(0)
axes[0].set_xlabel("feature set")
axes[0].set_yticks(range(len(TISSUES)), TISSUES)
for i in range(piv.shape[0]):
    for j in range(3):
        axes[0].text(j, i, f"{piv.values[i,j]:.2f}", ha="center", va="center",
                     color="white", fontsize=7)
axes[0].set_title("In-tissue sex prediction (donor-split)\nAUROC, logistic regression")
fig.colorbar(im, ax=axes[0], shrink=0.8)
# panel B: cross-tissue transfer
piv2 = det.pivot(index="test_tissue", columns="train_tissue", values="auc")
piv2 = piv2.reindex(index=TISSUES, columns=TISSUES)
im2 = axes[1].imshow(piv2.values, cmap="viridis", vmin=0.5, vmax=1.0)
axes[1].set_xticks(range(12), TISSUES, rotation=45, ha="right")
axes[1].set_yticks(range(12), TISSUES)
axes[1].set_title("Cross-tissue transfer (train A -> test B)\nautosome-only top-500 DE genes")
fig.colorbar(im2, ax=axes[1], shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_ml_performance.png")); plt.close(fig)
print("fig3 done")

# ---------- Fig 4: effect concordance ----------
d1 = pd.read_csv(os.path.join(RES, "de_Blood.csv")).set_index("gene")
d2 = pd.read_csv(os.path.join(RES, "de_Heart.csv")).set_index("gene")
common = d1.index.intersection(d2.index)
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(d1.loc[common, "log2FC_adj"], d2.loc[common, "log2FC_adj"],
           s=2, alpha=0.25, color="#555")
sig = (d1.loc[common, "q_sex"] < 0.05) & (d2.loc[common, "q_sex"] < 0.05)
ax.scatter(d1.loc[common][sig]["log2FC_adj"], d2.loc[common][sig]["log2FC_adj"],
           s=5, color="#c9599b", label=f"sig in both (n={sig.sum()})")
r = np.corrcoef(d1.loc[common, "log2FC_adj"], d2.loc[common, "log2FC_adj"])[0, 1]
ax.set_xlabel("Blood adj. log2FC (F-M)"); ax.set_ylabel("Heart adj. log2FC (F-M)")
ax.set_title(f"Effect-size concordance r={r:.2f}")
ax.legend(markerscale=3)
fig.savefig(os.path.join(FIG, "fig4_concordance.png")); plt.close(fig)
print("fig4 done")
print("[done] figures complete")
