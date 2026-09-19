# Distributed and Scale-Dependent Sex-Associated Transcriptional Information Across Human Tissues

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Official code repository and reproducible pipeline for the manuscript:
**"Distributed and scale-dependent sex-associated transcriptional information across human tissues"**

**Author:** Chenhao Zhao (Independent Researcher)  
**Contact:** `acnatgi@proton.me`  
**Preprint & Submission Lineage:** Nature Communications / bioRxiv

---

## 🔬 Overview

This repository contains all 44 deterministic Python scripts required to reproduce the analyses, statistical audits, cross-tissue machine learning pipelines, and publication figures from raw expression matrices.

### Core Discoveries:
1. **Genome-wide Autosomal Distribution & Portability**: Autosomal sex-associated signals are not restricted to discrete lists of DE genes. Removing all FDR-significant genes leaves prediction intact, and genes with marginal $p > 0.5$ accumulate robust predictive AUROC ($> 0.95$).
2. **Latent Compressibility**: The diffuse signal is efficiently captured by ~100 principal components, matching thousands of randomly drawn individual genes.
3. **Scale-Dependent Single-Cell Recoverability**: In single-cell pseudobulk, apparent sex signal strength is monotonically conditioned by cell aggregation scale ($N = 15, 30, 50, 100$).

---

## 📁 Repository Structure

```text
├── scripts/
│   ├── 01_preprocess.py               # Raw counts/TPM filtering & log2 transformation
│   ├── 02_de_analysis.py              # Covariate-adjusted OLS differential expression
│   ├── 03_ml_cross_tissue.py          # Donor-disjoint L2 logistic regression
│   ├── 05_figures.py                  # Publication-grade plotting routines
│   ├── 06_exp1_donor_disjoint.py      # Strict donor-exclusion cross-tissue transfer
│   ├── 09b_exp6_tcga_parallel.py      # External validation in TCGA tumour-adjacent normals
│   ├── 11_rt_perm_dose.py             # 5,000-permutation null & random gene dose-response
│   ├── 17_d1_nonsig_dose.py           # Non-significant (p > 0.5) accumulation curves
│   ├── 22_d4_pc_vs_gene.py            # Principal components vs. individual gene panels
│   ├── 23_d7_genespace_infospace.py   # Information geometry and residual manifolds
│   ├── m36_fixed_cells.py             # Fixed-N single-cell pseudobulk downsampling
│   ├── m36_analyze.py                 # Partial R2 variance decomposition (cells vs. depth)
│   └── split_stability.py             # 50 independent donor-split stability verification
├── figures/
│   ├── Figure_1.png                   # Bulk expression landscape & cross-tissue transfer
│   ├── Figure_2.png                   # Information geometry, dose-response & PC compression
│   ├── Figure_3.png                   # Single-cell pseudobulk aggregation scale dependence
│   └── Figure_4.png                   # Measurement scale covariates & interpretation bounds
├── requirements.txt                   # Python environment dependencies
└── README.md                          # Repository documentation
```

---

## ⚙️ Installation & Requirements

To set up the computational environment, Python 3.9+ is recommended:

```bash
git clone https://github.com/MCzhao2006/distributed-sex-transcriptome.git
cd distributed-sex-transcriptome
pip install -r requirements.txt
```

---

## 📊 Data Availability

The analyses utilize publicly available transcriptomic resources:
- **GTEx v10**: dbGaP accession [phs000424.v10.p2](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000424.v10.p2)
- **TCGA Pan-Cancer**: Accessible via the [GDC Data Portal](https://portal.gdc.cancer.gov/)
- **OneK1K PBMC**: GEO accession [GSE196830](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE196830)
- **PBMC_Indonesia**: GEO accession [GSE143640](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE143640)
- **KPMP Kidney Atlas**: Accessible via [KPMP Data Release](https://www.kpmp.org/)

---

## 📄 License and Citation

This code is licensed under the MIT License - see the LICENSE file for details.
