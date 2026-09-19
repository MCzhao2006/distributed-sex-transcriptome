# -*- coding: utf-8 -*-
"""Step 1: Build the analysis-ready expression matrix.
- Parse Gencode v39 GTF -> gene_id -> chromosome map
- Load 12 per-tissue GTEx v10 TPM GCTs
- Intersect samples with sex annotations; align genes across tissues
- Save one parquet per tissue (float32)
Memory strategy: process one tissue at a time; free before next.
"""
import gzip, os, re
import pandas as pd
import numpy as np

ROOT = r"F:\nature"
RAW = os.path.join(ROOT, "data", "raw")
ANN = os.path.join(ROOT, "data", "annot")
OUT = os.path.join(ROOT, "data", "proc")
os.makedirs(OUT, exist_ok=True)

TISSUES = {
    "whole_blood": "Blood",
    "muscle_skeletal": "Muscle",
    "lung": "Lung",
    "thyroid": "Thyroid",
    "skin_sun_exposed_lower_leg": "Skin",
    "esophagus_mucosa": "Esophagus",
    "artery_tibial": "Artery",
    "adipose_subcutaneous": "AdiposeSubq",
    "nerve_tibial": "Nerve",
    "heart_left_ventricle": "Heart",
    "colon_transverse": "Colon",
    "adipose_visceral_omentum": "AdiposeVisc",
}

# ---------- 1. gene -> chromosome map ----------
print("[1] parsing GTF ...", flush=True)
gene2chr = {}
with open(os.path.join(ANN, "gencode.v39.genes.gtf"), encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if p[2] != "gene":
            continue
        gid = re.search(r'gene_id "([^"]+)"', p[8]).group(1)
        gene2chr[gid] = p[0]          # chr1 ... chrX chrY chrM
print(f"    genes in GTF: {len(gene2chr)}", flush=True)
pd.Series(gene2chr).rename("chr").to_csv(os.path.join(OUT, "gene2chr.csv"))

# ---------- 2. sample metadata ----------
print("[2] sample metadata ...", flush=True)
ph = pd.read_csv(os.path.join(ANN, "SubjectPhenotypesDS.txt"), sep="\t")
sa = pd.read_csv(os.path.join(ANN, "SampleAttributesDS.txt"), sep="\t", low_memory=False)
gtex = sa[sa.SAMPID.astype(str).str.match(r"^GTEX-")].copy()
gtex["SUBJID"] = "GTEX-" + gtex.SAMPID.str.split("-").str[1]
meta = gtex.merge(ph[["SUBJID", "SEX", "AGE", "DTHHRDY"]], on="SUBJID", how="inner")
meta = meta[meta.SEX.isin([1, 2])]
meta["sex"] = (meta.SEX == 2).astype(int)          # 0=male, 1=female
meta = meta[["SAMPID", "SUBJID", "SMTSD", "sex", "AGE", "SMRIN", "DTHHRDY"]]
meta = meta.set_index("SAMPID")
meta.to_csv(os.path.join(OUT, "sample_meta.csv"))
print(f"    annotated samples: {len(meta)} ({(meta.sex==1).sum()} F / {(meta.sex==0).sum()} M)", flush=True)

# ---------- 3. per-tissue matrix build ----------
summary = []
sexcol = meta["sex"]
for tkey, tname in TISSUES.items():
    fn = os.path.join(RAW, f"gene_tpm_v10_{tkey}.gct.gz")
    print(f"[3] {tname}: reading {os.path.basename(fn)}", flush=True)
    with gzip.open(fn, "rt") as f:
        f.readline()                                   # version line
        dims = f.readline().split()                    # n_genes n_samples
        header = f.readline().rstrip("\n").split("\t")
        sample_ids = header[2:]
        ncol = len(header)
        # single pass chunked read: col0 = gene id, cols2.. = TPM
        reader = pd.read_csv(f, sep="\t", header=None, skiprows=0,
                             usecols=list(range(ncol)), chunksize=3000,
                             dtype={0: str}, engine="c")
        idx, mats = [], []
        for chunk in reader:
            idx.append(chunk.iloc[:, 0].to_numpy())
            mats.append(chunk.iloc[:, 2:].to_numpy(dtype=np.float32))
        X = np.vstack(mats)
        genes = np.concatenate(idx)
        del mats
    X = X.T                                            # samples x genes
    keep = np.array([s in meta.index for s in sample_ids])
    X = X[keep]
    sid_kept = [s for s, k in zip(sample_ids, keep) if k]
    df = pd.DataFrame(X, index=sid_kept, columns=genes)
    df.index.name = "SAMPID"
    df.to_parquet(os.path.join(OUT, f"tpm_{tname}.parquet"))
    n_m = int((sexcol.loc[sid_kept] == 0).sum()); n_f = int((sexcol.loc[sid_kept] == 1).sum())
    summary.append({"tissue": tname, "samples": df.shape[0], "genes": df.shape[1],
                    "males": n_m, "females": n_f})
    print(f"    -> {df.shape[0]} samples x {df.shape[1]} genes (M={n_m} F={n_f}) saved", flush=True)
    del df, X

pd.DataFrame(summary).to_csv(os.path.join(OUT, "matrix_summary.csv"), index=False)
print("[done] all tissues processed", flush=True)
