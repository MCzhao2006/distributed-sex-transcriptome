# -*- coding: utf-8 -*-
"""Red-Team A5+A6: multitarget sanity + marker-based composition residualization.

A5: predict sex / age / RIN / ischemic time from the SAME autosomal expression
    (donor-split). If sex AUC >> other targets, signal is sex-specific, not
    generic metadata leakage.
A6: cell-composition proxy correction with marker gene-set scores (no external
    tools): per-sample composition scores from tissue-specific marker panels,
    then residualize expression on TRAIN donors only and re-run sex ML.

Outputs:
  results/rt_multitarget.csv
  results/rt_composition_corrected.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score

ROOT = r"F:\nature"
PROC = os.path.join(ROOT, "data", "proc")
RES = os.path.join(ROOT, "results")
ANN = os.path.join(ROOT, "data", "annot")

TISSUES = ["Blood", "Muscle", "Lung", "Thyroid", "Skin", "Esophagus", "Artery",
           "AdiposeSubq", "Nerve", "Heart", "Colon", "AdiposeVisc"]

MARKERS = {
    "immune":   ["PTPRC","CD3D","CD3E","CD8A","CD4","CD19","MS4A1","CD68","LYZ","NKG7"],
    "endothel": ["PECAM1","VWF","CLDN5","CDH5","KDR"],
    "fibro":    ["COL1A1","COL1A2","DCN","LUM","VIM"],
    "epithel":  ["EPCAM","KRT8","KRT18","KRT19"],
    "adipoc":   ["ADIPOQ","PLIN1","LPL","FABP4"],
    "myocyt":   ["MYH2","ACTN3","CKM","TNNI2","MYH7","TPM2"],
    "eryth":    ["HBB","HBA1","HBA2","ALAS2"],
    "mito":     ["MT-CO1","MT-CO2","MT-ND1","MT-CO3"],
    "ribo":     ["RPLP0","RPS27A","RPL13A","RPSA"],
}

def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"))

def get_split(meta, idx, seed=7, frac=0.6):
    donors = meta.loc[idx, "SUBJID"].to_numpy()
    rng = np.random.RandomState(seed)
    uniq = pd.unique(donors); rng.shuffle(uniq)
    tr = set(uniq[:int(len(uniq) * frac)])
    return pd.Series(donors).isin(tr).to_numpy()

def build_scores(X, sym_of):
    """Marker mean-expression scores per sample."""
    out = {}
    for name, mks in MARKERS.items():
        cols = [i for mk in mks for i in np.where(sym_of == mk)[0]]
        if len(cols) >= 3:
            out[name] = X[:, cols].mean(axis=1)
    return pd.DataFrame(out, index=X.index if hasattr(X, "index") else None)

def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    sa = pd.read_csv(os.path.join(ANN, "SampleAttributesDS.txt"), sep="\t", low_memory=False)
    sa["SUBJID"] = "GTEX-" + sa.SAMPID.str.split("-").str[1]
    sa = sa.set_index("SAMPID")
    import re
    sym_map = {}
    with open(os.path.join(ANN, "gencode.v39.genes.gtf"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.split("\t")
            if p[2] != "gene": continue
            gid = re.search(r'gene_id "([^"]+)"', p[8]).group(1)
            nm = re.search(r'gene_name "([^"]+)"', p[8])
            sym_map[gid] = nm.group(1) if nm else gid

    # ================= A5: multitarget =================
    rows5 = []
    for t in ["Muscle", "Thyroid", "Blood"]:
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        Y = np.log2(X.to_numpy(dtype=np.float32) + 1.0)
        m = meta.loc[X.index]
        tr_mask = get_split(meta, X.index)
        # targets
        sex = m.sex.to_numpy(int)
        age = m.AGE.map({"20-29":25,"30-39":35,"40-49":45,"50-59":55,"60-69":65,"70-79":75}).fillna(60).to_numpy(float)
        rin = m.SMRIN.fillna(m.SMRIN.median()).to_numpy(float)
        isch = pd.to_numeric(sa.loc[X.index, "SMTSISCH"], errors="coerce")
        isch = isch.fillna(isch.median()).to_numpy(float)
        # sex (classification)
        clf = make_clf(); clf.fit(Y[tr_mask], sex[tr_mask])
        p = clf.predict_proba(Y[~tr_mask])[:, 1]
        auc_sex = roc_auc_score(sex[~tr_mask], p)
        # continuous targets with Ridge (regression R2 on test)
        for tname, tv in [("age", age), ("RIN", rin), ("ischemic_time", isch)]:
            reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            reg.fit(Y[tr_mask], tv[tr_mask])
            r2 = r2_score(tv[~tr_mask], reg.predict(Y[~tr_mask]))
            rows5.append({"tissue": t, "target": tname,
                          "metric": "R2_test" if tname != "sex" else "AUC_test",
                          "value": round(r2 if tname != "sex" else auc_sex, 4)})
        rows5.append({"tissue": t, "target": "sex", "metric": "AUC_test",
                      "value": round(auc_sex, 4)})
        print(f"[A5] {t}: sex AUC={auc_sex:.3f}, age R2={rows5[-4]['value']}, "
              f"RIN R2={rows5[-3]['value']}, isch R2={rows5[-2]['value']}", flush=True)
    pd.DataFrame(rows5).to_csv(os.path.join(RES, "rt_multitarget.csv"), index=False)

    # ================= A6: composition residualization =================
    rows6 = []
    for t in ["Muscle", "Thyroid", "Blood"]:
        Xdf = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        Xdf = Xdf.loc[:, (Xdf >= 1.0).mean(axis=0) >= 0.20]
        genes = Xdf.columns.to_numpy()
        sym_of = np.array([sym_map.get(g, g) for g in genes])
        Y = np.log2(Xdf.to_numpy(dtype=np.float32) + 1.0)
        y = meta.loc[Xdf.index, "sex"].to_numpy(int)
        tr_mask = get_split(meta, Xdf.index)

        scores = build_scores(Y, sym_of)                    # n x n_marks
        # residualize each gene on composition scores (fit TRAIN only)
        S_tr = scores.loc[tr_mask].to_numpy(float)
        S_all = scores.to_numpy(float)
        mu_tr = Y[tr_mask].mean(axis=0)
        # add intercept
        S_tr1 = np.column_stack([np.ones(len(S_tr)), S_tr])
        S_all1 = np.column_stack([np.ones(len(S_all)), S_all])
        coef = np.linalg.pinv(S_tr1.T @ S_tr1) @ S_tr1.T @ Y[tr_mask].astype(np.float64)
        Yres = (Y.astype(np.float64) - S_all1 @ coef).astype(np.float32)

        # autosomal mask
        import sys as _s; _s.path.insert(0, os.path.join(ROOT, "scripts"))
        g2c = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]
        auto = ~g2c.reindex(genes).fillna("NA").isin(["chrX","chrY","chrM"]).to_numpy()

        for tag, Ymat in [("raw", Y), ("composition_residualized", Yres)]:
            clf = make_clf(); clf.fit(Ymat[np.ix_(tr_mask, auto)], y[tr_mask])
            p = clf.predict_proba(Ymat[np.ix_(~tr_mask, auto)])[:, 1]
            auc = roc_auc_score(y[~tr_mask], p)
            rows6.append({"tissue": t, "variant": tag, "n_markers": scores.shape[1],
                          "auc": round(auc, 4)})
            print(f"[A6] {t} {tag}: AUC={auc:.3f}", flush=True)
    pd.DataFrame(rows6).to_csv(os.path.join(RES, "rt_composition_corrected.csv"), index=False)
    print("[done] RT-A5/A6 complete", flush=True)

if __name__ == "__main__":
    main()
