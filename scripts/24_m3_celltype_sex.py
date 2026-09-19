# -*- coding: utf-8 -*-
"""M3 v1 (chatGPT 5.1): within-cell-type sex prediction - the decisive M3 gate.

PRE-REGISTERED DESIGN (fixed before looking at results):
  Data unit: donor x cell_type pseudo-bulk (donor = independent unit)
  Cohorts: KD_LivingDonor (19 donors: 10F/9M, 18 cell types, normal kidney)
           + PBMC_Indonesia (normal blood) if download completes
  Features: autosomal genes only (symbol->chr via GENCODE v39)
  Per cell type with >=4 donors per sex:
    - donor-grouped 5-fold CV (folds disjoint in donors)
    - L2 logistic (C=0.1) on CPM+log2 pseudo-bulk
    - AUC = mean over folds; 200 donor-level permutations -> empirical p
  DECISION RULE (fixed in advance):
    a cell type is POSITIVE if  perm_p < 0.05  AND  auc > 0.65
    M3 verdict:
      >=50% tested cell types positive  -> intrinsic autosomal signal SUPPORTED
      <=20% positive                    -> composition-only SUPPORTED
      otherwise                         -> MIXED (report which types)

Output: results/m3_celltype_sex.csv, results/m3_summary.txt
"""
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = r"F:\nature"
SCD = os.path.join(ROOT, "data", "singlecell")
RES = os.path.join(ROOT, "results")

def build_sym2chr():
    sym2chr = {}
    with open(os.path.join(ROOT, "data", "annot", "gencode.v39.genes.gtf"),
              encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.split("\t")
            if p[2] != "gene": continue
            nm = re.search(r'gene_name "([^"]+)"', p[8]).group(1)
            sym2chr[nm] = p[0]
    return sym2chr

def pseudo_bulk(path):
    with h5py.File(path, "r") as hf:
        _o = hf["obs"]
        def cats(n):
            c = [x.decode() if isinstance(x, bytes) else x
                 for x in _o[n]["categories"][:]]
            return pd.Categorical.from_codes(_o[n]["codes"][:], c)
        obs = pd.DataFrame({"donor": cats("donor_id"), "sex": cats("sex"),
                            "ct": cats("cell_type")})
        _g = hf["raw/X"] if "raw/X" in hf else hf["X"]
        nnz = _g["data"].shape[0]
        CH = 40_000_000
        data = np.empty(nnz, dtype=np.float32)
        ind = np.empty(nnz, dtype=np.int32)
        for s in range(0, nnz, CH):
            e = min(s + CH, nnz)
            data[s:e] = _g["data"][s:e]
            ind[s:e] = _g["indices"][s:e]
        ptr = _g["indptr"][:].astype(np.int64)
        n_cells, n_genes = (int(x) for x in _g.attrs["shape"])
        X = sp.csr_matrix((data, ind, ptr), shape=(n_cells, n_genes))
        _v = hf["var"]
        if "gene_symbols" in _v or "feature_name" in _v:
            k = "gene_symbols" if "gene_symbols" in _v else "feature_name"
            vc = [x.decode() if isinstance(x, bytes) else x for x in _v[k]["categories"][:]]
            syms = np.array([vc[c] for c in _v[k]["codes"][:]])
        else:
            syms = np.array([x.decode() if isinstance(x, bytes) else x
                             for x in _v["_index"][:]])
    smap = {"female": 1, "male": 0}
    obs["sex_i"] = obs.sex.astype(str).str.lower().map(smap)
    keep = obs.sex_i.notna().to_numpy()
    obs = obs[keep].reset_index(drop=True)
    X = X[np.where(keep)[0]]
    keys, rows = [], []
    for (d, ct), gidx in obs.groupby(["donor", "ct"], observed=True).groups.items():
        gidx = np.asarray(list(gidx))
        rows.append(np.asarray(X[gidx].sum(axis=0)).ravel())
        keys.append((d, ct))
    sex_lu = obs.drop_duplicates("donor").set_index("donor")["sex_i"]
    M = np.vstack(rows)
    donors = np.array([k[0] for k in keys])
    cts = np.array([k[1] for k in keys])
    sexes = np.array([sex_lu[d] for d in donors], dtype=int)
    return M, donors, cts, sexes, syms

def cpm_log(M):
    lib = M.sum(axis=1, keepdims=True)
    return np.log2(M / np.maximum(lib, 1) * 1e6 + 1.0).astype(np.float32)

def cv_auc(Yg, y, groups):
    gkf = GroupKFold(n_splits=5)
    aucs = []
    for tr_i, te_i in gkf.split(Yg, y, groups):
        if len(set(y[te_i])) < 2: continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(Yg[tr_i], y[tr_i])
        p = clf.predict_proba(Yg[te_i])[:, 1]
        aucs.append(roc_auc_score(y[te_i], p))
    return float(np.mean(aucs)) if aucs else np.nan

def main():
    sym2chr = build_sym2chr()
    auto_of = lambda syms: np.array([sym2chr.get(s, "NA") not in ("chrX", "chrY", "chrM")
                                     for s in syms])
    files = [("kidney_LivingDonor", os.path.join(SCD, "KD_LivingDonor.h5ad"))]
    pbmc = os.path.join(SCD, "PBMC_Indonesia.h5ad")
    if os.path.exists(pbmc) and os.path.getsize(pbmc) > 1e9:
        files.append(("blood_Indonesia", pbmc))

    all_rows = []
    for cohort, path in files:
        print(f"\n=== {cohort} ===", flush=True)
        M, donors, cts, sexes, syms = pseudo_bulk(path)
        auto = auto_of(syms)
        L = cpm_log(M)
        print(f"    units={M.shape[0]}, donors={len(set(donors))}, auto_genes={auto.sum()}", flush=True)
        sex_by_donor = dict(zip(donors, sexes))
        ud_all = pd.unique(donors)
        for ct in pd.unique(cts):
            m = cts == ct
            s = sexes[m]
            if min(np.bincount(s)) < 4:
                continue
            Yg = L[np.ix_(m, auto)]
            y = s
            g = donors[m]
            auc = cv_auc(Yg, y, g)
            if np.isnan(auc):
                continue
            rng = np.random.RandomState(100)
            hits = 0
            for i in range(200):
                perm_map = dict(zip(ud_all, rng.permutation([sex_by_donor[d] for d in ud_all])))
                yp = np.array([perm_map[d] for d in g])
                a0 = cv_auc(Yg, yp, g)
                if a0 >= auc: hits += 1
            p_emp = (hits + 1) / 201
            verdict = "POSITIVE" if (p_emp < 0.05 and auc > 0.65) else "negative"
            all_rows.append({"cohort": cohort, "cell_type": ct,
                             "n_donors": int(m.sum()),
                             "n_female": int(s.sum()), "n_male": int((s == 0).sum()),
                             "auc_cv": round(auc, 4), "perm_p": round(p_emp, 4),
                             "verdict": verdict})
            print(f"    {ct[:38]:38s} n={m.sum()} auc={auc:.3f} p={p_emp:.3f} {verdict}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RES, "m3_celltype_sex.csv"), index=False)
    lines = ["M3 v1 verdict (pre-registered rule: POSITIVE if p<0.05 & auc>0.65)"]
    for cohort, sub in df.groupby("cohort"):
        pos = (sub.verdict == "POSITIVE").mean()
        lines.append(f"  {cohort}: {int((sub.verdict=='POSITIVE').sum())}/{len(sub)} "
                     f"cell types POSITIVE ({pos:.0%})")
        if pos >= 0.5:
            lines.append("    -> intrinsic autosomal within-celltype signal SUPPORTED")
        elif pos <= 0.2:
            lines.append("    -> composition-only SUPPORTED (within-type signal absent)")
        else:
            lines.append("    -> MIXED; check which types drive it")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "m3_summary.txt"), "w", encoding="utf-8").write(txt)

if __name__ == "__main__":
    main()
