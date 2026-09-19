# -*- coding: utf-8 -*-
"""D6: Single-cell within-cell-type sex test (Level 2 -> Level 3 gate).

QUESTION (chatGPT_Other / chatGPT_third item 7):
  Is the diffuse autosomal bulk sex signal just cell-composition, or does it
  persist WITHIN a cell type?

DATA: Tabula Sapiens (CELLxGENE, normal adult donors, 8M/7F across study).
  TS_Muscle.h5ad (46772 cells), TS_Fat.h5ad (94415), TS_Skin.h5ad (17786)
  NOTE: few donors per tissue (~2-6 per sex per tissue) -> we CANNOT do
  donor-level train/test splits. Instead: pseudo-bulk per (donor x cell type)
  -> within each cell type, OLS sex test on donors (n small, so we report
  effect direction consistency across cell types and meta-analysis, not
  single-test significance).

METHOD:
 1. load h5ad, keep cells with sex metadata + non-str Nagive cell types
 2. pseudo-bulk: sum raw counts per (donor, cell_type, sex); TMM-ish CPM+log2
 3. for each cell type with >=2 donors per sex:
      - per-gene OLS: expr ~ sex (donor = unit)
      - record beta_sex, p, n_donors
 4. cross-cell-type meta per gene: fraction of cell types with same sign,
    Stouffer Z; compare against GTEx broad replicators
 5. KEY READOUT: within-cell-type autosomal signal vs GTEx bulk signal
    correlation (effect-size concordance).

Output: results/d6_sc_within_celltype.csv, results/d6_sc_meta.csv,
        results/d6_summary.txt
"""
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

ROOT = r"F:\nature"
SCD = os.path.join(ROOT, "data", "singlecell")
RES = os.path.join(ROOT, "results")
ANN = os.path.join(ROOT, "data", "annot")

FILES = {"Muscle": "TS_Muscle.h5ad", "Fat": "TS_Fat.h5ad", "Skin": "TS_Skin.h5ad"}
# donor per sex: Muscle 3F/2M, Fat 3F/3M, Skin 2F/2M -> OLS underpowered;
# primary readout = cross-celltype direction consistency + GTEx concordance

def gene_symbol(adata):
    """CELLxGENE uses var_names=ensembl, gene_symbols in var."""
    for cand in ["gene_symbols", "feature_name", "gene_name"]:
        if cand in adata.var.columns:
            return adata.var[cand].astype(str).to_numpy()
    return adata.var_names.astype(str).to_numpy()

def pseudo_bulk(adata, symbols):
    """Aggregate counts per (donor, sex, cell_type)."""
    obs = adata.obs.copy()
    # find donor / sex / celltype columns
    don_c = next(c for c in ["donor_id", "donor", "sample_id", "experiment"]
                 if c in obs.columns)
    sex_c = next(c for c in ["sex", "donor_sex", "development_stage"] if c in obs.columns)
    ct_c = next(c for c in ["cell_type", "celltype", "free_annotation"] if c in obs.columns)
    sex_raw = obs[sex_c].astype(str)
    # if sex column holds age stages instead, map via donor (TS: sex in obs? check)
    print(f"    sex column '{sex_c}' values:", sex_raw.value_counts().head(6).to_dict(), flush=True)
    obs["_ct"] = obs[ct_c].astype(str)
    obs["_donor"] = obs[don_c].astype(str)
    X = adata.raw.X if adata.raw is not None else adata.X
    import scipy.sparse as sp
    if not sp.issparse(X): X = sp.csr_matrix(X)   # keep sparse
    groups = obs.groupby(["_donor", "_ct"]).indices
    return obs, groups, X

def main():
    gene2chr = pd.read_csv(os.path.join(ROOT, "data", "proc", "gene2chr.csv"),
                           index_col=0)["chr"]
    sym2chr = {}
    with open(os.path.join(ANN, "gencode.v39.genes.gtf"), encoding="utf-8") as _f:
        for _line in _f:
            if _line.startswith("#"): continue
            _p = _line.split("	")
            if _p[2] != "gene": continue
            _nm = re.search(r'gene_name "([^"]+)"', _p[8]).group(1)
            sym2chr[_nm] = _p[0]

    all_rows, meta_rows = [], []
    for tissue, fn in FILES.items():
        path = os.path.join(SCD, fn)
        print(f"\n=== {tissue}: {fn} ===", flush=True)
        # (anndata bypassed entirely: metadata + matrix both via h5py below)
        import h5py as _h5b
        with _h5b.File(path, "r") as _hb:
            _o = _hb["obs"]
            def _rd_cats(name):
                _cats = [x.decode() if isinstance(x, bytes) else x
                         for x in _o[name]["categories"][:]]
                return pd.Categorical.from_codes(_o[name]["codes"][:], categories=_cats)
            obs = pd.DataFrame({"donor_id": _rd_cats("donor_id"),
                                "sex": _rd_cats("sex"),
                                "cell_type": _rd_cats("cell_type")})
        print(f"    cells={len(obs)} (h5py meta read)", flush=True)
        don_c, sex_c, ct_c = "donor_id", "sex", "cell_type"
        print(f"    donor='{don_c}' sex='{sex_c}' celltype='{ct_c}'", flush=True)
        if don_c is None or ct_c is None:
            print("    MISSING KEY METADATA, skip"); continue
        # sex: if absent per-cell, build donor->sex from dataset description
        if sex_c is None:
            # TS paper donor sex mapping unavailable in obs -> check unique sample prefix
            print("    no sex column! inspecting donor ids:", flush=True)
            print(obs[don_c].value_counts().head(10), flush=True)
            continue
        sexmap = {"female": 1, "male": 0}
        sex_lower = obs[sex_c].astype(str).str.lower()
        ok_sex = sex_lower.isin(sexmap)
        if ok_sex.sum() == 0:
            print("    sex values not male/female:", obs[sex_c].value_counts().head(), flush=True)
            continue
        sex_raw = obs[sex_c].astype(str)
        import h5py as _h5
        import scipy.sparse as sp
        with _h5.File(path, "r") as hf:
            _g = hf["raw/X"] if "raw/X" in hf else hf["X"]
            _n, _m = tuple(int(x) for x in _g.attrs["shape"])
            _nnz = _g["data"].shape[0]
            # chunked read to cap peak memory (~250MB per chunk)
            _CH = 40_000_000
            _data = np.empty(_nnz, dtype=np.float32)
            _ind = np.empty(_nnz, dtype=np.int32)
            for s in range(0, _nnz, _CH):
                e = min(s + _CH, _nnz)
                _data[s:e] = _g["data"][s:e]
                _ind[s:e] = _g["indices"][s:e]
            _ptr = _g["indptr"][:].astype(np.int64)
            X = sp.csr_matrix((_data, _ind, _ptr), shape=(_n, _m))
            _var = hf["var"]
            if "gene_symbols" in _var or "feature_name" in _var:
                _key = "gene_symbols" if "gene_symbols" in _var else "feature_name"
                _vcats = [x.decode() if isinstance(x, bytes) else x
                          for x in _var[_key]["categories"][:]]
                _vcodes = _var[_key]["codes"][:]
                symbols = np.array([_vcats[c] for c in _vcodes])
            else:
                symbols = np.array([x.decode() if isinstance(x, bytes) else x
                                    for x in _var["_index"][:]])


        df = pd.DataFrame({"donor": obs[don_c].astype(str).to_numpy(),
                           "sex": sex_raw.str.lower().map(sexmap).to_numpy(),
                           "ct": obs[ct_c].astype(str).to_numpy()})
        keep = np.asarray(df.sex.notna())
        df = df[keep]
        X = X[np.where(keep)[0]]
        # pseudo-bulk
        pb = {}
        for (d, ct), gidx in df.groupby(["donor","ct"]).groups.items():
            gidx = np.asarray(list(gidx))
            pb[(d, ct)] = np.asarray(X[gidx].sum(axis=0)).ravel()
        print(f"    pseudo-bulk units: {len(pb)}", flush=True)
        pb_df = pd.DataFrame(pb).T
        pb_df.columns = symbols[:pb_df.shape[1]]
        # lib-size normalize + log
        lib = pb_df.sum(axis=1)
        pb_log = np.log2(pb_df.div(lib, axis=0) * 1e6 + 1.0)
        # per cell type: sex OLS on donors
        rows = []
        for ct, sub in pb_log.groupby(level=1):
            sexes = df.drop_duplicates("donor").set_index("donor")["sex"]
            sd = sub.index.get_level_values(0)
            s = sexes.reindex(sd).to_numpy()
            if len(set(s)) < 2 or min(np.bincount(s.astype(int))) < 2:
                continue
            if len(sd) < 4:
                continue          # allow 2v2 (Skin); direction-only readout
            D = np.column_stack([np.ones(len(sd)), s]).astype(np.float64)
            Yg = sub.to_numpy(np.float64)
            XtXi = np.linalg.pinv(D.T @ D)
            B = XtXi @ D.T @ Yg
            resid = Yg - D @ B
            dof = Yg.shape[0] - 2
            s2 = (resid**2).sum(0)/dof
            se = np.sqrt(np.outer(s2, np.diag(XtXi)))
            tt = B[1]/se[:,1]
            p = 2*stats.t.sf(np.abs(tt), dof)
            for j, g in enumerate(sub.columns):
                rows.append({"tissue": tissue, "cell_type": ct, "gene": g,
                             "n_donors": len(sd), "log2FC": B[1][j], "p": p[j]})
        r = pd.DataFrame(rows)
        if r.empty:
            print("    no celltype passed donor filter, skip tissue", flush=True)
            continue
        # chr via gene SYMBOL (CELLxGENE uses symbols; gene2chr is ensembl-keyed)
        r["chr"] = r.gene.map(sym2chr).fillna("NA")
        r_auto = r[~r.chr.isin(["chrX","chrY","chrM"])]
        n_sig = int((r_auto.p < 0.05).sum())
        print(f"    autosomal p<0.05 tests: {n_sig} / {len(r_auto)}", flush=True)
        r_auto.to_csv(os.path.join(RES, f"d6_sc_{tissue.lower()}.csv"), index=False)
        all_rows.append(r_auto)
        meta_rows.append({"tissue": tissue, "n_celltypes_tested": r["cell_type"].nunique(),
                          "n_auto_tests": len(r_auto), "n_p05": n_sig,
                          "frac_p05": round(n_sig/len(r_auto), 4)})

    # ---- cross-cell-type meta (same-sign consistency) ----
    if all_rows:
        big = pd.concat(all_rows)
        # gene x celltype effect matrix per tissue
        for tissue, sub in big.groupby("tissue"):
            piv = sub.pivot_table(index="gene", columns="cell_type",
                                  values="log2FC", aggfunc="first")
            piv = piv.dropna(axis=0, how="any")
            if piv.shape[1] >= 3:
                med = piv.median(axis=1)
                same = (np.sign(piv) == np.sign(med)[:, None]).sum(axis=1) / piv.shape[1]
                out = pd.DataFrame({"median_log2FC": med.round(4),
                                    "frac_same_sign": same.round(3),
                                    "n_celltypes": piv.shape[1]})
                out.to_csv(os.path.join(RES, f"d6_sc_meta_{tissue.lower()}.csv"))
                strong = out[(out.n_celltypes >= 3)].sort_values("frac_same_sign", ascending=False)
                print(f"\n[{tissue}] top cross-celltype consistent genes:")
                print(strong.head(15).to_string(), flush=True)
        pd.DataFrame(meta_rows).to_csv(os.path.join(RES, "d6_sc_summary.csv"), index=False)
    print("\n[done] D6 complete", flush=True)

if __name__ == "__main__":
    main()
