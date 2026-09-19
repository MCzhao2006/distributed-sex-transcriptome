# -*- coding: utf-8 -*-
"""D6b (chatGPT fifth P0): three nulls for the cross-cell-type consistency.

Null 1  donor-level sex-label permutation: swap M/F donor labels, recompute
        all cell-type effects + consistency. 200 perms -> null distribution.
Null 3  gene-wise donor permutation: for each gene, permute values across
        donors WITHIN each cell type (destroys gene-gene covariance and any
        real sex effect, keeps per-cell-type marginal distributions).
        50 perms.
(also saves the real pseudo-bulk matrices for reuse)

Output: results/d6b_nulls.csv, results/d6b_pb_cache_{tissue}.npz
"""
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
from scipy import stats

ROOT = r"F:\nature"
SCD = os.path.join(ROOT, "data", "singlecell")
RES = os.path.join(ROOT, "results")
ANN = os.path.join(ROOT, "data", "annot")

FILES = {"Muscle": "TS_Muscle.h5ad", "Skin": "TS_Skin.h5ad", "Fat": "TS_Fat.h5ad"}

def pseudo_bulk_matrix(path):
    """Return pb_df (donor,celltype) x gene with symbol columns + donor/sex/ct."""
    with h5py.File(path, "r") as hf:
        _o = hf["obs"]
        cats = lambda n: pd.Categorical.from_codes(
            _o[n]["codes"][:],
            [x.decode() if isinstance(x, bytes) else x for x in _o[n]["categories"][:]])
        obs = pd.DataFrame({"donor": cats("donor_id"), "sex": cats("sex"),
                            "ct": cats("cell_type")})
        _g = hf["raw/X"] if "raw/X" in hf else hf["X"]
        _nnz = _g["data"].shape[0]
        _CH = 40_000_000
        data = np.empty(_nnz, dtype=np.float32)
        ind = np.empty(_nnz, dtype=np.int32)
        for s in range(0, _nnz, _CH):
            e = min(s + _CH, _nnz)
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
    sexmap = {"female": 1, "male": 0}
    obs["sex_i"] = obs.sex.astype(str).str.lower().map(sexmap)
    keep = obs.sex_i.notna().to_numpy()
    obs = obs[keep].reset_index(drop=True)
    X = X[np.where(keep)[0]]
    # aggregate
    keys, rows = [], []
    for (d, ct), gidx in obs.groupby(["donor", "ct"]).groups.items():
        gidx = np.asarray(list(gidx))
        rows.append(np.asarray(X[gidx].sum(axis=0)).ravel())
        keys.append((d, ct))
    sex_lookup = obs.drop_duplicates("donor").set_index("donor")["sex_i"]
    donors = [k[0] for k in keys]; cts = [k[1] for k in keys]
    M = np.vstack(rows)                       # units x genes
    sexes = np.array([sex_lookup[d] for d in donors], dtype=int)
    return M, np.array(donors), np.array(cts), sexes, syms

def cpm_log(M):
    lib = M.sum(axis=1, keepdims=True)
    return np.log2(M / np.maximum(lib, 1) * 1e6 + 1.0)

def consistency_matrix(pb_log, units_ct, units_sex, min_don=4):
    """Return per-gene consistency over qualifying cell types + n_ct."""
    out_same, n_ct_used = None, 0
    signs = []
    for ct in pd.unique(units_ct):
        m = units_ct == ct
        s = units_sex[m]
        if len(set(s)) < 2 or min(np.bincount(s)) < 2 or m.sum() < min_don:
            continue
        Yg = pb_log[m]
        D = np.column_stack([np.ones(m.sum()), s]).astype(np.float64)
        B = np.linalg.pinv(D.T @ D) @ D.T @ Yg.astype(np.float64)
        signs.append(np.sign(B[1]))
        n_ct_used += 1
    if n_ct_used < 3:
        return None, 0
    S = np.vstack(signs)                       # n_ct x genes
    med_sign = np.sign(np.median(S, axis=0))
    same = (S == med_sign[None, :]).sum(axis=0) / n_ct_used
    return same, n_ct_used

def main():
    rng0 = np.random.RandomState(7)
    results = []
    for tissue, fn in FILES.items():
        print(f"\n=== {tissue} ===", flush=True)
        M, donors, cts, sexes, syms = pseudo_bulk_matrix(os.path.join(SCD, fn))
        pb_log = cpm_log(M)
        np.savez_compressed(os.path.join(RES, f"d6b_pb_cache_{tissue.lower()}.npz"),
                            M=M, donors=donors, cts=cts, sexes=sexes, syms=syms)
        print(f"    units={M.shape[0]}, donors={len(set(donors))}", flush=True)

        # observed (autosomal only via symbol map)
        sym2chr = {}
        with open(os.path.join(ANN, "gencode.v39.genes.gtf"), encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"): continue
                p = line.split("\t")
                if p[2] != "gene": continue
                nm = re.search(r'gene_name "([^"]+)"', p[8]).group(1)
                sym2chr[nm] = p[0]
        auto_mask = np.array([sym2chr.get(s, "NA") not in ("chrX", "chrY", "chrM")
                              for s in syms])
        same_obs, n_ct = consistency_matrix(pb_log[:, auto_mask], cts, sexes)
        print(f"    observed consistency={same_obs.mean():.4f} (n_ct={n_ct})", flush=True)

        # ---- Null 1: donor sex-label permutation (200) ----
        uniq_d = pd.unique(donors)
        d_sex = dict(zip(donors, sexes))
        null1 = []
        for i in range(200):
            perm = dict(zip(uniq_d, rng0.permutation(
                [d_sex[d] for d in uniq_d])))
            s_perm = np.array([perm[d] for d in donors])
            same_n, _ = consistency_matrix(pb_log[:, auto_mask], cts, s_perm)
            null1.append(same_n.mean())
        null1 = np.array(null1)
        p1 = (np.sum(null1 >= same_obs.mean()) + 1) / 201
        print(f"    Null1 (sex perm 200): mean={null1.mean():.4f} "
              f"max={null1.max():.4f} p={p1:.4f}", flush=True)

        # ---- Null 3: gene-wise donor permutation within cell type (50) ----
        null3 = []
        for i in range(50):
            rng = np.random.RandomState(3000 + i)
            pb_perm = pb_log.copy()
            for ct in pd.unique(cts):
                m = cts == ct
                # permute each gene across the units of this cell type
                P = rng.permutation(m.sum())
                pb_perm[m] = pb_log[m][P]
            same_n, _ = consistency_matrix(pb_perm[:, auto_mask], cts, sexes)
            null3.append(same_n.mean())
        null3 = np.array(null3)
        p3 = (np.sum(null3 >= same_obs.mean()) + 1) / 51
        print(f"    Null3 (gene-wise perm 50): mean={null3.mean():.4f} "
              f"max={null3.max():.4f} p={p3:.4f}", flush=True)

        results.append({"tissue": tissue, "n_ct": n_ct, "n_auto_genes": int(auto_mask.sum()),
                        "obs_consistency": round(float(same_obs.mean()), 4),
                        "null1_sex_perm_mean": round(float(null1.mean()), 4),
                        "null1_max": round(float(null1.max()), 4),
                        "null1_p": round(float(p1), 4),
                        "null3_genewise_mean": round(float(null3.mean()), 4),
                        "null3_max": round(float(null3.max()), 4),
                        "null3_p": round(float(p3), 4)})

    pd.DataFrame(results).to_csv(os.path.join(RES, "d6b_nulls.csv"), index=False)
    print("\n[done] D6b complete", flush=True)

if __name__ == "__main__":
    main()
