# -*- coding: utf-8 -*-
"""M3.1 (chatGPT eighth): corrected M3 statistical engine + dual-cohort replication.

Fixes vs M3 v1 (per review):
 1. Permutation: 200 -> configurable 5000 (N_PERM env; default here 1000 for
    tractability on battery power, final run should set N_PERM=5000).
 2. CV: donor-grouped 5-fold replaced by leave-one-donor-out (LODO) when
    donors/sex < 12; AUC pooled over held-out donors (pooling predictions
    then computing one AUC = more stable than averaging 5 tiny fold-AUCs).
 3. Cohorts flagged 'exploratory' when <12 donors/sex (review: don't fake
    precision); replication verdict comes from effect ALIGNMENT across
    KD_LivingDonor & KD_Mature, not single-cohort p.
 4. VERDICT RULE (pre-registered):
    - cohort n<12/sex  -> 'exploratory' (no composition-only claim allowed)
    - replication = same-sign cell-type effects across cohorts with both
      |AUC-0.5| >= 0.10
    - composition-only claim ALLOWED only if both cohorts' high-power test
      (>=12/sex, e.g. PBMC n=199) is also negative.

Output: results/m32_celltypes_two_sided.csv, results/m32_summary.txt
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

ROOT = r"F:\nature"
SCD = os.path.join(ROOT, "data", "singlecell")
RES = os.path.join(ROOT, "results")
N_PERM = int(os.environ.get("N_PERM", "1000"))
RNG = np.random.RandomState(11)

def build_sym2chr():
    sym2chr = {}
    with open(os.path.join(ROOT, "data", "annot", "gencode.v39.genes.gtf"),
              encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.split("\t")
            if p[2] != "gene": continue
            sym2chr[re.search(r'gene_name "([^"]+)"', p[8]).group(1)] = p[0]
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
        rows.append(np.asarray(X[np.asarray(list(gidx))].sum(axis=0)).ravel())
        keys.append((d, ct))
    sex_lu = obs.drop_duplicates("donor").set_index("donor")["sex_i"]
    return (np.vstack(rows),
            np.array([k[0] for k in keys]),
            np.array([k[1] for k in keys]),
            np.array([sex_lu[k[0]] for k in keys], dtype=int),
            syms)

def cpm_log(M):
    lib = M.sum(axis=1, keepdims=True)
    return np.log2(M / np.maximum(lib, 1) * 1e6 + 1.0).astype(np.float32)

def lodo_auc(Yg, y, donors):
    """Leave-one-donor-out: pool predictions, single AUC (stable for small n)."""
    preds = np.full(len(y), np.nan)
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        if len(set(y[tr])) < 2: continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(Yg[tr], y[tr])
        preds[i] = clf.predict_proba(Yg[i:i+1])[:, 1]
    ok = ~np.isnan(preds)
    if len(set(y[ok])) < 2: return np.nan, preds
    return float(roc_auc_score(y[ok], preds[ok])), preds

def auc_strength(auc):
    """Two-sided strength: max(AUC, 1-AUC). 0.5=no signal, 1.0=perfect (either direction)."""
    return max(auc, 1.0 - auc)

def perm_p_lodo(Yg, y, donors, n_perm, obs_auc):
    """M3.2: TWO-SIDED donor-level permutation on auc_strength."""
    obs_s = auc_strength(obs_auc)
    ud = pd.unique(donors)
    d_sex = dict(zip(donors, y))
    hits = 0
    for i in range(n_perm):
        pm = dict(zip(ud, RNG.permutation([d_sex[d] for d in ud])))
        yp = np.array([pm[d] for d in donors])
        a0, _ = lodo_auc(Yg, yp, donors)
        if auc_strength(a0) >= obs_s: hits += 1
    return (hits + 1) / (n_perm + 1)

def _one_ct_file(args):
    fn, n_perm = args
    d = np.load(fn, allow_pickle=True)
    Yg, y, g = d.item()["Yg"], d.item()["y"], d.item()["g"]
    auc, _ = lodo_auc(Yg, y, g)
    if np.isnan(auc):
        return None
    p_emp = perm_p_lodo(Yg, y, g, n_perm, auc)
    return auc, p_emp

def main():
    sym2chr = build_sym2chr()
    cohorts = [("KD_LivingDonor", os.path.join(SCD, "KD_LivingDonor.h5ad")),
               ("KD_Mature",      os.path.join(SCD, "KD_Mature.h5ad"))]
    pbmc = os.path.join(SCD, "PBMC_Indonesia.h5ad")
    if os.path.exists(pbmc) and os.path.getsize(pbmc) > 3e9:
        cohorts.append(("PBMC_Indonesia", pbmc))

    rows = []
    for cohort, path in cohorts:
        print(f"\n=== {cohort} ===", flush=True)
        M, donors, cts, sexes, syms = pseudo_bulk(path)
        auto = np.array([sym2chr.get(s, "NA") not in ("chrX","chrY","chrM") for s in syms])
        L = cpm_log(M)
        n_f = int((sexes == 1).sum()); n_m = int((sexes == 0).sum())
        tag = "" if min(n_f, n_m) >= 12 else "  [EXPLORATORY: <12/sex]"
        print(f"    units={M.shape[0]} donors={len(set(donors))} F={n_f} M={n_m} "
              f"auto={auto.sum()}{tag}", flush=True)
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp(prefix="m31_")
        jobs = []
        for ct in pd.unique(cts):
            m = cts == ct
            s = sexes[m]
            if min(np.bincount(s)) < 4: continue
            Yg = L[np.ix_(m, auto)]
            n_perm = N_PERM if min(n_f, n_m) >= 12 else min(N_PERM, 500)
            fn = os.path.join(tmpdir, re.sub(r"[^A-Za-z0-9_]", "_", ct)[:60] + ".npy")
            np.save(fn, {"Yg": Yg, "y": s, "g": donors[m]})
            jobs.append((ct, fn, n_perm, int(m.sum()), n_f, n_m,
                         "confirmatory" if min(n_f, n_m) >= 12 else "exploratory"))
        print(f"    {len(jobs)} cell types staged", flush=True)
        from concurrent.futures import ProcessPoolExecutor
        # PBMC matrices ~30x larger -> fewer workers to avoid OOM
        n_w = 2 if ('Indonesia' in cohort or 'PBMC' in cohort) else 8
        with ProcessPoolExecutor(max_workers=n_w) as ex:
            futs = {ex.submit(_one_ct_file, (fn, n_perm)): ct
                    for ct, fn, n_perm, nd, nf, nm, dsg in jobs}
            for fut in futs:
                ct = futs[fut]
                res = fut.result()
                if res is None: continue
                auc, p_emp = res
                job = next(j for j in jobs if j[0] == ct)
                _, _, nperm, nd, nf, nm, dsg = job
                strength = auc_strength(auc)
                direction = "female-high" if auc > 0.5 else ("male-high" if auc < 0.5 else "none")
                rows.append({"cohort": cohort, "cell_type": ct,
                             "n_donors": nd, "n_female": nf, "n_male": nm,
                             "design": dsg, "auc_raw": round(auc, 4),
                             "auc_strength": round(strength, 4), "direction": direction,
                             "perm_n": nperm, "perm_p_two_sided": round(p_emp, 4)})
                print(f"    {ct[:40]:40s} n={nd:4d} raw={auc:.3f} S={strength:.3f} "
                      f"{direction:11s} p({nperm})={p_emp:.4f}", flush=True)
        shutil.rmtree(tmpdir, ignore_errors=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "m32_celltypes_two_sided.csv"), index=False)

    # ---- cross-cohort alignment (KD_LivingDonor vs KD_Mature) ----
    lines = [f"M3.2 TWO-SIDED (N_PERM={N_PERM}) — POSITIVE if perm_p<0.05 AND strength>0.65"]
    a = df[df.cohort == "KD_LivingDonor"].set_index("cell_type")
    b = df[df.cohort == "KD_Mature"].set_index("cell_type")
    common = a.index.intersection(b.index)
    rep_same, rep_opp = [], []
    for ct in common:
        Sa = max(a.loc[ct, "auc_raw"], 1 - a.loc[ct, "auc_raw"])
        Sb = max(b.loc[ct, "auc_raw"], 1 - b.loc[ct, "auc_raw"])
        same_dir = (np.sign(a.loc[ct, "auc_raw"] - 0.5) ==
                    np.sign(b.loc[ct, "auc_raw"] - 0.5))
        if Sa >= 0.65 and Sb >= 0.65:
            (rep_same if same_dir else rep_opp).append(ct)
    lines.append(f"  KD replication (strength>=0.65 both): {len(rep_same)} same-dir "
                 f"+ {len(rep_opp)} opposite-dir / {len(common)} common")
    if rep_same:
        lines.append("  SAME-direction: " + ", ".join(rep_same[:12]))
    if rep_opp:
        lines.append("  OPPOSITE-direction (context-dependent): " + ", ".join(rep_opp[:12]))
    # confirmatory cohort verdict
    conf = df[df.design == "confirmatory"]
    for cohort, sub in conf.groupby("cohort"):
        pos = int(((sub.perm_p_two_sided < 0.05) & (sub.auc_strength > 0.65)).sum())
        lines.append(f"  {cohort} (confirmatory, n>=12/sex): {pos}/{len(sub)} POSITIVE (two-sided)")
    if len(conf) == 0:
        lines.append("  (no confirmatory cohort this run)")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(RES, "m32_summary.txt"), "w", encoding="utf-8").write(txt)

if __name__ == "__main__":
    main()
