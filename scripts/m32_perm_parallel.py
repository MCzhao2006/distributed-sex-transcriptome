# -*- coding: utf-8 -*-
"""M3.2-P — permutation-parallel two-sided within-cell-type sex signal.

Same statistics as m32_pbmc.py (two-sided S=max(AUC,1-AUC), donor-level
permutation, strict chr1-22 autosomes, LODO pooled AUC), but parallelised at
the PERMUTATION level so that a 128-core machine actually uses 128 cores.

Architecture
  Pass 1  (one task per cell type)      : observed LODO AUC -> strength
  Pass 2  (N_CHUNKS tasks per cell type): permutation hits for its share
  Aggregate: p = (sum_hits + 1) / (N_PERM + 1)

This is statistically identical to the serial version: permutations are
independent, so splitting them across processes and summing hits gives the
same p-value (up to RNG stream realisation).

Env:
  NATURE_ROOT   working dir                     (default: script's dir)
  N_PERM        permutations per cell type      (default 500)
  N_CHUNKS      chunks per cell type            (default 8)
  M32_WORKERS   parallel processes              (default: cpu_count-1)
  ONLY_COHORT   e.g. PBMC_Indonesia             (default: all available)

Outputs:
  results/m31_celltypes.csv
  results/m31_summary.txt
"""
import hashlib
import os
import re
import shutil
import tempfile

import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NATURE_ROOT") or os.path.dirname(_HERE)  # script lives in <ROOT>/scripts
SCD = os.path.join(ROOT, "data", "singlecell")
RES = os.path.join(ROOT, "results")

N_PERM = int(os.environ.get("N_PERM", "500"))
N_CHUNKS = int(os.environ.get("N_CHUNKS", "8"))
MIN_RUN_DONORS_PER_SEX = int(os.environ.get("MIN_RUN_DONORS_PER_SEX", "4"))
MIN_DONORS_PER_SEX = int(os.environ.get("MIN_DONORS_PER_SEX", "12"))
AUTOSOMES = {f"chr{i}" for i in range(1, 23)}


# ---------------------------------------------------------------- utilities

def stable_seed(*parts, base_seed=11):
    text = "|".join(map(str, parts))
    x = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    return int((x + base_seed) % (2**32 - 1))


def auc_strength(auc):
    if np.isnan(auc):
        return np.nan
    return float(max(auc, 1.0 - auc))


def auc_direction(auc, tol=1e-12):
    if np.isnan(auc):
        return "NA"
    if auc > 0.5 + tol:
        return "forward"
    if auc < 0.5 - tol:
        return "reverse"
    return "chance"


def build_sym2chr():
    sym2chr = {}
    gtf = os.path.join(ROOT, "data", "annot", "gencode.v39.genes.gtf")
    with open(gtf, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] != "gene":
                continue
            m = re.search(r'gene_name "([^"]+)"', p[8])
            if m and m.group(1):
                sym2chr[m.group(1)] = p[0]
    return sym2chr


# ---------------------------------------------------------------- data I/O

def pseudo_bulk(path):
    """donor x cell_type pseudobulk from a CELLxGENE h5ad (h5py, streaming)."""
    with h5py.File(path, "r") as hf:
        _o = hf["obs"]

        def cats(n):
            c = [x.decode() if isinstance(x, bytes) else x
                 for x in _o[n]["categories"][:]]
            return pd.Categorical.from_codes(_o[n]["codes"][:], c)

        obs = pd.DataFrame({"donor": cats("donor_id"),
                            "sex": cats("sex"),
                            "ct": cats("cell_type")})
        _g = hf["raw/X"] if "raw/X" in hf else hf["X"]
        n_cells, n_genes = (int(x) for x in _g.attrs["shape"])
        ptr = _g["indptr"][:].astype(np.int64)
        nnz = _g["data"].shape[0]

        _v = hf["var"]
        if "gene_symbols" in _v or "feature_name" in _v:
            k = "gene_symbols" if "gene_symbols" in _v else "feature_name"
            vc = [x.decode() if isinstance(x, bytes) else x
                  for x in _v[k]["categories"][:]]
            syms = np.array([vc[c] for c in _v[k]["codes"][:]])
        else:
            syms = np.array([x.decode() if isinstance(x, bytes) else x
                             for x in _v["_index"][:]])

        # ---- sex filter + bucket map (inside the open file block) ----
        smap = {"female": 1, "male": 0}
        obs["sex_i"] = obs["sex"].astype(str).str.lower().map(smap)
        keep_pos = np.where(obs.sex_i.notna().to_numpy())[0]

        key_order = sorted({(d, ct) for d, ct in
                            zip(obs["donor"].to_numpy()[keep_pos],
                                obs["ct"].to_numpy()[keep_pos])})
        key_id = {k: i for i, k in enumerate(key_order)}
        row2b = np.array([key_id[(d, ct)] for d, ct in
                          zip(obs["donor"].to_numpy()[keep_pos],
                              obs["ct"].to_numpy()[keep_pos])], dtype=np.int64)

        # ---- streaming scatter-add: keep the h5 handle alive ----
        M = np.zeros((len(key_order), n_genes), dtype=np.float32)
        CH = 40_000_000
        for s in range(0, nnz, CH):
            e = min(s + CH, nnz)
            rows = np.searchsorted(ptr, np.arange(s, e), side="right") - 1
            buck = row2b[rows]
            np.add.at(M, (buck, _g["indices"][s:e]), _g["data"][s:e])

    sex_of = obs.drop_duplicates("donor").set_index("donor")["sex_i"]
    donors = np.array([k[0] for k in key_order])
    cts = np.array([k[1] for k in key_order])
    sexes = np.array([sex_of[d] for d in donors], dtype=int)
    return M, donors, cts, sexes, syms


def cpm_log(M):
    lib = M.sum(axis=1, keepdims=True)
    return np.log2(M / np.maximum(lib, 1) * 1e6 + 1.0).astype(np.float32)


# ---------------------------------------------------------------- LODO

def lodo_auc(Yg, y, donors):
    """Leave-one-donor-out, predictions pooled, single AUC."""
    donors = np.asarray(donors)
    preds = np.full(len(y), np.nan)
    for d in pd.unique(donors):
        te = donors == d
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(Yg[tr], y[tr])
        preds[te] = clf.predict_proba(Yg[te])[:, 1]
    ok = ~np.isnan(preds)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], preds[ok]))


# ---------------------------------------------------------------- workers

def _observed(fn, seed):
    d = np.load(fn, allow_pickle=True).item()
    auc = lodo_auc(d["Yg"], d["y"], d["g"])
    return (float(auc) if not np.isnan(auc) else None,
            auc_strength(auc) if not np.isnan(auc) else None)


def _perm_chunk(args):
    """Count how many permutations reach the observed strength."""
    fn, n_perm_chunk, seed, obs_strength = args
    d = np.load(fn, allow_pickle=True).item()
    Yg, y, donors = d["Yg"], d["y"], d["g"]
    rng = np.random.RandomState(seed)
    ud = pd.unique(donors)
    donor_sex = {}
    for dd in ud:
        yy = y[donors == dd]
        donor_sex[dd] = int(yy[0])
    base = np.array([donor_sex[dd] for dd in ud])
    hits = 0
    for _ in range(n_perm_chunk):
        yp_map = dict(zip(ud, rng.permutation(base)))
        yp = np.array([yp_map[dd] for dd in donors])
        if len(np.unique(yp)) < 2:
            continue
        a = lodo_auc(Yg, yp, donors)
        if np.isnan(a):
            continue
        if auc_strength(a) >= obs_strength:
            hits += 1
    return hits


# ---------------------------------------------------------------- main

def main():
    os.makedirs(RES, exist_ok=True)
    try:
        n_cpu = len(os.sched_getaffinity(0))      # respects cgroup/container limits
    except AttributeError:
        n_cpu = os.cpu_count() or 4
    n_w = int(os.environ.get("M32_WORKERS", max(1, n_cpu - 1)))
    print("=" * 62)
    print("M3.2-P permutation-parallel two-sided cell-type sex analysis")
    print(f"ROOT={ROOT}")
    print(f"N_PERM={N_PERM}  N_CHUNKS={N_CHUNKS}  WORKERS={n_w}  CPUs={n_cpu}")
    print("statistic: S=max(AUC,1-AUC); autosomes chr1-22 only")
    print("=" * 62, flush=True)

    sym2chr = build_sym2chr()
    print(f"symbol->chr map: {len(sym2chr):,} genes", flush=True)

    cohorts = [("KD_LivingDonor", os.path.join(SCD, "KD_LivingDonor.h5ad")),
               ("KD_Mature", os.path.join(SCD, "KD_Mature.h5ad"))]
    pbmc = os.path.join(SCD, "PBMC_Indonesia.h5ad")
    if os.path.exists(pbmc) and os.path.getsize(pbmc) > 3e9:
        cohorts.append(("PBMC_Indonesia", pbmc))
    only = os.environ.get("ONLY_COHORT", "")
    if only:
        cohorts = [c for c in cohorts if c[0] == only]

    rows = []
    for cohort, path in cohorts:
        print(f"\n=== {cohort} ===", flush=True)
        M, donors, cts, sexes, syms = pseudo_bulk(path)
        chrom = np.array([sym2chr.get(str(s)) for s in syms], dtype=object)
        auto = np.array([c in AUTOSOMES for c in chrom])
        L = cpm_log(M)
        nf = len(pd.unique(donors[sexes == 1]))
        nm = len(pd.unique(donors[sexes == 0]))
        print(f"    genes={len(syms):,} autosomal={auto.sum():,} "
              f"units={M.shape[0]:,} donors={len(pd.unique(donors))} F={nf} M={nm}",
              flush=True)

        tmpdir = tempfile.mkdtemp(prefix="m32p_")
        jobs, meta = [], {}
        for ct in pd.unique(cts):
            m = cts == ct
            cd, cs = donors[m], sexes[m]
            f_d = pd.unique(cd[cs == 1])
            m_d = pd.unique(cd[cs == 0])
            if min(len(f_d), len(m_d)) < MIN_RUN_DONORS_PER_SEX:
                continue
            design = ("confirmatory" if min(len(f_d), len(m_d)) >= MIN_DONORS_PER_SEX
                      else "exploratory")
            Yg = L[np.ix_(m, auto)]
            fn = os.path.join(tmpdir, re.sub(r"[^A-Za-z0-9_]", "_", str(ct))[:60] + ".npy")
            np.save(fn, {"Yg": Yg, "y": cs, "g": cd})
            meta[fn] = (ct, len(pd.unique(cd)), len(f_d), len(m_d), design)
            jobs.append(fn)
        print(f"    {len(jobs)} cell types staged", flush=True)

        # ---- pass 1: observed AUC per cell type ----
        obs = {}
        with ProcessPoolExecutor(max_workers=min(n_w, len(jobs))) as ex:
            futs = {ex.submit(_observed, fn, stable_seed(cohort, meta[fn][0], "obs")): fn
                    for fn in jobs}
            for f in as_completed(futs):
                fn = futs[f]
                res = f.result()
                ct = meta[fn][0]
                obs[fn] = res
                print(f"    [obs] {ct[:42]:42s} AUC={res[0]:.3f} "
                      f"S={res[1]:.3f}" if res[0] is not None else f"    [obs] {ct} NA",
                      flush=True)
        jobs = [fn for fn in jobs if obs.get(fn, (None,))[0] is not None]

        # ---- pass 2: permutation hits in chunks ----
        per = max(1, N_PERM // N_CHUNKS)
        tasks = []
        for fn in jobs:
            S = obs[fn][1]
            for ci in range(N_CHUNKS):
                n_c = per if ci < N_CHUNKS - 1 else max(1, N_PERM - per * (N_CHUNKS - 1))
                tasks.append((fn, n_c, stable_seed(cohort, meta[fn][0], ci, "perm"), S))
        print(f"    launching {len(tasks)} permutation tasks", flush=True)
        hits = {fn: 0 for fn in jobs}
        done = 0
        with ProcessPoolExecutor(max_workers=n_w) as ex:
            futs = {ex.submit(_perm_chunk, t): t[0] for t in tasks}
            for f in as_completed(futs):
                fn = futs[f]
                hits[fn] += f.result()
                done += 1
                if done % max(1, len(tasks) // 10) == 0:
                    print(f"    ... {done}/{len(tasks)} chunk tasks done", flush=True)

        for fn in jobs:
            ct, nd, nf_, nm_, design = meta[fn]
            auc, S = obs[fn]
            p = (hits[fn] + 1) / (N_PERM + 1)
            rows.append({"cohort": cohort, "cell_type": ct, "n_donors": nd,
                         "n_female": nf_, "n_male": nm_, "design": design,
                         "auc_lodo": round(auc, 6), "auc_strength": round(S, 6),
                         "direction": auc_direction(auc), "perm_n": N_PERM,
                         "perm_hits": hits[fn], "perm_p_two_sided": round(p, 6)})
            print(f"    {ct[:40]:40s} donors={nd:4d} F={nf_:3d} M={nm_:3d} "
                  f"AUC={auc:.3f} S={S:.3f} dir={auc_direction(auc):7s} "
                  f"p({N_PERM})={p:.4f} [{design}]", flush=True)

        shutil.rmtree(tmpdir, ignore_errors=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "m31_celltypes.csv"), index=False)

    lines = ["M3.2-P permutation-parallel two-sided analysis",
             f"N_PERM={N_PERM}  N_CHUNKS={N_CHUNKS}  WORKERS={n_w}",
             "statistic S=max(AUC,1-AUC); autosomes chr1-22; unique-donor counts"]
    if not df.empty:
        conf = df[df.design == "confirmatory"]
        for cohort, sub in conf.groupby("cohort"):
            pos = int(((sub.perm_p_two_sided < 0.05) & (sub.auc_strength >= 0.65)).sum())
            lines.append(f"  {cohort} (confirmatory): {pos}/{len(sub)} POSITIVE "
                         f"(p<0.05 & S>=0.65)")
        if conf.empty:
            lines.append("  no confirmatory cell types in this run")
        a = df[df.cohort == "KD_LivingDonor"].set_index("cell_type")
        b = df[df.cohort == "KD_Mature"].set_index("cell_type")
        common = a.index.intersection(b.index)
        same, opp = [], []
        for ct in common:
            Sa, Sb = a.loc[ct, "auc_strength"], b.loc[ct, "auc_strength"]
            if Sa >= 0.60 and Sb >= 0.60:
                if a.loc[ct, "direction"] == b.loc[ct, "direction"]:
                    same.append(ct)
                else:
                    opp.append(ct)
        lines.append(f"  KD common={len(common)} strong={len(same)+len(opp)} "
                     f"same-direction={len(same)} opposite={len(opp)}")
        if same:
            lines.append("  same-direction: " + ", ".join(same[:20]))
        lines.append(f"  exploratory analyses: {int((df.design=='exploratory').sum())}")

    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    with open(os.path.join(RES, "m31_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
