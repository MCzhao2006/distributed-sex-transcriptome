# -*- coding: utf-8 -*-
"""m36_fixed_cells.py — M3.6: fixed cells-per-unit downsampling experiment.

Question (chatGPT_11.md)
  Apparent sex AUC rises steeply with cells-per-pseudobulk-unit (Spearman +0.83
  PBMC, +0.94 OneK1K). Is that (A) measurement precision, or (B) a real
  cell-composition signal? And beyond either, is there residual sex-associated
  expression signal at a matched measurement scale?

Design
  Cap every donor x cell_type unit at a FIXED N cells (15/30/50/100) plus the
  native (uncapped) reference, over many independent random seeds.
  For each (cohort, cell_type, N) report across seeds:
      AUC, S = max(AUC, 1-AUC), median + 95% CI
  and, at the same N, a DONOR-LABEL PERMUTATION NULL.

Why the null is not optional
  Fixing N removes the cell-composition axis by construction (every donor now
  contributes the same number of cells), so any AUC left over is expression-
  based. But a low AUC at N=15 is equally consistent with "real signal, heavily
  attenuated" and "no signal at all". Only the excess over the matched-N null
  separates those two. Without it the curve cannot be interpreted.

Guards that are easy to get wrong (all three bit us or would have)
  * K (PCA dim) must be << n. K=199 with n=199 collapsed to AUC~0.2-0.4.
    Rule: K = max(3, min(K_PCA_MAX, n_donors // 2)). Different cohorts therefore
    use different K -- a real confound, reported explicitly, not hidden.
  * A cap of N is only honest if the units actually HAD N cells. KD cohorts
    average ~103 cells/unit, so "N=100" would silently be "native" for them.
    Each (cell_type, cap) is skipped unless >= CAP_COVERAGE of units reach N,
    and the achieved median is recorded either way.
  * Staging all seeds before modelling would hold ~1500 x 26 MB in RAM. Seeds
    are therefore processed one at a time: stage -> pool -> checkpoint.

Usage
  NATURE_ROOT=F:/nature python scripts/m36_fixed_cells.py
  env: M36_COHORTS / M36_NSEEDS / M36_WORKERS / M36_NLEVELS / M36_NNULLSEEDS
       M36_NPERM / M36_KMAX / M36_CAPCOV
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NATURE_ROOT") or os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from m32_perm_parallel import (  # noqa: E402
    cpm_log, build_sym2chr, auc_strength, auc_direction,
    AUTOSOMES, MIN_RUN_DONORS_PER_SEX, stable_seed)

from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.preprocessing import StandardScaler      # noqa: E402
from sklearn.decomposition import PCA                 # noqa: E402
from sklearn.pipeline import make_pipeline            # noqa: E402
from sklearn.metrics import roc_auc_score             # noqa: E402

SCD = os.path.join(ROOT, "data", "singlecell")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "m36_fixed_cells.csv")

COHORTS = os.environ.get("M36_COHORTS", "PBMC_Indonesia,KD_Mature,KD_LivingDonor").split(",")
N_LEVELS = [int(x) for x in os.environ.get("M36_NLEVELS", "15,30,50,100").split(",")]
N_SEEDS = int(os.environ.get("M36_NSEEDS", "50"))
N_NULL_SEEDS = int(os.environ.get("M36_NNULLSEEDS", "10"))
N_PERM = int(os.environ.get("M36_NPERM", "20"))
K_PCA_MAX = int(os.environ.get("M36_KMAX", "100"))
WORKERS = int(os.environ.get("M36_WORKERS", "10"))
CAP_COVERAGE = float(os.environ.get("M36_CAPCOV", "0.8"))
C_L2 = 0.1
UNCAPPED = 10 ** 9


def read_obs(path):
    with h5py.File(path, "r") as hf:
        o = hf["obs"]

        def cats(n):
            c = [x.decode() if isinstance(x, bytes) else x for x in o[n]["categories"][:]]
            return np.asarray(pd.Categorical.from_codes(o[n]["codes"][:], c)).astype(str)
        donor, sex, ct = cats("donor_id"), cats("sex"), cats("cell_type")
        _g = hf["raw/X"] if "raw/X" in hf else hf["X"]
        n_cells = int(_g.attrs["shape"][0])
        _v = hf["var"]
        if "gene_symbols" in _v or "feature_name" in _v:
            k = "gene_symbols" if "gene_symbols" in _v else "feature_name"
            vc = [x.decode() if isinstance(x, bytes) else x for x in _v[k]["categories"][:]]
            syms = np.array([vc[c] for c in _v[k]["codes"][:]])
        else:
            syms = np.array([x.decode() if isinstance(x, bytes) else x for x in _v["_index"][:]])
    return donor, sex, ct, n_cells, syms


def unit_and_rank(n_cells_total, donor, ct, sex, seed):
    keep = np.isin(sex, ["female", "male"])
    if keep.sum() != n_cells_total:
        raise SystemExit(f"REFUSING: {n_cells_total - keep.sum()} cells lack a sex label; "
                         "the global-cell-id -> unit indexing below would silently corrupt.")
    key_order = sorted({(d, c) for d, c in zip(donor, ct)})
    kid = {k: i for i, k in enumerate(key_order)}
    unit = np.array([kid[(d, c)] for d, c in zip(donor, ct)], dtype=np.int64)
    rng = np.random.RandomState(seed)
    rkey = rng.random(len(unit))
    order = np.lexsort((rkey, unit))
    su = unit[order]
    _u, first, cnt = np.unique(su, return_index=True, return_counts=True)
    rank = np.empty(len(unit), dtype=np.int64)
    rank[order] = np.arange(len(order)) - np.repeat(first, cnt)
    return unit, rank, key_order


def scatter_caps(path, unit, rank, caps, n_units, n_genes):
    """One streaming pass -> one pseudobulk matrix per cap.

    The caps are NESTED thresholds (a cell with rank r belongs to every cap C
    with r < C), so scattering straight into each cap adds the same nonzero up
    to len(caps) times: for caps (15,30,50,100) that measured ~2.8 np.add.at
    passes over 1.18B nonzeros, and np.add.at is the whole cost of this function.

    Instead each cell is assigned to the single disjoint bucket it falls in and
    the caps are recovered by a cumulative sum. Same matrices, one pass of
    np.add.at, and cells past the largest cap are never touched at all.
    """
    INF = 10 ** 8
    fin = sorted(c for c in caps if c < INF)
    has_inf = any(c >= INF for c in caps)
    ordered = fin + ([INF] if has_inf else [])
    k = len(ordered)
    if fin:
        # rank r belongs to caps j where r < fin[j]; the smallest such j is the
        # bucket index. side='right' is essential: rank == cap must NOT count
        # (side='left' would include it and shift every boundary by one).
        bid = np.searchsorted(np.asarray(fin, dtype=np.int64), rank, side="right")
        if not has_inf:
            bid = np.where(bid >= len(fin), -1, bid)
    else:
        bid = np.zeros(len(rank), dtype=np.int64)
    inc = bid >= 0                        # False for cells past the largest cap
    print(f"    [scatter] buckets={ordered}  cells kept={int(inc.sum()):,}"
          f" / {len(rank):,}", flush=True)

    with h5py.File(path, "r") as hf:
        _g = hf["raw/X"] if "raw/X" in hf else hf["X"]
        ptr = _g["indptr"][:].astype(np.int64)
        nnz = _g["data"].shape[0]
        B = [np.zeros((n_units, n_genes), dtype=np.float32) for _ in range(k)]
        CH = 8_000_000
        for s in range(0, nnz, CH):
            e = min(s + CH, nnz)
            rows = np.searchsorted(ptr, np.arange(s, e), side="right") - 1
            ok = inc[rows]
            if not ok.any():
                continue
            buck = unit[rows][ok]
            idx = _g["indices"][s:e][ok]
            dat = _g["data"][s:e][ok]
            bb = bid[rows][ok]
            for b in range(k):
                m = bb == b
                if m.any():
                    np.add.at(B[b], (buck[m], idx[m]), dat[m])
        for j in range(1, k):             # cap j = buckets 0..j
            B[j] += B[j - 1]
    out = {}
    for c in caps:
        out[c] = B[k - 1] if c >= INF else B[fin.index(c)]
    return out


def _k_for(n_donors):
    return int(max(3, min(K_PCA_MAX, n_donors // 2)))


def _auc_lodo(S, y, donors):
    donors = np.asarray(donors)
    preds = np.full(len(y), np.nan)
    for d in pd.unique(donors):
        te = donors == d
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C_L2, max_iter=2000))
        clf.fit(S[tr], y[tr])
        preds[te] = clf.predict_proba(S[te])[:, 1]
    ok = ~np.isnan(preds)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], preds[ok]))


def _job(args):
    Yg, y, donors, k, do_null, perm_seed, n_perm = args
    pcs = PCA(n_components=k, svd_solver="randomized", random_state=0)
    S = pcs.fit_transform(StandardScaler().fit_transform(Yg))
    auc = _auc_lodo(S, y, donors)
    nulls = []
    if do_null:
        rng = np.random.RandomState(perm_seed)
        ud = pd.unique(donors)
        base = np.array([int(y[donors == d][0]) for d in ud])
        for _ in range(n_perm):
            mp = dict(zip(ud, rng.permutation(base)))
            yp = np.array([mp[d] for d in donors])
            if len(np.unique(yp)) < 2:
                continue
            a = _auc_lodo(S, yp, donors)
            if not np.isnan(a):
                nulls.append(float(a))
    return auc, nulls


def main():
    os.makedirs(RES, exist_ok=True)
    print("=" * 74)
    print("M3.6 fixed-cells-per-unit downsampling")
    print(f"cohorts={COHORTS} N={N_LEVELS} seeds={N_SEEDS} "
          f"null={N_NULL_SEEDS}x{N_PERM}perm Kmax={K_PCA_MAX} "
          f"cap_coverage>={CAP_COVERAGE} workers={WORKERS}")
    print("=" * 74, flush=True)

    sym2chr = build_sym2chr()
    rows = []
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        rows = prev.to_dict("records")
        print(f"[resume] {len(rows)} rows already computed -> will be rewritten", flush=True)

    for cohort in COHORTS:
        path = os.path.join(SCD, f"{cohort}.h5ad")
        if not os.path.exists(path):
            print(f"[skip] {cohort}: not present locally", flush=True); continue
        donor, sex, ct, n_cells, syms = read_obs(path)
        auto = np.array([sym2chr.get(str(s)) in AUTOSOMES for s in syms])
        n_genes = len(syms)
        cts_all = pd.unique(ct)
        key_order0 = sorted({(d, c) for d, c in zip(donor, ct)})
        n_units = len(key_order0)
        cts_u0 = np.array([k[1] for k in key_order0])
        donors_u0 = np.array([k[0] for k in key_order0])
        sex_of = pd.Series(sex).groupby(pd.Series(donor)).first()
        y_u0 = np.array([1 if str(sex_of[d]).lower() == "female" else 0 for d in donors_u0])
        caps = N_LEVELS + [UNCAPPED]
        seeds = [stable_seed(cohort, "m36seed", i) for i in range(N_SEEDS)]
        print(f"\n=== {cohort} ===  cells={n_cells:,} units={n_units} "
              f"cell_types={len(cts_all)} genes={n_genes:,} autosomal={int(auto.sum()):,}",
              flush=True)

        t_cohort = time.time()
        for si, sd in enumerate(seeds):
            t0 = time.time()
            unit, rank, key_order = unit_and_rank(n_cells, donor, ct, sex, sd)
            M = scatter_caps(path, unit, rank, caps, n_units, n_genes)
            # cells actually kept per unit, per cap (drives both the honesty check
            # and the reported achieved N)
            kept = {C: np.bincount(unit[rank < C], minlength=n_units) for C in caps}
            t_scatter = time.time() - t0

            tasks = []
            for C in caps:
                Lb = cpm_log(M[C])
                lib = M[C].sum(axis=1)
                for ctv in cts_all:
                    m = cts_u0 == ctv
                    y = y_u0[m]
                    if min(np.bincount(y)) < MIN_RUN_DONORS_PER_SEX:
                        continue
                    ach = kept[C][m]
                    cov = float((ach >= C).mean()) if C != UNCAPPED else 1.0
                    if C != UNCAPPED and cov < CAP_COVERAGE:
                        continue           # "N=100" would really be "native" here
                    n_units_ct = len(np.unique(donors_u0[m]))
                    k = _k_for(n_units_ct)
                    tasks.append(((Lb[np.ix_(m, auto)], y, donors_u0[m], k,
                                   si < N_NULL_SEEDS,
                                   stable_seed(cohort, ctv, C, si, "perm"), N_PERM),
                                  {"cohort": cohort, "cell_type": ctv, "cap_n": C,
                                   "cap_label": "native" if C == UNCAPPED else f"N{C}",
                                   "seed": int(sd), "k_pca": k,
                                   "n_units": n_units_ct,
                                   "n_donors": int(len(np.unique(donors_u0[m]))),
                                   "n_female": int((y == 1).sum()),
                                   "n_male": int((y == 0).sum()),
                                   "achieved_median_cells": float(np.median(ach)),
                                   "cap_coverage": round(cov, 3),
                                   "median_log_lib": round(float(np.median(np.log1p(lib[m]))), 3)}))
            del M, kept

            got = 0
            if tasks:
                with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                    futs = {ex.submit(_job, t[0]): t[1] for t in tasks}
                    for f in as_completed(futs):
                        meta = futs[f]
                        auc, nulls = f.result()
                        if np.isnan(auc):
                            continue
                        rec = dict(meta)
                        rec.update({
                            "auc": round(auc, 5), "S": round(auc_strength(auc), 5),
                            "direction": auc_direction(auc), "null_n": len(nulls),
                            "null_median": round(float(np.median(nulls)), 5) if nulls else np.nan,
                            "null_p95": round(float(np.percentile(nulls, 95)), 5) if nulls else np.nan,
                        })
                        rows.append(rec)
                        got += 1
            print(f"  [seed {si+1}/{N_SEEDS}] scatter {t_scatter:.0f}s  "
                  f"{len(tasks)} jobs -> {got} rows  (total {len(rows)})", flush=True)
            if (si + 1) % 5 == 0:
                pd.DataFrame(rows).to_csv(OUT, index=False)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  [cohort done] {cohort} in {(time.time()-t_cohort)/60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\n[done] {len(df)} rows -> {OUT}", flush=True)
    if len(df):
        g = df.groupby(["cohort", "cap_label"]).agg(
            n=("S", "size"), S_median=("S", "median"),
            S_lo=("S", lambda s: s.quantile(.025)), S_hi=("S", lambda s: s.quantile(.975)),
            cells=("achieved_median_cells", "median"), K=("k_pca", "first"))
        print(g.to_string())


if __name__ == "__main__":
    main()
