# -*- coding: utf-8 -*-
"""m36_shard.py — seed-sharded driver for M3.6 on a many-core machine.

Why this exists
  The single-process m36_fixed_cells.py does one streaming scatter pass per seed.
  That pass measured ~25 min on the cloud box for PBMC's 725M nonzeros (75 s on
  the local NVMe), so 100 seeds of OneK1K would need 40+ hours serially. Seeds
  are fully independent, so the only fix is to run several at once.

  Each shard processes seeds[LO:HI] and writes its own CSV; the parent merges.
  Memory per shard is the binding constraint, so it is checked up front:
      per shard = n_caps * n_units * n_autosomal_genes * 4 bytes
  OneK1K (22033 units x 32826 autosomal genes x 5 caps) is ~14.5 GB, so ~10
  shards fit comfortably in a 503 GB box while still cutting wall time ~10x.

Usage
  NATURE_ROOT=/root/nature M36_COHORTS=OneK1K M36_SEED_LO=0 M36_SEED_HI=10 \
      python scripts/m36_shard.py
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NATURE_ROOT") or os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import m36_fixed_cells as M  # noqa: E402
from m32_perm_parallel import cpm_log, build_sym2chr, AUTOSOMES, MIN_RUN_DONORS_PER_SEX  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

SEED_LO = int(os.environ.get("M36_SEED_LO", "0"))
SEED_HI = int(os.environ.get("M36_SEED_HI", str(M.N_SEEDS)))
N_SEEDS = M.N_SEEDS
N_LEVELS = M.N_LEVELS
N_NULL_SEEDS = M.N_NULL_SEEDS
N_PERM = M.N_PERM
CAP_COVERAGE = M.CAP_COVERAGE
WORKERS = M.WORKERS
UNCAPPED = M.UNCAPPED
# 'native' is seed-invariant (rank < inf is always all-True), so every shard would
# recompute an identical matrix -- 20% of both memory and scatter time for nothing.
# Run it once in a dedicated single-seed job instead.
INCLUDE_NATIVE = os.environ.get("M36_INCLUDE_NATIVE", "1") == "1"


def main():
    print(f"[shard] seeds [{SEED_LO}, {SEED_HI}) of {N_SEEDS}; inner pool = {WORKERS}",
          flush=True)
    sym2chr = build_sym2chr()
    jobs = []
    for cohort in M.COHORTS:
        path = os.path.join(M.SCD, f"{cohort}.h5ad")
        if not os.path.exists(path):
            print(f"[skip] {cohort}: {path} missing", flush=True); continue
        donor, sex, ct, n_cells, syms = M.read_obs(path)
        auto = np.array([sym2chr.get(str(s)) in AUTOSOMES for s in syms])
        n_genes = len(syms)
        key_order0 = sorted({(d, c) for d, c in zip(donor, ct)})
        n_units = len(key_order0)
        est_gb = len(N_LEVELS + [UNCAPPED]) * n_units * int(auto.sum()) * 4 / 2**30
        print(f"[{cohort}] cells={n_cells:,} units={n_units} autosomal={int(auto.sum()):,} "
              f"-> per-seed matrices ~{est_gb:.1f} GB", flush=True)
        if est_gb > float(os.environ.get("M36_MAX_GB", "20")):
            print(f"  !! exceeds M36_MAX_GB; reduce caps or split further", flush=True)
        cts_u0 = np.array([k[1] for k in key_order0])
        donors_u0 = np.array([k[0] for k in key_order0])
        sex_of = pd.Series(sex).groupby(pd.Series(donor)).first()
        y_u0 = np.array([1 if str(sex_of[d]).lower() == "female" else 0 for d in donors_u0])
        seeds_all = [M.stable_seed(cohort, "m36seed", i) for i in range(N_SEEDS)]

        rows = []
        for si in range(SEED_LO, min(SEED_HI, N_SEEDS)):
            sd = seeds_all[si]
            unit, rank, key_order = M.unit_and_rank(n_cells, donor, ct, sex, sd)
            caps = list(N_LEVELS) + ([UNCAPPED] if INCLUDE_NATIVE else [])
            Mx = M.scatter_caps(path, unit, rank, caps, n_units, n_genes)
            kept = {C: np.bincount(unit[rank < C], minlength=n_units) for C in caps}
            tasks = []
            for C in caps:
                Lb = cpm_log(Mx[C])
                lib = Mx[C].sum(axis=1)
                for ctv in pd.unique(cts_u0):
                    m = cts_u0 == ctv
                    y = y_u0[m]
                    if min(np.bincount(y)) < MIN_RUN_DONORS_PER_SEX:
                        continue
                    ach = kept[C][m]
                    cov = float((ach >= C).mean()) if C != UNCAPPED else 1.0
                    if C != UNCAPPED and cov < CAP_COVERAGE:
                        continue
                    k = M._k_for(int(len(np.unique(donors_u0[m]))))
                    # Spread the null seeds evenly across shards. Using `si < NNULL`
                    # would pile every null onto the first couple of shards and make
                    # those the makespan bottleneck (~3 h vs ~40 min).
                    stride = max(1, N_SEEDS // max(1, N_NULL_SEEDS))
                    do_null = N_NULL_SEEDS > 0 and (si % stride == 0)
                    tasks.append(((Lb[np.ix_(m, auto)], y, donors_u0[m], k,
                                   do_null,
                                   M.stable_seed(cohort, ctv, C, si, "perm"), N_PERM),
                                  {"cohort": cohort, "cell_type": ctv, "cap_n": C,
                                   "cap_label": "native" if C == UNCAPPED else f"N{C}",
                                   "seed": int(sd), "k_pca": k,
                                   "n_units": int(len(np.unique(donors_u0[m]))),
                                   "n_donors": int(len(np.unique(donors_u0[m]))),
                                   "n_female": int((y == 1).sum()),
                                   "n_male": int((y == 0).sum()),
                                   "achieved_median_cells": float(np.median(ach)),
                                   "cap_coverage": round(cov, 3),
                                   "median_log_lib": round(float(np.median(np.log1p(lib[m]))), 3)}))
            del Mx, kept
            got = 0
            if tasks:
                with ProcessPoolExecutor(max_workers=min(WORKERS, len(tasks))) as ex:
                    futs = {ex.submit(M._job, t[0]): t[1] for t in tasks}
                    for f in as_completed(futs):
                        meta = futs[f]
                        auc, nulls = f.result()
                        if np.isnan(auc):
                            continue
                        rec = dict(meta)
                        rec.update({
                            "auc": round(auc, 5), "S": round(M.auc_strength(auc), 5),
                            "direction": M.auc_direction(auc), "null_n": len(nulls),
                            "null_median": round(float(np.median(nulls)), 5) if nulls else np.nan,
                            "null_p95": round(float(np.percentile(nulls, 95)), 5) if nulls else np.nan,
                        })
                        rows.append(rec); got += 1
            print(f"  [seed {si}] {len(tasks)} jobs -> {got} rows", flush=True)

        if rows:
            out = os.path.join(M.RES, f"m36_fixed_cells_{cohort}_part{SEED_LO}.csv")
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"[shard] wrote {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
