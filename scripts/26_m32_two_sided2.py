# -*- coding: utf-8 -*-
"""
M3.1 — Dual-cohort cell-type sex-signal analysis
Corrected statistical engine.

Major fixes:
 1. Two-sided permutation test:
      AUC_strength = max(AUC, 1 - AUC)
    so AUC=0.02 is correctly recognized as strong reverse-direction
    classification rather than a null result.

 2. Donor counting:
    n_female / n_male are counted as UNIQUE DONORS within each
    cell type, not pseudobulk rows.

 3. Strict autosome definition:
    only chr1-chr22 are retained.
    Unknown / NA chromosome mappings are NOT treated as autosomes.

 4. Independent RNG per cell type:
    permutation streams are deterministically seeded by cohort/cell type,
    avoiding identical RNG streams across multiprocessing workers.

 5. LODO:
    one pseudobulk row per donor × cell type, therefore leave-one-donor-out
    is valid within each cell type.

 6. Cross-cohort replication:
    signal strength is evaluated using |AUC-0.5|.
    Direction is reported separately.

Outputs:
    results/m31_celltypes.csv
    results/m31_summary.txt
"""

import os
import re
import warnings
import hashlib
import shutil
import tempfile

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


# ============================================================
# Configuration
# ============================================================

ROOT = r"F:\nature"
SCD = os.path.join(ROOT, "data", "singlecell")
RES = os.path.join(ROOT, "results")

# Final manuscript run should normally use:
#     set N_PERM=5000
#
# Default kept at 1000 for screening / debugging.
N_PERM = int(os.environ.get("N_PERM", "1000"))

# Minimum donors per sex required for confirmatory analysis.
MIN_DONORS_PER_SEX = int(os.environ.get("MIN_DONORS_PER_SEX", "12"))

# Minimum donors per sex required to even run a cell type.
MIN_RUN_DONORS_PER_SEX = int(os.environ.get("MIN_RUN_DONORS_PER_SEX", "4"))

# Strict autosomes only.
AUTOSOMES = {f"chr{i}" for i in range(1, 23)}


# ============================================================
# Utilities
# ============================================================

def stable_seed(*parts, base_seed=11):
    """
    Deterministic independent seed for each cohort/cell type.

    Important:
    Do not rely on a single global RandomState when using
    ProcessPoolExecutor. Independent jobs otherwise risk receiving
    identical RNG streams.
    """
    text = "|".join(map(str, parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    x = int.from_bytes(digest[:8], "little", signed=False)
    return int((x + base_seed) % (2**32 - 1))


def auc_strength(auc):
    """
    Direction-invariant predictive strength.

    AUC 0.5 = no discrimination
    AUC 1.0 = perfect forward discrimination
    AUC 0.0 = perfect reverse discrimination

    Therefore:
        strength = max(AUC, 1-AUC)
    """
    if np.isnan(auc):
        return np.nan
    return float(max(auc, 1.0 - auc))


def auc_direction(auc, tol=1e-12):
    """
    Direction of raw AUC.
    """
    if np.isnan(auc):
        return "NA"

    if auc > 0.5 + tol:
        return "forward"

    if auc < 0.5 - tol:
        return "reverse"

    return "chance"


# ============================================================
# Annotation
# ============================================================

def build_sym2chr():
    """
    Build gene-symbol -> chromosome mapping from GENCODE v39.

    Only chromosome labels chr1-chr22 will later be considered
    autosomal. Unknown mappings are excluded.
    """
    sym2chr = {}

    gtf = os.path.join(
        ROOT,
        "data",
        "annot",
        "gencode.v39.genes.gtf"
    )

    with open(gtf, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue

            p = line.rstrip("\n").split("\t")

            if len(p) < 9:
                continue

            if p[2] != "gene":
                continue

            m = re.search(r'gene_name "([^"]+)"', p[8])

            if not m:
                continue

            symbol = m.group(1)
            chrom = p[0]

            # Only store valid symbols.
            if symbol:
                sym2chr[symbol] = chrom

    return sym2chr


# ============================================================
# Single-cell input
# ============================================================

def pseudo_bulk(path):
    """
    Load h5ad and construct donor × cell-type pseudobulk.

    Returns:
        M       : pseudobulk count matrix
        donors  : donor ID for each row
        cts     : cell type for each row
        sexes   : donor sex for each row
        syms    : gene symbols
    """

    with h5py.File(path, "r") as hf:

        # ----------------------------
        # obs
        # ----------------------------
        _o = hf["obs"]

        def cats(n):
            c = [
                x.decode() if isinstance(x, bytes) else x
                for x in _o[n]["categories"][:]
            ]

            return pd.Categorical.from_codes(
                _o[n]["codes"][:],
                c
            )

        obs = pd.DataFrame({
            "donor": cats("donor_id"),
            "sex": cats("sex"),
            "ct": cats("cell_type")
        })

        # ----------------------------
        # expression matrix
        # ----------------------------
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

        n_cells, n_genes = (
            int(x)
            for x in _g.attrs["shape"]
        )

        X = sp.csr_matrix(
            (data, ind, ptr),
            shape=(n_cells, n_genes)
        )

        # ----------------------------
        # gene symbols
        # ----------------------------
        _v = hf["var"]

        if "gene_symbols" in _v or "feature_name" in _v:

            k = (
                "gene_symbols"
                if "gene_symbols" in _v
                else "feature_name"
            )

            vc = [
                x.decode() if isinstance(x, bytes) else x
                for x in _v[k]["categories"][:]
            ]

            syms = np.array([
                vc[c]
                for c in _v[k]["codes"][:]
            ])

        else:

            syms = np.array([
                x.decode() if isinstance(x, bytes) else x
                for x in _v["_index"][:]
            ])

    # ========================================================
    # Sex encoding
    # ========================================================

    smap = {
        "female": 1,
        "male": 0
    }

    obs["sex_i"] = (
        obs["sex"]
        .astype(str)
        .str.lower()
        .map(smap)
    )

    keep = obs.sex_i.notna().to_numpy()

    obs = obs[
        keep
    ].reset_index(drop=True)

    # NOTE: do NOT subset X rows here — sparse fancy-index on 46k x 36k
    # allocates a full copy (~2.7GB). Groupby below uses positional indices
    # into the ORIGINAL matrix, with the mapping rebuilt via np.where(keep).

    # ========================================================
    # Pseudobulk:
    # one row = donor × cell type
    # ========================================================

    keys = []
    rows = []

    groups = obs.groupby(
        ["donor", "ct"],
        observed=True
    ).groups

    # positional indices in the ORIGINAL cell order
    orig_pos = np.where(keep)[0]

    for (d, ct), gidx in groups.items():

        idx = orig_pos[
            np.asarray(
                list(gidx),
                dtype=int
            )
        ]

        rows.append(
            np.asarray(
                X[idx].sum(axis=0)
            ).ravel()
        )

        keys.append(
            (d, ct)
        )

    sex_lu = (
        obs
        .drop_duplicates("donor")
        .set_index("donor")["sex_i"]
    )

    M = np.vstack(rows)

    donors = np.array([
        k[0]
        for k in keys
    ])

    cts = np.array([
        k[1]
        for k in keys
    ])

    sexes = np.array([
        sex_lu[k[0]]
        for k in keys
    ], dtype=int)

    return (
        M,
        donors,
        cts,
        sexes,
        syms
    )


# ============================================================
# Normalization
# ============================================================

def cpm_log(M):
    """
    CPM + log2 transform.
    """
    lib = M.sum(
        axis=1,
        keepdims=True
    )

    return np.log2(
        M /
        np.maximum(lib, 1)
        * 1e6
        + 1.0
    ).astype(
        np.float32
    )


# ============================================================
# LODO
# ============================================================

def lodo_auc(Yg, y, donors):
    """
    Leave-one-donor-out.

    Within a single cell type, each donor contributes exactly
    one pseudobulk row. Therefore leave-one-row-out is equivalent
    to leave-one-donor-out.

    Predictions are pooled first, and ONE AUC is calculated
    across all held-out donors.
    """

    donors = np.asarray(donors)

    unique_donors = pd.unique(donors)

    preds = np.full(
        len(y),
        np.nan,
        dtype=float
    )

    for d in unique_donors:

        te = donors == d
        tr = ~te

        # Need both classes in training data.
        if len(np.unique(y[tr])) < 2:
            continue

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                max_iter=2000
            )
        )

        clf.fit(
            Yg[tr],
            y[tr]
        )

        preds[te] = clf.predict_proba(
            Yg[te]
        )[:, 1]

    ok = ~np.isnan(preds)

    if ok.sum() < 2:
        return np.nan, preds

    if len(np.unique(y[ok])) < 2:
        return np.nan, preds

    auc = roc_auc_score(
        y[ok],
        preds[ok]
    )

    return float(auc), preds


# ============================================================
# Two-sided donor permutation
# ============================================================

def perm_p_lodo(
    Yg,
    y,
    donors,
    n_perm,
    obs_auc,
    seed
):
    """
    Two-sided permutation test.

    Test statistic:

        S = max(AUC, 1-AUC)

    This treats AUC=0.0 and AUC=1.0 as equally strong
    discrimination with opposite directions.

    Donor labels are permuted at donor level.

    P-value:

        (hits + 1) / (n_perm + 1)

    """

    rng = np.random.RandomState(
        seed
    )

    donors = np.asarray(donors)
    y = np.asarray(y)

    ud = pd.unique(
        donors
    )

    # --------------------------------------------------------
    # Donor -> observed sex
    # --------------------------------------------------------

    donor_sex = {}

    for d in ud:

        yy = y[
            donors == d
        ]

        if len(np.unique(yy)) != 1:
            raise ValueError(
                f"Donor {d} has inconsistent sex labels."
            )

        donor_sex[d] = int(
            yy[0]
        )

    observed_strength = auc_strength(
        obs_auc
    )

    hits = 0

    sex_values = np.array([
        donor_sex[d]
        for d in ud
    ])

    for _ in range(
        n_perm
    ):

        permuted_sex = rng.permutation(
            sex_values
        )

        perm_map = dict(
            zip(
                ud,
                permuted_sex
            )
        )

        yp = np.array([
            perm_map[d]
            for d in donors
        ])

        # If permutation accidentally creates only one class,
        # skip it. This is unlikely when both sexes have enough
        # donors, but makes the function robust.
        if len(np.unique(yp)) < 2:
            continue

        auc_perm, _ = lodo_auc(
            Yg,
            yp,
            donors
        )

        if np.isnan(auc_perm):
            continue

        perm_strength = auc_strength(
            auc_perm
        )

        if perm_strength >= observed_strength:
            hits += 1

    return (
        hits + 1
    ) / (
        n_perm + 1
    )


# ============================================================
# Worker
# ============================================================

def _one_ct_file(args):
    """
    Worker process.

    Args:
        fn
        n_perm
        seed
    """

    fn, n_perm, seed = args

    d = np.load(
        fn,
        allow_pickle=True
    )

    obj = d.item()

    Yg = obj["Yg"]
    y = obj["y"]
    donors = obj["g"]

    auc, _ = lodo_auc(
        Yg,
        y,
        donors
    )

    if np.isnan(auc):
        return None

    strength = auc_strength(
        auc
    )

    direction = auc_direction(
        auc
    )

    p_emp = perm_p_lodo(
        Yg,
        y,
        donors,
        n_perm,
        auc,
        seed
    )

    return (
        auc,
        strength,
        direction,
        p_emp
    )


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        RES,
        exist_ok=True
    )

    print(
        "============================================================",
        flush=True
    )

    print(
        "M3.1 — corrected dual-cohort cell-type analysis",
        flush=True
    )

    print(
        f"N_PERM={N_PERM}",
        flush=True
    )

    print(
        f"MIN_DONORS_PER_SEX={MIN_DONORS_PER_SEX}",
        flush=True
    )

    print(
        "Two-sided statistic: AUC_strength=max(AUC,1-AUC)",
        flush=True
    )

    print(
        "Autosomes: chr1-chr22 ONLY",
        flush=True
    )

    print(
        "============================================================",
        flush=True
    )

    # --------------------------------------------------------
    # Annotation
    # --------------------------------------------------------

    sym2chr = build_sym2chr()

    print(
        f"Loaded GENCODE symbol→chromosome map: "
        f"{len(sym2chr):,} genes",
        flush=True
    )

    # --------------------------------------------------------
    # Cohorts
    # --------------------------------------------------------

    import os as _os
    _only = _os.environ.get("ONLY_COHORT", "")
    cohorts = [
        (
            "KD_LivingDonor",
            os.path.join(
                SCD,
                "KD_LivingDonor.h5ad"
            )
        ),
        (
            "KD_Mature",
            os.path.join(
                SCD,
                "KD_Mature.h5ad"
            )
        )
    ]
    if _only:
        cohorts = [c for c in cohorts if c[0] == _only]

    pbmc = os.path.join(
        SCD,
        "PBMC_Indonesia.h5ad"
    )

    if (
        os.path.exists(pbmc)
        and os.path.getsize(pbmc) > 3e9
    ):
        cohorts.append(
            (
                "PBMC_Indonesia",
                pbmc
            )
        )

    rows = []

    # ========================================================
    # Cohort loop
    # ========================================================

    for cohort, path in cohorts:

        print(
            f"\n=== {cohort} ===",
            flush=True
        )

        M, donors, cts, sexes, syms = pseudo_bulk(
            path
        )

        # ----------------------------------------------------
        # STRICT autosomes
        # ----------------------------------------------------

        chromosomes = np.array([
            sym2chr.get(
                str(s),
                None
            )
            for s in syms
        ], dtype=object)

        auto = np.array([
            c in AUTOSOMES
            for c in chromosomes
        ])

        n_unknown = int(
            pd.isna(chromosomes).sum()
        )

        print(
            f"    genes={len(syms):,}",
            flush=True
        )

        print(
            f"    autosomal={auto.sum():,}",
            flush=True
        )

        print(
            f"    unknown_chr={n_unknown:,}",
            flush=True
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        L = cpm_log(
            M
        )

        total_donors = len(
            pd.unique(donors)
        )

        total_female = len(
            pd.unique(
                donors[
                    sexes == 1
                ]
            )
        )

        total_male = len(
            pd.unique(
                donors[
                    sexes == 0
                ]
            )
        )

        print(
            f"    pseudobulk units={M.shape[0]:,}",
            flush=True
        )

        print(
            f"    unique donors={total_donors}",
            flush=True
        )

        print(
            f"    donors: F={total_female} "
            f"M={total_male}",
            flush=True
        )

        tmpdir = tempfile.mkdtemp(
            prefix="m31_"
        )

        jobs = []

        # ====================================================
        # Cell-type staging
        # ====================================================

        for ct in pd.unique(cts):

            m = (
                cts == ct
            )

            # ------------------------------------------------
            # IMPORTANT:
            # Count UNIQUE DONORS, not pseudobulk rows.
            # ------------------------------------------------

            ct_donors = donors[m]
            ct_sexes = sexes[m]

            female_donors = pd.unique(
                ct_donors[
                    ct_sexes == 1
                ]
            )

            male_donors = pd.unique(
                ct_donors[
                    ct_sexes == 0
                ]
            )

            n_f = len(
                female_donors
            )

            n_m = len(
                male_donors
            )

            n_d = len(
                pd.unique(ct_donors)
            )

            # ------------------------------------------------
            # Minimum donor requirement
            # ------------------------------------------------

            if min(
                n_f,
                n_m
            ) < MIN_RUN_DONORS_PER_SEX:

                print(
                    f"    SKIP {ct[:45]:45s} "
                    f"F={n_f} M={n_m} "
                    f"(too few donors)",
                    flush=True
                )

                continue

            # ------------------------------------------------
            # Design label
            # ------------------------------------------------

            design = (
                "confirmatory"
                if min(
                    n_f,
                    n_m
                ) >= MIN_DONORS_PER_SEX
                else "exploratory"
            )

            # ------------------------------------------------
            # Permutation count
            #
            # Exploratory cohorts can use fewer permutations
            # during screening.
            # ------------------------------------------------

            if design == "confirmatory":
                n_perm = N_PERM
            else:
                n_perm = min(
                    N_PERM,
                    500
                )

            # ------------------------------------------------
            # Data
            # ------------------------------------------------

            Yg = L[
                np.ix_(
                    m,
                    auto
                )
            ]

            safe_ct = re.sub(
                r"[^A-Za-z0-9_]",
                "_",
                str(ct)
            )[:60]

            fn = os.path.join(
                tmpdir,
                safe_ct + ".npy"
            )

            np.save(
                fn,
                {
                    "Yg": Yg,
                    "y": ct_sexes,
                    "g": ct_donors
                }
            )

            seed = stable_seed(
                cohort,
                ct,
                base_seed=11
            )

            jobs.append(
                (
                    ct,
                    fn,
                    n_perm,
                    n_d,
                    n_f,
                    n_m,
                    design,
                    seed
                )
            )

        print(
            f"    {len(jobs)} cell types staged",
            flush=True
        )

        # ====================================================
        # Multiprocessing
        # ====================================================

        from concurrent.futures import (
            ProcessPoolExecutor
        )

        # PBMC: per-worker PEAK (data+sklearn copies) ~800MB; 3 workers = 2.4GB safe
        n_w = 3 if ('Indonesia' in cohort or 'PBMC' in cohort) else 8

        with ProcessPoolExecutor(
            max_workers=n_w
        ) as ex:

            futs = {}

            for (
                ct,
                fn,
                n_perm,
                nd,
                nf,
                nm,
                design,
                seed
            ) in jobs:

                fut = ex.submit(
                    _one_ct_file,
                    (
                        fn,
                        n_perm,
                        seed
                    )
                )

                futs[fut] = (
                    ct,
                    n_perm,
                    nd,
                    nf,
                    nm,
                    design,
                    seed
                )

            for fut in futs:

                (
                    ct,
                    nperm,
                    nd,
                    nf,
                    nm,
                    design,
                    seed
                ) = futs[fut]

                try:

                    res = fut.result()

                except Exception as e:

                    print(
                        f"    ERROR {ct}: {e}",
                        flush=True
                    )

                    continue

                if res is None:
                    continue

                (
                    auc,
                    strength,
                    direction,
                    p_emp
                ) = res

                row = {
                    "cohort": cohort,
                    "cell_type": ct,

                    "n_donors": nd,
                    "n_female": nf,
                    "n_male": nm,

                    "design": design,

                    "auc_lodo": round(
                        auc,
                        6
                    ),

                    "auc_strength": round(
                        strength,
                        6
                    ),

                    "direction": direction,

                    "perm_n": nperm,

                    "perm_p_two_sided": round(
                        p_emp,
                        6
                    ),

                    "perm_seed": seed
                }

                rows.append(
                    row
                )

                print(
                    f"    {ct[:40]:40s} "
                    f"donors={nd:4d} "
                    f"F={nf:3d} "
                    f"M={nm:3d} "
                    f"AUC={auc:.3f} "
                    f"S={strength:.3f} "
                    f"dir={direction:7s} "
                    f"p({nperm})={p_emp:.4f} "
                    f"[{design}]",
                    flush=True
                )

        shutil.rmtree(
            tmpdir,
            ignore_errors=True
        )

    # ========================================================
    # Save cell-type results
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    out_csv = os.path.join(
        RES,
        "m31_celltypes.csv"
    )

    df.to_csv(
        out_csv,
        index=False
    )

    # ========================================================
    # Summary
    # ========================================================

    lines = []

    lines.append(
        "M3.1 corrected dual-cohort analysis"
    )

    lines.append(
        f"N_PERM={N_PERM}"
    )

    lines.append(
        "Permutation statistic: "
        "S=max(AUC,1-AUC)"
    )

    lines.append(
        "Autosome definition: chr1-chr22 only"
    )

    lines.append(
        "Donor counts: unique donor IDs"
    )

    # --------------------------------------------------------
    # KD replication
    # --------------------------------------------------------

    if not df.empty:

        a = df[
            df.cohort ==
            "KD_LivingDonor"
        ].set_index(
            "cell_type"
        )

        b = df[
            df.cohort ==
            "KD_Mature"
        ].set_index(
            "cell_type"
        )

        common = a.index.intersection(
            b.index
        )

        rep_same_direction = []
        rep_any_direction = []

        for ct in common:

            auc_a = float(
                a.loc[
                    ct,
                    "auc_lodo"
                ]
            )

            auc_b = float(
                b.loc[
                    ct,
                    "auc_lodo"
                ]
            )

            s_a = auc_strength(
                auc_a
            )

            s_b = auc_strength(
                auc_b
            )

            strong = (
                s_a >= 0.60
                and
                s_b >= 0.60
            )

            direction_a = auc_direction(
                auc_a
            )

            direction_b = auc_direction(
                auc_b
            )

            if strong:

                rep_any_direction.append(
                    ct
                )

                if (
                    direction_a ==
                    direction_b
                ):

                    rep_same_direction.append(
                        ct
                    )

        lines.append(
            f"  KD common cell types: "
            f"{len(common)}"
        )

        lines.append(
            f"  KD strong replication "
            f"(strength>=0.60): "
            f"{len(rep_any_direction)}/"
            f"{len(common)}"
        )

        lines.append(
            f"  KD same-direction replication: "
            f"{len(rep_same_direction)}/"
            f"{len(common)}"
        )

        if rep_same_direction:

            lines.append(
                "  same-direction types: "
                +
                ", ".join(
                    rep_same_direction[:20]
                )
            )

    # --------------------------------------------------------
    # Confirmatory cohort summary
    # --------------------------------------------------------

    if not df.empty:

        conf = df[
            df.design ==
            "confirmatory"
        ]

        if len(conf):

            for cohort, sub in conf.groupby(
                "cohort"
            ):

                # Signal-positive criterion:
                #
                #   permutation significant
                #   AND
                #   direction-invariant strength >= 0.65
                #
                pos = int(
                    (
                        (
                            sub[
                                "perm_p_two_sided"
                            ] < 0.05
                        )
                        &
                        (
                            sub[
                                "auc_strength"
                            ] >= 0.65
                        )
                    ).sum()
                )

                lines.append(
                    f"  {cohort}: "
                    f"{pos}/{len(sub)} "
                    f"confirmatory-positive "
                    f"(p<0.05 & strength>=0.65)"
                )

        else:

            lines.append(
                "  No confirmatory cell types "
                "in this run."
            )

    # --------------------------------------------------------
    # Exploratory summary
    # --------------------------------------------------------

    if not df.empty:

        exp = df[
            df.design ==
            "exploratory"
        ]

        lines.append(
            f"  Exploratory results: "
            f"{len(exp)} cell-type/cohort analyses"
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    txt = "\n".join(
        lines
    )

    print(
        "\n" + txt,
        flush=True
    )

    out_txt = os.path.join(
        RES,
        "m31_summary.txt"
    )

    with open(
        out_txt,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            txt
        )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()