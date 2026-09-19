# -*- coding: utf-8 -*-
"""Experiment 6: External validation on TCGA normal solid tissue.

Cross-platform test: train L2-logistic on GTEx (donor-split, autosome top-500
DE genes per tissue), test on TCGA *normal* tissue STAR counts (TMM-free:
log2(TPM+1) computed identically). Platform/batch differ entirely from GTEx,
so this is a true external-cohort check.

GTEx->TCGA normal mapping (closest tissue analogs, sex-balanced projects):
  Blood      -> (skip: no TCGA blood normal)
  Muscle     -> (skip)
  Lung       -> TCGA-LUAD + TCGA-LUSC normals
  Thyroid    -> TCGA-THCA normals
  Skin       -> TCGA-BRCA normals (skin-anchor breast; also largest F cohort)
  Artery     -> (skip)
  AdiposeSubq-> TCGA-BRCA normals (adipose-rich tissue)
  Nerve      -> (skip)
  Heart      -> (skip)
  Colon      -> TCGA-COAD normals
  Esophagus  -> TCGA-ESCA normals
  AdiposeVisc-> TCGA-UCEC normals (omentum-adjacent)

Sex labels: demographic.sex_at_birth (male/female).
Data: GDC API /data endpoint, STAR 2-pass gene counts -> TPM via gene lengths
(gencode v36 lengths shipped in GDC star_counts header not available; we use
TPM = counts / gene_len_kb / library_scale computed with GDC gene lengths from
the gdc gene reference file).

To keep memory/time bounded: fetch only the top-500 GTEx genes per model.
"""
import os, io, json, gzip, time, urllib.request, urllib.parse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = r"F:\nature"
PROC = os.path.join(ROOT, "data", "proc")
RES = os.path.join(ROOT, "results")
PROXY = {"https": "socks5h://127.0.0.1:7890", "http": "socks5h://127.0.0.1:7890"}
opener = urllib.request.build_opener(urllib.request.ProxyHandler(PROXY))

API = "https://api.gdc.cancer.gov"

PAIRS = {
    "Thyroid":   ["TCGA-THCA"],
    "Lung":      ["TCGA-LUAD", "TCGA-LUSC"],
    "Colon":     ["TCGA-COAD"],
    "Esophagus": ["TCGA-ESCA"],
    "Skin":      ["TCGA-BRCA"],
    "AdiposeSubq": ["TCGA-BRCA"],
    "AdiposeVisc": ["TCGA-UCEC"],
}

def gdc_json(path, params=None, tries=5):
    url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
    for i in range(tries):
        try:
            with opener.open(url, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"    retry {i+1} ({e})", flush=True); time.sleep(4)
    raise RuntimeError(url)

def fetch_normal_samples(projects):
    """Return list of (case_id, sample_id, gender, project)."""
    out = []
    flt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "samples.sample_type",
                                 "value": ["Solid Tissue Normal"]}},
        {"op": "in", "content": {"field": "project.project_id", "value": projects}},
    ]}
    params = {"filters": json.dumps(flt),
              "fields": "samples.submitter_id,demographic.sex_at_birth,project.project_id",
              "size": "500", "from": "0", "format": "JSON"}
    while True:
        d = gdc_json("/cases", params)
        for h in d["data"]["hits"]:
            g = h.get("demographic", {}).get("sex_at_birth", "")
            for s in h.get("samples", []):
                out.append((h["id"], s["submitter_id"], g, h["project"]["project_id"]))
        pg = d["data"]["pagination"]
        if pg["from"] + pg["count"] >= pg["total"]:
            break
        params["from"] = str(pg["from"] + pg["count"])
    return out

def fetch_star_counts(sample_ids):
    """Find STAR count files for given samples; return {sample: file_uuid}."""
    res = {}
    for i in range(0, len(sample_ids), 20):
        chunk = sample_ids[i:i+20]
        flt = {"op": "and", "content": [
            {"op": "in", "content": {"field": "cases.samples.submitter_id", "value": chunk}},
            {"op": "in", "content": {"field": "files.data_type",
                                     "value": ["Gene Expression Quantification"]}},
        ]}
        d = gdc_json("/files", {"filters": json.dumps(flt),
                                "fields": "file_id,file_name,cases.samples.submitter_id",
                                "size": "500", "format": "JSON"})
        for h in d["data"]["hits"]:
            if "rna_seq.augmented_star_gene_counts" not in h.get("file_name", ""):
                continue
            for c in h.get("cases", []):
                for s in c.get("samples", []):
                    sid = s.get("submitter_id")
                    if sid in set(chunk) and sid not in res:
                        res[sid] = h["file_id"]
    return res

def download_counts(uuid, gene_list):
    """Download a STAR counts tsv; return DataFrame [gene, tpm] for gene_list."""
    url = f"{API}/data/{uuid}"
    for i in range(5):
        try:
            with opener.open(url, timeout=300) as r:
                raw = r.read()
            # columns: gene_id gene_name gene_type unstranded stranded_first
            #          stranded_second tpm_unstranded fpkm fpkm_uq
            df = pd.read_csv(io.BytesIO(raw), sep="\t", comment="#",
                             header=None,
                             names=["gene", "name", "type", "cts", "sf", "ss",
                                    "tpm", "fpkm", "fpkmuq"],
                             usecols=[0, 6], on_bad_lines="skip")
            df = df[df.gene.str.startswith("ENSG", na=False)]
            df = df[df.gene.isin(gene_list)]
            return df
        except Exception as e:
            print(f"    dl retry {i+1}: {str(e)[:80]}", flush=True); time.sleep(5)
    return None

def main():
    # ---- GTEx models ----
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]
    models = {}
    for t in PAIRS:
        X = pd.read_parquet(os.path.join(PROC, f"tpm_{t}.parquet"))
        X = X.loc[:, (X >= 1.0).mean(axis=0) >= 0.20]
        de = pd.read_csv(os.path.join(RES, f"de_{t}.csv"))
        de_auto = de[de.chr.isin([f"chr{i}" for i in range(1, 23)])]
        top = de_auto.reindex(de_auto.log2FC_adj.abs().sort_values(
            ascending=False).index).head(500)["gene"].to_numpy()
        top = np.intersect1d(top, X.columns)
        Y = np.log2(X[top].to_numpy(dtype=np.float32) + 1.0)
        y = meta.loc[X.index, "sex"].to_numpy(dtype=int)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(Y, y)
        models[t] = {"genes": top, "clf": clf}
        print(f"[gtex model] {t}: {len(top)} autosomal DE genes", flush=True)

    results = []
    for gtex_t, projects in PAIRS.items():
        print(f"\n=== GTEx {gtex_t} -> {projects} ===", flush=True)
        genes = models[gtex_t]["genes"]; clf = models[gtex_t]["clf"]
        cases = fetch_normal_samples(projects)
        # one entry per sample (gender from case)
        samples = [(sid, g) for _, sid, g, _ in cases if g in ("male", "female")]
        print(f"    normal samples: {len(samples)}", flush=True)
        smap = fetch_star_counts([s for s, _ in samples])
        print(f"    star count files: {len(smap)}", flush=True)
        mat = {}
        for sid, g in samples:
            if sid not in smap: continue
            df = download_counts(smap[sid], set(genes.tolist()))
            if df is None or df.empty: continue
            df = df.groupby("gene", as_index=False)["tpm"].sum()
            mat[sid] = (g, df.set_index("gene")["tpm"])
        if len(mat) < 20:
            print("    too few files, skipping"); continue
        # build matrix
        ids = list(mat.keys())
        M = pd.DataFrame({sid: mat[sid][1] for sid in ids}).T
        M = M.reindex(columns=genes).fillna(0.0)
        ytc = np.array([mat[s][0] == "female" for s in ids]).astype(int)
        Xtc = np.log2(M.to_numpy(dtype=np.float32) + 1.0)
        p = clf.predict_proba(Xtc)[:, 1]
        auc = roc_auc_score(ytc, p)
        auprc = average_precision_score(ytc, p)
        bacc = (((p > .5)[ytc == 1].mean()) + ((p <= .5)[ytc == 0].mean())) / 2
        results.append({"gtex_train": gtex_t, "tcga_projects": ",".join(projects),
                        "n_test": len(ids), "n_female": int(ytc.sum()),
                        "auc": round(auc, 4), "auprc": round(auprc, 4),
                        "bal_acc": round(bacc, 4)})
        print(f"    -> GTEx{gtex_t} vs TCGA-normal: AUC={auc:.3f} AUPRC={auprc:.3f} bACC={bacc:.3f}", flush=True)

    pd.DataFrame(results).to_csv(os.path.join(RES, "exp6_tcga_validation.csv"), index=False)
    print("\n[done] exp6 complete", flush=True)

if __name__ == "__main__":
    main()
