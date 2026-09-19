# -*- coding: utf-8 -*-
"""Exp6 v2: TCGA external validation with curl-based parallel downloads.

Fixes vs v1:
- Downloads via curl subprocess (handles socks5 + retries robustly, resumable)
- Thread pool (8 workers) -> fast even for 1500+ files
- Files cached to disk; re-runs skip existing files
- Same GTEx models as v1 (top-500 autosomal DE genes, donor-split training)
- TPM column used directly (GDC augmented star counts provide tpm_unstranded)
"""
import os, io, json, time, subprocess, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = r"F:\nature"
PROC = os.path.join(ROOT, "data", "proc")
RES = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "data", "tcga_cache")
os.makedirs(CACHE, exist_ok=True)
API = "https://api.gdc.cancer.gov"
PROXY = "socks5h://127.0.0.1:7890"

PAIRS = {
    "Thyroid":   (["TCGA-THCA"], "Thyroid"),
    "Lung":      (["TCGA-LUAD", "TCGA-LUSC"], "Lung"),
    "Colon":     (["TCGA-COAD"], "Colon"),
    "Esophagus": (["TCGA-ESCA"], "Esophagus"),
    "Skin":      (["TCGA-BRCA"], "Skin"),
    "AdiposeSubq": (["TCGA-BRCA"], "AdiposeSubq"),
    "AdiposeVisc": (["TCGA-UCEC"], "AdiposeVisc"),
}

def gdc_json(path, params=None, tries=6):
    url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
    for i in range(tries):
        try:
            cmd = ["curl", "-s", "--max-time", "120", "--proxy", PROXY, url]
            out = subprocess.run(cmd, capture_output=True, timeout=150).stdout
            return json.loads(out.decode())
        except Exception as e:
            time.sleep(3)
    raise RuntimeError(url)

def fetch_normal_samples(projects):
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
                out.append((h["id"], s["submitter_id"], g))
        pg = d["data"]["pagination"]
        if pg["from"] + pg["count"] >= pg["total"]:
            break
        params["from"] = str(pg["from"] + pg["count"])
    return out

def fetch_star_counts(sample_ids):
    """Return {sample_uuid_submitter: file_uuid} for augmented star counts."""
    res = {}
    chunks = [sample_ids[i:i+40] for i in range(0, len(sample_ids), 40)]
    for chunk in chunks:
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
        # paginate if needed
        pg = d["data"]["pagination"]
        while pg["from"] + pg["count"] < pg["total"]:
            params = {"filters": json.dumps(flt),
                      "fields": "file_id,file_name,cases.samples.submitter_id",
                      "size": "500", "from": str(pg["from"] + pg["count"]),
                      "format": "JSON"}
            d = gdc_json("/files", params)
            for h in d["data"]["hits"]:
                if "rna_seq.augmented_star_gene_counts" not in h.get("file_name", ""):
                    continue
                for c in h.get("cases", []):
                    for s in c.get("samples", []):
                        sid = s.get("submitter_id")
                        if sid in set(chunk) and sid not in res:
                            res[sid] = h["file_id"]
            pg = d["data"]["pagination"]
    return res

def curl_download(uuid, dst):
    """Download GDC data file via curl (cached)."""
    if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
        return True
    url = f"{API}/data/{uuid}"
    for i in range(4):
        rc = subprocess.run(["curl", "-s", "--max-time", "600", "--proxy", PROXY,
                             "-o", dst, url]).returncode
        if rc == 0 and os.path.exists(dst) and os.path.getsize(dst) > 10_000:
            return True
        time.sleep(2)
    return False

def extract_tpm(path, gene_set):
    """Read star counts tsv, return {gene: tpm} for genes in gene_set."""
    try:
        df = pd.read_csv(path, sep="\t", comment="#", header=None,
                         names=["gene", "name", "type", "cts", "sf", "ss",
                                "tpm", "fpkm", "fpkmuq"],
                         usecols=[0, 6], on_bad_lines="skip", engine="c")
        df = df[df.gene.str.startswith("ENSG", na=False) & df.gene.isin(gene_set)]
        return dict(zip(df.gene, df.tpm))
    except Exception:
        return None

def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    # ---------- GTEx models ----------
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
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(Y, y)
        models[t] = {"genes": set(top), "glist": top, "clf": clf}
        print(f"[gtex model] {t}: {len(top)} genes", flush=True)

    results = []
    for gtex_t, (projects, _) in PAIRS.items():
        print(f"\n=== GTEx {gtex_t} -> {projects} ===", flush=True)
        mdl = models[gtex_t]
        cases = fetch_normal_samples(projects)
        samples = [(sid, g) for _, sid, g in cases if g in ("male", "female")]
        print(f"    normal samples: {len(samples)}", flush=True)
        smap = fetch_star_counts([s for s, _ in samples])
        print(f"    star files: {len(smap)}", flush=True)

        # parallel download
        todo = [(sid, uuid) for sid, uuid in smap.items()]
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(curl_download, uuid,
                              os.path.join(CACHE, uuid + ".tsv")): sid
                    for sid, uuid in todo}
            done = 0
            for f in futs:
                f.result(); done += 1
                if done % 100 == 0:
                    print(f"    downloaded {done}/{len(todo)}", flush=True)
        print(f"    download complete: {done}", flush=True)

        sex_lookup = dict(samples)
        mat, sexes = {}, {}
        for sid, uuid in todo:
            path = os.path.join(CACHE, uuid + ".tsv")
            tpms = extract_tpm(path, mdl["genes"])
            if not tpms: continue
            mat[sid] = tpms
            sexes[sid] = sex_lookup[sid]
        if len(mat) < 20:
            print("    too few extracted, skipping"); continue
        ids = list(mat)
        M = pd.DataFrame(mat).T.reindex(columns=mdl["glist"]).fillna(0.0)
        ytc = np.array([sexes[s] == "female" for s in ids]).astype(int)
        Xtc = np.log2(M.to_numpy(dtype=np.float32) + 1.0)
        p = mdl["clf"].predict_proba(Xtc)[:, 1]
        auc = roc_auc_score(ytc, p)
        auprc = average_precision_score(ytc, p)
        bacc = (((p > .5)[ytc == 1].mean()) + ((p <= .5)[ytc == 0].mean())) / 2
        results.append({"gtex_train": gtex_t, "tcga": ",".join(projects),
                        "n_test": len(ids), "n_female": int(ytc.sum()),
                        "auc": round(auc, 4), "auprc": round(auprc, 4),
                        "bal_acc": round(bacc, 4)})
        print(f"    -> AUC={auc:.3f} AUPRC={auprc:.3f} bACC={bacc:.3f} (n={len(ids)}, F={ytc.sum()})", flush=True)

    pd.DataFrame(results).to_csv(os.path.join(RES, "exp6_tcga_validation.csv"), index=False)
    print("\n[done] exp6 v2 complete", flush=True)

if __name__ == "__main__":
    main()
