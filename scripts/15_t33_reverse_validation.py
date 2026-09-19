# -*- coding: utf-8 -*-
"""T3-3: Reverse external validation TCGA -> GTEx (chatGPT item 13).

Train on TCGA normal tissue (cached star counts, tpm_unstranded), donor-level
split within TCGA; test on GTEx tissue samples from donors NOT in TCGA training
(by construction disjoint - different cohorts entirely).

Design per mapping (mirror of exp6):
  TCGA-THCA normals -> GTEx Thyroid
  TCGA-LUAD/LUSC    -> GTEx Lung
  TCGA-COAD         -> GTEx Colon
  TCGA-ESCA         -> GTEx Esophagus
  TCGA-BRCA         -> GTEx Skin  (single-sex issue: keep but flag)
Features: top-500 autosomal DE genes computed ON TCGA TRAIN DONORS ONLY.

Output: results/t3_3_reverse_validation.csv
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
API = "https://api.gdc.cancer.gov"
PROXY = "socks5h://127.0.0.1:7890"

PAIRS = {
    "Thyroid": (["TCGA-THCA"], "Thyroid"),
    "Lung":    (["TCGA-LUAD", "TCGA-LUSC"], "Lung"),
    "Colon":   (["TCGA-COAD"], "Colon"),
    "Esophagus": (["TCGA-ESCA"], "Esophagus"),
    "Skin":    (["TCGA-BRCA"], "Skin"),
}

def gdc_json(path, params=None, tries=6):
    url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
    for _ in range(tries):
        try:
            out = subprocess.run(["curl", "-s", "--max-time", "120", "--proxy",
                                  PROXY, url], capture_output=True,
                                 timeout=150).stdout
            return json.loads(out.decode())
        except Exception:
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
              "fields": "samples.submitter_id,demographic.sex_at_birth",
              "size": "500", "from": "0", "format": "JSON"}
    while True:
        d = gdc_json("/cases", params)
        for h in d["data"]["hits"]:
            g = h.get("demographic", {}).get("sex_at_birth", "")
            for s in h.get("samples", []):
                out.append((s["submitter_id"], g))
        pg = d["data"]["pagination"]
        if pg["from"] + pg["count"] >= pg["total"]:
            break
        params["from"] = str(pg["from"] + pg["count"])
    return out

def fetch_star_counts(sample_ids):
    res = {}
    for i in range(0, len(sample_ids), 40):
        chunk = sample_ids[i:i+40]
        flt = {"op": "and", "content": [
            {"op": "in", "content": {"field": "cases.samples.submitter_id", "value": chunk}},
            {"op": "in", "content": {"field": "files.data_type",
                                     "value": ["Gene Expression Quantification"]}},
        ]}
        params = {"filters": json.dumps(flt),
                  "fields": "file_id,file_name,cases.samples.submitter_id",
                  "size": "500", "from": "0", "format": "JSON"}
        while True:
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
            if pg["from"] + pg["count"] >= pg["total"]:
                break
            params["from"] = str(pg["from"] + pg["count"])
    return res

def extract_tpm(path, gene_set):
    try:
        df = pd.read_csv(path, sep="\t", comment="#", header=None,
                         names=["gene", "name", "type", "cts", "sf", "ss",
                                "tpm", "fpkm", "fpkmuq"],
                         usecols=[0, 6], on_bad_lines="skip", engine="c")
        df = df[df.gene.str.startswith("ENSG", na=False) & df.gene.isin(gene_set)]
        return dict(zip(df.gene, df.tpm))
    except Exception:
        return None

def curl_download(uuid, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
        return True
    url = f"{API}/data/{uuid}"
    for _ in range(4):
        rc = subprocess.run(["curl", "-s", "--max-time", "600", "--proxy", PROXY,
                             "-o", dst, url]).returncode
        if rc == 0 and os.path.exists(dst) and os.path.getsize(dst) > 10_000:
            return True
        time.sleep(2)
    return False

def main():
    meta = pd.read_csv(os.path.join(PROC, "sample_meta.csv"), index_col=0)
    gene2chr = pd.read_csv(os.path.join(PROC, "gene2chr.csv"), index_col=0)["chr"]
    results = []
    for tkey, (projects, gtex_t) in PAIRS.items():
        print(f"\n=== TCGA {projects} -> GTEx {gtex_t} ===", flush=True)
        # ---- TCGA side ----
        cases = fetch_normal_samples(projects)
        samples = [(s, g) for s, g in cases if g in ("male", "female")]
        smap = fetch_star_counts([s for s, _ in samples])
        print(f"    tcga normals: {len(samples)}, star files: {len(smap)}", flush=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(curl_download, uuid, os.path.join(CACHE, uuid + ".tsv"))
                    for uuid in smap.values()]
            for f in futs: f.result()
        # build full gene-space TCGA matrix is too big; first compute train-only
        # DE we need gene expression for ALL genes of TCGA train split.
        # Strategy: read all cached files once, keep ENSG rows with any symbol
        tcga_sex = dict(samples)
        ids = list(smap)
        # donor split inside TCGA (submitter sample prefix = patient)
        pat = np.array([s[:12] for s in ids])   # TCGA-XX-XXXX patient prefix
        rng = np.random.RandomState(7)
        upat = pd.unique(pat); rng.shuffle(upat)
        tr_pat = set(upat[:int(len(upat)*0.6)])
        tr_mask = pd.Series(pat).isin(tr_pat).to_numpy()
        print(f"    tcga split: train {tr_mask.sum()} / test {(~tr_mask).sum()} patients={len(upat)}", flush=True)

        # pass 1: read train files, collect gene tpm matrix (top DE selection)
        def read_file(uuid):
            return extract_tpm_all(os.path.join(CACHE, uuid + ".tsv"))
        def extract_tpm_all(path):
            try:
                df = pd.read_csv(path, sep="\t", comment="#", header=None,
                                 names=["gene", "name", "type", "cts", "sf", "ss",
                                        "tpm", "fpkm", "fpkmuq"],
                                 usecols=[0, 6], on_bad_lines="skip", engine="c")
                df = df[df.gene.str.startswith("ENSG", na=False)]
                return dict(zip(df.gene, df.tpm))
            except Exception:
                return None
        tr_ids = [i for i, m in zip(ids, tr_mask) if m]
        te_ids = [i for i, m in zip(ids, tr_mask) if not m]
        # read all files (cached; fast local)
        all_dicts = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            for sid, d in zip(ids, ex.map(read_file, [smap[i] for i in ids])):
                if d: all_dicts[sid] = d
        print(f"    parsed files: {len(all_dicts)}", flush=True)
        all_genes = sorted(set().union(*[set(v) for v in all_dicts.values()]))
        M = pd.DataFrame({sid: pd.Series(all_dicts[sid]) for sid in all_dicts}).T
        M = M.reindex(columns=all_genes).astype(np.float32)
        M = M.groupby(level=0).sum()  # dedupe rows if any
        # rebuild labels AFTER index normalization
        m_idx = M.index
        ytc = np.array([tcga_sex[s] == "female" for s in m_idx]).astype(int)
        is_tr = pd.Series([s[:12] in tr_pat for s in m_idx]).to_numpy()
        from scipy import stats
        from statsmodels.stats.multitest import multipletests
        Y = np.log2(M.to_numpy() + 1.0)
        D = np.column_stack([np.ones(is_tr.sum()), ytc[is_tr]]).astype(np.float64)
        Ytr = Y[is_tr].astype(np.float64)
        XtX_inv = np.linalg.pinv(D.T @ D)
        B = XtX_inv @ D.T @ Ytr
        resid = Ytr - D @ B
        dof = Ytr.shape[0] - D.shape[1]
        s2 = (resid**2).sum(axis=0)/dof
        se = np.sqrt(np.outer(s2, np.diag(XtX_inv)))
        p = 2*stats.t.sf(np.abs(B[1]/se[:,1]), dof)
        chr_of = gene2chr.reindex(all_genes).fillna("NA").to_numpy()
        auto = ~np.isin(chr_of, ["chrX","chrY","chrM"])
        _, q, _, _ = multipletests(p, method="fdr_bh")
        de = pd.DataFrame({"gene": all_genes, "auto": auto, "log2FC": B[1], "q": q})
        de_auto = de[de.auto]
        top = de_auto.reindex(de_auto.log2FC.abs().sort_values(ascending=False).index).head(500)["gene"].tolist()
        print(f"    tcga train-only DE done, top={len(top)}", flush=True)

        # ---- GTEx side (test) ----
        Xg = pd.read_parquet(os.path.join(PROC, f"tpm_{gtex_t}.parquet"))
        Xg = Xg.loc[:, (Xg >= 1.0).mean(axis=0) >= 0.20]
        common = [g for g in top if g in set(Xg.columns)]
        if len(common) < 100:
            print("    too few common genes, skip"); continue
        cols = np.array([np.where(Xg.columns == g)[0][0] for g in common])
        Ygt = np.log2(Xg.to_numpy(dtype=np.float32) + 1.0)[:, cols]
        ygt = meta.loc[Xg.index, "sex"].to_numpy(int)
        # TCGA-train model on common genes
        tr_cols = np.array([all_genes.index(g) for g in common])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(Y[np.ix_(is_tr, tr_cols)], ytc[is_tr])
        pg = clf.predict_proba(Ygt)[:, 1]
        auc = roc_auc_score(ygt, pg)
        auprc = average_precision_score(ygt, pg)
        results.append({"tcga_train": ",".join(projects), "gtex_test": gtex_t,
                        "n_tcga_train": int(is_tr.sum()),
                        "n_gtex_test": len(ygt), "n_gtex_female": int(ygt.sum()),
                        "n_features": len(common),
                        "auc": round(auc, 4), "auprc": round(auprc, 4)})
        print(f"    -> TCGA->GTEx AUC={auc:.3f} AUPRC={auprc:.3f} (n={len(ygt)})", flush=True)
    pd.DataFrame(results).to_csv(os.path.join(RES, "t3_3_reverse_validation.csv"), index=False)
    print("[done] T3-3 complete", flush=True)

if __name__ == "__main__":
    main()
