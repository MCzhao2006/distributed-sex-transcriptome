# -*- coding: utf-8 -*-
"""lit_query.py — novelty-audit literature query helper (Europe PMC REST + local filtering).

WHY THIS EXISTS
  Europe PMC's free-text relevance ranking is poor for *concept-level* queries:
  a query like `(autosomal) AND ("sex prediction")` returns 99 hits of which ~0 are
  on-topic, because the ranking is dominated by recent unrelated papers.
  This tool fetches a large page and then filters LOCALLY by regex over title+abstract,
  so the operator sees only the records that actually mention the concept.

NETWORK
  This environment's WebSearch/WebFetch are unavailable. The machine does have internet
  via a SOCKS5 proxy. Usage:
      python scripts/lit_query.py --proxy socks5h://127.0.0.1:7891 --claim N1

  Verified reachable through that proxy (2026-09-18):
      www.ebi.ac.uk (Europe PMC REST) OK | api.crossref.org OK | api.openalex.org OK
      api.unpaywall.org OK | www.biorxiv.org OK
      pmc.ncbi.nlm.nih.gov / eutils.ncbi.nlm.nih.gov / web.archive.org  TIMEOUT
      www.science.org / discovery.dundee.ac.uk  Cloudflare 403

OUTPUT
  Prints matched records (title / authors / year / DOI / PMCID / OA) to stdout.
  Does NOT write to results/ — this is an audit tool, not an analysis product.

USAGE
  python scripts/lit_query.py --proxy <socks> --claim N1
  python scripts/lit_query.py --proxy <socks> --query 'your query' --filter 'regex'
  python scripts/lit_query.py --proxy <socks> --fetch PMC5961118     # full text XML -> text
"""
import argparse
import html
import json
import re
import subprocess
import sys

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

# claim -> (query, local filter regex over "title || abstract")
CLAIMS = {
    "N1": ('(autosomal OR autosome) AND ("sex prediction" OR "predict sex" OR "sex classifier")',
           r"autosom\w*.{0,80}(predict|classif|discriminat)|(predict|classif|discriminat)\w*.{0,80}autosom"),
    "N3": ('("number of genes" OR "gene number" OR "gene subset" OR "random genes") AND '
           '(prediction OR classifier OR "prediction accuracy")',
           r"(number of genes|gene subset|random genes|gene panel).{0,90}(accurac|AUC|predict|classif)"),
    "N4": ('("non-significant" OR "non significant" OR "non-DE" OR "nonsignificant" OR "null genes") '
           'AND (predict OR predictive OR classifier)',
           r"(non-?significant|non-?DE|null genes).{0,90}(predict|classif|AUC)"),
    "N5": ('(compressib OR "low-dimensional" OR "dimensionality reduction") AND '
           '("sex differences" OR "sex-biased" OR "sex-associated")',
           r"(compressib|low-dimensional|dimensionality).{0,90}(sex)|(sex).{0,90}(compressib|low-dimensional)"),
    "N6": ('("cross-tissue" OR "across tissues" OR transfer) AND (predict OR classifier) AND '
           '("sex differences" OR "sex-biased" OR sex)',
           r"(cross-?tissue|transfer|portab).{0,90}(sex|predict)"),
    "N7": ('(asymmetr OR asymmetric) AND (transfer OR portability) AND (expression OR transcriptom)',
           r"asymmetr\w*.{0,90}(transfer|portab|tissue)"),
    "N11": ('("within cell type" OR "per cell type" OR "cell-type-specific") AND sex AND '
            '(downsampl OR scaling OR "cell number")',
            r"(cell type|cell-type).{0,90}(sex).{0,90}(downsampl|scal|number of cells)|downsampl\w*.{0,90}sex"),
    "N12": ('("cell number" OR "cells per" OR "number of cells") AND '
            '(confound OR correlat OR associat) AND (single-cell OR pseudobulk)',
            r"(cell number|cells per|number of cells).{0,90}(correlat|associat|confound)"),
    "N13": ('(residualis OR residualiz OR "regress out" OR "adjust for") AND '
            '("cell composition" OR "cell-type composition" OR "cellular composition") AND sex',
            r"(cell composition|cellular composition|cell-type composition).{0,90}(residual|regress|adjust)"),
    "N14": ('(recoverability OR "information geometry" OR framework) AND '
            '("sex differences" OR "sex-associated" OR "sex-biased") AND (tissue OR transcriptom)',
            r"recoverab\w*|information geometry"),
}


def curl(url, proxy=None, timeout=60, ua=True):
    cmd = ["curl", "-s", "-L", "--max-time", str(timeout)]
    if proxy:
        cmd += ["--proxy", proxy]
    if ua:
        cmd += ["-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"]
    cmd.append(url)
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"curl failed rc={p.returncode} for {url}")
    return p.stdout


def search(query, proxy, pagesize=100, sort="CITED desc"):
    """Europe PMC search. NOTE: default relevance ranking is recency-dominated and
    returns unrelated 2025-26 papers for concept-level queries. Sorting by citation
    count surfaces the landmark work instead — always use sort=CITED desc for audits."""
    import urllib.parse
    url = (f"{EPMC}?query={urllib.parse.quote(query)}"
           f"&format=json&pageSize={pagesize}&resultType=core")
    if sort:
        url += f"&sort={urllib.parse.quote(sort)}"
    raw = curl(url, proxy)
    return json.loads(raw.decode("utf-8", "replace"))


def run_claim(claim, proxy, pagesize=100, show_all=False):
    query, filt = CLAIMS[claim]
    d = search(query, proxy, pagesize)
    res = d.get("resultList", {}).get("result", [])
    print(f"=== {claim} ===")
    print(f"query : {query}")
    print(f"filter: {filt}")
    print(f"hitCount={d.get('hitCount')}  fetched={len(res)}")
    rx = re.compile(filt, re.I)
    shown = 0
    for r in res:
        blob = f"{r.get('title') or ''} || {r.get('abstractText') or ''}"
        if not show_all and not rx.search(blob):
            continue
        shown += 1
        print("  ---")
        print("   title:", (r.get("title") or "")[:150])
        print("   auth :", (r.get("authorString") or "")[:80])
        print("   meta :", r.get("journalInfo", {}).get("journal", {}).get("title"),
              r.get("pubYear"), "| DOI:", r.get("doi"), "| PMCID:", r.get("pmcid"),
              "| OA:", r.get("isOpenAccess"))
    print(f"  >>> matched {shown} / {len(res)} fetched\n")
    return shown


def fetch_fulltext(pmcid, proxy):
    raw = curl(FULLTEXT.format(pmcid=pmcid), proxy, timeout=90)
    t = raw.decode("utf-8", "replace")
    t = re.sub(r"<(title|p|sec|abstract|fig|caption)[^>]*>", "\n\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="socks5h://127.0.0.1:7891")
    ap.add_argument("--claim", help="one of: " + ", ".join(CLAIMS))
    ap.add_argument("--all", action="store_true", help="run every claim")
    ap.add_argument("--query")
    ap.add_argument("--filter", default=".")
    ap.add_argument("--fetch", help="PMCID -> dump plain text")
    ap.add_argument("--pagesize", type=int, default=100)
    ap.add_argument("--show-all", action="store_true")
    a = ap.parse_args()

    if a.fetch:
        sys.stdout.reconfigure(encoding="utf-8")
        print(fetch_fulltext(a.fetch, a.proxy))
        return
    if a.query:
        d = search(a.query, a.proxy, a.pagesize)
        res = d.get("resultList", {}).get("result", [])
        rx = re.compile(a.filter, re.I)
        print(f"hitCount={d.get('hitCount')} fetched={len(res)}")
        for r in res:
            blob = f"{r.get('title') or ''} || {r.get('abstractText') or ''}"
            if not a.show_all and not rx.search(blob):
                continue
            print("  -", (r.get("title") or "")[:140])
            print("    ", (r.get("authorString") or "")[:70], "|", r.get("pubYear"),
                  "| DOI:", r.get("doi"), "| PMCID:", r.get("pmcid"), "| OA:", r.get("isOpenAccess"))
        return
    if a.all:
        for c in CLAIMS:
            run_claim(c, a.proxy, a.pagesize, a.show_all)
        return
    if a.claim:
        run_claim(a.claim, a.proxy, a.pagesize, a.show_all)
        return
    ap.print_help()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
