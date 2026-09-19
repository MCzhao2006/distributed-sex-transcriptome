# -*- coding: utf-8 -*-
"""GTEx v10 per-tissue TPM matrix downloader (single file, stdlib only).
Downloads per-tissue gene TPM GCT files from the public adult-gtex GCS bucket
through a socks5 proxy, with resume + size verification."""
import os, sys, time, urllib.request, json

BASE = "https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/tpms-by-tissue/"
OUT = r"F:\nature\data\raw"
PROXY = "socks5h://127.0.0.1:7890"   # user-provided proxy for speed

# 12 tissues chosen to cover major organ systems with balanced M/F counts
TISSUES = [
    "whole_blood", "muscle_skeletal", "lung", "thyroid",
    "skin_sun_exposed_lower_leg", "esophagus_mucosa", "artery_tibial",
    "adipose_subcutaneous", "nerve_tibial", "heart_left_ventricle",
    "colon_transverse", "adipose_visceral_omentum",
]

# socks5 support without PySocks: spawn curl instead (curl is bundled with git-bash
# and honors ALL_PROXY). Simplest robust approach: call curl via os.system.
def fetch(url, dst, tries=4):
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print(f"  [skip] {os.path.basename(dst)} exists ({os.path.getsize(dst)//2**20} MB)")
        return True
    for attempt in range(1, tries+1):
        cmd = (f'curl -s --max-time 1800 --proxy {PROXY} -o "{dst}" "{url}"')
        rc = os.system(cmd)
        if rc == 0 and os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            print(f"  [ok] {os.path.basename(dst)} {os.path.getsize(dst)//2**20} MB (try {attempt})")
            return True
        print(f"  [retry {attempt}] rc={rc} size={os.path.getsize(dst) if os.path.exists(dst) else 0}")
        time.sleep(3)
    return False

def main():
    os.makedirs(OUT, exist_ok=True)
    status = {}
    for t in TISSUES:
        name = f"gene_tpm_v10_{t}.gct.gz"
        url = BASE + name
        dst = os.path.join(OUT, name)
        print(f"[{t}]")
        status[t] = fetch(url, dst)
    print(json.dumps(status, indent=0))
    sys.exit(0 if all(status.values()) else 1)

if __name__ == "__main__":
    main()
