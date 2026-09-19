# -*- coding: utf-8 -*-
"""AUDIT DIAGNOSTIC ONLY -- tau sensitivity to the capped N=4000 point.

`d1_saturation_fit.csv` fitted AUC(k)=0.5+alpha*(1-exp(-k/tau)) using the
REQUESTED feature count k. For Muscle and Blood the N=4000 point was
silently capped at the pool size, so the k actually used was too large.
Refit using the ACHIEVED k and compare. Writes nothing.
"""
import numpy as np
import pandas as pd
import os

RES = os.path.join(r"F:\nature", "results")
REQ = [50, 100, 200, 500, 1000, 2000, 4000]
ACH = {"Muscle": [50, 100, 200, 500, 1000, 2000, 3690],
       "Thyroid": [50, 100, 200, 500, 1000, 2000, 4000],
       "Blood": [50, 100, 200, 500, 1000, 2000, 3504]}

d = pd.read_csv(os.path.join(RES, "d1_nonsig_dose.csv"))
print(f"{'tissue':8s} {'tau k=requested':>16s} {'tau k=achieved':>15s} {'delta':>8s} "
      f"{'alpha req':>10s} {'alpha ach':>10s}")
for t in ["Muscle", "Thyroid", "Blood"]:
    dd = d[d.tissue == t]
    aa = dd.auc_median.to_numpy(float) - 0.5

    def fit(kk):
        best = None
        for alpha in np.linspace(0.2, 0.6, 41):
            for tau in np.geomspace(50, 5000, 60):
                sse = ((aa - alpha * (1 - np.exp(-kk / tau))) ** 2).sum()
                if best is None or sse < best[0]:
                    best = (sse, alpha, tau)
        return best

    s1, a1, t1 = fit(np.array(REQ, float))
    s2, a2, t2 = fit(np.array(ACH[t], float))
    print(f"{t:8s} {t1:16.1f} {t2:15.1f} {t2 - t1:+8.1f} {a1:10.3f} {a2:10.3f}")

print("\n(SSE with requested k was the value written to d1_saturation_fit.csv)")
