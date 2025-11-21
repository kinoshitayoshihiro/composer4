#!/usr/bin/env python3
# Ensure analysis/bars.parquet has 'energy' and 'valence' columns.
import sys, pandas as pd, numpy as np, pathlib
p = pathlib.Path(sys.argv[1])  # path to analysis/bars.parquet
df = pd.read_parquet(p)
changed = False
if "energy" not in df.columns:
    if "loudness_db" in df.columns:
        ld = df["loudness_db"].values.astype(float)
        e = (ld - np.nanmin(ld)) / max(1e-6, (np.nanmax(ld)-np.nanmin(ld)))
        df["energy"] = np.clip(e, 0.0, 1.0)
    else:
        df["energy"] = 0.5
    changed = True
if "valence" not in df.columns:
    df["valence"] = 0.5
    changed = True
if changed:
    df.to_parquet(p, index=False)
print("OK: bars has columns:", len(df.columns))