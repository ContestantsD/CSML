import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description="encode the HCP phenotype table")
ap.add_argument("csv", help="phenotype CSV (one row per subject, 'Subject' column)")
ap.add_argument("subject_ids", help="offline subject_ids.npy defining the subject set")
ap.add_argument("out_dir", help="output directory")
args = ap.parse_args()
CSV, SID, OUT = args.csv, args.subject_ids, Path(args.out_dir)


def encode_age(s):
    if isinstance(s, str) and "-" in s:
        parts = s.split("-")
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            return np.nan
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


df = pd.read_csv(CSV, dtype=str)
df["Subject"] = df["Subject"].astype(str).str.zfill(6)
sids_raw = np.load(SID, allow_pickle=True)
sids = [str(s).split(".")[0].split("_")[0].zfill(6) for s in sids_raw]

cols = [c for c in df.columns if c != "Subject"]
out = np.full((len(sids), len(cols)), np.nan, dtype=np.float32)
missing_subj = []
for i, sid in enumerate(sids):
    row = df[df["Subject"] == sid]
    if len(row) == 0:
        missing_subj.append(sid)
        continue
    r = row.iloc[0]
    for j, c in enumerate(cols):
        v = r[c]
        if pd.isna(v) or v == "":
            continue
        if c == "Gender":
            out[i, j] = 0.0 if v.strip() == "M" else 1.0
        elif c == "Age":
            out[i, j] = encode_age(v)
        else:
            try:
                out[i, j] = float(v)
            except ValueError:
                out[i, j] = np.nan

OUT.mkdir(parents=True, exist_ok=True)
np.save(OUT / "phenotypes.npy", out)
(OUT / "phenotype_names.json").write_text(json.dumps(cols, ensure_ascii=False, indent=2))
np.save(OUT / "subject_ids.npy", np.asarray(sids, dtype=object))
print(f"N_subj={len(sids)} N_pheno={len(cols)} missing_subj={len(missing_subj)} out={OUT}")
for j, c in enumerate(cols):
    print(f"  {c:24s} finite={int(np.isfinite(out[:, j]).sum())}/{len(sids)}")
