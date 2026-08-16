import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_HD95_MM = 3.0
DEFAULT_ASD_MM = 0.8
DEFAULT_CC_LOW = 0.3


def dataset_of(row):
    src = (row.get("source") or "").strip()
    if not src:
        parts = (row.get("case") or "").split(".")
        if len(parts) > 1:
            src = parts[1]
    return src or "unknown"


def hemi_of(row):
    h = (row.get("hemi") or "").strip()
    if h:
        return h
    parts = (row.get("case") or "").split(".")
    if len(parts) > 3:
        return parts[3]
    return ""


def subject_of(row):
    s = (row.get("subject") or "").strip()
    if s:
        return s
    parts = (row.get("case") or "").split(".")
    if len(parts) > 2:
        return parts[2]
    return ""


def as_float(row, key):
    v = row.get(key, "")
    if v in ("", None):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def flag_case(row, hd95_mm, asd_mm, cc_low):
    reasons = []
    severity = "ok"
    if row.get("status") and row.get("status") != "ok":
        reasons.append(f"status:{row.get('status')}")
        severity = "error"

    b_hd95 = as_float(row, "baseline_hd95")
    b_asd = as_float(row, "baseline_asd")
    s_hd95 = as_float(row, "hd95")
    s_asd = as_float(row, "asd")
    s_cc = as_float(row, "cc")

    if np.isfinite(b_hd95) and b_hd95 > hd95_mm:
        reasons.append(f"baseline_hd95>{hd95_mm}")
    if np.isfinite(b_asd) and b_asd > asd_mm:
        reasons.append(f"baseline_asd>{asd_mm}")
    if np.isfinite(s_hd95) and s_hd95 > hd95_mm:
        reasons.append(f"stage2_hd95>{hd95_mm}")
    if np.isfinite(s_asd) and s_asd > asd_mm:
        reasons.append(f"stage2_asd>{asd_mm}")
    if np.isfinite(s_cc) and s_cc < cc_low:
        reasons.append(f"stage2_cc<{cc_low}")
    if row.get("stage2_improves_all_vs_baseline") == "False" and row.get("status") == "ok":
        reasons.append("stage2_not_improving")

    if reasons and severity != "error":
        if any(r.startswith("baseline_hd95") or r.startswith("baseline_asd") for r in reasons):
            severity = "delete"
        else:
            severity = "review"
    return reasons, severity


def stats(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": "", "median": "", "p95": "", "max": ""}
    return {"n": int(arr.size), "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "p95": round(float(np.quantile(arr, 0.95)), 4),
            "max": round(float(arr.max()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", action="append", required=True,
                    help="aggregate_case_rows.csv from the deformation run; repeatable")
    ap.add_argument("--hd95-mm", type=float, default=DEFAULT_HD95_MM)
    ap.add_argument("--asd-mm", type=float, default=DEFAULT_ASD_MM)
    ap.add_argument("--cc-low", type=float, default=DEFAULT_CC_LOW)
    ap.add_argument("--out", type=Path, required=True, help="report output directory")
    args = ap.parse_args()

    rows = []
    for p in args.cases:
        rows.extend(read_rows(p))
    print(f"loaded {len(rows)} cases from {len(args.cases)} file(s)", flush=True)

    flag_fields = ["dataset", "target", "source", "subject", "hemi", "status",
                   "baseline_hd95", "baseline_asd", "baseline_cc",
                   "stage2_hd95", "stage2_asd", "stage2_cc",
                   "stage2_improves_all_vs_baseline", "severity", "reasons", "case_dir"]
    flag_rows = []
    for row in rows:
        reasons, severity = flag_case(row, args.hd95_mm, args.asd_mm, args.cc_low)
        dataset = dataset_of(row)
        flag_rows.append({
            "dataset": dataset, "target": row.get("target", ""), "source": row.get("source", ""),
            "subject": subject_of(row), "hemi": hemi_of(row),
            "status": row.get("status", ""),
            "baseline_hd95": row.get("baseline_hd95", ""), "baseline_asd": row.get("baseline_asd", ""),
            "baseline_cc": row.get("baseline_cc", ""),
            "stage2_hd95": row.get("hd95", ""), "stage2_asd": row.get("asd", ""),
            "stage2_cc": row.get("cc", ""),
            "stage2_improves_all_vs_baseline": row.get("stage2_improves_all_vs_baseline", ""),
            "severity": severity, "reasons": ";".join(reasons),
            "case_dir": row.get("case_dir", ""),
        })

    write_csv(args.out / "deformed_case_flags.csv", flag_rows, flag_fields)
    write_csv(args.out / "deformed_delete_candidates.csv",
              [r for r in flag_rows if r["severity"] == "delete"], flag_fields)
    write_csv(args.out / "deformed_review_candidates.csv",
              [r for r in flag_rows if r["severity"] == "review"], flag_fields)

    groups = defaultdict(list)
    for row in rows:
        dataset = dataset_of(row)
        key = (dataset, hemi_of(row))
        groups[key].append(row)

    sum_fields = ["dataset", "hemi", "n", "n_ok", "n_delete", "n_review", "n_error",
                  "b_hd95_mean", "b_hd95_p95", "b_asd_mean", "s_hd95_mean", "s_hd95_p95",
                  "s_asd_mean", "s_cc_mean", "n_stage2_not_improving"]
    sum_rows = []
    for (dataset, hemi), group in sorted(groups.items()):
        flags = [r for r in flag_rows if r["dataset"] == dataset and r["hemi"] == hemi]
        sum_rows.append({
            "dataset": dataset, "hemi": hemi, "n": len(group),
            "n_ok": sum(1 for r in flags if r["severity"] == "ok"),
            "n_delete": sum(1 for r in flags if r["severity"] == "delete"),
            "n_review": sum(1 for r in flags if r["severity"] == "review"),
            "n_error": sum(1 for r in flags if r["severity"] == "error"),
            "b_hd95_mean": stats([as_float(r, "baseline_hd95") for r in group])["mean"],
            "b_hd95_p95": stats([as_float(r, "baseline_hd95") for r in group])["p95"],
            "b_asd_mean": stats([as_float(r, "baseline_asd") for r in group])["mean"],
            "s_hd95_mean": stats([as_float(r, "hd95") for r in group])["mean"],
            "s_hd95_p95": stats([as_float(r, "hd95") for r in group])["p95"],
            "s_asd_mean": stats([as_float(r, "asd") for r in group])["mean"],
            "s_cc_mean": stats([as_float(r, "cc") for r in group])["mean"],
            "n_stage2_not_improving": sum(1 for r in group if r.get("stage2_improves_all_vs_baseline") == "False"),
        })
    write_csv(args.out / "deformed_summary.csv", sum_rows, sum_fields)

    print("\n=== deformed summary ===")
    for s in sum_rows:
        print(s)
    print(f"\nflags: delete={sum(1 for r in flag_rows if r['severity']=='delete')} "
          f"review={sum(1 for r in flag_rows if r['severity']=='review')} "
          f"error={sum(1 for r in flag_rows if r['severity']=='error')}")


if __name__ == "__main__":
    main()
