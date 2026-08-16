import argparse
import csv
from collections import Counter
from pathlib import Path


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def by_dataset(rows, key="dataset"):
    return {r.get(key, ""): r for r in rows}


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-dir", type=Path, required=True,
                    help="stage 1 output directory (readability_summary.csv / all_readability.csv)")
    ap.add_argument("--stage2-dir", type=Path, required=True,
                    help="stage 2 output directory (mapped_qc_summary.csv / all_mapped_checks.csv / all_completeness_issues.csv)")
    ap.add_argument("--stage3-dir", type=Path, required=True,
                    help="stage 3 output directory (deformed_summary.csv / deformed_case_flags.csv)")
    ap.add_argument("--out", type=Path, required=True, help="report output directory")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    s1 = by_dataset(read_csv(args.stage1_dir / "readability_summary.csv"))
    s2 = by_dataset(read_csv(args.stage2_dir / "mapped_qc_summary.csv"))
    s3_rows = read_csv(args.stage3_dir / "deformed_summary.csv")
    s3 = {}
    for r in s3_rows:
        s3.setdefault(r.get("dataset", ""), []).append(r)
    completeness = read_csv(args.stage2_dir / "all_completeness_issues.csv")
    datasets = sorted(set(s1) | set(s2) | set(s3))

    roll_fields = [
        "dataset",
        "s1_n_vtk", "s1_readable", "s1_unreadable", "s1_vertex_mode",
        "s2_checked", "s2_loadable", "s2_nonfinite", "s2_non_watertight",
        "s2_degenerate", "s2_vertex_mode", "s2_missing_L", "s2_missing_R",
        "s3_sample_n", "s3_sample_delete", "s3_sample_review",
        "s3_b_hd95_mean", "s3_s_hd95_mean", "s3_coverage",
    ]
    roll = []
    for ds in datasets:
        a = s1.get(ds, {})
        b = s2.get(ds, {})
        s3ds = s3.get(ds, [])
        s3_n = sum(int(r.get("n", 0) or 0) for r in s3ds)
        s3_del = sum(int(r.get("n_delete", 0) or 0) for r in s3ds)
        s3_rev = sum(int(r.get("n_review", 0) or 0) for r in s3ds)
        b_hd95 = next((r.get("b_hd95_mean") for r in s3ds if r.get("b_hd95_mean") != ""), "")
        s_hd95 = next((r.get("s_hd95_mean") for r in s3ds if r.get("s_hd95_mean") != ""), "")
        roll.append({
            "dataset": ds,
            "s1_n_vtk": a.get("n_vtk", ""), "s1_readable": a.get("readable", ""),
            "s1_unreadable": a.get("unreadable", ""), "s1_vertex_mode": a.get("vertex_mode", ""),
            "s2_checked": b.get("checked_objs", ""), "s2_loadable": b.get("loadable", ""),
            "s2_nonfinite": b.get("nonfinite", ""), "s2_non_watertight": b.get("non_watertight", ""),
            "s2_degenerate": b.get("with_degenerate_faces", ""), "s2_vertex_mode": b.get("vertex_mode", ""),
            "s2_missing_L": b.get("missing_L", ""), "s2_missing_R": b.get("missing_R", ""),
            "s3_sample_n": s3_n, "s3_sample_delete": s3_del, "s3_sample_review": s3_rev,
            "s3_b_hd95_mean": b_hd95, "s3_s_hd95_mean": s_hd95,
            "s3_coverage": "sample_only" if s3_n and s3_n < int(b.get("checked_objs", 9999) or 9999) else ("none" if not s3_n else "full"),
        })
    write_csv(args.out / "three_stage_summary.csv", roll, roll_fields)

    actions = []
    for r in read_csv(args.stage1_dir / "all_readability.csv"):
        if r.get("readable") == "0" or r.get("readable") == 0:
            actions.append({"stage": "1_readability", "dataset": r.get("dataset", ""),
                            "severity": "blocker", "hemi": r.get("hemi", ""),
                            "item": r.get("path", ""), "reason": r.get("error_type", "") + ": " + r.get("error", "")[:120]})
    for r in read_csv(args.stage2_dir / "all_mapped_checks.csv"):
        if r.get("loadable") != "1":
            actions.append({"stage": "2_mapped", "dataset": r.get("dataset", ""), "severity": "blocker",
                            "hemi": r.get("hemi", ""), "item": r.get("path", ""), "reason": "not_loadable: " + r.get("error", "")[:120]})
            continue
        reasons = []
        if str(r.get("finite")) == "0":
            reasons.append("nonfinite_coords")
        if str(r.get("watertight")) == "0":
            reasons.append("not_watertight")
        if r.get("degenerate_faces") not in ("", "0"):
            reasons.append(f"degenerate_faces={r.get('degenerate_faces')}")
        if reasons:
            actions.append({"stage": "2_mapped", "dataset": r.get("dataset", ""), "severity": "review",
                            "hemi": r.get("hemi", ""), "item": r.get("path", ""), "reason": ";".join(reasons)})
    for r in completeness:
        if r.get("issue") == "missing_mapped_obj":
            actions.append({"stage": "2_mapped", "dataset": r.get("dataset", ""), "severity": "reprocess",
                            "hemi": r.get("hemi", ""), "item": r.get("stem", ""), "reason": "missing_mapped_obj (rerun MAPS)"})
    for r in read_csv(args.stage3_dir / "deformed_case_flags.csv"):
        if r.get("severity") in ("delete", "review", "error"):
            actions.append({"stage": "3_deformed", "dataset": r.get("dataset", ""), "severity": r.get("severity"),
                            "hemi": r.get("hemi", ""), "item": r.get("subject", ""), "reason": r.get("reasons", "")})

    write_csv(args.out / "action_items.csv", actions,
              ["stage", "dataset", "severity", "hemi", "item", "reason"])

    print("=== three-stage summary ===")
    for r in roll:
        print(f"{r['dataset']}: s1 unreadable={r['s1_unreadable']} | "
              f"s2 nonfinite={r['s2_nonfinite']} non_watertight={r['s2_non_watertight']} "
              f"degenerate={r['s2_degenerate']} missing(L/R)={r['s2_missing_L']}/{r['s2_missing_R']} | "
              f"s3[{r['s3_coverage']}] n={r['s3_sample_n']} del={r['s3_sample_delete']} rev={r['s3_sample_review']}")
    sev = Counter(a["severity"] for a in actions)
    print(f"\n=== action items: {dict(sev)} ===")
    for a in actions[:20]:
        print(f"  [{a['stage']}/{a['severity']}] {a['dataset']} {a['hemi']} {a['item']} -- {a['reason'][:80]}")
    if len(actions) > 20:
        print(f"  ... {len(actions) - 20} more in action_items.csv")


if __name__ == "__main__":
    main()
