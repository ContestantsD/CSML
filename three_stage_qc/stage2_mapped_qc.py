import argparse
import csv
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh

HEMI_RE = re.compile(r"[._-](LH|RH|L|R|left|right)[._-]", re.IGNORECASE)

CHECK_FIELDS = ["dataset", "hemi", "loadable", "finite", "n_vertices", "n_faces",
                "watertight", "euler_number", "degenerate_faces", "is_volume",
                "error_type", "error", "path"]


def infer_hemi_path(path):
    m = HEMI_RE.search(path.name)
    if m:
        h = m.group(1).upper()
        if h in ("L", "LH", "LEFT"):
            return "L"
        if h in ("R", "RH", "RIGHT"):
            return "R"
    p = str(path).lower().replace("\\", "/")
    if "left" in p or "/lh" in p:
        return "L"
    if "right" in p or "/rh" in p:
        return "R"
    return ""


def check_one(payload):
    dataset, path = payload
    path = Path(path)
    try:
        mesh = trimesh.load(path, process=False, maintain_order=True)
    except Exception as exc:
        return {"dataset": dataset, "hemi": infer_hemi_path(path), "path": str(path),
                "loadable": 0, "finite": "", "n_vertices": "", "n_faces": "",
                "watertight": "", "euler_number": "", "degenerate_faces": "",
                "is_volume": "", "error_type": type(exc).__name__, "error": repr(exc)[:500]}
    v = np.asarray(mesh.vertices, dtype=np.float64)
    finite = bool(np.isfinite(v).all())
    n_v, n_f = int(len(v)), int(len(mesh.faces))
    try:
        watertight = int(bool(mesh.is_watertight))
    except Exception:
        watertight = ""
    try:
        is_volume = int(bool(mesh.is_volume))
    except Exception:
        is_volume = ""
    try:
        euler = int(round(mesh.euler_number))
    except Exception:
        euler = ""
    try:
        areas = trimesh.triangles.area(mesh.triangles)
        n_degen = int((areas <= 1e-12).sum())
    except Exception:
        n_degen = ""
    return {"dataset": dataset, "hemi": infer_hemi_path(path), "path": str(path),
            "loadable": 1, "finite": int(finite), "n_vertices": n_v, "n_faces": n_f,
            "watertight": watertight, "euler_number": euler,
            "degenerate_faces": n_degen, "is_volume": is_volume,
            "error_type": "", "error": ""}


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def original_stems(root):
    by_hemi = {"L": set(), "R": set(), "": set()}
    for p in root.rglob("*.vtk"):
        by_hemi.setdefault(infer_hemi_path(p), set()).add(p.stem)
    return by_hemi


def check_dataset(name, ori_root, mapped_dir, args):
    ori = original_stems(ori_root)

    mapped_by_hemi = {"L": [], "R": []}
    for hemi_key, dirname in (("L", "LH"), ("R", "RH")):
        hemi_dir = mapped_dir / dirname
        if hemi_dir.exists():
            mapped_by_hemi[hemi_key] = sorted(hemi_dir.glob("*.obj"))

    jobs = []
    for hemi_key, objs in mapped_by_hemi.items():
        for obj in objs:
            jobs.append((name, obj))

    if args.limit:
        jobs = jobs[: args.limit]

    rows = []
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(check_one, j): j for j in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if i % 100 == 0:
                    print(f"  [{name}] {i}/{len(jobs)}", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            rows.append(check_one(j))
            if i % 100 == 0:
                print(f"  [{name}] {i}/{len(jobs)}", flush=True)

    missing_rows = []
    completeness = {}
    for hemi_key in ("L", "R"):
        mapped_stems = {Path(o).stem for o in mapped_by_hemi[hemi_key]}
        ori_stems = ori.get(hemi_key, set())
        ori_pool = ori_stems | ori.get("", set()) if not ori_stems else ori_stems
        missing = sorted(ori_pool - mapped_stems)
        extra = sorted(mapped_stems - ori_pool)
        completeness[hemi_key] = {
            "original_n": len(ori_pool), "mapped_n": len(mapped_stems),
            "missing_n": len(missing), "extra_n": len(extra),
        }
        for stem in missing:
            missing_rows.append({"dataset": name, "hemi": hemi_key, "stem": stem,
                                 "issue": "missing_mapped_obj"})
        for stem in extra:
            missing_rows.append({"dataset": name, "hemi": hemi_key, "stem": stem,
                                 "issue": "extra_mapped_no_original"})

    return rows, missing_rows, completeness, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True, metavar="NAME=ORI_DIR",
                    help="original VTK directory; repeatable")
    ap.add_argument("--mapped-root", type=Path, required=True,
                    help="root containing one mapped subdirectory per dataset")
    ap.add_argument("--subdir", action="append", default=[], metavar="NAME=SUBDIR",
                    help="mapped subdirectory for a dataset (default: dataset name)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap objs checked per dataset (0=all; completeness still uses full lists)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True, help="report output directory")
    args = ap.parse_args()

    ori_roots = dict(d.split("=", 1) for d in args.dataset)
    subdirs = dict(d.split("=", 1) for d in args.subdir)
    args.out.mkdir(parents=True, exist_ok=True)
    targets = list(ori_roots)
    all_rows, all_missing, summary = [], [], []

    for name in targets:
        print(f"[{name}] checking mapped OBJ ...", flush=True)
        mapped_dir = args.mapped_root / subdirs.get(name, name)
        rows, missing, completeness, elapsed = check_dataset(
            name, Path(ori_roots[name]), mapped_dir, args)
        write_csv(args.out / f"{name}_mapped_checks.csv", rows, CHECK_FIELDS)
        write_csv(args.out / f"{name}_completeness_issues.csv", missing,
                  ["dataset", "hemi", "stem", "issue"])

        n_load = sum(r["loadable"] for r in rows)
        n_finite_bad = sum(1 for r in rows if r["loadable"] == 1 and r["finite"] == 0)
        vc = Counter(r["n_vertices"] for r in rows if r["loadable"] == 1)
        n_non_watertight = sum(1 for r in rows if r["loadable"] == 1 and r["watertight"] == 0)
        n_degen = sum(1 for r in rows if r["loadable"] == 1 and r["degenerate_faces"] not in ("", 0))
        summary.append({
            "dataset": name,
            "checked_objs": len(rows), "loadable": n_load,
            "nonfinite": n_finite_bad,
            "vertex_mode": vc.most_common(1)[0][0] if vc else "",
            "vertex_distinct": len(vc),
            "non_watertight": n_non_watertight,
            "with_degenerate_faces": n_degen,
            "ori_L": completeness["L"]["original_n"], "mapped_L": completeness["L"]["mapped_n"],
            "missing_L": completeness["L"]["missing_n"], "extra_L": completeness["L"]["extra_n"],
            "ori_R": completeness["R"]["original_n"], "mapped_R": completeness["R"]["mapped_n"],
            "missing_R": completeness["R"]["missing_n"], "extra_R": completeness["R"]["extra_n"],
            "elapsed_sec": round(elapsed, 1),
        })
        all_rows.extend(rows)
        all_missing.extend(missing)
        print(f"[{name}] loadable={n_load}/{len(rows)} nonfinite={n_finite_bad} "
              f"non_watertight={n_non_watertight} degenerate={n_degen} | "
              f"missing L={completeness['L']['missing_n']} R={completeness['R']['missing_n']} "
              f"({summary[-1]['elapsed_sec']}s)", flush=True)

    write_csv(args.out / "all_mapped_checks.csv", all_rows, CHECK_FIELDS)
    write_csv(args.out / "all_completeness_issues.csv", all_missing,
              ["dataset", "hemi", "stem", "issue"])
    write_csv(args.out / "mapped_qc_summary.csv", summary,
              list(summary[0].keys()) if summary else ["dataset"])
    print("\n=== mapped qc summary ===")
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
