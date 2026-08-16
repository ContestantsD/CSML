import argparse
import csv
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CSG"))
from datagen_batch import read_vtk_polydata

HEMI_RE = re.compile(r"[._-](LH|RH|L|R|left|right)[._-]", re.IGNORECASE)
FIELDNAMES = ["dataset", "hemi", "readable", "error_type", "n_vertices",
              "n_faces", "n_scalars", "elapsed_ms", "scalar_names", "error", "path"]


def infer_hemi(path):
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


def probe(path):
    t0 = time.time()
    try:
        pts, faces, scalars = read_vtk_polydata(path)
        return {
            "readable": 1, "n_vertices": int(len(pts)), "n_faces": int(len(faces)),
            "n_scalars": int(len(scalars)), "scalar_names": ";".join(sorted(scalars)),
            "error_type": "", "error": "",
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as exc:
        return {
            "readable": 0, "n_vertices": "", "n_faces": "", "n_scalars": "",
            "scalar_names": "", "error_type": type(exc).__name__,
            "error": repr(exc)[:500],
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
        }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True, metavar="NAME=DIR",
                    help="original VTK directory; repeatable")
    ap.add_argument("--limit", type=int, default=0, help="per-dataset cap (0=all)")
    ap.add_argument("--out", type=Path, required=True, help="report output directory")
    args = ap.parse_args()

    datasets = dict(d.split("=", 1) for d in args.dataset)
    args.out.mkdir(parents=True, exist_ok=True)
    targets = list(datasets)
    all_rows, summary = [], []

    for name in targets:
        root = Path(datasets[name])
        vkts = sorted(root.rglob("*.vtk"))
        if args.limit:
            vkts = vkts[: args.limit]
        print(f"[{name}] probing {len(vkts)} VTK ...", flush=True)
        t0 = time.time()
        rows = []
        for i, p in enumerate(vkts, 1):
            r = probe(p)
            r.update({"dataset": name, "hemi": infer_hemi(p), "path": str(p)})
            rows.append(r)
            if i % 200 == 0:
                print(f"  [{name}] {i}/{len(vkts)}", flush=True)

        write_csv(args.out / f"{name}_readability.csv", rows, FIELDNAMES)

        n_ok = sum(r["readable"] for r in rows)
        vc = Counter(r["n_vertices"] for r in rows if r["readable"])
        fc = Counter(r["n_faces"] for r in rows if r["readable"])
        errs = Counter(r["error_type"] for r in rows if not r["readable"])
        by_hemi = Counter(r["hemi"] for r in rows)
        summary.append({
            "dataset": name, "n_vtk": len(rows), "readable": n_ok,
            "unreadable": len(rows) - n_ok,
            "hemi_breakdown": ";".join(f"{k}={v}" for k, v in sorted(by_hemi.items())),
            "vertex_mode": vc.most_common(1)[0][0] if vc else "",
            "vertex_distinct_counts": len(vc),
            "face_mode": fc.most_common(1)[0][0] if fc else "",
            "face_distinct_counts": len(fc),
            "error_breakdown": "; ".join(f"{k}={v}" for k, v in errs.items()) or "",
            "elapsed_sec": round(time.time() - t0, 1),
        })
        all_rows.extend(rows)
        print(f"[{name}] readable={n_ok}/{len(rows)} errors={dict(errs)} "
              f"({summary[-1]['elapsed_sec']}s)", flush=True)

    write_csv(args.out / "all_readability.csv", all_rows, FIELDNAMES)
    write_csv(args.out / "readability_summary.csv", summary,
              list(summary[0].keys()) if summary else ["dataset"])
    print("\n=== readability summary ===")
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
