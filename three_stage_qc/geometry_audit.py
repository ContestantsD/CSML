import argparse
import csv
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CSG"))
from datagen_batch import read_vtk_polydata

HEMI_RE = re.compile(r"[._-](LH|RH|L|R|left|right)[._-]", re.IGNORECASE)
EPS = 1e-9


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


def audit_one(payload):
    dataset, path = payload
    path = Path(path)
    t0 = time.time()
    rec = {
        "dataset": dataset, "hemi": infer_hemi(path), "path": str(path),
        "readable": 0, "finite": "", "n_vertices": "", "n_faces": "",
        "min_edge_len": "", "p1_edge_len": "", "min_face_area": "",
        "n_coincident_vertex_faces": "", "n_collinear_faces": "",
        "n_degenerate_faces": "", "n_zero_edges": "",
        "n_edges_lt_1e9": "", "n_edges_lt_1e6": "", "n_edges_lt_1e3": "",
        "n_edges_lt_0p01": "", "n_edges_lt_0p05": "",
        "n_faces_area_lt_1e9": "", "n_faces_area_lt_1e6": "",
        "n_faces_area_lt_1e4": "", "n_faces_area_lt_1e3": "",
        "flag": "", "error_type": "", "error": "",
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }
    try:
        pts, faces, _ = read_vtk_polydata(path)
    except Exception as exc:
        rec.update(error_type=type(exc).__name__, error=repr(exc)[:400], flag="read_error")
        return rec

    pts = np.asarray(pts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    finite = bool(np.isfinite(pts).all())
    rec.update(readable=1, finite=int(finite),
               n_vertices=int(len(pts)), n_faces=int(len(faces)))

    if faces.ndim != 2 or faces.shape[1] != 3:
        rec.update(flag="non_triangular_faces",
                   error=f"face_width={faces.shape[1] if faces.ndim == 2 else 'ragged'}")
        return rec
    if not finite:
        rec.update(flag="nonfinite_coords")
        return rec

    tri = pts[faces]
    e01 = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    e12 = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    e20 = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    all_edges = np.concatenate([e01, e12, e20])
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)

    coincident = (e01 < EPS) | (e12 < EPS) | (e20 < EPS)
    collinear = (area < EPS) & ~coincident
    n_coincident = int(coincident.sum())
    n_collinear = int(collinear.sum())
    n_zero_edges = int((all_edges < EPS).sum())

    rec.update(
        min_edge_len=float(all_edges.min()),
        p1_edge_len=float(np.quantile(all_edges, 0.01)),
        min_face_area=float(area.min()),
        n_coincident_vertex_faces=n_coincident,
        n_collinear_faces=n_collinear,
        n_degenerate_faces=n_coincident + n_collinear,
        n_zero_edges=n_zero_edges,
        n_edges_lt_1e9=int((all_edges < 1e-9).sum()),
        n_edges_lt_1e6=int((all_edges < 1e-6).sum()),
        n_edges_lt_1e3=int((all_edges < 1e-3).sum()),
        n_edges_lt_0p01=int((all_edges < 1e-2).sum()),
        n_edges_lt_0p05=int((all_edges < 0.05).sum()),
        n_faces_area_lt_1e9=int((area < 1e-9).sum()),
        n_faces_area_lt_1e6=int((area < 1e-6).sum()),
        n_faces_area_lt_1e4=int((area < 1e-4).sum()),
        n_faces_area_lt_1e3=int((area < 1e-3).sum()),
    )
    flags = []
    if n_coincident:
        flags.append("coincident_vertices")
    if n_collinear:
        flags.append("collinear_faces")
    rec["flag"] = ";".join(flags)
    rec["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
    return rec


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
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True, help="report output directory")
    args = ap.parse_args()

    datasets = dict(d.split("=", 1) for d in args.dataset)
    args.out.mkdir(parents=True, exist_ok=True)
    targets = list(datasets)
    all_rows, summary = [], []
    fields = list(audit_one(("__probe__", Path("__probe__"))).keys())

    for name in targets:
        root = Path(datasets[name])
        vkts = sorted(root.rglob("*.vtk"))
        if args.limit:
            vkts = vkts[: args.limit]
        print(f"[{name}] auditing {len(vkts)} VTK ...", flush=True)
        jobs = [(name, p) for p in vkts]
        rows = []
        t0 = time.time()
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(audit_one, j): j for j in jobs}
                for i, fut in enumerate(as_completed(futs), 1):
                    rows.append(fut.result())
                    if i % 200 == 0:
                        print(f"  [{name}] {i}/{len(jobs)}", flush=True)
        else:
            for i, j in enumerate(jobs, 1):
                rows.append(audit_one(j))
                if i % 200 == 0:
                    print(f"  [{name}] {i}/{len(jobs)}", flush=True)

        write_csv(args.out / f"{name}_geometry_audit.csv", rows, fields)

        flagged = [r for r in rows if r["flag"]]
        n_read_err = sum(1 for r in rows if r["flag"] == "read_error")
        n_coin = sum(1 for r in rows if "coincident_vertices" in r["flag"])
        n_col = sum(1 for r in rows if "collinear_faces" in r["flag"])
        summary.append({
            "dataset": name, "n_vtk": len(rows),
            "readable": sum(r["readable"] for r in rows),
            "flagged_total": len(flagged),
            "flagged_read_error": n_read_err,
            "flagged_coincident_vertices": n_coin,
            "flagged_collinear_faces": n_col,
            "total_coincident_faces": sum(r.get("n_coincident_vertex_faces") or 0 for r in rows),
            "total_collinear_faces": sum(r.get("n_collinear_faces") or 0 for r in rows),
            "min_edge_len_global": min((r["min_edge_len"] for r in rows if r["min_edge_len"] != ""), default=""),
            "min_face_area_global": min((r["min_face_area"] for r in rows if r["min_face_area"] != ""), default=""),
            "elapsed_sec": round(time.time() - t0, 1),
        })
        all_rows.extend(rows)
        print(f"[{name}] flagged={len(flagged)}/{len(rows)} "
              f"(coincident_verts={n_coin}, collinear={n_col}, read_err={n_read_err}) "
              f"({summary[-1]['elapsed_sec']}s)", flush=True)

    write_csv(args.out / "all_geometry_audit.csv", all_rows, fields)
    write_csv(args.out / "geometry_audit_summary.csv", summary,
              list(summary[0].keys()) if summary else ["dataset"])
    write_csv(args.out / "flagged_vtks.csv", [r for r in all_rows if r["flag"]], fields)
    print("\n=== geometry audit summary ===")
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
