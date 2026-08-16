import argparse
import os
import re
import numpy as np


def read_vtk(path):
    verts, faces = [], []
    mode, need = None, 0
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "POINTS":
                mode, need = "v", int(t[1]) * 3
                continue
            if t[0] == "POLYGONS":
                mode = "f"
                continue
            if mode == "v":
                verts.extend(float(x) for x in t)
                if len(verts) >= need:
                    mode = None
            elif mode == "f" and int(t[0]) == 3:
                faces.append([int(x) for x in t[1:4]])
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3), \
        np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def write_vtk(verts, faces, path):
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n")
        f.write("POINTS {} float\n".format(len(verts)))
        for p in verts:
            f.write("{} {} {}\n".format(float(p[0]), float(p[1]), float(p[2])))
        f.write("POLYGONS {} {}\n".format(len(faces), len(faces) * 4))
        for tri in faces:
            f.write("3 {} {} {}\n".format(int(tri[0]), int(tri[1]), int(tri[2])))


def read_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                verts.append([float(x) for x in t[1:4]])
            elif t[0] == "f":
                idx = [int(x.split("/")[0]) - 1 for x in t[1:4]]
                if len(idx) == 3:
                    faces.append(idx)
    return np.asarray(verts, dtype=np.float64), \
        np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def write_obj(verts, faces, path):
    with open(path, "w") as f:
        f.write("# converted surface\n")
        for p in verts:
            f.write("v {} {} {}\n".format(float(p[0]), float(p[1]), float(p[2])))
        for tri in faces:
            f.write("f {} {} {}\n".format(tri[0] + 1, tri[1] + 1, tri[2] + 1))


def convert_file(src, dst):
    if src.suffix.lower() == ".vtk" and dst.suffix.lower() == ".obj":
        v, fc = read_vtk(src)
        write_obj(v, fc, dst)
    elif src.suffix.lower() == ".obj" and dst.suffix.lower() == ".vtk":
        v, fc = read_obj(src)
        write_vtk(v, fc, dst)
    else:
        raise SystemExit(f"unsupported conversion: {src} -> {dst} (only .vtk<->.obj)")


def convert_batch(batch_dir, out_dir, subject=None, final_frame=-1):
    from pathlib import Path
    batch_dir = Path(batch_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (batch_dir / "final.vtk").exists():
        cases = [(subject or batch_dir.name, batch_dir)]
    else:
        cases = [(d.name, d) for d in sorted(batch_dir.iterdir())
                 if d.is_dir() and (d / "final.vtk").exists()]
        if subject:
            raise SystemExit("--subject applies to a single-case --batch directory")
    if not cases:
        raise SystemExit(f"no deformation outputs (final.vtk / tp_XX.vtk) under {batch_dir}")
    for subj, case_dir in cases:
        tp_idx = []
        for p in sorted(case_dir.glob("tp_*.vtk")):
            m = re.match(r"tp_(\d+)$", p.stem)
            if m:
                tp_idx.append((int(m.group(1)), p))
        last = max((i for i, _ in tp_idx), default=-1)
        ffinal = final_frame if final_frame >= 0 else last + 1
        for i, p in tp_idx:
            v, fc = read_vtk(p)
            write_obj(v, fc, out_dir / f"{subj}_{i}.obj")
        v, fc = read_vtk(case_dir / "final.vtk")
        write_obj(v, fc, out_dir / f"{subj}_{ffinal}.obj")
        print(f"[convert] {case_dir.name}: {len(tp_idx)} tp frames + final -> "
              f"{subj}_*.obj (final frame {ffinal})")


def main():
    ap = argparse.ArgumentParser(description="convert between legacy ASCII VTK PolyData and Wavefront OBJ")
    ap.add_argument("input", nargs="?", help="single .vtk or .obj file")
    ap.add_argument("output", nargs="?", help="single-file output path")
    ap.add_argument("--batch", default=None,
                    help="deformation output directory (single case, or root of per-subject case directories)")
    ap.add_argument("--subject", default=None,
                    help="subject id used in batch output file names (single-case --batch only)")
    ap.add_argument("--out-dir", default=None, help="batch output directory")
    ap.add_argument("--final-frame", type=int, default=-1,
                    help="frame number assigned to final.vtk (default: one past the last tp_XX index)")
    args = ap.parse_args()

    if args.batch:
        if not args.out_dir:
            raise SystemExit("--out-dir is required with --batch")
        convert_batch(args.batch, args.out_dir, args.subject, args.final_frame)
        return
    if not (args.input and args.output):
        raise SystemExit("give INPUT OUTPUT for a single conversion, or --batch/--out-dir")
    from pathlib import Path
    convert_file(Path(args.input), Path(args.output))
    print(f"[convert] {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
