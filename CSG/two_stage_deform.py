import argparse
import glob
import os
import re
import shutil
import subprocess

import numpy as np
import trimesh

STAGE1 = {"def_kw": 5.0, "att_kw": 2.0, "noise": 0.5}
STAGE2 = {"def_kw": 4.0, "att_kw": 1.0, "noise": 0.4}
N_TP = 10
MAX_ITER = 120

DEFORM_PY = "python"  # set from --deform-py in main()


def load_mesh(path):
    m = trimesh.load(path, process=False, force="mesh")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32)


def write_vtk(verts, faces, path):
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n")
        f.write("POINTS {} float\n".format(len(verts)))
        for p in verts:
            f.write("{:.6f} {:.6f} {:.6f}\n".format(p[0], p[1], p[2]))
        if len(faces):
            f.write("POLYGONS {} {}\n".format(len(faces), len(faces) * 4))
            for tri in faces:
                f.write("3 {} {} {}\n".format(int(tri[0]), int(tri[1]), int(tri[2])))


def write_xmls(workdir, cfg):
    os.makedirs(os.path.join(workdir, "data"), exist_ok=True)
    with open(os.path.join(workdir, "model.xml"), "w") as f:
        f.write(
            '<?xml version="1.0"?>\n<model>\n'
            '  <model-type>Registration</model-type>\n  <dimension>3</dimension>\n'
            '  <template>\n    <object id="surf">\n'
            '      <deformable-object-type>SurfaceMesh</deformable-object-type>\n'
            '      <attachment-type>current</attachment-type>\n'
            '      <kernel-type>keops</kernel-type>\n'
            '      <kernel-width>{att_kw}</kernel-width>\n'
            '      <noise-std>{noise}</noise-std>\n'
            '      <filename>data/source.vtk</filename>\n'
            '    </object>\n  </template>\n  <deformation-parameters>\n'
            '    <kernel-type>keops</kernel-type>\n'
            '    <kernel-width>{def_kw}</kernel-width>\n'
            '    <number-of-timepoints>{n_tp}</number-of-timepoints>\n'
            '  </deformation-parameters>\n</model>\n'.format(
                att_kw=cfg["att_kw"], noise=cfg["noise"],
                def_kw=cfg["def_kw"], n_tp=N_TP))
    with open(os.path.join(workdir, "data_set.xml"), "w") as f:
        f.write(
            '<?xml version="1.0"?>\n<data-set>\n  <subject id="target">\n'
            '    <visit id="target">\n'
            '      <filename object_id="surf">data/target.vtk</filename>\n'
            '    </visit>\n  </subject>\n</data-set>\n')
    with open(os.path.join(workdir, "optimization_parameters.xml"), "w") as f:
        f.write(
            '<?xml version="1.0"?>\n<optimization-parameters>\n'
            '  <optimization-method-type>ScipyLBFGS</optimization-method-type>\n'
            '  <max-iterations>{}</max-iterations>\n'
            '  <convergence-tolerance>1e-6</convergence-tolerance>\n'
            '  <save-every-n-iters>10</save-every-n-iters>\n'
            '  <gpu-mode>full</gpu-mode>\n'
            '  <use-sobolev-gradient>On</use-sobolev-gradient>\n'
            '</optimization-parameters>\n'.format(MAX_ITER))


def run_stage(workdir, source_vtk, target_vtk, cfg, driver):
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "data"), exist_ok=True)
    outdir = os.path.join(workdir, "output")
    shutil.copy2(source_vtk, os.path.join(workdir, "data", "source.vtk"))
    shutil.copy2(target_vtk, os.path.join(workdir, "data", "target.vtk"))
    write_xmls(workdir, cfg)
    env = dict(os.environ, CUDA_HOME=os.environ.get("CUDA_HOME", "/usr/local/cuda"),
               KMP_DUPLICATE_LIB_OK="TRUE")
    cmd = [DEFORM_PY, driver, "estimate",
           os.path.join(workdir, "model.xml"),
           os.path.join(workdir, "data_set.xml"),
           "-p", os.path.join(workdir, "optimization_parameters.xml"),
           "-o", outdir, "-v", "INFO"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    with open(os.path.join(workdir, "estimate.log"), "w") as f:
        f.write(r.stdout + "\n--- STDERR ---\n" + r.stderr)
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-15:])
        raise RuntimeError(f"estimate failed in {workdir}\n{tail}")
    flow = {}
    for fn in glob.glob(os.path.join(outdir, "*flow*surf*tp_*.vtk")):
        m = re.search(r"tp_(\d+)\.vtk$", fn)
        if m:
            flow[int(m.group(1))] = load_mesh(fn)[0]
    if not flow:
        raise RuntimeError(f"no flow timepoints in {outdir}")
    recon = list(glob.glob(os.path.join(outdir, "*Reconstruction*surf*.vtk")))
    recon_v = load_mesh(recon[0])[0] if recon else None
    return flow, recon_v


def main():
    global DEFORM_PY
    ap = argparse.ArgumentParser(description="Minimal two-stage deformetrica LDDMM reproduction.")
    ap.add_argument("--source", required=True, help="source mesh (.obj/.vtk) — the mapped surface")
    ap.add_argument("--target", required=True, help="target mesh (.vtk) — the original surface")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--driver", required=True, help="path to run_deformetrica.py")
    ap.add_argument("--deform-py", default=DEFORM_PY,
                    help="python interpreter used to run the deformetrica driver")
    args = ap.parse_args()
    DEFORM_PY = args.deform_py
    if not os.path.isfile(args.driver):
        raise SystemExit(f"driver not found: {args.driver}\n"
                         f"pass --driver /path/to/run_deformetrica.py")

    os.makedirs(args.out, exist_ok=True)
    sv, sf = load_mesh(args.source)
    tv, tf = load_mesh(args.target)
    data = os.path.join(args.out, "_data")
    os.makedirs(data, exist_ok=True)
    src_vtk = os.path.join(data, "source.vtk")
    tgt_vtk = os.path.join(data, "target.vtk")
    write_vtk(sv, sf, src_vtk)
    write_vtk(tv, tf, tgt_vtk)

    print("[stage1] broad kernels", STAGE1, flush=True)
    flow1, _ = run_stage(os.path.join(args.out, "stage1"), src_vtk, tgt_vtk, STAGE1, args.driver)
    s1_end = flow1[max(flow1)]
    s1_src = os.path.join(data, "stage1_endpoint.vtk")
    write_vtk(s1_end, sf, s1_src)

    print("[stage2] tight kernels", STAGE2, flush=True)
    flow2, recon2 = run_stage(os.path.join(args.out, "stage2"), s1_src, tgt_vtk, STAGE2, args.driver)
    final_v = recon2 if recon2 is not None else flow2[max(flow2)]

    write_vtk(final_v, sf, os.path.join(args.out, "final.vtk"))
    n = 0
    for t in sorted(flow2):
        if recon2 is not None and np.allclose(flow2[t], final_v, atol=1e-5):
            continue
        write_vtk(flow2[t], sf, os.path.join(args.out, "tp_{:02d}.vtk".format(t)))
        n += 1
    print("[done] wrote final.vtk + {} timepoints to {}".format(n, args.out), flush=True)


if __name__ == "__main__":
    main()
