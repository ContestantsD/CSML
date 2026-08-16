import argparse
import os
import numpy as np
from scipy.spatial import cKDTree


def main():
    ap = argparse.ArgumentParser(description="append patch-level IG attribution to 32k surface VTKs")
    ap.add_argument("--res-dir", required=True, help="directory containing ig_mean_LH.npy / ig_mean_RH.npy")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--subject", required=True, help="subject id of the source 32k VTK pair")
    ap.add_argument("--src-dir", required=True,
                    help="directory containing the source 32k VTKs")
    ap.add_argument("--lh-dir", required=True,
                    help="left-hemisphere offline feature directory (faces.npy / coordinates.npy)")
    ap.add_argument("--rh-dir", required=True,
                    help="right-hemisphere offline feature directory (faces.npy / coordinates.npy)")
    ap.add_argument("--src-pattern", default="{subject}_tfMRI_EMOTION_{tag}.vtk",
                    help="source VTK filename template; {subject} and {tag} (L or R) are substituted")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    src_vtk_dir = args.src_dir

    ig_all = np.concatenate([np.load(os.path.join(args.res_dir, 'ig_mean_%s.npy' % h)) for h in ['LH', 'RH']])
    Q99 = float(np.percentile(ig_all, 99))
    print('global Q99 (1024 patches): %.6f' % Q99)

    for hemi, tag in [('LH', 'L'), ('RH', 'R')]:
        feat_dir = args.lh_dir if hemi == 'LH' else args.rh_dir
        faces = np.load(os.path.join(feat_dir, 'faces.npy'))[0]
        coords = np.load(os.path.join(feat_dir, 'coordinates.npy'))[0]
        ig = np.load(os.path.join(args.res_dir, 'ig_mean_%s.npy' % hemi))

        face_centers = coords.reshape(-1, 3, 3).mean(axis=1)
        face_patch = np.repeat(np.arange(1024), 64)

        src = os.path.join(src_vtk_dir, args.src_pattern.format(subject=args.subject, tag=tag))
        lines = open(src, encoding='utf-8', errors='replace').read().splitlines()

        n_points = int(next(l.split()[1] for l in lines if l.startswith('POINTS')))
        i_pnt = next(i for i, l in enumerate(lines) if l.startswith('POINTS')) + 1
        verts = np.array([[float(x) for x in l.split()] for l in lines[i_pnt:i_pnt + n_points]], np.float32)
        assert len(verts) == n_points == 32492

        tree = cKDTree(face_centers)
        d, nidx = tree.query(verts, k=1)
        v_patch = face_patch[nidx]
        ig_v = ig[v_patch].astype(np.float32)
        ig_n = np.minimum(ig_v / Q99, 1.0).astype(np.float32)

        out = os.path.join(args.out_dir, '%s_%s_ig_patch_32k.vtk' % (args.subject, hemi))
        with open(out, 'w') as fp:
            fp.write('\n'.join(lines))
            fp.write('\nSCALARS ig_patch float 1\nLOOKUP_TABLE default\n')
            for v in ig_v:
                fp.write('%.8g\n' % v)
            fp.write('SCALARS ig_norm float 1\nLOOKUP_TABLE default\n')
            for v in ig_n:
                fp.write('%.8g\n' % v)
        print('%s: appended ig_patch [%.4g, %.4g] + ig_norm [%.4g, %.4g] to %s (%.1f MB)' % (
            hemi, ig_v.min(), ig_v.max(), ig_n.min(), ig_n.max(), out, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
