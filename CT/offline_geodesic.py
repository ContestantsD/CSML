import sys, time, argparse
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from multiprocessing import Pool

N_PATCHES = 1024
PATCH_SIZE = 64

_G = {}


def _init(feat_dir, vmax):
    _G["faces"] = np.load(Path(feat_dir) / "faces.npy", mmap_mode="r")
    _G["coords"] = np.load(Path(feat_dir) / "coordinates.npy", mmap_mode="r")
    _G["vmax"] = vmax


def subj_geodesic(i):
    faces_i = np.asarray(_G["faces"][i])
    obj_faces = faces_i.transpose(1, 0, 2).reshape(-1, 3)
    oc = np.asarray(_G["coords"][i]).transpose(1, 0, 2).reshape(-1, 9).reshape(-1, 3, 3)
    vmax = _G["vmax"]
    all_vi = obj_faces.reshape(-1)
    all_xyz = oc.reshape(-1, 3)
    order = np.argsort(all_vi, kind="stable")
    svi = all_vi[order]
    uniq_mask = np.concatenate(([True], svi[1:] != svi[:-1]))
    pos = np.full((vmax, 3), np.nan, dtype=np.float64)
    pos[svi[uniq_mask]] = all_xyz[order[uniq_mask]]
    a, b, c = obj_faces[:, 0], obj_faces[:, 1], obj_faces[:, 2]
    R = np.concatenate([a, b, c, a, b, c]); C = np.concatenate([b, c, a, c, a, b])
    W = np.linalg.norm(pos[R] - pos[C], axis=1)
    G = coo_matrix((W, (R, C)), shape=(vmax, vmax)).tocsr()
    face_ids = np.arange(N_PATCHES * PATCH_SIZE, dtype=np.int64).reshape(PATCH_SIZE, N_PATCHES).T
    pc = pos[obj_faces[face_ids]].mean(axis=(1, 2))
    valid_idx = np.where(np.isfinite(pos[:, 0]))[0]
    reps = valid_idx[cKDTree(pos[valid_idx]).query(pc, k=1)[1]]
    dist = dijkstra(G, indices=reps.tolist(), directed=False)
    geo = dist[:, reps]
    return i, ((geo + geo.T) / 2.0).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feat_dir")
    ap.add_argument("limit", nargs="?", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--subject-list", default="",
                    help="file of subject ids (one per line); only compute these rows")
    args = ap.parse_args()
    feat_dir = Path(args.feat_dir)
    faces = np.load(feat_dir / "faces.npy", mmap_mode="r")
    N, P, ps, _ = faces.shape
    global N_PATCHES
    N_PATCHES = P
    assert ps == PATCH_SIZE, f"patch_size {ps} != {PATCH_SIZE} (shape {faces.shape})"
    vmax = int(faces.max()) + 1
    n = N if args.limit <= 0 else min(args.limit, N)
    sids = np.load(feat_dir / "subject_ids.npy", allow_pickle=True)
    if args.subject_list:
        allow = {ln.split()[0] for ln in open(args.subject_list) if ln.strip()}
        idx_list = [i for i in range(n) if str(sids[i]) in allow]
        print(f"[geodesic] subject-list {len(allow)} requested -> {len(idx_list)}/{n} matched", flush=True)
    else:
        idx_list = list(range(n))
    print(f"[geodesic] {feat_dir} N={N} P={P} ps={ps} vmax={vmax} (computing {len(idx_list)}, {args.workers} workers)", flush=True)

    out_path = feat_dir / "geodesic.npy"
    geo = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(N, P, P))
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(str(feat_dir), vmax)) as pool:
        done = 0
        for i, g in pool.imap_unordered(subj_geodesic, idx_list, chunksize=4):
            geo[i] = g; done += 1
            if done % 50 == 0 or done == n:
                el = time.time() - t0; eta = el / done * (n - done)
                print(f"  [{done}/{n}] elapsed={el:.0f}s eta={eta:.0f}s", flush=True)
    geo.flush()
    print("=== geodesic self-check ===", flush=True)
    hop = np.load(feat_dir / "hop.npy", mmap_mode="r")
    for i in range(min(3, n)):
        g = np.asarray(geo[i]).astype(np.float32); h = np.asarray(hop[i])
        print(f"  subj[{i}] geo=[{g[g > 0].min():.2f},{g.max():.2f}]mm hop=[{int(h.min())},{int(h.max())}] "
              f"diag0={(np.diag(g) < 0.1).all()} size~{N*P*P*2/1e9:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
