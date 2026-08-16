from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from patch_geometry import (
    N_PATCHES,
    DEPTH,
    PATCH_SIZE,
    read_vtk_mesh,
    read_obj,
    build_graph,
    patch_corner_vertex_ids,
    patch_centers,
    patch_adjacency,
    anchor_cost,
    center_cost,
)

DEFAULT_DEFORMED_FRAMES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]


def _safe_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return vectors / norms


def compute_face10_encoding(points, faces, patch_size, target_patches):
    face_vertices = points[faces]
    face_centers = face_vertices.mean(axis=1).astype(np.float32, copy=False)
    edges01 = face_vertices[:, 1] - face_vertices[:, 0]
    edges02 = face_vertices[:, 2] - face_vertices[:, 0]
    cross = np.cross(edges01, edges02)
    face_areas = (0.5 * np.linalg.norm(cross, axis=1)).astype(np.float32, copy=False)
    face_normals = _safe_normalize(cross).astype(np.float32, copy=False)

    def _angles(u, v):
        denom = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        denom = np.where(denom > 0, denom, 1.0)
        cos = np.clip(np.sum(u * v, axis=1) / denom, -1.0, 1.0)
        return np.arccos(cos)

    a0 = _angles(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    a1 = _angles(face_vertices[:, 0] - face_vertices[:, 1], face_vertices[:, 2] - face_vertices[:, 1])
    a2 = _angles(face_vertices[:, 0] - face_vertices[:, 2], face_vertices[:, 1] - face_vertices[:, 2])
    face_angles = np.stack([a0, a1, a2], axis=1).astype(np.float32, copy=False)

    vertex_normals = np.zeros_like(points, dtype=np.float64)
    for c in range(3):
        np.add.at(vertex_normals, faces[:, c], face_normals * face_angles[:, c:c + 1])
    vertex_normals = _safe_normalize(vertex_normals).astype(np.float32, copy=False)

    face_curvs = np.vstack([
        np.sum(vertex_normals[faces[:, 0]] * face_normals, axis=1),
        np.sum(vertex_normals[faces[:, 1]] * face_normals, axis=1),
        np.sum(vertex_normals[faces[:, 2]] * face_normals, axis=1),
    ]).astype(np.float32, copy=False)

    feats = np.vstack([
        face_areas[np.newaxis, :],
        face_normals.T,
        np.sort(face_angles, axis=1).T,
        np.sort(face_curvs, axis=0),
    ]).astype(np.float32, copy=False)

    face_count = faces.shape[0]
    patch_count = int(np.ceil(face_count / patch_size))
    if patch_count != target_patches:
        raise ValueError(f"patch_count {patch_count} != target_patches {target_patches} (face_count={face_count})")
    if patch_count * patch_size != face_count:
        raise ValueError(f"face_count {face_count} != patch_count*patch_size {patch_count * patch_size}")

    index_grid = np.arange(face_count, dtype=np.int64).reshape(patch_size, patch_count).T
    feats_patch = feats[:, index_grid]
    centers_patch = face_centers[index_grid]
    faces_patch = faces[index_grid].astype(np.int32, copy=False)
    coords_patch = face_vertices[index_grid].reshape(patch_count, patch_size, 9).astype(np.float32, copy=False)
    return feats_patch, centers_patch, coords_patch, faces_patch, np.int32(face_count)


def compute_hop_matrix(adj_edges, n_patch):
    rows = np.concatenate([adj_edges[:, 0], adj_edges[:, 1]])
    cols = np.concatenate([adj_edges[:, 1], adj_edges[:, 0]])
    data = np.ones(len(rows), dtype=np.int8)
    g = coo_matrix((data, (rows, cols)), shape=(n_patch, n_patch)).tocsr()
    hop = dijkstra(g, directed=False, unweighted=True)
    hop[~np.isfinite(hop)] = -1
    return hop.astype(np.int8)


def extract_from_loaded(vtk_path, obj_vertices, obj_faces):
    source_points, _ = read_vtk_mesh(str(vtk_path))
    tree = cKDTree(source_points)
    triplets = np.empty((N_PATCHES, 3), dtype=np.int64)
    max_dist = 0.0
    for pid in range(N_PATCHES):
        corners = patch_corner_vertex_ids(obj_faces, pid)
        dists, ids = tree.query(obj_vertices[corners], k=1)
        ids = np.sort(ids.astype(np.int64))
        if len(set(ids.tolist())) != 3:
            raise RuntimeError(f"duplicate source anchors in patch {pid} ({vtk_path})")
        triplets[pid] = ids
        max_dist = max(max_dist, float(np.max(dists)))
    centers = patch_centers(obj_vertices, obj_faces)
    adj = patch_adjacency(obj_faces)
    return triplets, centers, adj, max_dist


def open_arrays(out_dir, n, target_patches, patch_size):
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "data": np.lib.format.open_memmap(out_dir / "data.npy", mode="w+", dtype=np.float32, shape=(n, 10, target_patches, patch_size)),
        "coordinates": np.lib.format.open_memmap(out_dir / "coordinates.npy", mode="w+", dtype=np.float32, shape=(n, target_patches, patch_size, 9)),
        "centers": np.lib.format.open_memmap(out_dir / "centers.npy", mode="w+", dtype=np.float32, shape=(n, target_patches, patch_size, 3)),
        "faces": np.lib.format.open_memmap(out_dir / "faces.npy", mode="w+", dtype=np.int32, shape=(n, target_patches, patch_size, 3)),
        "face_counts": np.lib.format.open_memmap(out_dir / "face_counts.npy", mode="w+", dtype=np.int32, shape=(n,)),
        "canon_map": np.lib.format.open_memmap(out_dir / "canonical_mapping.npy", mode="w+", dtype=np.int64, shape=(n, target_patches)),
        "hop": np.lib.format.open_memmap(out_dir / "hop.npy", mode="w+", dtype=np.int8, shape=(n, target_patches, target_patches)),
    }


def self_check(out_dir, n, subject_ids, objs, vtks):
    print("\n=== self-check ===")
    data = np.load(out_dir / "data.npy", mmap_mode="r")
    coords = np.load(out_dir / "coordinates.npy", mmap_mode="r")
    centers = np.load(out_dir / "centers.npy", mmap_mode="r")
    faces = np.load(out_dir / "faces.npy", mmap_mode="r")
    counts = np.load(out_dir / "face_counts.npy")
    canon_map = np.load(out_dir / "canonical_mapping.npy")
    sids = np.load(out_dir / "subject_ids.npy", allow_pickle=True)

    assert data.shape[0] == n == len(sids) == len(objs), f"count mismatch: data={data.shape[0]} n={n} sids={len(sids)} objs={len(objs)}"
    print(f"[1] count: N={n} OK")

    assert coords.shape[0] == centers.shape[0] == faces.shape[0] == canon_map.shape[0] == n
    assert canon_map.shape[1] == N_PATCHES
    print(f"[2] shape alignment OK  data={data.shape} canon_map={canon_map.shape}")

    finite = np.isfinite(data).all()
    area_pos = (data[:, 0] > 0).all()
    ang = data[:, 4:7]
    angles_ok = ((ang >= 0).all() and (ang <= np.pi).all())
    n_sliver = int((ang <= 0).sum()) + int((ang >= np.pi).sum())
    curvs_ok = ((data[:, 7:10] >= -1.0 - 1e-4).all() and (data[:, 7:10] <= 1.0 + 1e-4).all())
    print(f"[3] feature health: finite={finite} area>0={area_pos} angles in [0,pi]={angles_ok} curvs in [-1,1]={curvs_ok}")
    print(f"    sliver (degenerate triangles): {n_sliver}/{ang.size}={n_sliver/ang.size:.2e}")
    if not (finite and area_pos and angles_ok and curvs_ok):
        print("    [WARN] feature health check not fully passed")

    from scipy.sparse import coo_matrix
    n_sample = min(5, n)
    edge_hops_all = []
    for i in range(n_sample):
        m = canon_map[i]
        is_perm = len(set(m.tolist())) == N_PATCHES
        if not is_perm:
            print(f"    [4] subject {sids[i]} canonical_mapping is not a permutation!")
    perm_ok = all(len(set(canon_map[i].tolist())) == N_PATCHES for i in range(n))
    print(f"[4] canonical_mapping permutation: {perm_ok} (all {n} subjects)")

    meta_path = out_dir / "reference_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"[5] reference anchor max_dist={meta.get('ref_max_anchor_dist','?')} "
              f"(reference={meta.get('reference_obj','?')}, anchor_dist time={meta.get('anchor_dist_sec','?')}s)")

    hop = np.load(out_dir / "hop.npy", mmap_mode="r")
    hop0 = np.asarray(hop[0])
    diag_ok = (np.diag(hop0) == 0).all()
    sym_ok = (hop0 == hop0.T).all()
    connected = (hop0 >= 0).all()
    print(f"[6] hop: shape={hop.shape} diag=0:{diag_ok} symmetric:{sym_ok} connected:{connected} "
          f"range=[{int(hop0.min())},{int(hop0.max())}]")
    print("=== self-check done ===")


def scan_mapped(obj_dir, vtk_dir, args):
    objs = sorted(obj_dir.glob("*.obj"))
    if not objs:
        sys.exit(f"no obj under {obj_dir}")

    sids = []
    pairs = []
    missing_vtk = []
    import re as _re
    _pat = _re.compile(args.sid_regex) if args.sid_regex else None
    _keep = set(Path(args.sid_list).read_text(encoding="utf-8").split()) if args.sid_list else None
    _skipped = 0
    for p in objs:
        if _pat:
            m = _pat.search(p.name)
            if not m:
                _skipped += 1
                continue
            sid = m.group("sid")
        else:
            sid = p.name.split(".")[0]
        if _keep is not None and sid not in _keep:
            _skipped += 1
            continue
        hemi_short = args.hemi[0]
        vtk_path = vtk_dir / (args.name_template.format(sid=sid, obj_stem=p.stem, hemi=args.hemi, hemi_short=hemi_short) + ".vtk")
        if not vtk_path.exists():
            missing_vtk.append(str(vtk_path))
            continue
        sids.append(sid)
        pairs.append((p, vtk_path))
    if _skipped:
        print(f"[sid-list] filtered {_skipped} obj (not in list)")
    if missing_vtk:
        print(f"[warn] {len(missing_vtk)} obj missing vtk, skipped; processing {len(pairs)}")
    return sids, pairs


def scan_deformed(obj_dir, frames, mapped_sids, sid_list):
    sid_to_idx = {str(s): i for i, s in enumerate(mapped_sids)}
    keep = set(Path(sid_list).read_text(encoding="utf-8").split()) if sid_list else None
    if not obj_dir.exists():
        sys.exit(f"no deformed dir {obj_dir}")

    entries = []
    skipped_frame = missing_mapped = 0
    for p in sorted(obj_dir.glob("*.obj")):
        try:
            base, fstr = p.stem.rsplit("_", 1)
            frame = int(fstr)
        except ValueError:
            continue
        sid = base.split(".")[0]
        if frame not in frames:
            skipped_frame += 1
            continue
        if sid not in sid_to_idx:
            missing_mapped += 1
            continue
        if keep is not None and sid not in keep:
            continue
        entries.append((sid_to_idx[sid], frame, p, sid))
    print(f"deformed: {len(entries)} (subj,frame) pairs; skipped_frame={skipped_frame} missing_mapped={missing_mapped}")
    if not entries:
        sys.exit("no (subj,frame) pairs")

    entries.sort()
    rows = [(midx, p) for midx, _, p, _ in entries]
    sids = [sid for _, _, _, sid in entries]
    frame_ids = [frame for _, frame, _, _ in entries]
    return rows, frame_ids, sids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-dir", required=True,
                        help="surface root containing {hemi}/ subdirectories (mapped or deformed obj)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--surface-type", choices=["mapped", "deformed"], required=True)
    parser.add_argument("--hemi", choices=["LH", "RH"], required=True)
    parser.add_argument("--target", default="b1024_d3")
    parser.add_argument("--limit", type=int, default=0, help=">0: process only first N subjects (debug)")
    parser.add_argument("--sid-list", default="", help="subject ID list file (one base sid per line); only process listed objs")
    parser.add_argument("--vtk-dir", default="", help="[mapped] vtk directory for the corresponding hemisphere")
    parser.add_argument("--name-template", default="{sid}.white_MSMAll.32k_{hemi}", help="[mapped] filename template (without extension)")
    parser.add_argument("--alpha-center", type=float, default=0.25)
    parser.add_argument("--ref-obj", default="",
                        help="[mapped] template mapped .obj whose patches define the canonical slots")
    parser.add_argument("--ref-vtk", default="",
                        help="[mapped] template native .vtk carrying the fs_LR vertex identities")
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument("--sid-regex", default="", help="[mapped] regex to extract base sid from obj filename (must contain group 'sid')")
    parser.add_argument("--mapped-feature-dir", default="",
                        help="[deformed] mapped feature output dir (canonical_mapping.npy / hop.npy / subject_ids.npy)")
    parser.add_argument("--frames", default=",".join(map(str, DEFAULT_DEFORMED_FRAMES)),
                        help="[deformed] comma-separated frame numbers, default even frames 0-18")
    args = parser.parse_args()

    settings = {"b1024_d3": (1024, 3)}
    if args.target not in settings:
        raise ValueError(f"unsupported target {args.target}")
    n_patches, depth = settings[args.target]
    patch_size = 4 ** depth
    assert depth == DEPTH and patch_size == PATCH_SIZE, f"depth/patch_size mismatch: {depth}/{patch_size}"
    import patch_geometry as _pg
    _pg.N_PATCHES = n_patches
    global N_PATCHES
    N_PATCHES = n_patches

    obj_dir = Path(args.surface_dir) / args.hemi
    out_dir = Path(args.out_dir)

    if args.surface_type == "mapped":
        if not args.vtk_dir:
            sys.exit("--vtk-dir is required for --surface-type mapped")
        vtk_dir = Path(args.vtk_dir)
        sids, pairs = scan_mapped(obj_dir, vtk_dir, args)
        if args.limit > 0:
            pairs = pairs[: args.limit]
            sids = sids[: args.limit]
        if not args.ref_obj or not args.ref_vtk:
            sys.exit("--ref-obj and --ref-vtk are required for --surface-type mapped (canonical template)")
        ref_obj, ref_vtk = Path(args.ref_obj), Path(args.ref_vtk)
        if not ref_obj.is_file() or not ref_vtk.is_file():
            sys.exit(f"canonical reference not found: {ref_obj} / {ref_vtk}")

        n = len(pairs)
        print(f"[{args.hemi}] type=mapped subjects={n} patch_size={patch_size} target_patches={n_patches}")
        print(f"out_dir={out_dir}")
        t0 = time.time()
        ref_obj_v, ref_obj_f = read_obj(str(ref_obj))
        canon_triplets, canon_centers, canon_adj, ref_max_dist = extract_from_loaded(ref_vtk, ref_obj_v, ref_obj_f)
        unique_canon = np.array(sorted(set(canon_triplets.reshape(-1).tolist())), dtype=np.int64)
        ref_vtk_points, ref_vtk_faces = read_vtk_mesh(str(ref_vtk))
        ref_vtk_graph = build_graph(len(ref_vtk_points), ref_vtk_faces)
        anchor_dist = dijkstra(ref_vtk_graph, directed=False, unweighted=True, indices=unique_canon).astype(np.float32)
        anchor_dist_sec = time.time() - t0
        print(f"canonical template: unique_anchors={len(unique_canon)} max_anchor_dist={ref_max_dist:.3e} "
              f"anchor_dist time={anchor_dist_sec:.1f}s")

        arrs = open_arrays(out_dir, n, n_patches, patch_size)

        t_start = time.time()
        for i, ((obj_path, vtk_path), sid) in enumerate(zip(pairs, sids)):
            obj_v, obj_f = read_obj(str(obj_path))
            if len(obj_f) != n_patches * patch_size:
                sys.exit(f"{obj_path}: face count {len(obj_f)} != {n_patches * patch_size}")
            feats, cen, coord, fidx, fcount = compute_face10_encoding(obj_v, obj_f, patch_size, n_patches)
            raw_triplets, raw_centers, raw_adj, max_dist = extract_from_loaded(vtk_path, obj_v, obj_f)
            ca = anchor_cost(canon_triplets, raw_triplets, anchor_dist, unique_canon)
            cc = center_cost(canon_centers, raw_centers)
            cost = ca / (np.median(ca) + 1e-6) + args.alpha_center * cc / (np.median(cc) + 1e-6)
            rows, cols = linear_sum_assignment(cost)
            raw_to_canon = np.empty(n_patches, dtype=np.int64)
            raw_to_canon[cols] = rows
            arrs["data"][i] = feats
            arrs["coordinates"][i] = coord
            arrs["centers"][i] = cen
            arrs["faces"][i] = fidx
            arrs["face_counts"][i] = fcount
            arrs["canon_map"][i] = raw_to_canon
            arrs["hop"][i] = compute_hop_matrix(raw_adj, n_patches)
            if (i + 1) % 10 == 0 or i == 0 or i == n - 1:
                elapsed = time.time() - t_start
                eta = elapsed / (i + 1) * (n - i - 1)
                print(f"  [{i + 1}/{n}] {sid} max_anchor_dist={max_dist:.3e} "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        for k, v in arrs.items():
            v.flush()
        np.save(out_dir / "subject_ids.npy", np.asarray(sids, dtype=object))
        meta = {
            "surface_type": "mapped", "dataset_dir": str(args.surface_dir), "hemi": args.hemi,
            "target": args.target,
            "n_subjects": n, "reference_obj": str(ref_obj), "reference_vtk": str(ref_vtk),
            "alpha_center": args.alpha_center, "ref_max_anchor_dist": float(ref_max_dist),
            "anchor_dist_sec": round(anchor_dist_sec, 1), "total_sec": round(time.time() - t_start, 1),
        }
        (out_dir / "reference_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"\ndone: {n} subjects, total time {time.time() - t_start:.0f}s")

        if not args.skip_self_check:
            self_check(out_dir, n, sids, [p for p, _ in pairs], [v for _, v in pairs])

    else:
        if not args.mapped_feature_dir:
            sys.exit("--mapped-feature-dir is required for --surface-type deformed")
        frames = sorted(set(int(x) for x in args.frames.split(",")))
        print(f"selected frames: {frames}")

        mapped_dir = Path(args.mapped_feature_dir)
        canon_all = np.load(mapped_dir / "canonical_mapping.npy")
        hop_all = np.load(mapped_dir / "hop.npy", mmap_mode="r")
        mapped_sids = np.load(mapped_dir / "subject_ids.npy", allow_pickle=True)
        print(f"mapped: {len(mapped_sids)} subjects, source {mapped_dir}")

        rows, frame_ids, sids = scan_deformed(obj_dir, frames, mapped_sids, args.sid_list)
        if args.limit > 0:
            rows, frame_ids, sids = rows[: args.limit], frame_ids[: args.limit], sids[: args.limit]

        n = len(rows)
        print(f"[{args.hemi}] type=deformed pairs={n} frames={len(frames)} patch_size={patch_size} target_patches={n_patches}")
        print(f"out_dir={out_dir}")

        arrs = open_arrays(out_dir, n, n_patches, patch_size)

        t_start = time.time()
        for i, ((mapped_idx, obj_path), sid, frame) in enumerate(zip(rows, sids, frame_ids)):
            obj_v, obj_f = read_obj(str(obj_path))
            if len(obj_f) != n_patches * patch_size:
                sys.exit(f"{obj_path}: face count {len(obj_f)} != {n_patches * patch_size}")
            feats, cen, coord, fidx, fcount = compute_face10_encoding(obj_v, obj_f, patch_size, n_patches)
            arrs["data"][i] = feats
            arrs["coordinates"][i] = coord
            arrs["centers"][i] = cen
            arrs["faces"][i] = fidx
            arrs["face_counts"][i] = fcount
            arrs["canon_map"][i] = canon_all[mapped_idx]
            arrs["hop"][i] = hop_all[mapped_idx]
            if (i + 1) % 50 == 0 or i == 0 or i == n - 1:
                elapsed = time.time() - t_start
                eta = elapsed / (i + 1) * (n - i - 1)
                print(f"  [{i + 1}/{n}] {sid} frame={frame} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        for k, v in arrs.items():
            v.flush()
        np.save(out_dir / "subject_ids.npy", np.asarray(sids, dtype=object))
        np.save(out_dir / "frame_ids.npy", np.asarray(frame_ids, dtype=np.int32))
        meta = {
            "surface_type": "deformed", "dataset_dir": str(args.surface_dir), "hemi": args.hemi,
            "target": args.target, "frames": frames,
            "mapped_feature_dir": str(mapped_dir), "n_pairs": n,
            "total_sec": round(time.time() - t_start, 1),
        }
        (out_dir / "reference_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"\ndone: {n} (subj,frame) pairs, total time {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
