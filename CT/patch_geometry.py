from collections import Counter
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
import vtk


N_PATCHES = 2048
DEPTH = 3
PATCH_SIZE = 4 ** DEPTH


@lru_cache(maxsize=256)
def read_vtk_mesh(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    pts = poly.GetPoints()
    if pts is None:
        raise RuntimeError(f"no points in {path}")

    points = np.empty((pts.GetNumberOfPoints(), 3), dtype=np.float32)
    for i in range(pts.GetNumberOfPoints()):
        points[i] = pts.GetPoint(i)

    faces = []
    id_list = vtk.vtkIdList()
    polys = poly.GetPolys()
    polys.InitTraversal()
    while polys.GetNextCell(id_list):
        if id_list.GetNumberOfIds() == 3:
            faces.append([id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)])
    return points, np.asarray(faces, dtype=np.int64)


def read_obj(path):
    verts = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                tri = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
                if len(tri) == 3:
                    faces.append(tuple(tri))
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def build_graph(n_vertices, faces):
    edges = set()
    for tri in faces:
        a, b, c = map(int, tri)
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add(tuple(sorted((u, v))))
    rows, cols = [], []
    for a, b in edges:
        rows.extend([a, b])
        cols.extend([b, a])
    vals = np.ones(len(rows), dtype=np.float32)
    return coo_matrix((vals, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()


def patch_corner_vertex_ids(faces, patch_id):
    face_ids = patch_id + np.arange(PATCH_SIZE, dtype=np.int64) * N_PATCHES
    degree = Counter()
    used = set()
    for tri in faces[face_ids]:
        a, b, c = map(int, tri)
        used.update((a, b, c))
        for u, v in ((a, b), (b, c), (c, a)):
            degree[u] += 1
            degree[v] += 1
    corners = [v for v in used if degree[v] == 2]
    if len(corners) != 3:
        raise RuntimeError(f"patch {patch_id} has {len(corners)} corners")
    return corners


def patch_centers(obj_vertices, obj_faces):
    face_ids = np.arange(N_PATCHES * PATCH_SIZE, dtype=np.int64).reshape(PATCH_SIZE, N_PATCHES).T
    return obj_vertices[obj_faces[face_ids]].mean(axis=(1, 2))


def patch_adjacency(obj_faces):
    edge_owner = {}
    adj = set()
    for patch_id in range(N_PATCHES):
        for t in range(PATCH_SIZE):
            tri = obj_faces[patch_id + t * N_PATCHES]
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge = tuple(sorted((int(a), int(b))))
                previous = edge_owner.get(edge)
                if previous is None:
                    edge_owner[edge] = patch_id
                elif previous != patch_id:
                    adj.add(tuple(sorted((previous, patch_id))))
    return np.asarray(sorted(adj), dtype=np.int64)


def anchor_cost(canon_triplets, raw_triplets, anchor_dist, unique_canon):
    row_of = {int(v): i for i, v in enumerate(unique_canon)}
    canon_rows = np.vectorize(lambda x: row_of[int(x)])(canon_triplets)
    raw_flat = raw_triplets.reshape(-1)
    cost = np.empty((N_PATCHES, N_PATCHES), dtype=np.float32)
    for start in range(0, N_PATCHES, 256):
        stop = min(N_PATCHES, start + 256)
        block = anchor_dist[canon_rows[start:stop]][:, :, raw_flat]
        d = block.reshape(stop - start, 3, N_PATCHES, 3).transpose(0, 2, 1, 3)
        raw_to_canon = d.min(axis=2).mean(axis=2)
        canon_to_raw = d.min(axis=3).mean(axis=2)
        cost[start:stop] = 0.5 * (raw_to_canon + canon_to_raw)
    return cost


def center_cost(canon_centers, raw_centers):
    diff = canon_centers[:, None, :] - raw_centers[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)).astype(np.float32)
