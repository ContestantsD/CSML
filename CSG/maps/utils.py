from typing import List
import numpy as np

def check_duplicated(V) -> bool:
    np.unique(V, return_counts=True)
    for i in range(V.shape[0]):
        for j in range(i+1, V.shape[0]):
            if np.abs(V[i] - V[j]).sum() < 1e-7:
                return i, j
    return False

def maximal_independent_set(vids, faces, vertex_faces, n_vertices=None) -> List:
    if n_vertices is None:
        mark = {}
        is_marked = lambda v: mark.get(v, False)
        mark_vertices = lambda vertices: mark.update((int(v), True) for v in vertices)
    else:
        mark = np.zeros(int(n_vertices), dtype=bool)
        is_marked = lambda v: mark[int(v)]
        mark_vertices = lambda vertices: mark.__setitem__(vertices, True)

    mis = []
    for v in vids:
        v = int(v)
        if not is_marked(v):
            mis.append(v)
            for fid in vertex_faces[v]:
                mark_vertices(faces[fid])
    return mis
