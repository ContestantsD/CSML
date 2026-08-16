from maps.utils import maximal_independent_set
from collections import defaultdict, deque
import gc
from itertools import islice
from time import time
from typing import Dict, List, Set

import networkx as nx
import numpy as np
from sortedcollections import ValueSortedDict
from tqdm import tqdm
from trimesh import PointCloud
from trimesh import Trimesh

from .geometry import face_areas, min_triangle_angles
from .geometry import plane_from_points
from .geometry import to_barycentric, from_barycenteric
from .geometry import CDT, MVT
from .geometry import one_ring_neighbor_uv


def percentile_rank(values):
    values = np.asarray(values)
    if values.size == 0:
        return values.astype(float)

    unique, inverse = np.unique(values, return_inverse=True)
    if unique.size == 1:
        return np.zeros_like(values, dtype=float)

    unique_ranks = np.linspace(0.0, 1.0, unique.size)
    return unique_ranks[inverse]


def topological_distance(n_vertices, faces, seeds):
    adj = [set() for _ in range(n_vertices)]
    for face in faces:
        for i, u in enumerate(face):
            v = face[(i + 1) % len(face)]
            adj[u].add(v)
            adj[v].add(u)

    dist = np.full(n_vertices, -1, dtype=np.int32)
    queue = deque()
    for seed in seeds:
        if 0 <= seed < n_vertices and dist[seed] < 0:
            dist[seed] = 0
            queue.append(seed)

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(v)

    return dist


class Mesh:
    def __init__(self, vertices, faces):
        mesh = Trimesh(vertices, faces, process=False, maintain_order=True)
        self.verts = vertices.copy()
        self.faces = faces.copy()
        self.vertex_faces = [set(F) for F in mesh.vertex_faces]
        for fs in self.vertex_faces:
            if -1 in fs:
                fs.remove(-1)

        self.V = self.verts.shape[0]
        self.F = self.faces.shape[0]
        self.vmask = np.ones(self.V, dtype=bool)
        self.fmask = np.ones(self.F, dtype=bool)
        self.active_faces = self.F

    def neighbors(self, i: int) -> Set[int]:
        N = set()
        for f in self.vertex_faces[i]:
            N.add(self.faces[f, 0])
            N.add(self.faces[f, 1])
            N.add(self.faces[f, 2])
        N.remove(i)
        return N

    def one_ring_neighbors(self, i: int) -> List[int]:
        try:
            return self._one_ring_neighbors_fast(i)
        except Exception:
            return self._one_ring_neighbors_networkx(i)

    def _one_ring_neighbors_fast(self, i: int) -> List[int]:
        ring_adj = defaultdict(set)
        incident_faces = [fid for fid in self.vertex_faces[i] if self.fmask[fid]]
        if not incident_faces:
            raise ValueError("vertex has no active incident faces")

        for fid in incident_faces:
            face = self.faces[fid]
            others = [int(v) for v in face if v != i]
            if len(others) != 2 or others[0] == others[1]:
                raise ValueError("invalid incident face")
            a, b = others
            if not self.vmask[a] or not self.vmask[b]:
                raise ValueError("inactive ring vertex")
            ring_adj[a].add(b)
            ring_adj[b].add(a)

        if any(len(neighbors) != 2 for neighbors in ring_adj.values()):
            raise ValueError("one-ring is not a simple cycle")

        start = next(iter(ring_adj))
        cycle = [start]
        seen = {start}
        prev = None
        curr = start
        while True:
            next_candidates = [v for v in ring_adj[curr] if v != prev]
            if prev is None:
                if len(next_candidates) != 2:
                    raise ValueError("invalid cycle start")
                nxt = next_candidates[0]
            else:
                if len(next_candidates) != 1:
                    raise ValueError("cycle branch detected")
                nxt = next_candidates[0]

            if nxt == start:
                break
            if nxt in seen:
                raise ValueError("cycle closes before visiting all vertices")
            cycle.append(nxt)
            seen.add(nxt)
            prev, curr = curr, nxt

        if len(cycle) != len(ring_adj):
            raise ValueError("cycle did not visit all vertices")

        return self._orient_one_ring(i, cycle)

    def _one_ring_neighbors_networkx(self, i: int) -> List[int]:
        G = nx.Graph()
        for f in self.vertex_faces[i]:
            if not self.fmask[f]:
                continue
            G.add_edge(self.faces[f, 0], self.faces[f, 1])
            G.add_edge(self.faces[f, 1], self.faces[f, 2])
            G.add_edge(self.faces[f, 2], self.faces[f, 0])
        cycle = nx.cycle_basis(G.subgraph(G[i]))[0]
        return self._orient_one_ring(i, cycle)

    def _orient_one_ring(self, i: int, cycle: List[int]) -> List[int]:
        u, v = cycle[0], cycle[1]
        for f in self.vertex_faces[i]:
            if not self.fmask[f]:
                continue
            if u in self.faces[f] and v in self.faces[f]:
                clockwise = (
                    (u == self.faces[f, 0] and v == self.faces[f, 1])
                    or (u == self.faces[f, 1] and v == self.faces[f, 2])
                    or (u == self.faces[f, 2] and v == self.faces[f, 0])
                )
                if not clockwise:
                    cycle = cycle[::-1]
                break
        else:
            raise Exception("Impossible")
        return cycle

    def add_vertex(self, vertex):
        if self.V + 1 > self.verts.shape[0]:
            self.vmask = np.append(self.vmask, np.zeros_like(self.vmask))
            self.verts = np.append(self.verts, np.zeros_like(self.verts), axis=0)
        self.verts[self.V] = vertex
        self.vmask[self.V] = True
        self.vertex_faces.append(set())
        self.V += 1

    def remove_face(self, fid):
        if not self.fmask[fid]:
            return
        self.fmask[fid] = False
        self.active_faces -= 1
        for v in self.faces[fid]:
            self.vertex_faces[v].remove(fid)

    def remove_faces(self, fids):
        active_fids = [fid for fid in fids if self.fmask[fid]]
        if not active_fids:
            return
        self.fmask[active_fids] = False
        self.active_faces -= len(active_fids)
        for fid in active_fids:
            for v in self.faces[fid]:
                self.vertex_faces[v].remove(fid)

    def add_faces(self, new_faces):
        for v0, v1, v2 in new_faces:
            assert v0 != v1 and v1 != v2 and v2 != v0

        if self.F + len(new_faces) > self.faces.shape[0]:
            self.fmask = np.append(self.fmask, np.zeros_like(self.fmask))
            self.faces = np.append(self.faces, np.zeros_like(self.faces), axis=0)
        self.faces[self.F : self.F + len(new_faces)] = new_faces
        self.fmask[self.F : self.F + len(new_faces)] = True
        self.active_faces += len(new_faces)

        for fid, face in enumerate(new_faces):
            for v in face:
                self.vertex_faces[v].add(fid + self.F)

        self.F += len(new_faces)

    def remove_vertex(self, i, new_faces, neighbors=None):
        old_faces = self.vertex_faces[i]
        self.remove_faces(list(self.vertex_faces[i]))
        self.add_faces(new_faces)

        for k in neighbors:
            for f in old_faces:
                if f in self.vertex_faces[k]:
                    self.vertex_faces[k].remove(f)
            for j, face in enumerate(new_faces):
                if k in face:
                    self.vertex_faces[k].add(j + self.F - len(new_faces))


class BaseMesh(Mesh):
    def __init__(self, vertices, faces):
        super().__init__(vertices, faces)
        self.face_distortion = {i: 1 for i in range(self.F)}

    def assign_initial_vertex_weights(self):
        Q = {}
        for i in range(self.V):
            Q[i] = np.zeros([4, 4])
            for fid in self.vertex_faces[i]:
                plane = plane_from_points(self.verts[self.faces[fid]])
                plane = plane.reshape([1, 4])
                Q[i] += plane.T @ plane

        vertex_weights = ValueSortedDict()
        for i in range(self.V):
            vertex_weights[i] = self.compute_vertex_weights(i, Q)
        return vertex_weights, Q

    def compute_vertex_weights(self, i, Q: Dict):
        weight = 0
        coord = np.ones([1, 4])
        for v in self.neighbors(i):
            coord[0, :3] = self.verts[v]
            weight += coord @ Q[i] @ coord.T
        return weight

    def is_validate_removal(self, i: int, neighbors, new_faces, new_edges):
        if not self.is_manifold(new_edges, neighbors):
            return False
        return True

    def is_manifold(self, new_edges, neighbors):
        if len(new_edges) == 0:
            return True

        edge_set = {
            tuple(sorted((int(a), int(b))))
            for a, b in np.asarray(new_edges, dtype=np.int64)
        }
        seen_faces = set()
        for v in neighbors:
            for f in self.vertex_faces[v]:
                if f in seen_faces or not self.fmask[f]:
                    continue
                seen_faces.add(f)
                a, b, c = (int(x) for x in self.faces[f])
                if (
                    tuple(sorted((a, b))) in edge_set
                    or tuple(sorted((b, c))) in edge_set
                    or tuple(sorted((c, a))) in edge_set
                ):
                    return False
        return True


class ParamMesh(Mesh):
    def __init__(self, vertices, faces):
        super().__init__(vertices, faces)

        self.xyz = vertices.copy()
        self.baries = defaultdict(dict)
        self.on_edge = defaultdict(dict)

    def add_xyz(self, points_uv, uv, edge):
        if self.xyz.shape != self.verts.shape:
            self.xyz = np.append(self.xyz, np.zeros_like(self.xyz), axis=0)

        a = np.array(points_uv[edge[1]]) - points_uv[edge[0]]
        b = np.array(uv) - points_uv[edge[0]]
        t = np.linalg.norm(b) / np.linalg.norm(a)
        self.xyz[self.V - 1] = t * self.xyz[edge[1]] + (1 - t) * self.xyz[edge[0]]

    def is_watertight(self) -> bool:
        verts = self.verts[self.vmask]
        faces = self.faces[self.fmask]
        mesh = Trimesh(verts, faces, process=False)
        return mesh.is_watertight and mesh.is_winding_consistent

    def split_triangles_on_segments(self, points_uv: Dict, points_on_ring: Dict, lines):
        for line in lines:
            candidate_faces = set()
            for v in points_uv:
                if v not in points_on_ring:
                    for fid in self.vertex_faces[v]:
                        if all([u in points_uv for u in self.faces[fid]]):
                            candidate_faces.add(fid)

            intersections = defaultdict(list)
            point_on_edge = {}
            for fid in candidate_faces:
                v0, v1, v2 = self.faces[fid]
                for edge in (
                    tuple(sorted([v0, v1])),
                    tuple(sorted([v1, v2])),
                    tuple(sorted([v2, v0])),
                ):
                    ret = self.intersect(points_uv, edge, line)
                    if ret is None:
                        continue
                    if isinstance(ret[0], tuple):
                        new_vertex = point_on_edge.get(edge)
                        if new_vertex is None:
                            self.add_vertex([0, 0, 0])
                            self.add_xyz(points_uv, ret[0], edge)
                            new_vertex = self.V - 1
                            points_uv[new_vertex] = ret[0]
                            self.on_edge[new_vertex] = line
                            point_on_edge[edge] = new_vertex
                        ret[1] = new_vertex
                    ret += (edge,)
                    if ret not in intersections[fid]:
                        intersections[fid].append(ret)

            for fid, its in intersections.items():
                if len(its) == 3:
                    u0 = [x[1] for x in its if x[0] is None]
                    u = [x[1] for x in its if x[0] is not None]
                    if len(u0) != 2:
                        assert len(u0) == 2, "len(u0) != 2"
                    assert u0[0] == u0[1], "u0[0] != u0[1]"
                    self.split_into_two_triangle(fid, u0[0], u[0])
                elif len(its) == 2:
                    if its[0][0] is not None and its[1][0] is not None:
                        self.split_into_tri_trap(fid, its[0][1:], its[1][1:], points_uv)
                    elif (its[0][0] is None) ^ (its[1][0] is None):
                        raise Exception("[Impossible]")
                elif len(its) == 1:
                    raise Exception("[Impossible]")

    def split_into_two_triangle(self, fid, u0, u):
        v0, v1, v2 = self.faces[fid]
        self.remove_face(fid)
        if v1 == u0:
            v0, v1, v2 = v1, v2, v0
        elif v2 == u0:
            v0, v1, v2 = v2, v0, v1
        assert v0 == u0, "v0 != u0"
        self.add_faces([[v0, v1, u], [v0, u, v2]])

    def split_into_tri_trap(self, fid, edge0, edge1, points_uv):
        v0, v1, v2 = self.faces[fid]
        self.remove_face(fid)
        u0, (e0v0, e0v1) = edge0
        u1, (e1v0, e1v1) = edge1

        def on_edge(x0, x1):
            if sorted([x0, x1]) == sorted([e0v0, e0v1]):
                return u0
            if sorted([x0, x1]) == sorted([e1v0, e1v1]):
                return u1
            return None

        e0 = on_edge(v0, v1)
        e1 = on_edge(v1, v2)
        e2 = on_edge(v2, v0)

        if e0 is None:
            choice_a = [[v2, e2, e1], [v0, v1, e1], [v0, e1, e2]]
            choice_b = [[v2, e2, e1], [v1, e1, e2], [v1, e2, v0]]
        elif e1 is None:
            choice_a = [[v0, e0, e2], [v1, e2, e0], [v1, v2, e2]]
            choice_b = [[v0, e0, e2], [v2, e2, e0], [v2, e0, v1]]
        elif e2 is None:
            choice_a = [[v1, e1, e0], [v2, e0, e1], [v2, v0, e0]]
            choice_b = [[v1, e1, e0], [v0, e1, v2], [v0, e0, e1]]
        else:
            raise Exception("[Impossible]")

        def make_triangle(vids):
            return np.array([points_uv[vids[0]], points_uv[vids[1]], points_uv[vids[2]]])

        min_a = min(min_triangle_angles(make_triangle(choice_a[1])),
                    min_triangle_angles(make_triangle(choice_a[2])))
        min_b = min(min_triangle_angles(make_triangle(choice_b[1])),
                    min_triangle_angles(make_triangle(choice_b[2])))

        if min_a > min_b:
            return self.add_faces(choice_a)
        else:
            return self.add_faces(choice_b)

    @staticmethod
    def _cross_2d(a, b):
        return float(a[0] * b[1] - a[1] * b[0])

    @classmethod
    def _segment_intersection_point(cls, p0, p1, q0, q1, eps=1e-10):
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        q0 = np.asarray(q0, dtype=np.float64)
        q1 = np.asarray(q1, dtype=np.float64)

        r = p1 - p0
        s = q1 - q0
        denom = cls._cross_2d(r, s)
        if abs(denom) <= eps:
            return None

        qp = q0 - p0
        t = cls._cross_2d(qp, s) / denom
        u = cls._cross_2d(qp, r) / denom
        if t < -eps or t > 1.0 + eps or u < -eps or u > 1.0 + eps:
            return None

        t = min(1.0, max(0.0, t))
        point = p0 + t * r
        return (float(point[0]), float(point[1]))

    def intersect(self, points_uv, edge, line):
        line = sorted(line)
        edge = sorted(edge)

        if line == edge:
            return None
        if line[0] in edge:
            return [None, line[0]]
        if line[1] in edge:
            return [None, line[1]]

        point = self._segment_intersection_point(
            points_uv[edge[0]], points_uv[edge[1]],
            points_uv[line[0]], points_uv[line[1]],
        )
        if point is None:
            return None
        return [point, -1]


class MAPS:
    def __init__(self, vertices, faces, base_size, timeout=None, verbose=False,
                 pable_ids=None, anatomy_weight=1.0, protect_pable=True,
                 soft_last_pable=False):
        self.mesh = Trimesh(vertices, faces, process=False, maintain_order=True)
        self.base = BaseMesh(vertices, faces)
        self.param = ParamMesh(vertices, faces)

        self.base_size = base_size
        self.verbose = verbose
        self.timeout = timeout
        self.anatomy_weight = anatomy_weight
        self.protect_pable = protect_pable
        self.soft_last_pable = soft_last_pable

        self.param_tri_verts = defaultdict(list)
        self._candidate_cache = {}
        self.pable_ids = self.normalize_pable_ids(pable_ids)
        self.protected_vertices = np.zeros(self.base.V, dtype=bool)
        if (self.protect_pable or self.soft_last_pable) and self.pable_ids.size > 0:
            self.protected_vertices[self.pable_ids] = True
        self.c_ana = self.compute_anatomy_cost(self.pable_ids)

        self.decimate_to(self.base_size)
        self.base_size = self.base.active_faces

    def normalize_pable_ids(self, pable_ids):
        if pable_ids is None:
            return np.array([], dtype=np.int64)
        pable_ids = np.asarray(pable_ids, dtype=np.int64).reshape(-1)
        pable_ids = pable_ids[(0 <= pable_ids) & (pable_ids < self.base.V)]
        return np.unique(pable_ids)

    def compute_anatomy_cost(self, pable_ids):
        if pable_ids is None or len(pable_ids) == 0:
            return None
        dist = topological_distance(self.base.V, self.base.faces[: self.base.F], pable_ids)
        reachable = dist >= 0
        c_ana = np.zeros(self.base.V, dtype=np.float64)
        c_ana[reachable] = 1.0 - percentile_rank(dist[reachable])
        return c_ana

    def decimate(self):
        self.decimate_to(self.base_size)

    def release_soft_last_pable_vertices(self, vertex_weights, dirty):
        if not self.soft_last_pable or self.protect_pable:
            return False
        release_ids = [int(i) for i in self.pable_ids
                       if self.base.vmask[i] and self.protected_vertices[i]]
        if not release_ids:
            return False
        self.protected_vertices[release_ids] = False
        for i in release_ids:
            vertex_weights[i] = self.initial_vertex_weight(i)
            dirty.add(i)
        return True

    def candidate_signature(self, i: int):
        if not self.base.vmask[i]:
            return None
        return tuple(sorted(self.base.vertex_faces[i]))

    def invalidate_candidate_cache(self, vids):
        for v in vids:
            self._candidate_cache.pop(int(v), None)

    def cache_invalidation_ids(self, center: int, neighbors):
        vids = {int(center)}
        frontier = {int(v) for v in neighbors}
        vids.update(frontier)
        for v in list(frontier):
            if 0 <= v < self.base.V and self.base.vmask[v]:
                vids.update(int(u) for u in self.base.neighbors(v))
        return vids

    def build_decimation_candidate(self, i: int):
        signature = self.candidate_signature(i)
        if signature is None or self.protected_vertices[i]:
            return {"signature": signature, "valid": False}

        try:
            neighbors = self.base.one_ring_neighbors(i)
            neighbors_uv = one_ring_neighbor_uv(neighbors, self.base.verts, i)
            new_faces, new_edges = CDT(neighbors, neighbors_uv)

            if not self.base.is_validate_removal(i, neighbors, new_faces, new_edges):
                neighbors_uv = np.array([uv / np.linalg.norm(uv) for uv in neighbors_uv])
                for v in neighbors:
                    trial_faces, trial_edges = MVT(v, neighbors)
                    if self.base.is_validate_removal(i, neighbors, trial_faces, trial_edges):
                        new_faces, new_edges = trial_faces, trial_edges
                        break
                else:
                    return {"signature": signature, "valid": False}
        except Exception:
            return {"signature": signature, "valid": False}

        return {
            "signature": signature,
            "valid": True,
            "neighbors": tuple(int(v) for v in neighbors),
            "neighbors_uv": np.asarray(neighbors_uv, dtype=np.float64),
            "new_faces": np.asarray(new_faces, dtype=np.int64),
            "new_edges": tuple((int(a), int(b)) for a, b in new_edges),
        }

    def get_decimation_candidate(self, i: int):
        signature = self.candidate_signature(i)
        cached = self._candidate_cache.get(i)
        if cached is not None and cached.get("signature") == signature:
            return cached
        candidate = self.build_decimation_candidate(i)
        self._candidate_cache[i] = candidate
        return candidate

    def decimate_to(self, base_size):
        self.base_size = base_size
        if self.base.active_faces <= self.base_size:
            self.base_size = self.base.active_faces
            return

        start_time = time()
        vertex_weights = ValueSortedDict()
        for i in range(self.base.V):
            if self.protected_vertices[i] or not self.base.vmask[i]:
                continue
            vertex_weights[i] = self.initial_vertex_weight(i)
        dirty = set(vertex_weights.keys())

        with tqdm(total=self.base.active_faces - self.base_size, disable=not self.verbose, mininterval=10.0) as pbar:
            while self.base.active_faces > self.base_size:
                try:
                    first_vertex = next(iter(vertex_weights.keys()))
                except StopIteration:
                    first_vertex = None
                if first_vertex is None or not np.isfinite(vertex_weights[first_vertex]):
                    if self.release_soft_last_pable_vertices(vertex_weights, dirty):
                        continue
                    return

                for i in islice(vertex_weights.keys(), 512):
                    if i in dirty:
                        vertex_weights[i] = self.compute_vertex_weight(i)
                        dirty.remove(i)

                vw = list(vertex_weights.keys())
                if len(vw) == 0 or not np.isfinite(vertex_weights[vw[0]]):
                    if self.release_soft_last_pable_vertices(vertex_weights, dirty):
                        continue
                    return
                mis = maximal_independent_set(
                    vw, self.base.faces, self.base.vertex_faces, self.base.V
                )
                progressed = False
                for i in mis:
                    if i not in vertex_weights or not self.base.vmask[i]:
                        continue
                    if not np.isfinite(vertex_weights[i]):
                        continue
                    if self.timeout is not None and time() - start_time > self.timeout:
                        return
                    try:
                        neighbors = self.base.one_ring_neighbors(i)
                    except Exception:
                        vertex_weights[i] = float("inf")
                        continue
                    old_face_count = self.base.active_faces
                    if self.try_decimate_base_vertex(i):
                        self.base.vmask[i] = 0
                        vertex_weights.pop(i)

                        for k in neighbors:
                            if k in vertex_weights and self.base.vmask[k]:
                                vertex_weights[k] = self.initial_vertex_weight(k)
                                dirty.add(k)

                        pbar.update(old_face_count - self.base.active_faces)
                        progressed = True
                        if self.base.active_faces <= self.base_size:
                            return
                    else:
                        vertex_weights[i] = float("inf")

                if not progressed:
                    if len(mis) == 0:
                        if self.release_soft_last_pable_vertices(vertex_weights, dirty):
                            continue
                        return
                    continue

        self.base_size = self.base.active_faces

    def initial_vertex_weight(self, i: int):
        if self.protected_vertices[i]:
            return float("inf")
        if self.c_ana is None:
            return 0.0
        return self.anatomy_weight * self.c_ana[i]

    def compute_vertex_weight(self, i: int):
        if self.protected_vertices[i]:
            return float("inf")
        candidate = self.get_decimation_candidate(i)
        if not candidate["valid"]:
            return float("inf")

        old_faces = list(self.base.vertex_faces[i])
        old_areas = face_areas(self.base.verts, self.base.faces[old_faces]).sum()
        if old_areas <= 1e-12:
            return float("inf")

        new_areas = face_areas(self.base.verts, candidate["new_faces"]).sum()
        area_ratio = new_areas / old_areas
        geo_cost = area_ratio / (area_ratio + 1.0)

        if self.c_ana is None:
            return geo_cost

        return geo_cost + self.anatomy_weight * self.c_ana[i]

    def try_decimate_base_vertex(self, i: int) -> bool:
        if self.protected_vertices[i]:
            return False
        candidate = self.get_decimation_candidate(i)
        if not candidate["valid"]:
            return False

        neighbors = list(candidate["neighbors"])
        neighbors_uv = candidate["neighbors_uv"]
        new_faces = candidate["new_faces"]
        new_edges = candidate["new_edges"]

        ring_uv = {n: neighbors_uv[k] for k, n in enumerate(neighbors)}
        ring_uv[i] = [0, 0]

        invalidate_ids = self.cache_invalidation_ids(i, neighbors)
        self.reparameterize(i, ring_uv, new_faces, new_edges)
        self.base.remove_vertex(i, new_faces, neighbors)
        self.invalidate_candidate_cache(invalidate_ids)

        return True

    def reparameterize(self, i: int, ring_uv: Dict, new_faces, new_edges):
        points_uv = ring_uv.copy()
        neighbors = set([k for k in ring_uv.keys() if k != i])
        points_on_ring = neighbors.copy()

        for fid in self.base.vertex_faces[i]:
            face = self.base.faces[fid]
            face_uv = [ring_uv[face[0]], ring_uv[face[1]], ring_uv[face[2]]]
            for v in self.param_tri_verts[fid]:
                if v not in points_uv:
                    points_uv[v] = from_barycenteric(face_uv, self.param.baries[v][fid])
                    if v in self.param.on_edge:
                        edge = self.param.on_edge[v]
                        if edge[0] in neighbors and edge[1] in neighbors:
                            points_on_ring.add(v)
                del self.param.baries[v][fid]

        self.param.split_triangles_on_segments(points_uv, points_on_ring, new_edges)

        for v, uv in points_uv.items():
            self.uv_to_xyz_tri(v, uv, ring_uv, new_faces)

        return True

    def uv_to_xyz_tri(self, v: int, uv, verts_uv: Dict, faces: List):
        def in_triangle(point, triangle):
            max_s = np.abs(triangle).max()
            point = point / max_s
            triangle = triangle / max_s
            n1 = np.cross(point - triangle[0], triangle[1] - triangle[0])
            n2 = np.cross(point - triangle[1], triangle[2] - triangle[1])
            n3 = np.cross(point - triangle[2], triangle[0] - triangle[2])
            n1 = 0 if abs(n1) < 1e-10 else n1
            n2 = 0 if abs(n2) < 1e-10 else n2
            n3 = 0 if abs(n3) < 1e-10 else n3
            return ((n1 >= 0) and (n2 >= 0) and (n3 >= 0)) or (
                (n1 <= 0) and (n2 <= 0) and (n3 <= 0)
            )

        found = False
        for f, face in enumerate(faces):
            triangle_uv = [verts_uv[face[0]], verts_uv[face[1]], verts_uv[face[2]]]
            if in_triangle(uv, triangle_uv):
                point_bary = to_barycentric(uv, triangle_uv)
                assert np.abs(point_bary).sum() <= 2
                point_xyz = from_barycenteric(self.base.verts[face], point_bary)
                tri = f + self.base.F
                self.param_tri_verts[tri].append(v)
                self.param.baries[v][tri] = point_bary
                self.param.verts[v] = point_xyz
                found = True

        assert found

    def mesh_upsampling(self, depth) -> Trimesh:
        sub_verts, sub_faces = self.subdivide(depth)
        sub_verts = self.parameterize(sub_verts)
        return Trimesh(sub_verts, sub_faces, process=False, maintain_order=True)

    def subdivide(self, depth):
        verts = self.base.verts[self.base.vmask]
        vmaps = np.cumsum(self.base.vmask) - 1
        faces = self.base.faces[self.base.fmask]
        faces = vmaps[faces]

        for _ in range(depth):
            nV = verts.shape[0]
            nF = faces.shape[0]
            edges_d = np.concatenate(
                [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
            )
            edges_d = np.sort(edges_d, axis=1)
            edges_u, F2E = np.unique(edges_d, axis=0, return_inverse=True)
            new_verts = (verts[edges_u[:, 0]] + verts[edges_u[:, 1]]) / 2
            verts = np.concatenate([verts, new_verts], axis=0)

            E2 = F2E[:nF] + nV
            E0 = F2E[nF : nF * 2] + nV
            E1 = F2E[nF * 2 :] + nV
            faces = np.concatenate(
                [
                    np.stack([faces[:, 0], E2, E1], axis=-1),
                    np.stack([faces[:, 1], E0, E2], axis=-1),
                    np.stack([faces[:, 2], E1, E0], axis=-1),
                    np.stack([E0, E1, E2], axis=-1),
                ],
                axis=0,
            )

        return verts, faces

    def parameterize(self, points, chunk_size=5000, min_chunk_size=512):
        param_verts = self.param.verts[: self.param.V]
        param_faces = self.param.faces[self.param.fmask]
        param_mesh = Trimesh(param_verts, param_faces, process=False)

        start = 0
        chunk_size = max(1, int(chunk_size))
        min_chunk_size = max(1, int(min_chunk_size))
        current_chunk_size = min(chunk_size, max(len(points), 1))
        while start < len(points):
            stop = min(start + current_chunk_size, len(points))
            try:
                closest_points, _, triangle_id = param_mesh.nearest.on_surface(points[start:stop])
            except MemoryError:
                if current_chunk_size <= min_chunk_size:
                    raise
                current_chunk_size = max(min_chunk_size, current_chunk_size // 2)
                gc.collect()
                continue

            for offset, point in enumerate(closest_points):
                i = start + offset
                face = param_faces[triangle_id[offset]]
                triangle = self.param.verts[face]
                xyz = self.param.xyz[face]
                try:
                    bary = to_barycentric(point, triangle)
                    points[i] = from_barycenteric(xyz, bary)
                except np.linalg.LinAlgError:
                    points[i] = xyz.mean(axis=0)

            del closest_points, triangle_id
            start = stop

        return points
