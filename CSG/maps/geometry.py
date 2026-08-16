from typing import List
import numpy as np
import triangle as tr


def to_barycentric(points, triangle):
    points = np.array(points)
    triangle = np.array(triangle)

    if triangle.shape[1] == 3:
        a, b, c = np.linalg.solve(triangle.T, points)
    elif triangle.shape[1] == 2:
        A = np.vstack([triangle[1] - triangle[0], triangle[2] - triangle[0]])
        b, c = np.linalg.solve(A.T, points - triangle[0])
        a = 1 - b - c
    else:
        raise Exception("Invalid")

    eps = 1e-5

    return np.array([a, b, c])


def from_barycenteric(attr, bary):
    attr = np.array(attr)
    bary = np.array(bary)
    if len(attr.shape) == 1:
        return (attr * bary).sum()
    elif len(attr.shape) == 2:
        return (attr * bary[:, None]).sum(axis=0)
    else:
        raise Exception("Invalid")


def CDT(vids, vertices):
    vids = np.asarray(vids, dtype=np.int64)
    V = vertices.shape[0]
    ring = [(i, (i + 1) % V) for i in range(V)]
    data = {"vertices": vertices, "segments": ring}
    result = tr.triangulate(data, "pe")

    triangles = np.asarray(result["triangles"], dtype=np.int64)
    new_faces = vids[triangles]

    edges = np.asarray(result.get("edges", []), dtype=np.int64)
    if edges.size == 0:
        new_edges = np.empty((0, 2), dtype=np.int64)
    else:
        ring_edges = {tuple(edge) for edge in ring}
        keep = [
            (int(a), int(b)) not in ring_edges and (int(b), int(a)) not in ring_edges
            for a, b in edges
        ]
        new_edges = vids[edges[np.asarray(keep, dtype=bool)]]

    return new_faces, new_edges

def MVT(v, neighbors):
    edges = set()
    for i in range(len(neighbors)):
        j = i + 1 if i + 1 < len(neighbors) else 0
        edges.add((neighbors[i], neighbors[j]))
        edges.add((neighbors[j], neighbors[i]))

    new_faces = []
    new_edges = set()
    for i, u in enumerate(neighbors):
        j = i + 1 if i + 1 < len(neighbors) else 0
        w = neighbors[j]
        if u == v or w == v:
            continue
        new_faces.append([v, u, w])
        if not (v, u) in edges:
            new_edges.add((v, u))
        if not (v, w) in edges:
            new_edges.add((v, w))
    new_faces = np.array(new_faces)
    new_edges = np.array(list(new_edges))
    return new_faces, new_edges


def one_ring_neighbor_uv(
    neighbors: List[int],
    vertices: np.ndarray,
    i: int,
    return_angle=False,
    return_alpha=False,
):
    neighbors_p = neighbors
    neighbors_s = np.roll(neighbors_p, -1)
    vertices_p = vertices[neighbors_p]
    vertices_s = vertices[neighbors_s]
    direct_p = vertices_p - vertices[i]
    direct_s = vertices_s - vertices[i]
    length_p = np.sqrt((direct_p ** 2).sum(axis=1))
    length_s = np.sqrt((direct_s ** 2).sum(axis=1))
    direct_p = direct_p / length_p[:, np.newaxis]
    direct_s = direct_s / length_s[:, np.newaxis]

    angle_v = np.arccos((direct_p * direct_s).sum(axis=1))

    alpha = angle_v.sum()
    A = 2 * np.pi / alpha

    angle_v[1:] = np.cumsum(angle_v)[:-1]
    angle_v[0] = 0
    angle_v = angle_v * A

    u = np.power(length_p, A) * np.cos(angle_v)
    v = np.power(length_p, A) * np.sin(angle_v)

    uv = np.vstack([u, v]).transpose()

    if np.isnan(uv).any():
        raise Exception('Found NAN')

    ret = (uv,)
    if return_angle:
        ret += (angle_v,)
    if return_alpha:
        ret += (alpha,)

    if len(ret) == 1:
        ret = ret[0]
    return ret


def plane_from_points(points):
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]

    cp = np.cross(v1, v2)
    d = -np.dot(cp, points[2])

    l = np.linalg.norm(cp)
    cp /= l
    d /= l
    a, b, c = cp

    return np.array([a, b, c, d])


def vector_angle(A, B):
    cos_theta = np.dot(A, B) / np.linalg.norm(A) / np.linalg.norm(B)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def triangle_angles(triangle):
    a = vector_angle(triangle[1] - triangle[0], triangle[2] - triangle[0])
    b = vector_angle(triangle[2] - triangle[1], triangle[0] - triangle[1])
    c = np.pi - a - b
    return np.array([a, b, c])


def min_triangle_angles(triangle):
    return triangle_angles(triangle).min()


def face_areas(verts, faces):
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.empty(0, dtype=np.float64)
    triangles = verts[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    return np.linalg.norm(cross, axis=1) * 0.5
