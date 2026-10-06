import os
import math
import argparse
import numpy as np


def normalize(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def orient_faces_outward(vertices, faces):
    vertices = np.array(vertices, dtype=float)
    oriented = []

    for face in faces:
        pts = [vertices[i] for i in face]

        normal = np.zeros(3)
        for i in range(len(pts)):
            p = pts[i]
            q = pts[(i + 1) % len(pts)]
            normal += np.cross(p, q)

        centroid = np.mean(pts, axis=0)

        if np.dot(normal, centroid) < 0:
            face = list(reversed(face))

        oriented.append(face)

    return oriented


def scale_to_extent(vertices, target_extent):
    vertices = np.array(vertices, dtype=float)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    extent = np.max(maxs - mins)

    if extent == 0:
        return vertices

    vertices = vertices / extent * target_extent
    return vertices


def make_cube():
    v = [
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ]

    f = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]

    return v, orient_faces_outward(v, f)


def make_octahedron():
    v = [
        [ 1,  0,  0],
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0, -1,  0],
        [ 0,  0,  1],
        [ 0,  0, -1],
    ]

    f = [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ]

    return v, orient_faces_outward(v, f)


def make_icosahedron():
    phi = (1 + math.sqrt(5)) / 2

    v = [
        [-1,  phi, 0],
        [ 1,  phi, 0],
        [-1, -phi, 0],
        [ 1, -phi, 0],

        [0, -1,  phi],
        [0,  1,  phi],
        [0, -1, -phi],
        [0,  1, -phi],

        [ phi, 0, -1],
        [ phi, 0,  1],
        [-phi, 0, -1],
        [-phi, 0,  1],
    ]

    f = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],

        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],

        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],

        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    return v, orient_faces_outward(v, f)


def make_dodecahedron():
    """
    정십이면체를 정이십면체의 dual로 생성한다.
    정이십면체의 각 triangular face center가 정십이면체의 vertex가 된다.
    정이십면체의 각 vertex 주변의 5개 face center가 정십이면체의 pentagonal face가 된다.
    """
    ico_v, ico_f = make_icosahedron()
    ico_v = np.array(ico_v, dtype=float)

    centers = []
    for face in ico_f:
        c = np.mean(ico_v[face], axis=0)
        c = normalize(c)
        centers.append(c)

    centers = np.array(centers)

    dode_faces = []

    for vi in range(len(ico_v)):
        axis = normalize(ico_v[vi])

        adjacent_face_indices = [
            fi for fi, face in enumerate(ico_f)
            if vi in face
        ]

        # axis에 수직인 local basis 생성
        ref = np.array([0, 0, 1], dtype=float)
        if abs(np.dot(axis, ref)) > 0.9:
            ref = np.array([0, 1, 0], dtype=float)

        u = normalize(np.cross(axis, ref))
        w = normalize(np.cross(axis, u))

        angles = []
        for fi in adjacent_face_indices:
            p = centers[fi]
            p_proj = p - np.dot(p, axis) * axis
            angle = math.atan2(np.dot(p_proj, w), np.dot(p_proj, u))
            angles.append((angle, fi))

        ordered = [fi for angle, fi in sorted(angles)]
        dode_faces.append(ordered)

    return centers.tolist(), orient_faces_outward(centers.tolist(), dode_faces)


def write_obj(path, vertices, faces):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated regular polyhedron OBJ\n")

        for x, y, z in vertices:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")

        for face in faces:
            # OBJ index is 1-based
            idx = [str(i + 1) for i in face]
            f.write("f " + " ".join(idx) + "\n")


def generate_all(out_dir, size):
    os.makedirs(out_dir, exist_ok=True)

    makers = {
        "plain_cube": make_cube,
        "octahedron": make_octahedron,
        "dodecahedron": make_dodecahedron,
        "icosahedron": make_icosahedron,
    }

    for name, maker in makers.items():
        vertices, faces = maker()
        vertices = scale_to_extent(vertices, size)

        out_path = os.path.join(out_dir, f"{name}.obj")
        write_obj(out_path, vertices, faces)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="assets/generated")
    parser.add_argument("--size", type=float, default=0.08, help="max bbox extent in meters")
    args = parser.parse_args()

    generate_all(args.out, args.size)