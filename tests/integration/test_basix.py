"""Check IDMesh's actual face/edge orientation codes against Basix C++ transforms.

This test deliberately uses the fenics-basix Python bindings backed by the
C++ Basix implementation. Basix is test-only, never a core SDK dependency.
"""
import subprocess
import sys

import basix
import numpy as np


def verify():
    lines = subprocess.check_output([sys.argv[1]], text=True).splitlines()
    faces = []
    edges = []
    for line in lines:
        parts = line.split()
        if parts[0] == "FACE":
            faces.append(tuple(map(int, parts[1:])))
        elif parts[0] == "EDGE":
            edges.append(tuple(map(int, parts[1:])))
    assert len(faces) == 2 and faces[0][1] == faces[1][1], faces
    assert faces[0][4] == -faces[1][4], "shared facet induced signs"
    assert any(e[2] == 1 for e in edges), "reversed edge was not exercised"
    assert any(e[2] == 0 for e in edges), "forward edge was not exercised"

    element = basix.create_element(
        basix.ElementFamily.N1E, basix.CellType.tetrahedron, 2
    )
    transformations = element.entity_transformations()
    edge_flip = np.asarray(transformations["interval"])[0]
    face_rot, face_reflect = np.asarray(transformations["triangle"])

    def near(a, b):
        return np.allclose(a, b, rtol=1e-11, atol=1e-11)

    identity_edge = np.eye(edge_flip.shape[0])
    identity_face = np.eye(face_rot.shape[0])
    assert edge_flip.shape[0] > 1, "not a higher-order edge DOF block"
    assert face_rot.shape[0] > 0, "no high-order triangular face DOFs"
    assert near(edge_flip @ edge_flip, identity_edge)
    assert near(np.linalg.matrix_power(face_rot, 3), identity_face)
    assert near(face_reflect @ face_reflect, identity_face)
    assert near(
        face_reflect @ face_rot @ face_reflect,
        np.linalg.matrix_power(face_rot, 2),
    ), "Basix face symmetries do not form D3"

    def face_action(rotation, reversed_orientation):
        action = np.linalg.matrix_power(face_rot, rotation)
        if reversed_orientation:
            action = action @ face_reflect
        return action

    # The two adjacent tetrahedra induce distinct oriented views of their
    # common facet. A canonical face coefficient vector transformed through
    # either local view must agree exactly after restoring its orientation.
    canonical = np.arange(1, identity_face.shape[0] + 1, dtype=float)
    transforms = [face_action(face[2], bool(face[3])) for face in faces]
    local = [np.linalg.solve(matrix, canonical) for matrix in transforms]
    assert near(transforms[0] @ local[0], transforms[1] @ local[1])
    assert any(
        not near(np.linalg.matrix_power(edge_flip, bool(reversed_orientation)),
                 identity_edge)
        for _, _, reversed_orientation in edges
    ), "Basix edge reversal did not produce a nontrivial DOF transformation"
    print("Basix: high-order Nedelec edge/face transforms match oriented mesh data")


if __name__ == "__main__":
    verify()
