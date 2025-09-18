import numpy as np
from scipy.optimize import LinearConstraint


def get_flip_constrain():
    # ----------------------------
    # Linear constraints to avoid flips
    # ----------------------------
    # Example: ensure x0 < x4, x1 < x5, etc.
    eps = 0.01
    A = []
    b = []
    # x-axis constraints
    for i, j in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        row = np.zeros(24)
        row[i * 3 + 0] = -1  # -xi
        row[j * 3 + 0] = 1  # +xj
        A.append(row)
        b.append(-eps)

    # y-axis constraints
    for i, j in [(0, 2), (1, 3), (4, 6), (5, 7)]:
        row = np.zeros(24)
        row[i * 3 + 1] = -1
        row[j * 3 + 1] = 1
        A.append(row)
        b.append(-eps)

    # z-axis constraints
    for i, j in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        row = np.zeros(24)
        row[i * 3 + 2] = -1
        row[j * 3 + 2] = 1
        A.append(row)
        b.append(-eps)

    linear_constraint = LinearConstraint(np.array(A), b, np.inf)

    return linear_constraint


def volume_penalty(min_vol, new_mesh):
    """Penalty if any tetrahedron inside cube is inverted."""
    penalty = 0
    if min_vol > new_mesh.volume:
        penalty += 10**2

    return penalty


def volume_tetrahedron(p0, p1, p2, p3):
    """Signed volume of a tetrahedron."""
    return np.linalg.det(np.column_stack((p1 - p0, p2 - p0, p3 - p0))) / 6.0


def inside_out_penalty(corners):
    """Penalty if any tetrahedron inside cube is inverted."""
    tets = [(0, 1, 2, 4), (1, 2, 3, 7), (1, 4, 5, 7), (2, 4, 6, 7), (1, 2, 4, 7)]
    penalty = 0.0
    for i, j, k, l in tets:
        V = volume_tetrahedron(corners[i], corners[j], corners[k], corners[l])
        if V < 0:
            penalty += (-V) ** 2
    return penalty
