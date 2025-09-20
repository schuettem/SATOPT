from functools import partial

import numpy as np
from scipy.optimize import NonlinearConstraint

from utils import box_from_corners


def box_lengths(corners):
    """
    Given 8 vertices of a box (any orientation),
    return the 3 side lengths.
    """
    corners = np.asarray(corners)
    # take first corner as reference
    ref = corners[0]

    # compute distances to all other corners
    dists = np.linalg.norm(corners - ref, axis=1)

    # find the 3 smallest nonzero distances = edges
    edge_lengths = np.sort(np.unique(np.round(dists[dists > 1e-8], 8)))[:3]

    return edge_lengths


def get_obb_corners(mesh):
    obb = mesh.bounding_box_oriented
    T = obb.primitive.transform  # 4x4 homogeneous transform
    extents = obb.primitive.extents  # box side lengths

    # Build all 8 corners in local box space
    local_corners = (
        np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ]
        )
        * extents
    )  # scale by box side lengths

    # Convert to homogeneous coords
    local_corners_h = np.hstack([local_corners, np.ones((8, 1))])

    # Transform to world space
    world_corners = (T @ local_corners_h.T).T[:, :3]
    return world_corners


def volume_penalty(min_vol, new_mesh):
    """Penalty if any tetrahedron inside cube is inverted."""
    penalty = 0
    if min_vol > new_mesh.volume:
        penalty += 1000

    return penalty


def min_box_penalty(min_box, new_mesh):
    """Penalty if any tetrahedron inside cube is inverted."""

    obb_corners = get_obb_corners(new_mesh)
    box_mesh = box_from_corners(obb_corners)
    corners = box_mesh.vertices

    lengths = box_lengths(corners)

    penalty = 0

    if lengths[0] < min_box[0]:
        penalty += 100
    if lengths[1] < min_box[1]:
        penalty += 100
    if lengths[2] < min_box[2]:
        penalty += 100
    return penalty


# def volume_tetrahedron(p0, p1, p2, p3):
#     """Signed volume of a tetrahedron."""
#     return np.linalg.det(np.column_stack((p1 - p0, p2 - p0, p3 - p0))) / 6.0


# def inside_out_penalty(corners):
#     """Penalty if any tetrahedron inside cube is inverted."""
#     tets = [(0, 1, 2, 4), (1, 2, 3, 7), (1, 4, 5, 7), (2, 4, 6, 7), (1, 2, 4, 7)]
#     penalty = 0.0
#     for i, j, k, l in tets:
#         V = volume_tetrahedron(corners[i], corners[j], corners[k], corners[l])
#         if V < 0:
#             penalty += (-V) ** 2

#     return penalty


def trilinear_shape_derivatives(u, v, w):
    """Derivatives of the 8 shape functions wrt (u,v,w)."""
    dN_du = np.array(
        [
            -(1 - v) * (1 - w),
            -(1 - v) * w,
            -v * (1 - w),
            -v * w,
            (1 - v) * (1 - w),
            (1 - v) * w,
            v * (1 - w),
            v * w,
        ]
    )
    dN_dv = np.array(
        [
            -(1 - u) * (1 - w),
            -(1 - u) * w,
            (1 - u) * (1 - w),
            (1 - u) * w,
            -u * (1 - w),
            -u * w,
            u * (1 - w),
            u * w,
        ]
    )
    dN_dw = np.array(
        [
            -(1 - u) * (1 - v),
            (1 - u) * (1 - v),
            -(1 - u) * v,
            (1 - u) * v,
            -u * (1 - v),
            u * (1 - v),
            -u * v,
            u * v,
        ]
    )
    return dN_du, dN_dv, dN_dw


# def orientation_constraint(flat_control_points, sample_points=None):
#     """Compute Jacobian determinant at sample points."""
#     cp = flat_control_points.reshape((8, 3))  # 8x3 array

#     if sample_points is None:
#         sample_points = [
#             (0, 0, 0),
#             (1, 0, 0),
#             (0, 1, 0),
#             (0, 0, 1),
#             (1, 1, 0),
#             (1, 0, 1),
#             (0, 1, 1),
#             (1, 1, 1),
#             (0.5, 0.5, 0.5),
#         ]

#     vols = []
#     for u, v, w in sample_points:
#         dN_du, dN_dv, dN_dw = trilinear_shape_derivatives(u, v, w)

#         dx_du = np.sum(dN_du[:, None] * cp, axis=0)
#         dx_dv = np.sum(dN_dv[:, None] * cp, axis=0)
#         dx_dw = np.sum(dN_dw[:, None] * cp, axis=0)

#         J = np.stack([dx_du, dx_dv, dx_dw], axis=1)
#         vol = np.linalg.det(J)
#         vols.append(vol)

#     return np.array(vols)


# def get_nonflip_constraint():
#     return NonlinearConstraint(
#         orientation_constraint,
#         lb=1e-4,  # strictly positive determinant
#         ub=np.inf,
#     )


def general_orientation_constraint(flat_control_points, ffd, sample_points=None):
    """
    Non-flip constraint for arbitrary FFD grid.
    Checks Jacobian determinants of all unit cells.
    """
    nx, ny, nz = ffd.n_control_points
    cp = flat_control_points.reshape((nx, ny, nz, 3))

    if sample_points is None:
        # Default: corners + center of parametric cell
        sample_points = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
            (0.5, 0.5, 0.5),
        ]

    vols = []

    # Loop over all unit cells
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # corners of this cell
                cell_cp = cp[i : i + 2, j : j + 2, k : k + 2, :].reshape(
                    (8, 3)
                )  # 8 corners

                for u, v, w in sample_points:
                    # compute shape function derivatives
                    dN_du, dN_dv, dN_dw = trilinear_shape_derivatives(u, v, w)

                    dx_du = np.sum(dN_du[:, None] * cell_cp, axis=0)
                    dx_dv = np.sum(dN_dv[:, None] * cell_cp, axis=0)
                    dx_dw = np.sum(dN_dw[:, None] * cell_cp, axis=0)

                    J = np.stack([dx_du, dx_dv, dx_dw], axis=1)
                    vol = np.linalg.det(J)
                    vols.append(vol)

    return np.array(vols)


def get_general_nonflip_constraint(ffd):
    constrain = partial(general_orientation_constraint, ffd=ffd)
    return NonlinearConstraint(
        constrain,
        lb=1e-4,  # strictly positive determinant
        ub=np.inf,
    )
