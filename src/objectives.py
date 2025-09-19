from drag import compute_drag, compute_aoa_and_area
from constrains import volume_penalty
import trimesh
from pygem import FFD
import numpy as np


def create_init_mesh_ffd(radius, n_control_points):
    cube_side = radius * 10
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    ffd = FFD(n_control_points)
    ffd.box_length = [cube_side, cube_side, cube_side]
    ffd.box_origin = [-cube_side / 2, -cube_side / 2, -cube_side / 2]

    return mesh, ffd


def objective(flat_control_points, ffd, mesh, lookup_t, radius):
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)

    volume = 4.2

    control_points = flat_control_points.reshape(ffd.control_points().shape)

    old_disp = []
    old_disp.append(ffd.array_mu_x.flatten())
    old_disp.append(ffd.array_mu_y.flatten())
    old_disp.append(ffd.array_mu_z.flatten())
    old_disp = list(np.array(old_disp).T)

    new_disp = (control_points - ffd.control_points()) / ffd.box_length

    total_disp = old_disp + new_disp

    ffd.array_mu_x = total_disp[:, 0].reshape(ffd.array_mu_x.shape)
    ffd.array_mu_y = total_disp[:, 1].reshape(ffd.array_mu_y.shape)
    ffd.array_mu_z = total_disp[:, 2].reshape(ffd.array_mu_z.shape)

    new_vertices = ffd(mesh.vertices)
    mesh.vertices = new_vertices

    aoas, areas = compute_aoa_and_area(panels=mesh.faces, points=new_vertices)
    drag = compute_drag(aoas=aoas, areas=areas, lookup_table=lookup_t)

    obj = drag

    # print(volume, mesh.volume)
    obj += volume_penalty(volume, mesh)
    # obj += inside_out_penalty(ffd.control_points())
    # p = inside_out_penalty(ffd.control_points())

    print(drag, mesh.volume)
    return obj
