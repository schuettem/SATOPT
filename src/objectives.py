import numpy as np

from constrains import volume_penalty
from drag import compute_aoa_and_area


def objective(flat_control_points, ffd, init_mesh, drag_model, min_volume):
    mesh = init_mesh.copy()

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
    drag = drag_model.calc_drag(aoas=aoas, areas=areas)

    obj = drag

    obj += volume_penalty(min_volume, mesh)
    # obj += min_box_penalty([0, 3, 3], mesh)
    # obj += inside_out_penalty(ffd.control_points())
    # p = inside_out_penalty(ffd.control_points())

    print(drag, mesh.volume)
    return obj
