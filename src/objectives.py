from drag import compute_drag, compute_aoa_and_area
from constrains import volume_penalty


def objective(flat_control_points, mesh, ffd, lookup_t):
    control_points = flat_control_points.reshape((8, 3))

    disp = (control_points - ffd.control_points()) / ffd.box_length

    ffd.array_mu_x = disp[:, 0].reshape(ffd.array_mu_x.shape)
    ffd.array_mu_y = disp[:, 1].reshape(ffd.array_mu_y.shape)
    ffd.array_mu_z = disp[:, 2].reshape(ffd.array_mu_z.shape)

    new_vertices = ffd(mesh.vertices)
    new_mesh = mesh
    new_mesh.vertices = new_vertices

    aoas, areas = compute_aoa_and_area(panels=new_mesh.faces, points=new_vertices)
    drag = compute_drag(aoas=aoas, areas=areas, lookup_table=lookup_t)

    obj = drag

    # Add volume penalty
    # obj += 1000 * inside_out_penalty(control_points)

    obj += volume_penalty(mesh.volume, new_mesh)
    return obj
