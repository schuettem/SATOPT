from drag import compute_drag, compute_aoa_and_area
import numpy as np


def objective_free(x, init_mesh, ffd, lookup_t):
    shape = list(ffd.array_mu_x.shape)
    shape.extend([3])
    x = np.array(x)
    x = x.reshape(shape)

    ffd.array_mu_x = x[:, :, :, 0]
    ffd.array_mu_y = x[:, :, :, 1]
    ffd.array_mu_z = x[:, :, :, 2]

    new_vertices = ffd(init_mesh.vertices)
    new_mesh = init_mesh
    new_mesh.vertices = new_vertices

    if new_mesh.volume < 3.5:
        return 10**2
    else:
        aoas, areas = compute_aoa_and_area(panels=new_mesh.faces, points=new_vertices)
        drag = compute_drag(aoas=aoas, areas=areas, lookup_table=lookup_t)
        return drag


def objective_x(x, init_mesh, ffd, lookup_t):
    shape = list(ffd.array_mu_x.shape)
    x = np.array(x)
    x = x.reshape(shape)

    ffd.array_mu_z = x[:, :, :]

    new_vertices = ffd(init_mesh.vertices)
    new_mesh = init_mesh
    new_mesh.vertices = new_vertices

    if new_mesh.volume < 3.5:
        return 10**2
    else:
        aoas, areas = compute_aoa_and_area(panels=new_mesh.faces, points=new_vertices)
        drag = compute_drag(aoas=aoas, areas=areas, lookup_table=lookup_t)
        return drag


objective_fn = {
    "free": objective_free,
    "x": objective_x,
}


def deform_mesh(x, mesh, ffd, lookup_t):
    shape = list(ffd.array_mu_x.shape)
    shape.extend([3])
    x = np.array(x)
    x = x.reshape(shape)

    ffd.array_mu_x = x[:, :, :, 0]
    ffd.array_mu_y = x[:, :, :, 1]
    ffd.array_mu_z = x[:, :, :, 2]

    new_vertices = ffd(mesh.vertices)
    new_mesh = mesh
    new_mesh.vertices = new_vertices

    aoas, areas = compute_aoa_and_area(panels=new_mesh.faces, points=new_vertices)
    drag = compute_drag(aoas=aoas, areas=areas, lookup_table=lookup_t)

    return new_mesh, drag


def deform_mesh_x(x, mesh, ffd, lookup_t):
    shape = list(ffd.array_mu_x.shape)
    x = np.array(x)
    x = x.reshape(shape)

    ffd.array_mu_x = x[:, :, :]

    new_vertices = ffd(mesh.vertices)
    new_mesh = mesh
    new_mesh.vertices = new_vertices

    aoas, areas = compute_aoa_and_area(panels=new_mesh.faces, points=new_vertices)
    drag = compute_drag(aoas=aoas, areas=areas, lookup_table=lookup_t)

    return new_mesh, drag


deformatoin_fn = {
    "free": deform_mesh,
    "x": deform_mesh_x,
}
