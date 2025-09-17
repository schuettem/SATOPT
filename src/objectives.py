from drag import compute_drag, compute_aoa_and_area
import numpy as np


def volum_penalty(mesh, cutoff, penalty_faktor):
    penalty = 0
    if mesh.volume < cutoff:
        penalty = penalty_faktor
        
    return penalty

PENALTIES = {
    "volume": volum_penalty
}

def objective_free(x, init_mesh, ffd, lookup_t, penalty_fns: list=[]):
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

    aoas, areas = compute_aoa_and_area(panels=new_mesh.faces, points=new_vertices)
    drag = compute_drag(aoas=aoas, areas=areas, lookup_table=lookup_t)
    
    penalty = 0
    for peanlty_fn in penalty_fns:
        penalty += peanlty_fn(new_mesh, 3.5, 10**2)
        
    drag += penalty
    
    return drag


def objective_x(x, init_mesh, ffd, lookup_t, penalty_fns: list=[]):
    shape = list(ffd.array_mu_x.shape)
    x = np.array(x)
    x = x.reshape(shape)

    ffd.array_mu_z = x[:, :, :]

    new_vertices = ffd(init_mesh.vertices)
    new_mesh = init_mesh
    new_mesh.vertices = new_vertices

    aoas, areas = compute_aoa_and_area(panels=new_mesh.faces, points=new_vertices)
    drag = compute_drag(aoas=aoas, areas=areas, lookup_table=lookup_t)

    penalty = 0
    for peanlty_fn in penalty_fns:
        penalty += peanlty_fn(new_mesh, 3.5, 10**2)
        
    drag += penalty
    
    return drag


OBJECTIVES = {
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


DEFORMATIONS = {
    "free": deform_mesh,
    "x": deform_mesh_x,
}
