from functools import partial

from scipy.optimize import minimize

from analysis import plot_aoas, plot_mesh_c_points
from constrains import get_general_nonflip_constraint
from drag import compute_aoa_and_area
from objectives import objective


def run(drag_model, init_mesh, ffd, min_volume=0, opt_bond=None):
    mesh = init_mesh.copy()

    control_points = ffd.control_points()

    flat_control_points = control_points.flatten()

    nonlinear_constraint = get_general_nonflip_constraint(ffd)

    obj_fn = partial(
        objective, init_mesh=mesh, drag_model=drag_model, min_volume=min_volume
    )

    if opt_bond is not None:
        opt_bond = [(-opt_bond, opt_bond)] * flat_control_points.shape[0]

    for i in range(1):
        flat_control_points = flat_control_points
        res = minimize(
            obj_fn,
            flat_control_points,
            args=(ffd),
            method="SLSQP",
            constraints=[nonlinear_constraint],
            bounds=opt_bond,
            options={"ftol": 1e-6, "disp": True, "maxiter": 500},
        )
        print(i)

    control_points = res.x.reshape(control_points.shape)

    return control_points, ffd


def get_final_mesh(control_points, ffd, init_mesh):
    disp = (control_points - ffd.control_points()) / ffd.box_length

    ffd.array_mu_x = disp[:, 0].reshape(ffd.array_mu_x.shape)
    ffd.array_mu_y = disp[:, 1].reshape(ffd.array_mu_y.shape)
    ffd.array_mu_z = disp[:, 2].reshape(ffd.array_mu_z.shape)

    new_vertices = ffd(init_mesh.vertices)
    init_mesh.vertices = new_vertices

    plot_mesh_c_points(init_mesh, ffd)

    aoas, areas = compute_aoa_and_area(panels=init_mesh.faces, points=new_vertices)
    plot_aoas(init_mesh, aoas)

    return init_mesh
