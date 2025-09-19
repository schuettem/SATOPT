from drag import load_c_d_lookup_table
from scipy.optimize import minimize
from objectives import objective, create_init_mesh_ffd
from drag import compute_aoa_and_area
from analysis import plot_mesh_c_points, plot_aoas
from constrains import get_nonflip_constraint, get_general_nonflip_constraint
import os
from functools import partial



def run():
    radius = 4
    n_control_points = [3, 3, 3]
    
    
    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "aerodynamic_coefficients_panel_method.csv",
    )
    lookup_t = load_c_d_lookup_table(file_path)

    mesh, ffd = create_init_mesh_ffd(radius=radius, n_control_points=n_control_points)
    control_points = ffd.control_points()
    
    flat_control_points = control_points.flatten()
    
    # nonlinear_constraint = get_nonflip_constraint()
    nonlinear_constraint = get_general_nonflip_constraint(ffd)
    
    obj_fn = partial(objective, mesh=mesh, lookup_t=lookup_t, radius=radius)
    # bounds = [(-2, 2)] * flat_control_points.shape[0]

    for i in range(1):
        flat_control_points = flat_control_points
        res = minimize(
            obj_fn,
            flat_control_points,
            args=(ffd),
            method="SLSQP",
            constraints=[nonlinear_constraint],
            # bounds=bounds,
            options={"ftol": 1e-6, "disp": True, "maxiter": 500},
        )
        print(i)

    control_points = res.x.reshape(control_points.shape)

    mesh, ffd = create_init_mesh_ffd(radius=radius, n_control_points=n_control_points)
    disp = (control_points - ffd.control_points()) / ffd.box_length

    ffd.array_mu_x = disp[:, 0].reshape(ffd.array_mu_x.shape)
    ffd.array_mu_y = disp[:, 1].reshape(ffd.array_mu_y.shape)
    ffd.array_mu_z = disp[:, 2].reshape(ffd.array_mu_z.shape)
    
    new_vertices = ffd(mesh.vertices)
    mesh.vertices = new_vertices
    
    plot_mesh_c_points(mesh, ffd)

    aoas, areas = compute_aoa_and_area(panels=mesh.faces, points=new_vertices)
    plot_aoas(mesh, aoas)

    return mesh
