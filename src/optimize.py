import trimesh
from pygem import FFD
from drag import load_c_d_lookup_table
from scipy.optimize import minimize
from objectives import objective
from drag import compute_aoa_and_area, load_c_d_lookup_table
from analysis import plot_mesh_c_points, plot_aoas
from constrains import get_flip_constrain
import os


def run(
    radius=1
):
    
    cube_side = radius * 3
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    ffd = FFD([2, 2, 2])
    ffd.box_length = [cube_side, cube_side, cube_side]
    ffd.box_origin = [-cube_side/2, -cube_side/2, -cube_side/2]

    control_points = ffd.control_points()
    flat_control_points = control_points.flatten()
    
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "aerodynamic_coefficients_panel_method.csv")
    lookup_t = load_c_d_lookup_table(file_path)
    
    linear_constraint = get_flip_constrain()
    
    res = minimize(objective, flat_control_points, args=(mesh, ffd, lookup_t), method='SLSQP', constraints=[linear_constraint], options={'ftol':1e-12, 'disp':True, 'maxiter':500})

    control_points = res.x.reshape((8,3))

    disp = (control_points - ffd.control_points()) / ffd.box_length
    ffd.array_mu_x = disp[:, 0].reshape(ffd.array_mu_x.shape)
    ffd.array_mu_y = disp[:, 1].reshape(ffd.array_mu_y.shape)
    ffd.array_mu_z = disp[:, 2].reshape(ffd.array_mu_z.shape)
    
    new_vertices = ffd(mesh.vertices)
    new_mesh = mesh
    new_mesh.vertices = new_vertices
    
    plot_mesh_c_points(mesh, ffd)

    aoas, areas = compute_aoa_and_area(panels=new_mesh.faces, points=new_vertices)
    plot_aoas(mesh, aoas)
    new_mesh.show()