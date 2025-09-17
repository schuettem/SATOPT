import trimesh
from pygem import FFD
from drag import load_c_d_lookup_table
from skopt import gp_minimize
from functools import partial
from objectives import objective_fn, deformatoin_fn

def run(
    objective: str='x',
    radius: float=1.0
):
    
    length = radius * 2.5
    
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

    ffd = FFD([2, 2, 2])
    ffd.box_length = [length, length, length]
    ffd.box_origin = [-length/2, -length/2, -length/2]
    
    lookup_t = load_c_d_lookup_table(
        "aerodynamic_coefficients_panel_method.csv"
    )

    fn = partial(objective_fn[objective], init_mesh=mesh, ffd=ffd, lookup_t=lookup_t)

    if objective == "free":
        x = [(0.0, 3.0)] * ffd.array_mu_x.flatten().shape[0] * 3
    elif objective == "x":
        x = [(0.0, 3.0)] * ffd.array_mu_x.flatten().shape[0]
        
    res = gp_minimize(fn, x, n_calls=100)
    
    mesh, drag = deformatoin_fn[objective](x=res.x, mesh=mesh, ffd=ffd, lookup_t=lookup_t)

    return mesh, drag