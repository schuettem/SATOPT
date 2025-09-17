import trimesh
from pygem import FFD
from drag import load_c_d_lookup_table
from skopt import gp_minimize
from functools import partial
from objectives import OBJECTIVES, DEFORMATIONS, PENALTIES


def run(
    objective: str,
    radius: float = 1.0,
    displacement_range: tuple=(0, 2),
    n_calls: int=100,
    penalties: list=[],
):

    length = radius * 3

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

    ffd = FFD([2, 2, 2])
    ffd.box_length = [length, length, length]
    ffd.box_origin = [-length/2, -length/2, -length/2]

    lookup_t = load_c_d_lookup_table("aerodynamic_coefficients_panel_method.csv")


    penalty_fns = []
    for penalty in penalties:
        penalty_fns.append(PENALTIES[penalty])
    
    fn = partial(OBJECTIVES[objective], init_mesh=mesh, ffd=ffd, lookup_t=lookup_t, penalty_fns=penalty_fns)

    if objective == "free":
        x = [displacement_range] * ffd.array_mu_x.flatten().shape[0] * 3
    elif objective in ["x", "y", "z"]:
        x = [displacement_range] * ffd.array_mu_x.flatten().shape[0]

    res = gp_minimize(fn, x, n_calls=n_calls)

    mesh, drag = DEFORMATIONS[objective](
        x=res.x, mesh=mesh, ffd=ffd, lookup_t=lookup_t
    )

    return mesh, drag, ffd
