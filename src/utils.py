import numpy as np
import pandas as pd
import trimesh
from pygem import FFD
from scipy.interpolate import interp1d


def get_init_mesh_ffd(cp_cube_side, n_control_points, radius=None, obj_file=None):
    if obj_file is not None:
        mesh = trimesh.load(file_obj=obj_file)
    else:
        if radius is not None:
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        else:
            raise ValueError("you have to define a radius for generating a icosphere")

    ffd = FFD(n_control_points)
    ffd.box_length = [cp_cube_side, cp_cube_side, cp_cube_side]
    ffd.box_origin = [-cp_cube_side / 2, -cp_cube_side / 2, -cp_cube_side / 2]
    return mesh, ffd


def box_from_corners(corners):
    """
    Build a trimesh box mesh from 8 corner points.
    Assumes corners are ordered like in get_aabb_corners or get_obb_corners.
    """

    # Triangulated cube faces (12 triangles, CCW winding)
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],  # bottom
            [4, 6, 5],
            [5, 6, 7],  # top
            [0, 2, 4],
            [2, 6, 4],  # left
            [1, 5, 3],
            [3, 5, 7],  # right
            [0, 4, 1],
            [1, 4, 5],  # front
            [2, 3, 6],
            [3, 7, 6],  # back
        ]
    )

    return trimesh.Trimesh(vertices=corners, faces=faces, process=False)


def load_c_d_lookup_table(csv_file):
    """
    Load the drag coefficient lookup table from csv file
    """
    try:
        df = pd.read_csv(csv_file)
        c_d_ram_list = df["C_d_ram"].tolist()
        c_d_wake_list = df["C_d_wake"].tolist()
        aoa_list = df["AoA"].tolist()

        ram_interpolator = interp1d(
            aoa_list, c_d_ram_list, bounds_error=False, fill_value="extrapolate"
        )
        wake_interpolator = interp1d(
            aoa_list, c_d_wake_list, bounds_error=False, fill_value="extrapolate"
        )
        lookup_table = {
            "ram": ram_interpolator,
            "wake": wake_interpolator,
            "aoa_range": (np.min(aoa_list), np.max(aoa_list)),
        }
        return lookup_table
    except Exception as e:
        print(f"Error loading drag coefficient lookup table: {e}")
        return []
