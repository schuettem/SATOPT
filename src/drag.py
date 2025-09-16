import numpy as np
import pandas as pd
from  scipy.interpolate import interp1d


def compute_aoa_and_area(panels, points, incident_velocity=np.array([-1, 0, 0])):
    """
    Compute the angle of attack (AoA) and reference area from a list of panels.
    panels: information from which points (ids for points list) the panel is formed (dosen't change).
    points: list of points in 3D space.
    """
    # Loop through the panels to compute the area
    areas = []
    aoas = []
    for (i, panel) in enumerate(panels):
        p1, p2, p3 = points[panel[0]], points[panel[1]], points[panel[2]]
        vec1 = p2 - p1
        vec2 = p3 - p1
        normal = np.cross(vec1, vec2)

        # Calculate the area of the triangle formed by the three points
        area = 0.5 * np.linalg.norm(normal)
        areas.append(area)

        if area == 0:
            raise ValueError(f"Panel {i} has zero area, points: {p1}, {p2}, {p3}")

        # Determine if it is wake or ram based on the angle between normal and incident velocity
        normed_normal = normal / np.linalg.norm(normal)
        incident_velocity_normed = incident_velocity / np.linalg.norm(incident_velocity)

        dot = np.dot(normed_normal, incident_velocity_normed)
        dot = np.clip(dot, -1.0, 1.0)  # Ensure dot product is within valid range

        # Compute the angle of attack (AoA)
        # AoA = 90° means normal perpendicular to flow (grazing)
        # AoA = 0° means normal parallel to flow
        # AoA > 0° means ram panel (normal opposes flow)
        # AoA < 0° means wake panel (normal aligned with flow)
        aoa = np.arcsin(-dot)  # Use arcsin with negative dot for correct sign
        aoa = np.rad2deg(aoa)  # Convert to degrees (-90° to +90°)        # Determine wake/ram based on AoA sign

        if np.isnan(aoa):
            raise ValueError(f"NaN AoA computed for panel {i}, points: {p1}, {p2}, {p3}")

        aoas.append(aoa)

    return aoas, areas


def load_c_d_lookup_table(csv_file):
    """
    Load the drag coefficient lookup table from csv file
    """
    try:
        df = pd.read_csv(csv_file)
        c_d_ram_list = df['C_d_ram'].tolist()
        c_d_wake_list = df['C_d_wake'].tolist()
        aoa_list = df['AoA'].tolist()

        ram_interpolator = interp1d(aoa_list,
                                    c_d_ram_list,
                                    bounds_error=False,
                                    fill_value="extrapolate")
        wake_interpolator = interp1d(aoa_list,
                                     c_d_wake_list,
                                     bounds_error=False,
                                     fill_value="extrapolate")
        lookup_table = {
            'ram': ram_interpolator,
            'wake': wake_interpolator,
            'aoa_range': (np.min(aoa_list), np.max(aoa_list))
        }
        return lookup_table
    except Exception as e:
        print(f"Error loading drag coefficient lookup table: {e}")
        return []

def compute_drag(aoas, areas, lookup_table):
    """
    Compute the drag based on angle of attack and area.
    aoas: list of angles of attack in degrees.
    areas: list of areas corresponding to each angle of attack.
    lookup_table: lookup table for panels with AoAs for ram and wake drag coefficients.
    """
    total_drag = 0.0
    for aoa, area in zip(aoas, areas):
        if aoa < 0.0:  # Negative AoA indicates wake panel
            c_d = lookup_table['wake'](aoa)
        else:
            c_d = lookup_table['ram'](aoa)

        # Ensure c_d is a valid number
        if np.isnan(c_d) or np.isinf(c_d):
            print(f"Invalid drag coefficient for AoA {aoa}°: {c_d}")
            continue

        drag = c_d * area
        total_drag += drag

    return total_drag


def compute_elementwise_drag(aoas, areas, lookup_table):
    """
    Compute the drag for each panel based on its area and angle of attack.
    panels: information from which points (ids for points list) the panel is formed (dosen't change).
    points: list of points in 3D space.
    lookup_table: lookup table for panels with AoAs for ram and wake drag coefficients.
    """
    drag_per_panel = []
    for aoa, area in zip(aoas, areas):
        if aoa < 0.0:  # Negative AoA indicates wake panel
            c_d = lookup_table['wake'](aoa)
        else:
            c_d = lookup_table['ram'](aoa)

        # Ensure c_d is a valid number
        if np.isnan(c_d) or np.isinf(c_d):
            print(f"Invalid drag coefficient for AoA {aoa}°: {c_d}")
            continue

        drag = c_d * area
        drag_per_panel.append(drag)

    return drag_per_panel


if __name__ == "__main__":
    lookup_table = load_c_d_lookup_table('~/SATOPT/aerodynamic_coefficients_panel_method.csv')
    print("Drag Coefficient Lookup Table Loaded: ", lookup_table)
    # Test the lookup table
    aoa = 2
    wake_drag = lookup_table['wake'](aoa)
    ram_drag = lookup_table['ram'](aoa)
    print(f"AoA: {aoa}°, C_d_wake: {wake_drag:.3f}, C_d_ram: {ram_drag:.3f}")