import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pygem import FFD

# Import from drag.py
from drag import compute_aoa_and_area, load_c_d_lookup_table, compute_drag


def create_sphere(n_phi, n_theta, radius=0.5, center=(0.5, 0.5, 0.5)):
    """
    Create a mesh of a sphere with given resolution and radius.
    Create triangular panels for the sphere mesh containing the ids of the points forming each panel.
    """
    phi, theta = np.mgrid[
        0 : np.pi : complex(0, n_phi), 0 : 2 * np.pi : complex(0, n_theta)
    ]

    x = center[2] + radius * np.cos(phi)
    y = center[0] + radius * np.sin(phi) * np.cos(theta)
    z = center[1] + radius * np.sin(phi) * np.sin(theta)
    mesh_vertices = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    panels = []
    # Create triangular panels, avoiding degenerate triangles at poles
    for i in range(n_phi - 1):
        for j in range(n_theta):
            # Current row indices
            current = i * n_theta + j
            next_j = (j + 1) % n_theta
            current_next = i * n_theta + next_j

            # Next row indices
            below = (i + 1) * n_theta + j
            below_next = (i + 1) * n_theta + next_j

            # Skip degenerate triangles at the poles
            if i == 0:  # Top pole - only create one triangle per segment
                panels.append((current, below, below_next))  # Fixed winding
            elif i == n_phi - 2:  # Bottom pole - only create one triangle per segment
                panels.append((current, current_next, below))  # Fixed winding
            else:  # Middle sections - create two triangles per quad
                # First triangle: current -> below -> below_next (CCW from outside)
                panels.append((current, below, below_next))
                # Second triangle: current -> below_next -> current_next (CCW from outside)
                panels.append((current, below_next, current_next))

    return mesh_vertices, panels


def body_volume(vertices, panels):
    """
    Compute the volume of the body defined by the mesh vertices and panels.
    vertices: (N, 3) array of mesh points.
    panels: list of panels, where each panel is a tuple of point indices.
    """
    volume = 0.0
    for panel in panels:
        p1, p2, p3 = vertices[panel[0]], vertices[panel[1]], vertices[panel[2]]

        # Volume contribution using divergence theorem
        v1 = p2 - p1
        v2 = p3 - p1
        cross_product = np.cross(v1, v2)

        # The cross product gives us 2 * area * normal_vector
        # For outward normals, this should give positive volume
        centroid = (p1 + p2 + p3) / 3
        volume += np.dot(centroid, cross_product) / 6  # Divide by 6 = 2 * 3

    # If normals are correct (outward), volume should be positive
    # If volume is negative, normals are inward (mesh is inside-out)
    return abs(volume)  # Take absolute value as safety measure


def body_length(vertices):
    xs = vertices[:, 0]
    return np.max(xs) - np.min(xs)


def optimize_satellite_pygem(
    bbox_min,
    bbox_max,
    lattice_shape,
    L_max,
    V_min,
    n_iter,
    radius,
    center,
    n_phi=20,
    n_theta=20,
):
    """
    Optimize satellite using PyGem FFD with volume preservation
    """
    # Create PyGem FFD instance
    ffd = FFD()

    # Set up the FFD box
    ffd.box_length = np.array(bbox_max) - np.array(bbox_min)
    ffd.box_origin = np.array(bbox_min)
    ffd.n_control_points = lattice_shape

    # Load drag coefficient lookup table
    lookup_table = load_c_d_lookup_table(
        "plots_csv/aerodynamic_coefficients/aerodynamic_coefficients_panel_method.csv"
    )

    # Create original mesh vertices
    org_mesh_vertices, panels = create_sphere(n_phi, n_theta, radius, center)

    # Initialize original volume for preservation
    original_volume = body_volume(org_mesh_vertices, panels)
    print(f"Original volume: {original_volume:.2f}")

    def objective(control_points_displacement):
        """
        Objective function with PyGem FFD
        """
        # Apply displacement to control points
        ffd.array_mu_x = control_points_displacement[
            : lattice_shape[0] * lattice_shape[1] * lattice_shape[2]
        ].reshape(lattice_shape)
        ffd.array_mu_y = control_points_displacement[
            lattice_shape[0] * lattice_shape[1] * lattice_shape[2] : 2
            * lattice_shape[0]
            * lattice_shape[1]
            * lattice_shape[2]
        ].reshape(lattice_shape)
        ffd.array_mu_z = control_points_displacement[
            2 * lattice_shape[0] * lattice_shape[1] * lattice_shape[2] :
        ].reshape(lattice_shape)

        # Deform mesh
        deformed_vertices = ffd(org_mesh_vertices)

        # Compute AoA, areas, and wakes
        aoas, areas, wakes = compute_aoa_and_area(panels, deformed_vertices)

        # Compute drag
        drag = compute_drag(aoas, areas, wakes, lookup_table)

        return drag

    # Volume preservation constraint
    def volume_constraint(control_points_displacement):
        """
        Constraint to preserve volume using PyGem
        """
        # Apply displacement
        ffd.array_mu_x = control_points_displacement[
            : lattice_shape[0] * lattice_shape[1] * lattice_shape[2]
        ].reshape(lattice_shape)
        ffd.array_mu_y = control_points_displacement[
            lattice_shape[0] * lattice_shape[1] * lattice_shape[2] : 2
            * lattice_shape[0]
            * lattice_shape[1]
            * lattice_shape[2]
        ].reshape(lattice_shape)
        ffd.array_mu_z = control_points_displacement[
            2 * lattice_shape[0] * lattice_shape[1] * lattice_shape[2] :
        ].reshape(lattice_shape)

        # Deform mesh
        deformed_vertices = ffd(org_mesh_vertices)

        # Calculate current volume
        current_volume = body_volume(deformed_vertices, panels)

        # Volume preservation: current_volume should equal original_volume
        # Return positive when constraint is satisfied
        tolerance = 0.05 * original_volume  # 5% tolerance
        return tolerance - abs(current_volume - original_volume)

    # Length constraint
    def length_constraint(control_points_displacement):
        # Apply displacement
        ffd.array_mu_x = control_points_displacement[
            : lattice_shape[0] * lattice_shape[1] * lattice_shape[2]
        ].reshape(lattice_shape)
        ffd.array_mu_y = control_points_displacement[
            lattice_shape[0] * lattice_shape[1] * lattice_shape[2] : 2
            * lattice_shape[0]
            * lattice_shape[1]
            * lattice_shape[2]
        ].reshape(lattice_shape)
        ffd.array_mu_z = control_points_displacement[
            2 * lattice_shape[0] * lattice_shape[1] * lattice_shape[2] :
        ].reshape(lattice_shape)

        # Deform mesh
        deformed_vertices = ffd(org_mesh_vertices)

        # Length constraint
        return L_max - body_length(deformed_vertices)

    constraints = [
        {"type": "ineq", "fun": volume_constraint},
        {"type": "ineq", "fun": length_constraint},
    ]

    # Initial guess: no displacement
    n_control_points = lattice_shape[0] * lattice_shape[1] * lattice_shape[2]
    x0 = np.zeros(3 * n_control_points)  # x, y, z displacements

    # Bounds: limit displacement to prevent self-intersection
    bbox_size = np.array(bbox_max) - np.array(bbox_min)
    max_displacement = 0.1 * min(bbox_size)  # 10% of smallest dimension

    bounds = [(-max_displacement, max_displacement)] * len(x0)

    # Optimization callback
    iter_count = 0

    def callback(xk):
        nonlocal iter_count
        iter_count += 1

        # Apply displacement for evaluation
        ffd.array_mu_x = xk[:n_control_points].reshape(lattice_shape)
        ffd.array_mu_y = xk[n_control_points : 2 * n_control_points].reshape(
            lattice_shape
        )
        ffd.array_mu_z = xk[2 * n_control_points :].reshape(lattice_shape)

        deformed_vertices = ffd(org_mesh_vertices)

        current_drag = objective(xk)
        current_volume = body_volume(deformed_vertices, panels)
        current_length = body_length(deformed_vertices)

        print(
            f"[Iter {iter_count:3d}] Drag: {current_drag:.6f}, Volume: {current_volume:.2f}, Length: {current_length:.2f}"
        )

    # Run optimization
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={"maxiter": n_iter, "ftol": 1e-6, "disp": True},
    )

    # Apply final result
    ffd.array_mu_x = result.x[:n_control_points].reshape(lattice_shape)
    ffd.array_mu_y = result.x[n_control_points : 2 * n_control_points].reshape(
        lattice_shape
    )
    ffd.array_mu_z = result.x[2 * n_control_points :].reshape(lattice_shape)

    optimized_vertices = ffd(org_mesh_vertices)

    return optimized_vertices, ffd


if __name__ == "__main__":
    # Define bounding box
    bbox_min = [0.0, 0.0, 0.0]
    bbox_max = [10.0, 10.0, 10.0]

    # Geometry parameters
    radius = 5.0
    center = (5.0, 5.0, 5.0)

    # FFD lattice (smaller for stability)
    lattice_shape = (3, 3, 3)

    # Constraints
    L_max = 12.0
    V_min = 400.0  # Not used with volume preservation

    # Optimization parameters
    n_phi, n_theta = 8, 8
    n_iter = 10

    # Run optimization with PyGem
    optimized_vertices, ffd_instance = optimize_satellite_pygem(
        bbox_min,
        bbox_max,
        lattice_shape,
        L_max,
        V_min,
        n_iter,
        radius,
        center,
        n_phi,
        n_theta,
    )

    # Create original sphere for comparison
    org_vertices, panels = create_sphere(n_phi, n_theta, radius, center)

    # Plot comparison
    fig = plt.figure(figsize=(12, 5))

    # Original
    ax1 = fig.add_subplot(121, projection="3d")
    x_orig = org_vertices[:, 0].reshape(n_phi, n_theta)
    y_orig = org_vertices[:, 1].reshape(n_phi, n_theta)
    z_orig = org_vertices[:, 2].reshape(n_phi, n_theta)
    ax1.plot_wireframe(x_orig, y_orig, z_orig, color="blue", alpha=0.7)
    ax1.set_title("Original Sphere")
    ax1.set_box_aspect([1, 1, 1])

    # Optimized
    ax2 = fig.add_subplot(122, projection="3d")
    x_opt = optimized_vertices[:, 0].reshape(n_phi, n_theta)
    y_opt = optimized_vertices[:, 1].reshape(n_phi, n_theta)
    z_opt = optimized_vertices[:, 2].reshape(n_phi, n_theta)
    ax2.plot_wireframe(x_opt, y_opt, z_opt, color="red", alpha=0.7)
    ax2.set_title("Volume-Preserved Optimized")
    ax2.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()
