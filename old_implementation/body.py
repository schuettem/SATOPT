import numpy as np


def body_length(vertices):
    xs = vertices[:, 0]
    return np.max(xs) - np.min(xs)


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


def create_sphere(n_phi, n_theta, radius=0.5, center=(0.5, 0.5, 0.5)):
    """
    Create a mesh of a sphere with given resolution and radius.
    Create triangular panels for the sphere mesh containing the ids of the points forming each panel.
    """
    # Create phi from 0 to pi (includes both poles)
    phi = np.linspace(0, np.pi, n_phi)
    # Create theta from 0 to 2*pi but exclude the endpoint to avoid duplication
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    # Create meshgrid
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")

    x = center[0] + radius * np.cos(phi_grid)
    y = center[1] + radius * np.sin(phi_grid) * np.cos(theta_grid)
    z = center[2] + radius * np.sin(phi_grid) * np.sin(theta_grid)
    optimized_vertices = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

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
                panels.append((current, below, current_next))  # Fixed winding
            else:  # Middle sections - create two triangles per quad
                # First triangle: current -> below -> below_next (CCW from outside)
                panels.append((current, below, below_next))
                # Second triangle: current -> below_next -> current_next (CCW from outside)
                panels.append((current, below_next, current_next))

    return optimized_vertices, panels



def verify_normals_outward(vertices, panels, center=(0.0, 0.0, 0.0)):
    """
    Verify that mesh normals point outward from the center
    """
    outward_count = 0
    inward_count = 0

    for i, panel in enumerate(panels):
        p1, p2, p3 = vertices[panel[0]], vertices[panel[1]], vertices[panel[2]]

        # Compute normal using cross product (right-hand rule)
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)

        # Vector from center to triangle centroid
        triangle_center = (p1 + p2 + p3) / 3
        center_to_triangle = triangle_center - np.array(center)
        center_to_triangle = center_to_triangle / np.linalg.norm(center_to_triangle)

        # Check if normal points in same direction as center-to-triangle
        dot_product = np.dot(normal, center_to_triangle)
        if dot_product > 0:
            outward_count += 1
        else:
            inward_count += 1

    print(f"Normal check: {outward_count} outward, {inward_count} inward")
    return inward_count == 0  # All normals should point outward
