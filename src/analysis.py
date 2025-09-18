import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


def plot_mesh_c_points(mesh, ffd):
    control_points = ffd.control_points().T
    vertices = mesh.vertices

    # Create interactive 3D scatter plot
    fig = go.Figure()

    # Add main mesh points
    fig.add_trace(
        go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers",
            marker=dict(size=3, color="blue"),
            name="mesh",
        )
    )

    # Add control points
    fig.add_trace(
        go.Scatter3d(
            x=control_points[0, :],
            y=control_points[1, :],
            z=control_points[2, :],
            mode="markers",
            marker=dict(size=8, color="red"),
            name="control points",
        )
    )

    # Layout for axes
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=700,
        height=700,
        title="Interactive 3D Scatter",
    )

    fig.show()


def plot_aoas(mesh, aoas):
    # Plot comparison
    fig = plt.figure(figsize=(12, 5))

    # Second subplot: Surface colored by aoas
    ax2 = fig.add_subplot(111, projection="3d")

    # Create triangles for 3D plotting
    triangles = []
    colors = []
    for i, panel in enumerate(mesh.faces):
        # Get triangle vertices
        p1, p2, p3 = (
            mesh.vertices[panel[0]],
            mesh.vertices[panel[1]],
            mesh.vertices[panel[2]],
        )
        triangle = [p1, p2, p3]
        triangles.append(triangle)

        # Color based on aoas value
        aoa = aoas[i]
        # norme between 0 and 1 from -90 to 90 degrees
        aoa_normed = (aoa - np.min(aoas)) / (np.max(aoas) - np.min(aoas))

        color = plt.cm.viridis(aoa_normed)  # Use colormap to get color
        colors.append(color)

    # Create 3D polygon collection
    poly3d = Poly3DCollection(triangles, facecolors=colors, edgecolors="none")
    ax2.add_collection3d(poly3d)

    # Add colorbar for aoas values
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=-90, vmax=90)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, shrink=0.5, aspect=10)
    cbar.set_label("AoA per Panel")

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    plt.tight_layout()

    plt.show()
