import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib import cm

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
    """Plot mesh with faces colored by Angle of Attack using Plotly."""

    # Extract vertices and faces
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # Unpack vertex coordinates
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # Flatten face indices
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    # Normalize AoA values to [0, 1]
    aoa_normed = (np.array(aoas) - (-90)) / (90 - (-90))

    # Map normalized AoA to Viridis colormap
    viridis = cm.get_cmap("viridis")
    colors_rgba = viridis(aoa_normed)
    colors_rgb = (colors_rgba[:, :3] * 255).astype(np.uint8)
    colors_hex = ["rgb({},{},{})".format(r, g, b) for r, g, b in colors_rgb]

    # Create mesh plot
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            facecolor=colors_hex,
            showscale=True,
            colorbar=dict(title="AoA per Panel"),
            intensity=np.array(aoas),
            intensitymode="cell",
            colorscale="Viridis",
            cmin=-90,
            cmax=90,
            name="Mesh Surface",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
            lightposition=dict(x=0, y=0, z=0)
        )
    ])

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        title="Mesh Colored by Angle of Attack",
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()