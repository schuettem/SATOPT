
# Free Form Deformation

import numpy as np
from math import comb
import matplotlib.pyplot as plt


def bernsetein(n, i, t):
    """
        Compute the i-th Bernstein basis polynomial of degree n at t.
    """
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

class FFD:
    def __init__(self, bbox_min, bbox_max, lattice_shape):
        """
            bbox_min, bbox_max: array-like of shape (3,)
                Minimum and maximum corners of the deformation lattice.
            lattice_shape: tuple (l+1, m+1, n+1)
                Number of control points in each dimension
        """
        self.bmin = np.array(bbox_min, dtype=np.float32)
        self.bmax = np.array(bbox_max, dtype=np.float32)
        self.l, self.m, self.n = np.array(lattice_shape, dtype=np.int32) - 1

        # Initialize control points
        xs = np.linspace(self.bmin[0], self.bmax[0], self.l + 1)
        ys = np.linspace(self.bmin[1], self.bmax[1], self.m + 1)
        zs = np.linspace(self.bmin[2], self.bmax[2], self.n + 1)
        self.P = np.zeros((self.l + 1, self.m + 1, self.n + 1, 3), dtype=np.float32)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    self.P[i, j, k] = np.array([x, y, z], dtype=np.float32)

    def set_flat_control_points(self, flat_P):
        self.P = flat_P.reshape((self.l + 1, self.m + 1, self.n + 1, 3))

    def to_parametric(self, X):
        """
            Map a point X in global coords to (s, t, u) in [0, 1]^3.
            assuming an axis-aligned lattice.
        """
        return (X - self.bmin) / (self.bmax - self.bmin)

    def deform_point(self, s, t, u):
        """
            Apply the FFD blend to paramtric coords (s,t,u).
            Returns the the deformed global-space position
        """
        Xp = np.zeros(3, dtype=np.float32)
        for i in range(self.l + 1):
            Bi = bernsetein(self.l, i, s)
            for j in range(self.m + 1):
                Bj = bernsetein(self.m, j, t)
                for k in range(self.n + 1):
                    Bk = bernsetein(self.n, k, u)
                    Xp += Bi * Bj * Bk * self.P[i, j, k]
        return Xp

    def deform_mesh(self, vertices):
        """
            vertices: (N, 3) array of mesh points
            Returns deformed_vertices: (N, 3)
        """
        deformed_vertices = np.zeros_like(vertices, dtype=np.float32)
        for idx, X in enumerate(vertices):
            s, t, u = self.to_parametric(X)
            deformed_vertices[idx] = self.deform_point(s, t, u)
        return deformed_vertices

# Example usage:
if __name__ == "__main__":
    # DEfine a unit-cube bounding box
    bbox_min = [0.0, 0.0, 0.0]
    bbox_max = [1.0, 1.0, 1.0]

    # Create a 3x3x3 lattice
    ffd = FFD(bbox_min, bbox_max, lattice_shape=(4, 4, 4))

    # A simple mesh: unit sphere
    phi, theta = np.mgrid[0:np.pi:20j, 0:2 * np.pi:20j]
    x = 0.5 + 0.5 * np.sin(phi) * np.cos(theta)
    y = 0.5 + 0.5 * np.sin(phi) * np.sin(theta)
    z = 0.5 + 0.5 * np.cos(phi)
    mesh_verts = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # Plot the original mesh
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(x, y, z, alpha=0.7, color='blue', label='Original Mesh')
    ax.set_title('Original Mesh')
    # plot the control points
    ax.scatter(ffd.P[:, :, :, 0], ffd.P[:, :, :, 1], ffd.P[:, :, :, 2], color='green', label='Control Points')

    # Move one control point to deform the mesh
    ffd.P[1, 1, 3] += np.array([0.2, 0.2, 5], dtype=np.float32)
    ffd.P[2, 1, 3] += np.array([-0.2, 0.2, 5], dtype=np.float32)
    ffd.P[1, 2, 3] += np.array([0.2, -0.2, 5], dtype=np.float32)
    ffd.P[2, 2, 3] += np.array([-0.2, -0.2, 5], dtype=np.float32)

    # Deform the mesh
    new_verts = ffd.deform_mesh(mesh_verts)

    # Reshape back to grid form
    x_new = new_verts[:, 0].reshape(phi.shape)
    y_new = new_verts[:, 1].reshape(phi.shape)
    z_new = new_verts[:, 2].reshape(phi.shape)

    # Plot the deformed mesh
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x_new, y_new, z_new, alpha=0.7, color='red', label='Deformed Mesh')
    ax2.set_title('Deformed Mesh')
    # plot the control points
    ax2.scatter(ffd.P[:, :, :, 0], ffd.P[:, :, :, 1], ffd.P[:, :, :, 2], color='green', label='Control Points')

    # Set equal aspect ratio for both plots
    ax.set_box_aspect([1,1,1])  # Add this line for the first subplot
    ax2.set_box_aspect([1,1,1])  # Add this line for the second subplot

    plt.show()


