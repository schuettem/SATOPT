import numpy as np
from ffd import FFD

# Create a simple test case
bbox_min = [-1, -1, -1]
bbox_max = [1, 1, 1]
ffd = FFD(bbox_min, bbox_max, lattice_shape=(3, 3, 3))

# Simple triangle to test
vertices = np.array(
    [
        [0.0, 0.0, 0.0],  # Center
        [0.2, 0.0, 0.0],  # Right
        [0.0, 0.2, 0.0],  # Up
    ],
    dtype=np.float32,
)

# Calculate original normal
v1 = vertices[1] - vertices[0]  # Right vector
v2 = vertices[2] - vertices[0]  # Up vector
original_normal = np.cross(v1, v2)
print("Original triangle:")
print(f"  v0: {vertices[0]}")
print(f"  v1: {vertices[1]}")
print(f"  v2: {vertices[2]}")
print(f"  Edge vectors: v1={v1}, v2={v2}")
print(f"  Normal: {original_normal} (length: {np.linalg.norm(original_normal):.3f})")

# Apply some deformation - let's move a control point
print(f"\nControl point before: {ffd.P[1, 1, 1]}")
ffd.P[1, 1, 1] += np.array([0.5, -0.3, 0.1])  # Move center control point
print(f"Control point after: {ffd.P[1, 1, 1]}")

# Deform the triangle
deformed_vertices = ffd.deform_mesh(vertices)

# Calculate deformed normal
v1_def = deformed_vertices[1] - deformed_vertices[0]
v2_def = deformed_vertices[2] - deformed_vertices[0]
deformed_normal = np.cross(v1_def, v2_def)

print("\nDeformed triangle:")
print(f"  v0: {deformed_vertices[0]}")
print(f"  v1: {deformed_vertices[1]}")
print(f"  v2: {deformed_vertices[2]}")
print(f"  Edge vectors: v1={v1_def}, v2={v2_def}")
print(f"  Normal: {deformed_normal} (length: {np.linalg.norm(deformed_normal):.3f})")

# Check if normal flipped
dot_product = np.dot(original_normal, deformed_normal)
print("\nNormal comparison:")
print(f"  Dot product of normals: {dot_product:.3f}")
if dot_product > 0:
    print("  → Normals point in similar direction")
else:
    print("  → Normals point in opposite directions (FLIPPED!)")

# Test with extreme deformation that causes inversion
print("\n=== Testing with extreme deformation ===")
ffd2 = FFD(bbox_min, bbox_max, lattice_shape=(3, 3, 3))
# Create a strong inversion by moving control points in opposite directions
ffd2.P[1, 1, 1] += np.array([-2.0, 0.0, 0.0])  # Pull center left
ffd2.P[2, 1, 1] += np.array([2.0, 0.0, 0.0])  # Push right even more right

deformed_vertices2 = ffd2.deform_mesh(vertices)
v1_def2 = deformed_vertices2[1] - deformed_vertices2[0]
v2_def2 = deformed_vertices2[2] - deformed_vertices2[0]
deformed_normal2 = np.cross(v1_def2, v2_def2)

dot_product2 = np.dot(original_normal, deformed_normal2)
print("Extreme deformation:")
print(f"  Original normal: {original_normal}")
print(f"  Deformed normal: {deformed_normal2}")
print(f"  Dot product: {dot_product2:.3f}")
if dot_product2 > 0:
    print("  → Normals still point in similar direction")
else:
    print("  → Normals FLIPPED due to extreme deformation!")
