import laspy
import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree

def extract_prune_points(yaml_path):
    """Extract prune points from YAML."""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    prune_points = []
    base_directions = []
    base_points = []

    for entry in data.values():
        prune_points.append(entry["prune_point"])
        base_directions.append(entry["pruned_branch_base_direction"])
        base_points.append(entry["pruned_branch_base_point"])

    prune_points = np.array(prune_points, dtype=float)
    base_directions = np.array(base_directions, dtype=float)
    base_points = np.array(base_points, dtype=float)
    return prune_points, base_directions, base_points

def filter_by_axis(pcd,
                    x_min=None, x_max=None,
                    y_min=None, y_max=None,
                    z_min=None, z_max=None
                ):
    points = np.asarray(pcd.points)
    mask = np.ones(points.shape[0], dtype=bool)
    if x_min is not None:
        mask &= points[:, 0] >= x_min
    if x_max is not None:
        mask &= points[:, 0] <= x_max
    if y_min is not None:
        mask &= points[:, 1] >= y_min
    if y_max is not None:
        mask &= points[:, 1] <= y_max
    if z_min is not None:
        mask &= points[:, 2] >= z_min
    if z_max is not None:
        mask &= points[:, 2] <= z_max
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    return filtered_pcd

# -------------------- User Paths --------------------
yaml_path = "/home/marcus/IMML/prune_point_identifier/results/pruned_branches/all_branches_info.yaml"
las_path  = "/home/marcus/Downloads/2025-01-23-10-03-48.segmented.colour.las"
# -----------------------------------------------------

# Load prune points
prune_points, _, _ = extract_prune_points(yaml_path)

# Load LAS file
las = laspy.read(las_path)
points = np.vstack((las.x, las.y, las.z)).T

# Extract and normalize RGB
if hasattr(las, "red"):
    colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
else:
    colors = np.zeros_like(points)

# Recolor bluish points to brownish gray
blue_threshold = 0.35
dominance_factor = 1.1
mask_blue = (colors[:, 2] > blue_threshold) & \
            (colors[:, 2] > dominance_factor * colors[:, 0]) & \
            (colors[:, 2] > dominance_factor * colors[:, 1])
brownish_gray = np.array([80, 70, 60]) / 255.0
colors[mask_blue] = brownish_gray

# -------------------- Create main point cloud --------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# -------------------- Apply rotation + initial translation --------------------
theta_deg = 33.0
translation = np.array([0.03, -0.075, 0.0])  # initial overlay adjustment
theta_rad = np.radians(theta_deg)
R = np.array([
    [np.cos(theta_rad), -np.sin(theta_rad), 0.0],
    [np.sin(theta_rad),  np.cos(theta_rad), 0.0],
    [0.0,                0.0,               1.0]
])

pcd.rotate(R, center=(0, 0, 0))
pcd.translate(translation)

# -------------------- Combine all points to compute centroid --------------------
all_points = np.vstack((np.asarray(pcd.points), prune_points))
centroid = all_points.mean(axis=0)
pcd.translate(-centroid)
prune_points_centered = prune_points - centroid

# -------------------- Filter prune points: y <= 0 --------------------
prune_points_filtered = prune_points_centered[prune_points_centered[:, 1] <= 0]

# -------------------- Remove nearby duplicates --------------------
tree = cKDTree(prune_points_filtered)
threshold = 0.01
to_remove = set()

for i, point in enumerate(prune_points_filtered):
    if i in to_remove:
        continue
    neighbors = tree.query_ball_point(point, r=threshold)
    for n in neighbors:
        if n != i:
            to_remove.add(n)

prune_points_unique = np.delete(prune_points_filtered, list(to_remove), axis=0)

# -------------------- Create spheres for prune points --------------------
marker_radius = 0.02
spheres = []
for p in prune_points_unique:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius, resolution=18)
    sphere.translate(p)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    sphere.compute_vertex_normals()
    spheres.append(sphere)

# -------------------- Coordinate frame --------------------
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.2,
    origin=[0, 0, 0]
)

# -------------------- Visualize --------------------
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Centered LAS + Prune Points (Deduplicated)", width=1400, height=900)
vis.add_geometry(pcd)
for s in spheres:
    vis.add_geometry(s)
# vis.add_geometry(frame)

opt = vis.get_render_option()
opt.point_size = 1.0
opt.background_color = np.array([0.2, 0.2, 0.25])
# opt.background_color = np.array([1.0, 1.0, 1.0])

vis.run()
vis.destroy_window()
