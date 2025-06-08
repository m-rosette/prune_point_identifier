import open3d as o3d
import numpy as np
import os


class PrunedBranchIdentifier:
    """
    PrunedBranchIdentifier is a utility class for identifying, isolating, and visualizing pruned branches from 3D point cloud data in relation to a reference mesh.
    This class provides a workflow for:
        - Loading a triangle mesh and a point cloud from file.
        - Filtering the point cloud based on spatial constraints (e.g., minimum Z value).
        - Computing distances from each point in the point cloud to the mesh surface using raycasting.
        - Identifying candidate pruned branch points based on distance thresholds and statistical outlier removal.
        - Clustering the candidate points to isolate individual pruned branches using DBSCAN.
        - Optionally refining clusters, saving isolated branches, and visualizing results alongside the mesh.
        mesh_path (str): Path to the triangle mesh file.
        point_cloud_path (str): Path to the point cloud file.
        z_min (float, optional): Minimum Z value for filtering the point cloud. Defaults to 0.03.
        prune_point_dist_threshold (float, optional): Distance threshold for identifying pruned branch points. Defaults to 0.05.
        branch_cluster_distance_threshold (float, optional): Maximum allowed distance from a cluster centroid to the mesh for it to be considered a valid branch. Defaults to 0.5.
        outlier_nb_neighbors (int, optional): Number of neighbors for statistical outlier removal. Defaults to 20.
        outlier_std_ratio (float, optional): Standard deviation ratio for statistical outlier removal. Defaults to 2.0.
        dbscan_eps (float, optional): Epsilon parameter for DBSCAN clustering. Defaults to 0.04.
        dbscan_min_points (int, optional): Minimum number of points for DBSCAN clustering. Defaults to 10.
        mesh_color (np.ndarray, optional): RGB color for the mesh visualization. Defaults to np.array([0.6, 0.8, 1.0]).
        point_color (np.ndarray, optional): RGB color for the point cloud visualization. Defaults to np.array([0.1, 0.8, 0.1]).
        save_directory_prefix (str, optional): Prefix for directories where isolated branches are saved. Defaults to "results/pruned_branches/branch".
    Attributes:
        mesh (o3d.geometry.TriangleMesh): The loaded mesh object.
        pcd (o3d.geometry.PointCloud): The loaded and filtered point cloud.
        scene (o3d.t.geometry.RaycastingScene): Raycasting scene for distance computations.
        pcd_points (np.ndarray): Numpy array of point cloud coordinates.
        pruned_branch_pcd (o3d.geometry.PointCloud): Point cloud containing candidate pruned branch points.
    Typical usage:
        identifier = PrunedBranchIdentifier(mesh_path, point_cloud_path)
        identifier.main(vis=True, save_branches=True)
    """
    def __init__(
        self,
        mesh_path: str,
        point_cloud_path: str,
        z_min: float = 0.03,
        prune_point_dist_threshold: float = 0.05,
        branch_cluster_distance_threshold: float = 0.5,
        outlier_nb_neighbors: int = 20,
        outlier_std_ratio: float = 2.0,
        dbscan_eps: float = 0.04,
        dbscan_min_points: int = 10,
        mesh_color: np.ndarray = np.array([0.6, 0.8, 1.0]),
        point_color: np.ndarray = np.array([0.1, 0.8, 0.1]),
        save_directory_prefix: str = "results/pruned_branches/branch",
    ):
        
        """
        Initialize internal variables and thresholds.
        """
        # File paths
        self.mesh_path = mesh_path
        self.point_cloud_path = point_cloud_path

        # Thresholds & parameters
        self.z_min = z_min
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_points = dbscan_min_points
        self.prune_point_dist_threshold = prune_point_dist_threshold
        self.branch_cluster_distance_threshold = branch_cluster_distance_threshold
        self.mesh_color = mesh_color
        self.point_color = point_color
        self.save_directory_prefix = save_directory_prefix

        # Internal containers
        self.mesh = None
        self.pcd = None
        self.scene = None
        self.pcd_points = None
        self.pruned_branch_pcd = None

        # Load mesh and point cloud
        self.load_mesh()
        self.load_point_cloud(filter_z=True)

    def load_mesh(self):
        # Load your triangle mesh
        self.mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color(self.mesh_color)
    
    def load_point_cloud(self, filter_z=True):
        """
        Loads a point cloud from the file specified by self.point_cloud_path and optionally filters it by the Z axis.

        Args:
            filter_z (bool, optional): If True, filters the loaded point cloud to include only points with Z values
                greater than or equal to self.z_min. Defaults to True.

        Notes:
            - Sets self.pcd to the loaded (and possibly filtered) point cloud.
            - Colors all points in the point cloud with self.point_color.

        """
        self.pcd = o3d.io.read_point_cloud(self.point_cloud_path) 
        if filter_z:
            self.pcd = self.filter_by_axis(self.pcd, z_min=self.z_min)
        self.pcd.paint_uniform_color(self.point_color)

    def filter_by_axis(self,
                        pcd,
                        x_min=None, x_max=None,
                        y_min=None, y_max=None,
                        z_min=None, z_max=None
                    ):
        """
        Filters a point cloud by specified axis-aligned bounding box limits.

        Parameters:
            pcd (o3d.geometry.PointCloud): The input point cloud to filter.
            x_min (float, optional): Minimum x-value to include. Points with x < x_min are excluded.
            x_max (float, optional): Maximum x-value to include. Points with x > x_max are excluded.
            y_min (float, optional): Minimum y-value to include. Points with y < y_min are excluded.
            y_max (float, optional): Maximum y-value to include. Points with y > y_max are excluded.
            z_min (float, optional): Minimum z-value to include. Points with z < z_min are excluded.
            z_max (float, optional): Maximum z-value to include. Points with z > z_max are excluded.

        Returns:
            o3d.geometry.PointCloud: A new point cloud containing only the points within the specified bounds.
        """
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
    
    def raycast_pts_to_mesh(self):
        """
        Computes the unsigned distances from each point in the point cloud to the surface of the mesh using raycasting.

        This method converts the legacy mesh to an Open3D tensor mesh, adds it to a RaycastingScene,
        and then calculates the unsigned distance from each point in the point cloud to the nearest point on the mesh surface.

        Returns:
            np.ndarray: A 1D array of unsigned distances from each point in the point cloud to the mesh surface.
        """
        # Convert legacy mesh to tensor mesh for raycasting
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)

        # Create a RaycastingScene and add the mesh to it
        self.scene = o3d.t.geometry.RaycastingScene()
        mesh_id = self.scene.add_triangles(t_mesh)

        # Prepare query points from point cloud
        self.pcd_points = np.asarray(self.pcd.points, dtype=np.float32)
        query_points = o3d.core.Tensor(self.pcd_points, dtype=o3d.core.Dtype.Float32)

        # Compute distances to mesh surface
        unsigned_distances = self.scene.compute_distance(query_points)  # Tensor of shape (N,)
        distances = unsigned_distances.numpy()
        return distances
    
    def get_all_pruned_branch_pts(self, distances):
        """
        Identifies and processes pruned branch points from a point cloud based on distance filtering and outlier removal.

        This method performs the following steps:
        1. Filters points in the point cloud whose distances exceed a specified threshold (`self.prune_point_dist_threshold`).
        2. Creates a new point cloud containing only the filtered (pruned) points and assigns them a uniform color (`self.point_color`).
        3. Removes isolated points from the pruned point cloud using Statistical Outlier Removal, with parameters `self.outlier_nb_neighbors` and `self.outlier_std_ratio`.
        4. Stores the resulting cleaned point cloud in `self.pruned_branch_pcd`.

        Args:
            distances (np.ndarray): Array of distances corresponding to each point in the point cloud.

        Returns:
            None
        """
        # --- Distance-based filtering ---
        mask = distances > self.prune_point_dist_threshold

        # Create pruned point cloud
        pruned_branches = o3d.geometry.PointCloud()
        pruned_branches.points = o3d.utility.Vector3dVector(self.pcd_points[mask])
        pruned_branches.paint_uniform_color(self.point_color)

        # Remove isolated points using Statistical Outlier Removal
        self.pruned_branch_pcd, ind = pruned_branches.remove_statistical_outlier(
            nb_neighbors=self.outlier_nb_neighbors,
            std_ratio=self.outlier_std_ratio 
        )

    def refine_clusters_within(self, cluster, eps_list=None, min_points_list=None):
        """
        Attempts to find sub-clusters within a given cluster by varying DBSCAN parameters.
        Returns a list of sub-cluster point clouds (could be empty if no further clusters found).
        """
        if eps_list is None:
            eps_list = [self.dbscan_eps * 0.5, self.dbscan_eps * 0.75]
        if min_points_list is None:
            min_points_list = [max(3, self.dbscan_min_points // 2), max(2, self.dbscan_min_points // 4)]

        subclusters = []
        points = np.asarray(cluster.points)
        if len(points) == 0:
            return subclusters

        for eps in eps_list:
            for min_pts in min_points_list:
                labels = np.array(cluster.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False))
                max_label = labels.max()
                # Only consider if more than one sub-cluster is found
                if max_label > 0:
                    for i in range(max_label + 1):
                        mask = labels == i
                        if np.sum(mask) == 0:
                            continue
                        subcluster = cluster.select_by_index(np.where(mask)[0])
                        subclusters.append(subcluster)
                    return subclusters  # Return on first successful split
        return subclusters  # Empty if no further clusters found

    def isolate_individual_pruned_branches(self, save_branches=False):
        """
        Identifies and isolates individual pruned branches from a point cloud using DBSCAN clustering.
        This method clusters the points in `self.pruned_branch_pcd` using the DBSCAN algorithm with parameters
        `self.dbscan_eps` and `self.dbscan_min_points`. For each detected cluster, it computes the centroid and
        measures the distance from the centroid to a reference mesh (`self.scene`). Clusters whose centroids are
        within `self.branch_cluster_distance_threshold` of the mesh are considered valid pruned branches.
        Optionally, each valid branch can be saved as a separate point cloud file in a uniquely named directory.
        Args:
            save_branches (bool, optional): If True, saves each isolated branch as a separate PLY file in a
                directory prefixed by `self.save_directory_prefix`. Defaults to False.
        Returns:
            list: A list of Open3D point cloud objects, each representing an isolated pruned branch.
        """
        # Perform DBSCAN clustering
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
            labels = np.array(self.pruned_branch_pcd.cluster_dbscan(eps=self.dbscan_eps, min_points=self.dbscan_min_points, print_progress=True))

        max_label = labels.max()
        print(f"Found {max_label + 1} clusters.")

        max_label = labels.max()

        cluster_clouds = []
        branch_counter = 0
        for i in range(max_label + 1):
            mask = labels == i
            cluster = self.pruned_branch_pcd.select_by_index(np.where(mask)[0])

            # Compute cluster centroid
            cluster_points = np.asarray(cluster.points, dtype=np.float32)
            centroid = np.mean(cluster_points, axis=0, keepdims=True)
            centroid_tensor = o3d.core.Tensor(centroid, dtype=o3d.core.Dtype.Float32)

            # Distance from centroid to mesh
            centroid_distance = self.scene.compute_distance(centroid_tensor).item()

            if centroid_distance <= self.branch_cluster_distance_threshold:
                cluster.paint_uniform_color(np.random.rand(3)) # Random color
                cluster_clouds.append(cluster)
            
                if save_branches:
                    dir_path = f"{self.save_directory_prefix}_{branch_counter}"
                    os.makedirs(dir_path, exist_ok=True)
                    o3d.io.write_point_cloud(f"{dir_path}/branch.ply", cluster)
                    branch_counter += 1

        print(f"Filtered down to {len(cluster_clouds)} clusters.")
        return cluster_clouds
    
    def visualize_branches(self, branch_point_clouds):
        """
        Visualizes the pruned branches as point clouds along with the mesh using Open3D.

        Args:
            branch_point_clouds (list of o3d.geometry.PointCloud): 
                A list of Open3D PointCloud objects representing the pruned branches to be visualized.

        Displays:
            An interactive Open3D window showing the mesh and the provided branch point clouds.
        """
        # Visualize pruned points and mesh
        o3d.visualization.draw_geometries(
            [self.mesh] + branch_point_clouds,
            window_name="Pruned Branches and Mesh"
        )

    def main(self, vis=False, save_branches=False):
        """
        Main workflow for identifying and processing pruned branches.

        Args:
            vis (bool, optional): If True, visualize the isolated pruned branches. Defaults to False.
            save_branches (bool, optional): If True, save the isolated pruned branch point clouds. Defaults to False.

        Steps:
            1. Computes distances from raycast points to the mesh.
            2. Identifies all pruned branch points based on computed distances.
            3. Isolates individual pruned branches and optionally saves them.
            4. Optionally visualizes the isolated pruned branches.
        """
        distances = self.raycast_pts_to_mesh()
        self.get_all_pruned_branch_pts(distances)
        branch_point_clouds = self.isolate_individual_pruned_branches(save_branches=save_branches)
        if vis:
            self.visualize_branches(branch_point_clouds)


if __name__ == "__main__":
    # 1) Specify input file paths
    mesh_file = "results/tests/after_mesh.ply"
    pcd_file = "data/labeled_pt_clouds/preprune/2025-01-23-094626-ply-3dgs-ArtificialObjectRemoval.ply"


    # 2) Instantiate the pruner with desired thresholds
    pruner = PrunedBranchIdentifier(
        mesh_path=mesh_file,
        point_cloud_path=pcd_file,
        z_min=0.03,
        prune_point_dist_threshold=0.04,
        outlier_nb_neighbors=25,
        outlier_std_ratio=2.0,
        dbscan_eps=0.04,
        dbscan_min_points=20,
        save_directory_prefix="results/pruned_branches/branch"
    )
    
    pruner.main(vis=True, save_branches=True)