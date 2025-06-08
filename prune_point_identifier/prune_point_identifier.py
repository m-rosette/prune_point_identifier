import open3d as o3d
import numpy as np
import yaml
from skimage.morphology import skeletonize_3d
import networkx as nx
from scipy.interpolate import splprep, splev
import os
from tqdm import tqdm
import copy
import sys
from skan import csr
from pc_skeletor import LBC
setattr(np, "bool", bool)

# NOTE: This code modified the source code of pc_skeletor.laplacian.py by creating a class variable for self.n_neighbors and commenting out print statements.


class PrunePointIdentifier:
    """
    PrunePointIdentifier
    A class for identifying optimal prune points on 3D branch point clouds using skeletonization, spline fitting, and raycasting.
    This class provides a pipeline for processing 3D point clouds of branches, extracting their skeletons, fitting splines to the main axis, and determining the best prune point by raycasting against a mesh. It supports both voxel-based skeletonization and the Skeletor (LBC) method, and can process single or multiple branches in batch mode.
    Key Features:
    - Loads and preprocesses point clouds from .ply files.
    - Converts point clouds to voxel occupancy grids with adaptive voxel sizing.
    - Extracts skeletons using either 3D skeletonization or the LBC algorithm.
    - Converts skeletons to graphs and finds the longest path (main axis).
    - Fits parametric splines to the main axis and extracts endpoints and directions.
    - Evaluates endpoint density to select the most plausible branch base.
    - Performs raycasting from endpoints and within cones to find the optimal prune point on a mesh.
    - Saves results (base point, direction, prune point) to YAML files.
    - Provides visualization utilities for splines, rays, and prune points.
    - Supports batch processing of multiple branch point clouds.
        branch_num (int): Branch identifier number.
        mesh_path (str, optional): Path to the mesh file for raycasting. Defaults to "results/after_mesh.ply".
        parent_pcd_dir (str, optional): Directory containing branch point cloud subdirectories. Defaults to 'results/pruned_branches'.
        pcd_dir (str, optional): Directory containing the point cloud for the current branch. Defaults to a subdirectory of parent_pcd_dir.
        voxel_size (float, optional): Initial voxel size for downsampling and voxel grid creation. Defaults to 0.007.
        smoothness (float, optional): Spline smoothing parameter. Defaults to 0.03.
        cone_angle_deg (float, optional): Cone angle in degrees for raycasting. Defaults to 15.0.
        n_rings (int, optional): Number of concentric rings of rays in the cone. Defaults to 10.
        n_per_ring (int, optional): Number of rays per ring in the cone. Defaults to 20.
        cone_ray_length (float, optional): Length of rays cast within the cone. Defaults to 2.0.
    Attributes:
        pcd (o3d.geometry.PointCloud): The loaded and processed point cloud.
        occupancy_grid (np.ndarray): The voxel occupancy grid.
        skeleton (np.ndarray): The 3D skeleton array.
        G (networkx.Graph): The skeleton graph.
        longest_path (np.ndarray): The coordinates of the longest path in the skeleton.
        spine_points (np.ndarray): The points along the fitted spline.
        spline_derivs (np.ndarray): The derivatives (directions) along the spline.
        pruned_branch_base_point (np.ndarray): The selected base point for pruning.
        pruned_branch_base_direction (np.ndarray): The direction vector at the base point.
        frames (list): List of endpoint frames (origin and direction).
        found_prune_point (bool): Whether a valid prune point was found.
        best_hit_point (np.ndarray): The coordinates of the best prune point found by raycasting.
    """
    def __init__(self, 
                 branch_num, 
                 mesh_path=None,
                 parent_pcd_dir=None, 
                 pcd_dir=None, 
                 voxel_size=0.007, 
                 smoothness=0.03,
                 cone_angle_deg: float = 15.0,
                 n_rings: int = 10,
                 n_per_ring: int = 20,
                 cone_ray_length: float = 2.0):
        self.branch_num = branch_num
        self.voxel_size = voxel_size
        self.smoothness = smoothness
        self.before_pcd_path = 'data/labeled_pt_clouds/preprune/2025-01-23-094626-ply-3dgs-ArtificialObjectRemoval.ply'
        self.mesh_path = mesh_path or "results/after_mesh.ply"
        self.parent_pcd_dir = parent_pcd_dir or 'results/pruned_branches'
        self.pcd_dir = pcd_dir or f'{self.parent_pcd_dir}/branch_{branch_num}/'
        self.pcd = None
        self.occupancy_grid = None
        self.min_coords = None
        self.skeleton = None
        self.G = None
        self.longest_path = None
        self.spine_points = None
        self.spline_points = None
        self.spline_derivs = None
        self.pruned_branch_base_point = np.array([])
        self.pruned_branch_base_direction = np.array([])
        self.frames = None
        self.cone_angle_deg = cone_angle_deg
        self.n_rings = n_rings
        self.n_per_ring = n_per_ring
        self.cone_ray_length = cone_ray_length

        # Pre‐define colors
        self.color_best = [1.0, 0.0, 0.0]         # red for single best ray / highlight
        self.color_hit = [1.0, 0.0, 1.0]          # magenta for hits in best cone
        self.color_cone_pos = [0.0, 1.0, 0.0]     # green for +principal cone
        self.color_cone_neg = [1.0, 1.0, 0.0]     # yellow for –principal cone

        # Load mesh and build raycasting scene
        self.mesh, self.scene = self._load_mesh_and_scene(self.mesh_path)

        self.before_pcd = o3d.io.read_point_cloud(self.before_pcd_path)
        self.before_pcd = self.before_pcd.voxel_down_sample(self.voxel_size)
        self.before_pcd, _ = self.before_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Placeholders for raycast prune point hit results
        self.found_prune_point = False
        self.best_t = np.inf
        self.best_origin = None
        self.best_dir = None
        self.best_frame_idx = None
        self.best_cone_sign = None
        self.best_hit_point = np.array([None, None, None])

    def load_point_cloud(self, pcd_dir=None):
        """
        Loads a point cloud from a specified directory containing .ply files, applies voxel downsampling,
        and removes statistical outliers.
        Args:
            pcd_dir (str, optional): Path to the directory containing .ply point cloud files. If not provided,
                uses the instance's current `pcd_dir` attribute.
        Raises:
            FileNotFoundError: If the specified directory does not exist or contains no .ply files.
        Side Effects:
            - Sets `self.pcd_dir` to the provided directory if given.
            - Loads the first .ply file found in the directory into `self.pcd`.
            - Applies voxel downsampling using `self.voxel_size`.
            - Removes statistical outliers from the point cloud.
            - Stores a deep copy of the processed point cloud in `self.original_pcd`.
        """
        if pcd_dir is not None:
            self.pcd_dir = pcd_dir

        if not os.path.exists(self.pcd_dir):
            raise FileNotFoundError(f"No .ply file found in directory {self.pcd_dir}")
        
        ply_files = [f for f in os.listdir(self.pcd_dir) if f.endswith('.ply')]
        if not ply_files:
            raise FileNotFoundError(f"No .ply file found in directory {self.pcd_dir}")
        pcd_filename = ply_files[0]

        self.pcd = o3d.io.read_point_cloud(self.pcd_dir + pcd_filename)
        self.pcd = self.pcd.voxel_down_sample(self.voxel_size)
        self.pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        self.original_pcd = copy.deepcopy(self.pcd)

    def point_cloud_to_voxel_grid(self, min_voxel_size=0.001, min_occupancy=5):
        """
        Converts the point cloud (self.pcd) into a voxel occupancy grid using Open3D.
        This method attempts to create a voxel grid at the current self.voxel_size. If the number of occupied voxels is less than `min_occupancy`, 
        it iteratively decreases the voxel size by 0.001 until either the minimum occupancy is reached or the voxel size drops below `min_voxel_size`. 
        If the minimum voxel size is reached without sufficient occupancy, a RuntimeError is raised.
        The resulting occupancy grid is stored in `self.occupancy_grid`, and the minimum voxel coordinates are stored in `self.min_coords`.

        Args:
            min_voxel_size (float, optional): The minimum allowable voxel size. Defaults to 0.001.
            min_occupancy (int, optional): The minimum number of occupied voxels required. Defaults to 5.

        Raises:
            RuntimeError: If the minimum occupancy cannot be achieved before reaching the minimum voxel size.
        """
        while True:
            # Create an Open3D VoxelGrid at the current voxel_size
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd, self.voxel_size)
            voxels = voxel_grid.get_voxels()
            coords = np.array([v.grid_index for v in voxels])

            # If no voxels at all, or fewer than 10, attempt to shrink voxel_size
            if coords.size == 0 or coords.shape[0] < min_occupancy:
                # If we’re already at or below the minimum, stop and raise an error
                if self.voxel_size <= min_voxel_size:
                    raise RuntimeError(
                        f"Cannot obtain ≥{min_occupancy} voxels: voxel_size reached {self.voxel_size:.4f} with only {coords.shape[0]} voxels."
                    )
                # Otherwise, shrink by 0.001 and retry
                self.voxel_size -= 0.001
                print(f'Occupancy grid too thin, decreasing voxel_size to {self.voxel_size}')
                # Continue to next iteration with the new voxel_size
                continue

            # At this point, coords.shape[0] ≥ 10 → success
            break

        # Build occupancy_grid from these coords
        min_coords = coords.min(axis=0)
        coords -= min_coords
        grid_size = coords.max(axis=0) + 1
        occupancy_grid = np.zeros(grid_size, dtype=bool)
        occupancy_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = True

        self.occupancy_grid = occupancy_grid
        self.min_coords = min_coords

    def jitter_upsample(self, pcd, factor=2, noise_scale=0.001):
        """
        Upsamples a point cloud by duplicating points with added Gaussian noise (jitter).

        Args:
            pcd (o3d.geometry.PointCloud): The input point cloud to upsample.
            factor (int, optional): The number of jittered copies to generate for each point. Default is 2.
            noise_scale (float, optional): Standard deviation of the Gaussian noise added to each point. Default is 0.001.

        Returns:
            o3d.geometry.PointCloud: The upsampled point cloud containing the original and jittered points.
        """
        points = np.asarray(pcd.points)
        upsampled_points = []

        for _ in range(factor):
            noise = np.random.normal(scale=noise_scale, size=points.shape)
            noisy_points = points + noise
            upsampled_points.append(noisy_points)

        all_points = np.vstack([points] + upsampled_points)
        upsampled_pcd = o3d.geometry.PointCloud()
        upsampled_pcd.points = o3d.utility.Vector3dVector(all_points)
        return upsampled_pcd
    
    def _lbc_extract_skeleton_and_topology(self):
        """
        Extracts a skeleton graph and its topology from the point cloud using the LBC algorithm.
        This method performs the following steps:
        1. Ensures the point cloud has at least 30 points by upsampling if necessary.
        2. Initializes the LBC skeletonization algorithm with the current point cloud.
        3. Attempts to extract the skeleton, decrementing the number of neighbors (`n_neighbors`) if extraction fails due to insufficient points.
        4. Attempts to extract the skeleton topology, decrementing the topology parameter (`graph_k_n`) if extraction fails.
        5. Sets `self.G` to the resulting skeleton graph and stores a deep copy of the raw skeleton mesh in `self.lbc_skeleton` for optional visualization.
        Raises:
            RuntimeError: If skeleton extraction repeatedly fails even after reducing `n_neighbors`.
            ValueError: If topology extraction repeatedly fails even after reducing `graph_k_n`.
        """
        if len(self.pcd.points) < 30:
            self.pcd = self.jitter_upsample(self.pcd, factor=2, noise_scale=0.001)

        lbc = LBC(point_cloud=self.pcd)

        while True:
            try:
                lbc.extract_skeleton()
                break
            except RuntimeError as e:
                # This catches “k+1 > number of points” inside the Laplacian.
                # Decrement by 2 (arbitrary step) and try again.
                lbc.n_neighbors = max(lbc.n_neighbors - 2, 1)
                # print(f"[pc_skeletor]  LBC.extract_skeleton() failed: {e} → lowering n_neighbors to {lbc.n_neighbors}")

        # Attempt to extract topology, decrementing graph_k_n on failure:
        while True:
            try:
                lbc.extract_topology()
                self.G = lbc.skeleton_graph
                break
            except ValueError as e:
                lbc.graph_k_n -= 1
                # print(f"[pc_skeletor]  LBC.extract_topology() failed: {e} → lowering graph_k_n to {lbc.graph_k_n}")
        
        # keep the raw skeleton mesh for visualization if needed:
        self.lbc_skeleton = copy.deepcopy(lbc.skeleton)

    def skeletonize(self):
        self.skeleton = skeletonize_3d(self.occupancy_grid)

    def skeleton_to_graph(self):
        """
        Converts the 3D skeleton array to a NetworkX graph.
        Uses skan.csr.skeleton_to_csgraph for efficient skeleton graph extraction.
        """
        # skan expects a binary skeleton array
        graph, coordinates = csr.skeleton_to_csgraph(self.skeleton)
        # Convert the scipy sparse graph to a NetworkX graph
        self.G = nx.Graph(graph)
        # Optionally, you can store coordinates for mapping node indices to 3D positions
        self._skan_coordinates = coordinates

    def longest_path_in_skeleton(self):
        """
        Finds the longest shortest path (i.e., the diameter) in the skeleton graph `self.G` and stores the corresponding 3D coordinates.

        This method computes all-pairs shortest path lengths in the skeleton graph, identifies the pair of nodes with the maximum shortest 
        path distance, and reconstructs the path between them. It then maps the node indices along this path to their 3D coordinates using 
        `self._skan_coordinates`, and stores the resulting array in `self.longest_path`.

        Side Effects:
            - Sets `self.longest_path` to a NumPy array of shape (N, 3), where N is the number of nodes in the longest path.

        Assumes:
            - `self.G` is a NetworkX graph representing the skeleton.
            - `self._skan_coordinates` contains the 3D coordinates of the skeleton nodes, either as an (N, 3) array or as a tuple/list of three 1D arrays of length N.

        Returns:
            None
        """
        lengths = dict(nx.all_pairs_shortest_path_length(self.G))
        max_dist = 0
        max_pair = None
        for u, dist_dict in lengths.items():
            for v, dist in dist_dict.items():
                if dist > max_dist:
                    max_dist = dist
                    max_pair = (u, v)

        # If max_pair is not None, generate a path, else self.longest_path remains None
        if max_pair:
            path = nx.shortest_path(self.G, source=max_pair[0], target=max_pair[1])
            # self.longest_path = np.array(path)

            # Map each node index to its 3D coordinate in the skeleton voxel grid
            coords = self._skan_coordinates
            # Convert coords into an (num_nodes×3) array, regardless of original format
            coords_arr = np.array(coords)
            if coords_arr.ndim == 2 and coords_arr.shape[1] == 3:
                # already (N, 3)
                final_coords = coords_arr
            else:
                # assume coords is a tuple or list of three 1D arrays of length N
                final_coords = np.stack(coords, axis=1)  # now (N, 3)
            self.longest_path = final_coords[path]  # (len(path), 3)

    def compute_spline_points(self):
        """
        Computes the spline (spine) points for the point cloud.

        This method calculates the coordinates of the spline points by scaling the indices of the longest path
        with the voxel size, offsetting by the minimum bound of the point cloud, and centering within each voxel.
        The resulting points are stored in the `self.spine_points` attribute.

        Returns:
            None
        """
        min_bound = self.pcd.get_min_bound()
        self.spine_points = self.longest_path * self.voxel_size + min_bound + self.voxel_size / 2.0

    def fit_spline_to_points(self):
        """
        Fits a parametric spline to the set of spine points and computes its derivatives.

        This method uses the scipy.interpolate.splprep function to fit a B-spline to the points
        stored in self.spine_points. The spline order is chosen as the minimum of 3 and the number
        of points minus one. The fitted spline and its first derivative are then evaluated on a
        fine grid and stored in self.spline_points and self.spline_derivs, respectively.

        Raises:
            RuntimeError: If there are fewer than 2 spine points to fit a spline.
        """
        points = np.array(self.spine_points)
        m = points.shape[0]

        if m < 2:
            raise RuntimeError(f"Not enough spine points ({m}) to fit any spline.")
        # Pick spline order ≤ m-1, but at most 3
        k = min(3, m - 1)

        # Now splprep will succeed as long as m > k
        tck, u = splprep(points.T, s=self.smoothness, k=k)

        # Evaluate on a fine grid
        u_fine = np.linspace(0, 1, 200)
        spline = splev(u_fine, tck)
        spline_deriv = splev(u_fine, tck, der=1)

        self.spline_points = np.vstack(spline).T
        self.spline_derivs = np.vstack(spline_deriv).T      

    def extract_spline_endpoints(self):
        """
        Extracts the endpoints and their corresponding direction vectors from the spline.

        Sets the following instance attributes:
            - start_point: The first point of the spline.
            - end_point: The last point of the spline.
            - start_direction: The derivative (direction vector) at the start of the spline.
            - end_direction: The derivative (direction vector) at the end of the spline.
            - frames: A list of dictionaries, each containing the 'origin' (point) and 'direction' (vector) 
              for the start and end of the spline.

        Assumes that `self.spline_points` and `self.spline_derivs` are sequences with at least one element.
        """
        self.start_point = self.spline_points[0]
        self.end_point = self.spline_points[-1]
        self.start_direction = self.spline_derivs[0]
        self.end_direction = self.spline_derivs[-1]
        self.frames = [
            {"origin": self.start_point, "direction": self.start_direction},
            {"origin": self.end_point,   "direction": self.end_direction}
        ]
    
    def get_skeletor_points(self):
        """
        Identifies the two most extreme branch tips along the principal axis of a graph skeleton and computes their tangent directions.
        This method performs the following steps:
            1. Finds all nodes in the graph `self.G` with degree 1 (branch tips).
            2. Collects the 3D coordinates of these tip nodes.
            3. Centers the tip coordinates to compute the principal axis using eigenvalue decomposition of the covariance matrix.
            4. Projects the tip points onto the principal axis and identifies the two tips at the extremes.
            5. Stores the coordinates of these two tips in `self.spline_points`.
            6. For each tip, computes the tangent direction by finding the vector from the tip to its single neighbor, normalizing it, 
               and storing it in `self.spline_derivs`.
        Attributes Set:
            self.spline_points (list of np.ndarray): The 3D coordinates of the two extreme branch tips.
            self.spline_derivs (list of np.ndarray): The normalized tangent vectors at each tip.
        Assumes:
            - Each node in `self.G` has a 'pos' attribute with its 3D coordinates.
            - The graph `self.G` is connected and contains at least two tips.
        Returns:
            None
        """

        #  1. Find all nodes whose degree == 1 (those are the branch‐tips)
        end_nodes = [n for n, deg in self.G.degree() if deg == 1]

        #  2. Collect their coordinates
        tip_points = []
        for n in end_nodes:
            xyz = np.array(self.G.nodes[n]['pos'])
            tip_points.append(xyz)
        tip_points = np.vstack(tip_points)  # shape = (num_tips, 3)

        #  3. Center the tip‐points (for finding principal axis endpoints)
        mean_xyz = tip_points.mean(axis=0)
        centered = tip_points - mean_xyz

        #  4. Covariance & eigenvalue‐decomposition to find principal axis
        cov = np.cov(centered, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        principal_axis = eig_vecs[:, np.argmax(eig_vals)]  # unit 3D vector

        #  5. Project each centered tip onto that axis
        projections = centered.dot(principal_axis)   # (num_tips,)

        #  6. Find the two extremes along that axis
        idx_min = np.argmin(projections)
        idx_max = np.argmax(projections)

        #  7. Their 3D coordinates in world‐frame:
        end1 = tip_points[idx_min].astype(float)
        end2 = tip_points[idx_max].astype(float)
        self.spline_points = [end1, end2]

        #  (A) For each of those two tips, compute the tangent direction:
        self.spline_derivs = []
        for tip_coord, tip_node in zip(self.spline_points, [end_nodes[idx_min], end_nodes[idx_max]]):
            # (A.1) Find that tip’s single neighbor:
            nbr = next(iter(self.G.neighbors(tip_node)))
            nbr_coord = np.array(self.G.nodes[nbr]['pos'])

            # (A.2) Tangent = neighbor_pos − tip_pos
            v = nbr_coord - tip_coord
            v = v / np.linalg.norm(v)   # local z‐axis
            self.spline_derivs.append(v)

    def check_endpoint_surrounding_density(self, radius=0.15):
        """
        Evaluates the density of points surrounding each endpoint (frame origin) within a specified radius,
        and retains only the frame with the highest surrounding point density.

        For each frame in self.frames:
            - Computes the number of points in self.before_pcd within `radius` of the frame's origin using a KD-Tree.
            - Stores the count for each frame.

        After processing all frames:
            - Updates self.frames to contain only the frame with the maximum surrounding point count.
            - Sets self.pruned_branch_base_point and self.pruned_branch_base_direction to the origin and direction
              of the retained frame.

        Args:
            radius (float, optional): The radius within which to count neighboring points around each frame's origin.
                Defaults to 0.15.

        Side Effects:
            Modifies self.frames, self.pruned_branch_base_point, and self.pruned_branch_base_direction.
        """
        frame_num_points = []
        for idx, frame in enumerate(self.frames):
            origin = np.array(frame["origin"], dtype=float)
            principal = np.array(frame["direction"], dtype=float)
            principal /= np.linalg.norm(principal)

            # Build (or reuse) a KD-TreeFlann on the point cloud
            pcd_tree = o3d.geometry.KDTreeFlann(self.before_pcd)

            # search_radius_vector_3d returns:
            #   k = number of points found (including the center itself, if it coincides),
            #   idx = list of indices of those points
            k, idx, _ = pcd_tree.search_radius_vector_3d(origin, radius)

            frame_num_points.append(k)
        self.frames = [self.frames[np.argmax(frame_num_points)]]
        self.pruned_branch_base_point = self.frames[0]['origin']
        self.pruned_branch_base_direction = self.frames[0]['direction']

    def _generate_cone_directions(self, principal_axis: np.ndarray) -> np.ndarray:
        """
        Generates a set of unit-length direction vectors distributed within a cone centered around the given `principal_axis`.

        The cone is defined by `self.cone_angle_deg` (in degrees), and the directions are distributed in concentric rings (`self.n_rings`)
        with a specified number of directions per ring (`self.n_per_ring`). The vectors are returned in world coordinates, using a local
        orthonormal basis aligned with the `principal_axis`.

        Args:
            principal_axis (np.ndarray): The central axis of the cone (3D vector).

        Returns:
            np.ndarray: An array of shape (self.n_rings * self.n_per_ring, 3) containing unit direction vectors within the cone.
        """
        axis = principal_axis / np.linalg.norm(principal_axis)
        basis = self._build_basis(axis)

        cone_angle_rad = np.radians(self.cone_angle_deg)
        dirs = []

        for j in range(self.n_rings):
            θ = (j / max(self.n_rings - 1, 1)) * cone_angle_rad
            sin_θ = np.sin(θ)
            cos_θ = np.cos(θ)

            for k in range(self.n_per_ring):
                φ = 2.0 * np.pi * (k / self.n_per_ring)
                x_local = sin_θ * np.cos(φ)
                y_local = sin_θ * np.sin(φ)
                z_local = cos_θ
                dir_local = np.array([x_local, y_local, z_local], dtype=float)
                dir_world = basis @ dir_local
                dirs.append(dir_world / np.linalg.norm(dir_world))

        return np.stack(dirs, axis=0)
    
    def save_to_yaml(self, filename="branch_info.yaml"):
        """
        Saves branch and prune point information to a YAML file.

        Parameters:
            filename (str): Name of the YAML file to save the data to. Defaults to "branch_info.yaml".

        The saved YAML file contains:
            - pruned_branch_base_point: The base point of the pruned branch as a list.
            - pruned_branch_base_direction: The direction vector of the pruned branch base as a list.
            - prune_point: The coordinates of the best hit (prune) point as a list.

        Raises:
            AttributeError: If required attributes are missing from the object.
            IOError: If the file cannot be written.
        """
        data = {
            'pruned_branch_base_point': self.pruned_branch_base_point.tolist(),
            'pruned_branch_base_direction': self.pruned_branch_base_direction.tolist(),
            'prune_point': self.best_hit_point.tolist()
        }
        with open(self.pcd_dir + filename, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def _scan_for_best_hit(self) -> None:
        """
        Scans through all frames to find the best (closest) intersection point ("hit") by casting rays.

        For each frame, the method:
          1. Casts rays along the principal direction and its opposite.
          2. Casts multiple rays within a cone around the principal direction.
          3. Casts multiple rays within a cone around the opposite of the principal direction.

        For each ray, it checks for intersections with the scene and records the hit with the smallest distance (t value).
        If a valid hit is found, the method updates instance attributes with the details of the best hit, including:
          - Whether a prune point was found (`found_prune_point`)
          - The distance to the hit (`best_t`)
          - The origin and direction of the best ray (`best_origin`, `best_dir`)
          - The index of the frame and the cone sign (`best_frame_idx`, `best_cone_sign`)
          - The 3D coordinates of the hit point (`best_hit_point`)
        If no valid hit is found, sets `found_prune_point` to False.
        """
        best_t = np.inf
        best_origin = None
        best_dir = None
        best_frame_idx = None
        best_cone_sign = None

        for idx, frame in enumerate(self.frames):
            origin = np.array(frame["origin"], dtype=float)
            principal = np.array(frame["direction"], dtype=float)
            principal /= np.linalg.norm(principal)

            # 1) Principal ± rays
            for sign in (+1, -1):
                dir_vec = sign * principal
                ray = o3d.core.Tensor([[*origin, *dir_vec]], dtype=o3d.core.Dtype.Float32)
                ans = self.scene.cast_rays(ray)
                t_hit = float(ans["t_hit"][0].item())
                if np.isfinite(t_hit) and (t_hit < best_t):
                    best_t = t_hit
                    best_origin = origin.copy()
                    best_dir = dir_vec.copy()
                    best_frame_idx = idx
                    best_cone_sign = None

            # 2) Cone interior around +principal
            dirs_pos = self._generate_cone_directions(principal)
            origins_pos = np.tile(origin.reshape(1, 3), (dirs_pos.shape[0], 1))
            rays_np_pos = np.hstack([origins_pos, dirs_pos])
            rays_pos = o3d.core.Tensor(rays_np_pos, dtype=o3d.core.Dtype.Float32)
            ans_pos = self.scene.cast_rays(rays_pos)
            t_hits_pos = ans_pos["t_hit"].numpy()

            finite_mask_pos = np.isfinite(t_hits_pos)
            if np.any(finite_mask_pos):
                min_t_pos = np.min(t_hits_pos[finite_mask_pos])
                i_min = int(np.argmin(np.where(finite_mask_pos, t_hits_pos, np.inf)))
                if min_t_pos < best_t:
                    best_t = float(min_t_pos)
                    best_origin = origin.copy()
                    best_dir = dirs_pos[i_min].copy()
                    best_frame_idx = idx
                    best_cone_sign = +1

            # 3) Cone interior around -principal
            dirs_neg = self._generate_cone_directions(-principal)
            origins_neg = np.tile(origin.reshape(1, 3), (dirs_neg.shape[0], 1))
            rays_np_neg = np.hstack([origins_neg, dirs_neg])
            rays_neg = o3d.core.Tensor(rays_np_neg, dtype=o3d.core.Dtype.Float32)
            ans_neg = self.scene.cast_rays(rays_neg)
            t_hits_neg = ans_neg["t_hit"].numpy()

            finite_mask_neg = np.isfinite(t_hits_neg)
            if np.any(finite_mask_neg):
                min_t_neg = np.min(t_hits_neg[finite_mask_neg])
                i_min = int(np.argmin(np.where(finite_mask_neg, t_hits_neg, np.inf)))
                if min_t_neg < best_t:
                    best_t = float(min_t_neg)
                    best_origin = origin.copy()
                    best_dir = dirs_neg[i_min].copy()
                    best_frame_idx = idx
                    best_cone_sign = -1

        if best_origin is None or best_dir is None or not np.isfinite(best_t):
            self.found_prune_point = False
        else:
            # Store results in instance attributes
            self.found_prune_point = True
            self.best_t = best_t
            self.best_origin = best_origin
            self.best_dir = best_dir
            self.best_frame_idx = best_frame_idx
            self.best_cone_sign = best_cone_sign
            self.best_hit_point = self.best_origin + self.best_dir * self.best_t

    @staticmethod
    def _create_line_set(points, color=[1, 0, 0]):
        points = points.astype(float)
        lines = [[i, i + 1] for i in range(len(points) - 1)]
        colors = [color for _ in lines]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    @staticmethod
    def create_axis_frame(origin, direction, length=0.05):
        direction = direction / np.linalg.norm(direction)
        z_axis = direction
        up = np.array([0, 0, 1])
        if np.allclose(z_axis, up):
            up = np.array([1, 0, 0])  # Avoid singularity
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R = np.vstack([x_axis, y_axis, z_axis]).T
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=length)
        frame.rotate(R, center=(0, 0, 0))
        frame.translate(origin)
        return frame
    
    @staticmethod
    def get_child_directory_paths(parent_dir_path):
        result = []
        for name in os.listdir(parent_dir_path):
            full_path = os.path.join(parent_dir_path, name)
            if os.path.isdir(full_path):
                # Extract int after last underscore
                try:
                    num = int(name.split('_')[-1])
                except (ValueError, IndexError):
                    num = None
                result.append((full_path+'/', num))
        # Sort by num, ignoring entries where num is None (they go last)
        result.sort(key=lambda x: (x[1] is None, x[1]))
        return result
    
    @staticmethod
    def _build_basis(z_axis: np.ndarray) -> np.ndarray:
        """
        Builds an orthonormal basis [x, y, z = z_axis].
        """
        z = z_axis / np.linalg.norm(z_axis)
        tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.99 else np.array([0.0, 1.0, 0.0])
        x = np.cross(tmp, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return np.stack([x, y, z], axis=1)
    
    @staticmethod
    def _load_mesh_and_scene(mesh_path: str) -> tuple[o3d.geometry.TriangleMesh, o3d.t.geometry.RaycastingScene]:
        """
        Loads a mesh, computes normals, converts to t‐geometry, and builds a raycasting scene.
        """
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()

        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(tmesh)
        return mesh, scene
    
    @staticmethod
    def _create_colored_sphere(center: np.ndarray, radius: float, color: list[float]) -> o3d.geometry.TriangleMesh:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(center)
        sphere.paint_uniform_color(color)
        return sphere
    
    def _visualize_single_best_hit(self, show_branch_pt_cloud: bool) -> None:
        """
        Visualizes just the single best ray and hit point.
        """
        geoms = [self.mesh]

        line = self._create_line_set(np.array([self.best_origin, self.best_hit_point]), self.color_best)
        sphere = self._create_colored_sphere(self.best_hit_point, radius=0.005, color=self.color_best)
        geoms.extend([line, sphere])

        if show_branch_pt_cloud:
            geoms.append(self.original_pcd)

        o3d.visualization.draw_geometries(geoms)

    def _visualize_best_cone(self, show_branch_pt_cloud: bool) -> None:
        """
        Visualizes all rays in the winning cone, coloring each hit point as magenta spheres,
        misses as lines, and highlighting the single best hit.
        """
        frame = self.frames[self.best_frame_idx]
        origin = np.array(frame["origin"], dtype=float)
        principal = np.array(frame["direction"], dtype=float)
        principal /= np.linalg.norm(principal)

        # Determine axis and color for the best cone
        if self.best_cone_sign == +1:
            axis = principal
            cone_color = self.color_cone_pos
        elif self.best_cone_sign == -1:
            axis = -principal
            cone_color = self.color_cone_neg
        else:
            # Best hit was a principal ray; fall back to single‐ray visualization
            self._visualize_single_best_hit(show_branch_pt_cloud)
            return

        # Generate all directions in that cone
        dirs = self._generate_cone_directions(axis)
        origins = np.tile(origin.reshape(1, 3), (dirs.shape[0], 1))
        rays_np = np.hstack([origins, dirs])
        rays = o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32)
        ans = self.scene.cast_rays(rays)
        t_hits = ans["t_hit"].numpy()

        geoms = [self.mesh]

        for i in range(dirs.shape[0]):
            dir_vec = dirs[i]
            t_i = t_hits[i]

            if np.isfinite(t_i):
                hit_pt = origin + dir_vec * t_i
                sphere = self._create_colored_sphere(hit_pt, radius=0.005, color=self.color_hit)
                geoms.append(sphere)
                end_pt = hit_pt
            else:
                end_pt = origin + dir_vec * self.cone_ray_length

            line = self._create_line_set(np.array([origin, end_pt]), cone_color)
            geoms.append(line)

        # Highlight the single best hit
        sphere_best = self._create_colored_sphere(self.best_hit_point, radius=0.01, color=self.color_best)
        geoms.append(sphere_best)

        # if show_branch_pt_cloud:
        geoms.append(self.original_pcd)

        o3d.visualization.draw_geometries(geoms)

    def visualize_spline(self, skeletor=True):
        vis_objects = []
        if skeletor:
            vis_objects.append(self.lbc_skeleton)
        else:
            spline_pcd = o3d.geometry.PointCloud()
            spline_pcd.points = o3d.utility.Vector3dVector(self.spine_points)
            spline_pcd.paint_uniform_color([0, 0, 1])
            spline_line = self._create_line_set(np.array(self.spline_points), color=[1, 0, 0])
            vis_objects.append(spline_pcd)
            vis_objects.append(spline_line)

        start_axis = self.create_axis_frame(self.start_point, self.start_direction)
        end_axis = self.create_axis_frame(self.end_point, self.end_direction)

        vis_objects.append(self.pcd)
        vis_objects.append(start_axis)
        vis_objects.append(end_axis)

        o3d.visualization.draw_geometries(
            vis_objects,
            window_name="Branch Skeleton & Spline",
            width=1000,
            height=800,
            point_show_normal=False
        )

    def run_single_evaluation(self, visualize_spline=True, visualize_rays=False, visualize_prune_point=False, save_yaml=True):
        """
        Runs a single evaluation pipeline for prune point identification on a point cloud.

        This method performs the following steps:
            1. Loads the point cloud data.
            2. Converts the point cloud to a voxel grid.
            3. Skeletonizes the voxel grid.
            4. Converts the skeleton to a graph representation.
            5. Finds the longest path in the skeleton.
            6. If a longest path is found:
                - Computes spline points along the path.
                - Fits a spline to the computed points.
                - Extracts the endpoints of the spline.
                - Checks the density surrounding the endpoints.
                - Scans for the best prune point candidate.
            7. Optionally saves the results to a YAML file.
            8. Optionally visualizes the spline, rays, and/or prune point.

        Args:
            visualize_spline (bool, optional): If True, visualize the fitted spline. Defaults to True.
            visualize_rays (bool, optional): If True, visualize the best cone rays. Defaults to False.
            visualize_prune_point (bool, optional): If True, visualize the identified prune point. Defaults to False.
            save_yaml (bool, optional): If True, save the results to a YAML file. Defaults to True.

        Returns:
            None
        """
        self.load_point_cloud()
        self.point_cloud_to_voxel_grid()
        self.skeletonize()
        self.skeleton_to_graph()
        self.longest_path_in_skeleton()
        if self.longest_path is not None:
            self.compute_spline_points()
            self.fit_spline_to_points()
            self.extract_spline_endpoints()
            self.check_endpoint_surrounding_density()
            self._scan_for_best_hit()
        if save_yaml:
            self.save_to_yaml()
        if visualize_spline:
            self.visualize_spline(skeletor=False)
        if visualize_rays:
            self._visualize_best_cone(visualize_prune_point)
        if visualize_prune_point:
            self._visualize_single_best_hit(visualize_prune_point)
    
    def run_all_evaluations(self, visualize_spline=False, visualize_rays=False, visualize_prune_point=False, save_yaml=True):
        """
        Runs the full evaluation pipeline for all branch point cloud directories.

        This method processes each branch directory by loading its point cloud, converting it to a voxel grid,
        skeletonizing it, converting the skeleton to a graph, and finding the longest path in the skeleton.
        If a valid longest path is found, it computes and fits a spline, extracts endpoints, checks endpoint
        density, and scans for the best prune point. Optionally, results can be saved to a YAML file and
        various visualizations can be displayed.

        Args:
            visualize_spline (bool, optional): If True, visualize the fitted spline. Defaults to False.
            visualize_rays (bool, optional): If True, visualize the best cone rays. Defaults to False.
            visualize_prune_point (bool, optional): If True, visualize the best prune point. Defaults to False.
            save_yaml (bool, optional): If True, save results to a YAML file. Defaults to True.
        """
        child_dir_info = self.get_child_directory_paths(self.parent_pcd_dir)

        for dir_info in tqdm(child_dir_info, 
                         total=len(child_dir_info),
                         desc="Branches",
                         unit="branch",
                         file=sys.stdout,          # <- ensures output to terminal
                         leave=True,               # <- retains bar at end
                         dynamic_ncols=True):      # <- adjusts bar width to terminal
            dir_path, branch_id = dir_info
            self.branch_num = branch_id

            self.load_point_cloud(dir_path)
            self.point_cloud_to_voxel_grid()
            self.skeletonize()
            self.skeleton_to_graph()
            self.longest_path_in_skeleton()
            if self.longest_path is not None:
                self.compute_spline_points()
                self.fit_spline_to_points()
                self.extract_spline_endpoints()
                self.check_endpoint_surrounding_density()
                self._scan_for_best_hit()
            if save_yaml:
                self.save_to_yaml()
            if visualize_spline:
                self.visualize_spline(skeletor=False)
            if visualize_rays:
                self._visualize_best_cone(visualize_prune_point)
            if visualize_prune_point:
                self._visualize_single_best_hit(visualize_prune_point)

    def run_single_evalutation_pc_skeletor(self, visualize_spline=False, visualize_rays=False, visualize_prune_point=False, save_yaml=True):
        """
        Runs a single evaluation of the prune point identification pipeline using the Skeletor method.

        This method performs the following steps:
            1. Loads the point cloud data.
            2. Extracts the skeleton and topology using the LBC method.
            3. Retrieves Skeletor points.
            4. Extracts spline endpoints from the skeleton.
            5. Checks the density surrounding each endpoint.
            6. Scans for the best prune point candidate.

        Optionally, the method can save the results to a YAML file and visualize different stages of the process.

        Args:
            visualize_spline (bool, optional): If True, visualize the extracted spline. Defaults to False.
            visualize_rays (bool, optional): If True, visualize the rays and best cone if a prune point is found. Defaults to False.
            visualize_prune_point (bool, optional): If True, visualize the identified prune point if found. Defaults to False.
            save_yaml (bool, optional): If True, save the results to a YAML file. Defaults to True.

        Returns:
            None
        """
        self.load_point_cloud()
        self._lbc_extract_skeleton_and_topology()
        self.get_skeletor_points()
        self.extract_spline_endpoints()
        self.check_endpoint_surrounding_density()
        self._scan_for_best_hit()

        if save_yaml:
            self.save_to_yaml()
        if visualize_spline:
            self.visualize_spline(skeletor=True)
        if visualize_rays and self.found_prune_point:
            self._visualize_best_cone(visualize_prune_point)
        if visualize_prune_point and self.found_prune_point:
            self._visualize_single_best_hit(visualize_prune_point)

    def run_all_evaluations_pc_skeletor(self, save_yaml=True):
        """
        Runs the full evaluation pipeline for all branch point cloud directories, extracting skeletons,
        topology, and prune points, and aggregates the results.
        For each branch directory found under `self.parent_pcd_dir`, this method:
            - Loads the point cloud data.
            - Extracts skeleton and topology information.
            - Identifies key skeletor points.
            - Extracts spline endpoints.
            - Checks the density around endpoints.
            - Scans for the best prune point.
        Aggregates the results for all branches into a dictionary, and optionally saves the data as a YAML file.
        Args:
            save_yaml (bool, optional): If True, saves the aggregated branch data to 'all_branches_info.yaml'
                in the parent point cloud directory. Defaults to True.
        Side Effects:
            - Writes a YAML file with aggregated branch information if `save_yaml` is True.
            - Prints a message indicating the save location and number of branches processed.
        Returns:
            None
        """
        child_dir_info = self.get_child_directory_paths(self.parent_pcd_dir)

        all_branches_data = {}

        for dir_info in tqdm(child_dir_info, 
                         total=len(child_dir_info),
                         desc="Branches",
                         unit="branch",
                         file=sys.stdout,          # <- ensures output to terminal
                         leave=True,               # <- retains bar at end
                         dynamic_ncols=True):      # <- adjusts bar width to terminal
            dir_path, branch_id = dir_info
            self.branch_num = branch_id

            self.load_point_cloud(dir_path)
            self._lbc_extract_skeleton_and_topology()
            self.get_skeletor_points()
            self.extract_spline_endpoints()
            self.check_endpoint_surrounding_density()
            self._scan_for_best_hit()
        
            branch_dict = {
                'pruned_branch_base_point':        self.pruned_branch_base_point.tolist(),
                'pruned_branch_base_direction':    self.pruned_branch_base_direction.tolist(),
                'prune_point':                     self.best_hit_point.tolist()
            }

            all_branches_data[branch_id] = branch_dict

        if save_yaml:
            output_path = os.path.join(self.parent_pcd_dir, 'all_branches_info.yaml')
            with open(output_path, 'w') as f:
                yaml.dump(all_branches_data, f, default_flow_style=False)

            print(f"Saved aggregated YAML for {len(all_branches_data)} branches to:\n    {output_path}")


if __name__ == "__main__":
    branch_fit = PrunePointIdentifier(
        branch_num=33,
        voxel_size=0.008,
        smoothness=0.03,
        cone_angle_deg=30,
        n_rings=10,
        n_per_ring=20)

    # branch_fit.run_single_evaluation(visualize_prune_point=False, 
    #                                  visualize_rays=True, 
    #                                  visualize_spline=True, 
    #                                  save_yaml=False)
    # branch_fit.run_all_evaluations()

    # branch_fit.run_single_evalutation_pc_skeletor(visualize_prune_point=False, 
    #                                                 visualize_rays=True, 
    #                                                 visualize_spline=True, 
    #                                                 save_yaml=False)
    branch_fit.run_all_evaluations_pc_skeletor()