import open3d as o3d
import numpy as np
import trimesh
from trimesh.smoothing import filter_laplacian
import copy


class PointCloudToMesh:
    """
    A class for processing, aligning, and meshing 3D point clouds using Open3D and Trimesh.
    This class provides methods for loading, filtering, transforming, registering, upsampling,
    meshing, filling, cleaning, smoothing, and visualizing point clouds and meshes. It is designed
    for workflows where two point clouds (e.g., "before" and "after" scans) are compared, aligned,
    and converted to meshes.
    Args:
        before_path (str): Path to the "before" point cloud file.
        after_path (str): Path to the "after" point cloud file.
        theta_deg (float, optional): Rotation angle in degrees to apply to the "after" point cloud. Default is 33.
        y_trans (float, optional): Translation along the Y-axis to apply to the "after" point cloud. Default is -0.035.
    Attributes:
        before_path (str): Path to the "before" point cloud.
        after_path (str): Path to the "after" point cloud.
        theta (float): Rotation angle in radians.
        y_trans (float): Y-axis translation value.
        before_pcd (o3d.geometry.PointCloud or None): Loaded "before" point cloud.
        after_pcd (o3d.geometry.PointCloud or None): Loaded "after" point cloud.
    """
    def __init__(self, before_path, after_path, theta_deg=33, y_trans=-0.035):
        self.before_path = before_path
        self.after_path = after_path
        self.theta = np.radians(theta_deg)
        self.y_trans = y_trans
        self.before_pcd = None
        self.after_pcd = None

    def load_point_clouds(self):
        self.before_pcd = o3d.io.read_point_cloud(self.before_path)
        self.after_pcd = o3d.io.read_point_cloud(self.after_path)
    
    def load_single_point_cloud(self, filename):
        return o3d.io.read_point_cloud(filename)
    
    def downsample_pcd(self, pcd, voxel_size=0.01):
        # Fast voxel‐grid filter
        down = pcd.voxel_down_sample(voxel_size)
        return down

    def transform_after_cloud(self):
        R = np.array([
            [np.cos(self.theta), -np.sin(self.theta), 0],
            [np.sin(self.theta),  np.cos(self.theta), 0],
            [0,                  0,                   1]
        ])
        self.after_pcd.rotate(R, center=(0, 0, 0))
        self.after_pcd.translate([0, self.y_trans, 0])

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

    def apply_icp_registration(self, pre_icp_axis_filter=[0.9, 13.2, -2.5, 2.5, 0.03, 0.8]):
        """
        Applies Iterative Closest Point (ICP) registration to align the 'after' point cloud to the 'before' point cloud.
        This method first filters both the 'before' and 'after' point clouds using the specified axis-aligned bounding box filter.
        It then performs a two-stage ICP registration (coarse and fine) to compute the transformation that best aligns the filtered
        'after' point cloud to the filtered 'before' point cloud. The resulting transformation is applied to the top-level 'after' point cloud.
        Args:
            pre_icp_axis_filter (list[float], optional): A list of six float values specifying the axis-aligned bounding box
                filter in the format [min_x, max_x, min_y, max_y, min_z, max_z]. Defaults to [0.9, 13.2, -2.5, 2.5, 0.03, 0.8].
        Returns:
            None
        """
        # Copy and filter the before and after pcds to just the trucks with some small offshoot branches
        before_pcd = copy.deepcopy(self.before_pcd)
        after_pcd = copy.deepcopy(self.after_pcd)
        before_pcd = self.filter_by_axis(before_pcd, *pre_icp_axis_filter)
        after_pcd = self.filter_by_axis(after_pcd, *pre_icp_axis_filter)

        # Get coarse and fine ICP registration for filtered pcds
        reg_coarse = o3d.pipelines.registration.registration_icp(
            after_pcd, before_pcd, 0.05, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        reg_fine = o3d.pipelines.registration.registration_icp(
            after_pcd, before_pcd, 0.01, reg_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        # Apply resultant transform to top-level after pcd
        self.after_pcd.transform(reg_fine.transformation)

    def recolor_point_clouds(self, pcds):
        colors = [[1.0, 0.0, 0.0], [0.7, 0.7, 0.7]]
        pcds_colored = []

        for i, pcd in enumerate(pcds):
            pts = np.asarray(pcd.points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(colors[i], (pts.shape[0], 1)))
            pcds_colored.append(pcd)
            
        return pcds_colored
        
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
    
    def compute_mesh(self, pcd, alpha=0.005):
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        return mesh

    def fill_with_trimesh(self, o3d_mesh):
        """
        Fills holes in an Open3D TriangleMesh using trimesh and returns the filled mesh.

        This function converts the input Open3D TriangleMesh to a trimesh.Trimesh object,
        fills all holes automatically, removes degenerate faces, and then converts the
        result back to an Open3D TriangleMesh. Vertex normals are recomputed before returning.

        Args:
            o3d_mesh (o3d.geometry.TriangleMesh): The input Open3D triangle mesh to be filled.

        Returns:
            o3d.geometry.TriangleMesh: The filled Open3D triangle mesh with holes closed and degenerate faces removed.
        """
        # convert to trimesh
        tm = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices),
            faces=np.asarray(o3d_mesh.triangles),
            process=False
        )
        tm.fill_holes()             # fills all holes automatically
        tm.remove_degenerate_faces()
        # back to Open3D
        filled = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(tm.vertices),
            o3d.utility.Vector3iVector(tm.faces)
        )
        filled.compute_vertex_normals()
        return filled
    
    def cleanup_mesh(self, o3d_mesh, min_triangles=250):
        """
        Cleans up an Open3D mesh by removing duplicate, degenerate, and unreferenced elements,
        as well as filtering out small disconnected triangle clusters.

        This function performs the following operations on the input mesh:
            - Removes duplicated vertices.
            - Removes degenerate triangles.
            - Removes unreferenced vertices.
            - Removes non-manifold edges.
            - Clusters triangles into connected components and removes clusters with fewer than
              `min_triangles` triangles.
            - Removes any vertices that become unreferenced after triangle removal.

        Args:
            o3d_mesh (open3d.geometry.TriangleMesh): The input mesh to be cleaned.
            min_triangles (int, optional): The minimum number of triangles a cluster must have
                to be retained. Clusters with fewer triangles are removed. Default is 250.

        Returns:
            open3d.geometry.TriangleMesh: The cleaned mesh.
        """
        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_unreferenced_vertices()
        o3d_mesh.remove_non_manifold_edges()
        # Cluster the mesh into connected triangle components
        triangle_clusters, cluster_n_triangles, _ = o3d_mesh.cluster_connected_triangles()
        # Convert to numpy array for filtering
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        # Set a minimum number of triangles to keep (e.g., 100)
        triangles_to_keep = [
            i for i, num in enumerate(cluster_n_triangles) if num >= min_triangles
        ]
        # Create a mask for triangles to keep
        keep_mask = np.isin(triangle_clusters, triangles_to_keep)
        o3d_mesh.remove_triangles_by_mask(~keep_mask)
        # Remove now-unreferenced vertices (after triangle deletion)
        o3d_mesh.remove_unreferenced_vertices()
        return o3d_mesh

    def smooth_with_trimesh(self, o3d_mesh, iterations=10, lamb=0.3):
        """
        Smooths an Open3D TriangleMesh using Trimesh's Laplacian filter.

        This function converts an Open3D TriangleMesh to a Trimesh object, applies Laplacian smoothing,
        and converts the result back to an Open3D TriangleMesh.

        Args:
            o3d_mesh (o3d.geometry.TriangleMesh): The input mesh to be smoothed.
            iterations (int, optional): Number of smoothing iterations to perform. Default is 10.
            lamb (float, optional): Smoothing factor (lambda) controlling the amount of smoothing. Default is 0.3.

        Returns:
            o3d.geometry.TriangleMesh: The smoothed mesh as an Open3D TriangleMesh.
        """
        # Convert Open3D → Trimesh
        tm = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices),
            faces=np.asarray(o3d_mesh.triangles),
            process=True 
        )
        # Apply smoothing
        filter_laplacian(tm, lamb=lamb, iterations=iterations)
        # Back to Open3D
        smoothed = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(tm.vertices),
            o3d.utility.Vector3iVector(tm.faces)
        )
        smoothed.compute_vertex_normals()
        return smoothed
    
    def generate_smooth_mesh(self, pcd, visualize_mesh=True, output_path="output_mesh.ply"): 
        """
        Generates a smooth mesh from a given point cloud.
        This method performs several steps to convert a point cloud into a clean, smooth mesh:
        1. Upsamples the input point cloud using jittering to increase point density.
        2. Computes a mesh using the alpha shape algorithm.
        3. Fills holes in the mesh using trimesh utilities.
        4. Cleans up the mesh to remove artifacts and improve quality.
        5. Optionally visualizes the resulting mesh.
        6. Saves the mesh to a file.
        Args:
            pcd (open3d.geometry.PointCloud): The input point cloud to be meshed.
            visualize_mesh (bool, optional): If True, displays the mesh in a visualization window. Defaults to True.
            output_path (str, optional): Path to save the resulting mesh file. Defaults to "output_mesh.ply".
        Returns:
            None
        """
        print("[PCD] Upsampling point cloud...")
        pcd = self.jitter_upsample(pcd, factor=3, noise_scale=0.003)
        print("[Alpha Shape] Computing mesh...")
        mesh = self.compute_mesh(pcd, alpha=0.03)
        print("[Alpha Shape] Filling holes...")
        mesh = self.fill_with_trimesh(mesh)
        print("[Alpha Shape] Mesh cleanup")
        mesh = self.cleanup_mesh(mesh)
        # print("[Alpha Shape] Smoothing mesh...")
        # mesh = self.smooth_with_trimesh(mesh, iterations=15, lamb=0.3)

        if visualize_mesh:
            print("[Alpha Shape] Visualizing mesh...")
            o3d.visualization.draw_geometries([mesh])   
        
        print("[Alpha Shape] Saving mesh...")
        o3d.io.write_triangle_mesh(output_path, mesh)

    def visualize(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(self.after_pcd)
        vis.add_geometry(self.before_pcd)
        opt = vis.get_render_option()
        opt.point_size = 5.0
        vis.run()
        vis.destroy_window()

    def run(self):
        print("Loading point clouds...")
        self.load_point_clouds()

        # print("Downsampling 'before' point cloud...")
        # self.before_pcd = self.downsample_pcd(self.before_pcd, 0.005)
        # print("Downsampling 'after' point cloud...")
        # self.after_pcd = self.downsample_pcd(self.after_pcd, 0.005)

        # print("Transforming 'after' point cloud...")
        self.transform_after_cloud()

        # self.generate_smooth_mesh(self.before_pcd, output_path='/home/marcus/IMML/orchard_env/results/test_mesh_upscaled.ply')

        print("Filtering 'before' point cloud for positive Z values...")
        self.before_pcd = self.filter_by_axis(self.before_pcd, x_min=-1.0, z_min=0.03)
        print("Filtering 'after' point cloud for positive Z values...")
        self.after_pcd = self.filter_by_axis(self.after_pcd, x_min=-1.0, z_min=0.03)

        print("Applying ICP registration...")
        self.apply_icp_registration()

        self.generate_smooth_mesh(self.after_pcd, output_path='/home/marcus/IMML/orchard_env/results/tests/after_mesh.ply')

        # print("Recoloring point clouds...")
        # self.before_pcd, self.after_pcd = self.recolor_point_clouds([self.before_pcd, self.after_pcd])

        # print("Visualizing point clouds...")
        # self.visualize()


if __name__ == "__main__":
    comparer = PointCloudToMesh(
        # before_path="/home/marcus/IMML/orchard_env/data/BAcompare0405/variety1/xgrids/2025-01-23-094626-ply.ply",
        # after_path="/home/marcus/IMML/orchard_env/data/BAcompare0405/variety1/xgrids/2025-01-23-112250-ply.ply"
        before_path='/home/marcus/IMML/orchard_env/data/labeled_pt_clouds/preprune/2025-01-23-094626-ply-3dgs-ArtificialObjectRemoval.ply',
        after_path='/home/marcus/IMML/orchard_env/data/labeled_pt_clouds/postprune/2025-01-23-112250-ply-3dgs-ArtificialObjectRemoval.ply'
    )
    comparer.run()
    # comparer.simple()
