import open3d as o3d
import numpy as np
import matplotlib.cm as cm

# SETTINGS
ply_file = "/home/marcus/IMML/prune_point_identifier/data/labeled_pt_clouds/preprune/before_pcd_transformed.ply"
camera_json = "camera_view.json"
save_camera_view = False
window_width = 1600
window_height = 900

# LOAD PLY
pcd = o3d.io.read_point_cloud(ply_file)
points = np.asarray(pcd.points)
z_normalized = (points[:, 2] - points[:, 2].min()) / (points[:, 2].ptp())
cmap = cm.ScalarMappable(cmap="coolwarm").cmap
pcd.colors = o3d.utility.Vector3dVector(cmap(z_normalized)[:, :3])

# VISUALIZER
vis = o3d.visualization.Visualizer()
vis.create_window(width=window_width, height=window_height)
vis.add_geometry(pcd)
ctr = vis.get_view_control()

if save_camera_view:
    vis.run()
    param = ctr.convert_to_pinhole_camera_parameters()
    # Save intrinsic + extrinsic only
    o3d.io.write_pinhole_camera_parameters(camera_json, param)
    print("Camera saved!")
else:
    param = o3d.io.read_pinhole_camera_parameters(camera_json)

    # Ensure the window size matches
    param.intrinsic.width = window_width
    param.intrinsic.height = window_height

    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()

vis.destroy_window()
