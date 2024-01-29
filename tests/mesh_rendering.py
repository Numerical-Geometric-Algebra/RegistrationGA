import time
import numpy as np
import open3d as o3d
import copy

class Viewer3D(object):
    def __init__(self, title):
        self.CLOUD_NAME = 'cloud3d'
        self.first_cloud = True
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.main_vis = o3d.visualization.O3DVisualizer(title)
        app.add_window(self.main_vis)

    def tick(self):
        app = o3d.visualization.gui.Application.instance
        tick_return = app.run_one_tick()
        if tick_return:
            self.main_vis.post_redraw()
        return tick_return

    def update_cloud(self, geometries):
        if self.first_cloud:
            
            self.main_vis.add_geometry(self.CLOUD_NAME, geometries)
            self.main_vis.reset_camera_to_default()
            self.first_cloud = False
        else:
            self.main_vis.remove_geometry(self.CLOUD_NAME)
            self.main_vis.add_geometry(self.CLOUD_NAME, geometries)

            


viewer3d = Viewer3D("mytitle")
# pcd_data = o3d.data.DemoICPPointClouds()
pcd = o3d.io.read_point_cloud(f'/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper.ply')
bounds = pcd.get_axis_aligned_bounding_box()
extent = bounds.get_extent()

noisy_pcd = copy.deepcopy(pcd)
pts = np.asarray(pcd.points)
n_points = pts.shape[0]

color = np.array([[0,0,0.5]]*n_points)
noisy_pcd.colors = o3d.utility.Vector3dVector(color)
sigma = 0.005
mu = 0



while True:
    # Step 1) Perturb the cloud with a random walk to simulate an actual read
    # (based on https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/multiple_windows.py)
    
    pts = np.asarray(pcd.points)
    displacement = np.random.normal(mu,sigma,size=pts.shape)
    new_pts = pts + displacement

    noisy_pcd.points = o3d.utility.Vector3dVector(new_pts)

    # Step 2) Update the cloud and tick the GUI application
    viewer3d.update_cloud(noisy_pcd)
    viewer3d.tick()
    #time.sleep(0.1)