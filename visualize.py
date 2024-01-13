import open3d as o3d
import open3d.visualization.gui as gui
from time import sleep
import numpy as np
import copy



    


vis = o3d.visualization.Visualizer()
vis.create_window()

pcd = o3d.io.read_point_cloud(f'/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper.ply')

noisy_pcd = copy.deepcopy(pcd)
# apply_noise(pcd,noisy_pcd,0,0.003)

mu = 0
sigma = 0.001

def apply_noise(pcd,noisy_pcd):
    sigma = 0.001
    points = np.asarray(pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)

#np.asarray(pcd.points)
# vis.add_geometry(noisy_pcd)
# vis.poll_events()
# vis.update_renderer()
# opt = vis.get_render_option()
# opt.background_color = np.asarray([0, 0, 0])


def increase_sigma(vis):    
    apply_noise(pcd,noisy_pcd)
    vis.update_geometry()
    vis.update_renderer()
    vis.poll_events()
    vis.run()
    return False

def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False

print(np.asarray(pcd.colors))

vis.add_geometry(pcd)
# vis.update_geometry()
vis.update_renderer()
vis.poll_events()
vis.run()
vis.destroy_window()

# The point cloud is now shown

points = np.asarray(pcd.points)
points += np.random.normal(mu, sigma, size=points.shape)

pcd.points = o3d.utility.Vector3dVector(points)
#pcd.colors = o3d.utility.Vector3dVector([])

# vis.update_geometry()
vis.create_window()
vis.update_renderer()
vis.poll_events()
vis.run()

# The point cloud is no longer shown


# key_to_callback = {}
# key_to_callback[ord("K")] = change_background_to_black
# key_to_callback[ord("J")] = increase_sigma
# o3d.visualization.draw_geometries_with_key_callbacks([noisy_pcd], key_to_callback)

# opt = vis.get_render_option()
# opt.background_color = np.asarray([0, 0, 0])
# vis.update_renderer()

'''
ctrl = vis.get_view_control()
# This time we rotate the camera around the points and update the renderer
while vis.poll_events():
    # ctrl.rotate(5, 0)
    vis.update_renderer()


# We run the visualizater
vis.run()
# Once the visualizer is closed destroy the window and clean up
vis.destroy_window()
'''