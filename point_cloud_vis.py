#!/usr/bin/env python
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
import copy
from scipy.spatial.transform import Rotation
from cga3d_estimation import *
import algorithms
import multilinear_algebra as multiga

import os, pickle
from pathlib import Path


# Convert Rotor to a rotation matrix.
def rotor_to_rotation_matrix(R):
    axis = np.array(normalize(I*R(2)).list(1)[0][:3]) # compute axis from rotor
    angle = np.arccos(R(0))*2 # Compute angle from rotor
    rotation = Rotation.from_rotvec(angle * axis)

    # Get the rotation matrix
    Rot_matrix = rotation.as_matrix()

    return Rot_matrix

def load_obj(path):
    """
    read a dictionary from a pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_3DMatch_PCs(base_dir,item=0):
    config_path = os.path.join(base_dir,"config/train_info.pkl")
    config = load_obj(config_path)
    return config

def add_subdirectory(old_dir,subdir):
    base = os.path.basename(old_dir)
    directory = os.path.dirname(old_dir)
    new_path = os.path.join(directory, subdir, base)
    return new_path

def read_poses(path):
    file_ = open(path)
    data = file_.read()
    data_list = data.split("\n")
    data_list_ = [0]*len(data_list)

    for i in range(len(data_list)):
        data_list_[i] = data_list[i].split("\t")[:-1]
    
    rbm_txt = data_list_[1:-1]

    rbm = np.zeros([4,4])
    for i in range(4):
        for j in range(4):
            rbm[i][j] = float(rbm_txt[i][j])
    
    return rbm


def change_ext_subdir(path,subdir,ext_):
    pre, ext = os.path.splitext(path)
    path = pre + ext_
    return add_subdirectory(path,subdir)


def apply_rbm(pcd,rbm):
    pts = np.asarray(pcd.points)
    rot = rbm[:,:3][:3]
    trans = rbm[:,-1][:3]

    pts = pts@rot + trans
    pcd.points = o3d.utility.Vector3dVector(pts)


def get_point_clouds(cfg,base_dir,item=-1):
    index = np.argsort(cfg['overlap'])[item] # The most overlapping scenes
    src_path_ = os.path.join(base_dir,cfg['src'][index])
    tgt_path_ = os.path.join(base_dir,cfg['tgt'][index])
    print("Overlap:",cfg['overlap'][index])
    # print(src_path)
    # print(tgt_path)

    src_path = change_ext_subdir(src_path_,"fragments",".ply")
    tgt_path = change_ext_subdir(tgt_path_,"fragments",".ply")
    print(src_path_)
    print(cfg.keys())
    

    pose_src_path = change_ext_subdir(src_path_,"poses",".txt")
    pose_tgt_path = change_ext_subdir(tgt_path_,"poses",".txt")
    
    src_rbm = read_poses(pose_src_path)
    tgt_rbm = read_poses(pose_tgt_path)


    src_pcd = o3d.io.read_point_cloud(src_path)
    tgt_pcd = o3d.io.read_point_cloud(tgt_path)

    # rot = cfg['rot'][index]
    # trans = cfg['trans'][index]
    #print(rot)

    # apply_rbm(src_pcd,src_rbm)
    # apply_rbm(tgt_pcd,tgt_rbm)

    # R = rotmatrix_to_rotor(rot)
    # t = nparray_to_3dvga_vector_array(trans.T[0])
    # print(R)
    # print(t)
    
    '''
    pts = np.asarray(src_pcd.points)
    pts = pts@rot + trans.T
    src_pcd.points = o3d.utility.Vector3dVector(pts)
    '''

    # pts = np.asarray(tgt_pcd.points)
    # pts = pts@rot + trans.T
    # tgt_pcd.points = o3d.utility.Vector3dVector(pts)


    # pts = transform_numpy_cloud(src_pcd,R,t)
    
    # pts = transform_numpy_cloud(tgt_pcd,R,t)
    # tgt_pcd.points = o3d.utility.Vector3dVector(pts)
    return (src_pcd,tgt_pcd)
    

class PCViewer3D:
    def __init__(self,pcds,draw_primitives=[True,True,True],sigma_iter=0.0005,sigma=0.01,rdn_rbm=False,eig_grades = [1,2],compare_primitives=False):
        self.draw_primitives_bool = draw_primitives
        self.eig_grades = eig_grades
        self.sigma = sigma
        self.mu = 0
        self.pcd = pcds
        self.noisy_pcd = [copy.deepcopy(pcds[0]),copy.deepcopy(pcds[1])]
        self.P_lst = [0,0,0]
        self.sigma_iter = sigma_iter
        self.pcd += [0]
        self.noisy_pcd += [0]

        gui.Application.instance.initialize()
        w = gui.Application.instance.create_window("Open3D Example - Events",
                                               640, 480)
        self.window = w
        
        self.scene = gui.SceneWidget()
        self.scene.scene = o3d.visualization.rendering.Open3DScene(w.renderer)
        
        
        # self.scene.scene.enable_indirect_light(False)
        # self.scene.scene.enable_sun_light(False)

        # vis = o3d.visualization.Visualizer()
        # vis.get_render_option().line_width = 5
        # o3d.visualization.RenderOption.line_width = 20.0

        # self.scene.scene.get_render_option().line_width = 10

        w.add_child(self.scene)
        
        self.material = [0,0,0]
        # self.material = o3d.visualization.rendering.MaterialRecord()
        # self.material.shader = "defaultLit"

        # Color for each point cloud
        self.pc_color = [[213/255, 65/255, 0],[0, 0, 1],[34/255, 150/255, 0 ]]

        print(self.pc_color)

        self.material[0] = self.get_transparent_material(self.pc_color[0])
        self.material[1] = self.get_transparent_material(self.pc_color[1])
        self.material[2] = self.get_transparent_material(self.pc_color[2])

        self.scene.scene.add_geometry("Point Cloud0", self.noisy_pcd[0], self.material[0])
        # self.scene.scene.add_geometry("Point Cloud1", self.noisy_pcd[1], self.material)

        self.scene.setup_camera(60, self.scene.scene.bounding_box, (0, 0, 0))
        self.scene.set_on_key(self.on_key)
         

        self.primitive_material = [0,0,0]

        self.primitive_material[0] = self.get_transparent_material(self.pc_color[0])
        self.primitive_material[1] = self.get_transparent_material(self.pc_color[1])
        self.primitive_material[2] = self.get_transparent_material(self.pc_color[2])
        
        # self.transp_mat = self.get_transparent_material([0.0,0.5,0.5])
        # self.transp_mat_neg = self.get_transparent_material([0.5,0.5,0.0])
        # self.cylinder_mat = self.get_transparent_material([1.0,0.0,0.0])
        # self.cylinder_mat_neg = self.get_transparent_material([0.0,1.0,0.0])

        self.scene.scene.set_background([1.0,1.0,1.0,1.0])
        # self.scene.scene.show_axes(True)
        self.rdn_rbm = rdn_rbm
        
        self.n_points = np.asarray(self.pcd[0].points).shape[0]
        self.algorithm = None

        self.theta = 0
        self.t = 0
        self.camera_pos = [0,0,0]

        if self.rdn_rbm:
            self.T,self.R = gen_pseudordn_rigtr(100,1)
            self.t = -eo|self.T*2
        else:
            self.theta = 90*np.pi/180
            self.R = np.cos(self.theta/2) + e2*I*np.sin(self.theta/2)
            self.t = 0.0*e1 + 0.3*e2 + 0.3*e3
            self.T = 1 + (1/2)*einf*self.t

        # Initialize the estimated point cloud in green
        self.pcd[2] = copy.deepcopy(self.pcd[0])
        
        self.set_pc_color(0)
        self.set_pc_color(2)

        self.noisy_pcd[0] = copy.deepcopy(self.pcd[0])
        self.noisy_pcd[2] = copy.deepcopy(self.pcd[2])
        # color = np.array([[0.0,0.5,0.0]]*pts.shape[0])
        # self.pcd[2].colors = o3d.utility.Vector3dVector(color)

        # self.update_pc(0)
        # self.update_pc(1)
        
        self.compare_primitives = compare_primitives
        if not self.compare_primitives:
            self.update_rbm()
        else:
            self.draw_noisy_PCs()
        
    def set_pc_color(self,j):
        pts = np.asarray(self.pcd[j].points)
        n_points = pts.shape[0]
        color = np.array([self.pc_color[j]]*n_points)
        self.pcd[j].colors = o3d.utility.Vector3dVector(color)

    def draw_noisy_PCs(self):
        self.add_noise(self.pcd[0],self.noisy_pcd[0]) # Add noise to BLUE PC
        self.add_noise(self.pcd[1],self.noisy_pcd[1]) # Add noise to RED PC
        
        self.update_pc(0)
        self.update_pc(1)

        self.compute_primitives(0)
        self.compute_primitives(1)

        self.draw_primitives(0)
        self.draw_primitives(1)

    # Get relative translation error scale
    def get_scale_rte(self):
        self.rte_scale = np.max(abs(np.asarray(self.pcd[0].points)))


    # Adds noise to pcd then stores the result in noisy_pcd
    def add_noise(self,pcd,noisy_pcd):
        
        pts = np.copy(np.asarray(pcd.points))
        noise = np.random.normal(self.mu,self.sigma,size=pts.shape)
        pts += noise
        noisy_pcd.points = o3d.utility.Vector3dVector(pts)

    def update_rbm(self):
        if self.rdn_rbm:
            self.T,self.R = gen_pseudordn_rigtr(100,1)
            self.t = -eo|self.T*2
        else:
            self.R = np.cos(self.theta/2) + e2*I*np.sin(self.theta/2)
            self.T = 1 + (1/2)*einf*self.t
        # Applies a transformation to pcd[1] and stores it in noisy_pcd[1]
        pts = transform_numpy_cloud(self.pcd[1],self.R,self.t)
        self.pcd[1].points = o3d.utility.Vector3dVector(pts)
        self.update_model()

    def update_model(self):

        # Add noise to the point clouds
        self.add_noise(self.pcd[0],self.noisy_pcd[0]) # Add noise to BLUE PC
        self.add_noise(self.pcd[1],self.noisy_pcd[1]) # Add noise to RED PC
        
        self.estimate_rigtr()

        self.update_pc(0)
        # self.update_pc(1)
        self.update_pc(2)

        self.compute_primitives(2)

        self.draw_primitives(0)
        self.draw_primitives(2)
        
        self.draw_CeOM(0)
        self.draw_CeOM(2)

        
        # self.draw_primitives(1)

    def get_transparent_material(self,color):
        transp_mat = o3d.visualization.rendering.MaterialRecord()

        transp_mat.shader = 'defaultLitTransparency'
        # transp_mat.shader = 'defaultLitSSR'
        # transp_mat.base_color = [0, 70/255, 166/255, 0.7]
        # transp_mat.base_color = [145/255, 145/255, 145/255, 0.5]
        transp_mat.base_color = color + [0.7]

        transp_mat.base_roughness = 5.0
        transp_mat.base_reflectance = 1.0
        transp_mat.base_clearcoat = 1.0
        transp_mat.thickness = 5.0
        transp_mat.transmission = 3
        transp_mat.absorption_distance = 10
        # transp_mat.absorption_color = [0.5,0.5,0.5]

        # transp_mat.shader = 'defaultLitTransparency'
        # transp_mat.shader = "defaultLit"
        # transp_mat.base_color = color + [1]
        # print(transp_mat.base_color)

        # transp_mat.base_roughness = 1.0
        # transp_mat.base_reflectance = 0.0
        # transp_mat.base_clearcoat = 1.0
        # transp_mat.thickness = 5.0
        # transp_mat.transmission = 3
        # transp_mat.absorption_distance = 0
        # transp_mat.absorption_color = color
        return transp_mat

    def get_solid_material(self,color):
        solid_mat = o3d.visualization.rendering.MaterialRecord()
        # solid_mat.shader = 'defaultLitTransparency'
        # solid_mat.shader = "defaultLit"
        solid_mat.base_color = color + [1]

        # solid_mat.base_roughness = 100
        # solid_mat.base_reflectance = 0
        # solid_mat.base_clearcoat = 1.0
        # solid_mat.thickness = 1.0
        # solid_mat.transmission = 0
        # solid_mat.absorption_distance = 0
        # solid_mat.absorption_color = color
        return solid_mat

    def get_line_material(self,color):
        solid_mat = o3d.visualization.rendering.MaterialRecord()
        solid_mat.shader = "defaultLit"
        solid_mat.base_color = color + [10]
        solid_mat.base_roughness = 1
        solid_mat.base_reflectance = 0
        solid_mat.base_clearcoat = 0
        solid_mat.thickness = 10.0
        solid_mat.transmission = 0
        solid_mat.absorption_distance = 0
        solid_mat.absorption_color = color
        return solid_mat


    def update_pc(self,j):
        # print("Updating Point Cloud "+str(j))
        self.scene.scene.remove_geometry("Point Cloud"+str(j))
        self.scene.scene.add_geometry("Point Cloud"+str(j), self.noisy_pcd[j], self.get_solid_material(self.pc_color[j]))
        # print(self.pc_color[j])

    def run(self):
        gui.Application.instance.run()

    def on_key(self,e):
        if e.key == gui.KeyName.K:
            if e.type == gui.KeyEvent.UP:  
                self.sigma += self.sigma_iter
                if self.compare_primitives:
                    self.draw_noisy_PCs()
                else:
                    self.update_model()
        if e.key == gui.KeyName.J:
            if e.type == gui.KeyEvent.UP:  
                self.sigma -= self.sigma_iter
                if self.sigma < 0:
                    self.sigma = 0
                if self.compare_primitives:
                    self.draw_noisy_PCs()
                else:
                    self.update_model()

        if e.key == gui.KeyName.Q:
            gui.Application.instance.quit()
        if e.key == gui.KeyName.SPACE:
            if self.compare_primitives:
                self.draw_noisy_PCs()
            else:
                self.update_model()
        if e.key == gui.KeyName.C:
            self.scene.setup_camera(60, self.scene.scene.bounding_box, (0, 0, 0))

        if e.key == gui.KeyName.W:
            if e.type == gui.KeyEvent.UP: 
                self.t += 0.01*e1
                if not self.compare_primitives:
                    self.update_rbm()
                    self.update_model()
        if e.key == gui.KeyName.S:
            if e.type == gui.KeyEvent.UP: 
                self.t -= 0.01*e1
                if not self.compare_primitives:
                    self.update_rbm()
                    self.update_model()

        if e.key == gui.KeyName.A:
            if e.type == gui.KeyEvent.UP:
                self.theta += 1/np.pi
                if not self.compare_primitives:
                    self.update_rbm()
                    self.update_model()
        if e.key == gui.KeyName.D:
            if e.type == gui.KeyEvent.UP:
                self.theta -= 1/np.pi
                if not self.compare_primitives:
                    self.update_rbm()
                    self.update_model()

        return gui.Widget.EventCallbackResult.IGNORED
    
    def get_sphere(self,radius_sq,location):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=np.sqrt(abs(radius_sq)),resolution=20)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.7, 0.1, 0.1])  # To be changed to the point color.
        sphere = sphere.translate(location)
        if(radius_sq < 0):
            material = self.transp_mat_neg
        else:
            material = self.transp_mat
        return sphere,material

    def calculate_zy_rotation_for_arrow(self,vec):
        gamma = np.arctan2(vec[1], vec[0])
        Rz = np.array([
                        [np.cos(gamma), -np.sin(gamma), 0],
                        [np.sin(gamma), np.cos(gamma), 0],
                        [0, 0, 1]
                    ])

        vec = Rz.T @ vec

        beta = np.arctan2(vec[0], vec[2])
        Ry = np.array([
                        [np.cos(beta), 0, np.sin(beta)],
                        [0, 1, 0],
                        [-np.sin(beta), 0, np.cos(beta)]
                    ])
        return Rz, Ry
    
    def get_arrow(self, end, origin=np.array([0, 0, 0]), scale=1):
        vec = end - origin
        size = 0.25
        # size = np.sqrt(np.sum(vec**2))
        Rz, Ry = self.calculate_zy_rotation_for_arrow(vec)
        mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size/10 * scale,
        cone_height=size*0.4 * scale,
        cylinder_radius=size/30 * scale,
        cylinder_height=size*2*scale)
        mesh.rotate(Ry, center=np.array([0, 0, 0]))
        mesh.rotate(Rz, center=np.array([0, 0, 0]))
        mesh.translate(origin)
        mesh.compute_vertex_normals()
        return mesh


    def create_2d_torus(self,radius,thickness=0.001,num_points=500):
        # Create points on the circumference of the circle
        theta = np.linspace(0, 2*np.pi, num_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros_like(x)

        x1 = (radius - thickness) * np.cos(theta)
        y1 = (radius - thickness) * np.sin(theta)
        
        z2 = -np.ones_like(x)*0.0001

        points = np.r_[np.vstack((x, y, z)).T,np.vstack((x1, y1, z)).T]

        triangles = []
        for i in range(num_points):
            triangles.append([i, i+1, i + num_points])            

        for i in range(num_points-1):
            triangles.append([i + num_points + 1,i + num_points,i+1])

        triangles = np.array(triangles)

        torus_mesh = o3d.geometry.TriangleMesh()
        torus_mesh.vertices = o3d.utility.Vector3dVector(points)
        torus_mesh.triangles = o3d.utility.Vector3iVector(triangles)

        points = np.r_[np.vstack((x, y, z2)).T,np.vstack((x1, y1, z2)).T]

        torus_mesh1 = o3d.geometry.TriangleMesh()
        torus_mesh1.vertices = o3d.utility.Vector3dVector(points)
        torus_mesh1.triangles = o3d.utility.Vector3iVector(triangles)

        return torus_mesh,torus_mesh1,np.vstack((x, y, z)).T


    def create_circle(self,radius, position,normal,num_points=500):
        thickness = 0.002
        torus_mesh,torus_mesh1,points = self.create_2d_torus(radius,thickness,num_points)
    
        # Create triangles for filling the circle
        triangles = []
        for i in range(num_points):
            triangles.append([i, (i + 1) % num_points, num_points])

        # Convert points and triangles to Open3D format
        center_point = np.array([[0, 0, 0]])
        points = np.concatenate((points, center_point), axis=0)
        triangles = np.array(triangles)

        # Create a TriangleMesh object
        circle_mesh = o3d.geometry.TriangleMesh()
        circle_mesh.vertices = o3d.utility.Vector3dVector(points)
        circle_mesh.triangles = o3d.utility.Vector3iVector(triangles)

        normal = np.array(normal)
        z_axis = normal / np.linalg.norm(normal)
        x_axis = np.cross([0, 0, 1], z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

        # Apply the rotation matrix to the cylinder
        circle_mesh.rotate(rotation_matrix)
        torus_mesh.rotate(rotation_matrix)
        torus_mesh1.rotate(rotation_matrix)

        # Translate the line_set to the specified position
        circle_mesh.translate(position)
        torus_mesh.translate(position)
        torus_mesh1.translate(position)

        circle_mesh.compute_vertex_normals()
        torus_mesh.compute_vertex_normals()
        torus_mesh1.compute_vertex_normals()
    
        return circle_mesh,torus_mesh,torus_mesh1


    # Draw Center Of Mass (CeOM) of point cloud i
    def draw_CeOM(self,i):
        self.scene.scene.remove_geometry('CeOM'+str(i))

        x_pts = np.asarray(self.noisy_pcd[i].points)
        x_bar = x_pts.sum(axis=0)/self.n_points
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005,resolution=20)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.7, 0.1, 0.1])  # To be changed to the point color.
        sphere = sphere.translate(x_bar)
        material = self.material[i]
        self.scene.scene.add_geometry('CeOM'+str(i),sphere,material)


    def estimate_rigtr(self):
        
        if(self.algorithm is None):
            self.algorithm = self.get_default_algorithm()
        
        x_pts = np.asarray(self.noisy_pcd[0].points)
        y_pts = np.asarray(self.noisy_pcd[1].points)

        # Convert numpy array to multivector array
        x = nparray_to_3dvga_vector_array(x_pts)
        y = nparray_to_3dvga_vector_array(y_pts)
        
        # Use the chosen algorithm to estimate the RBM
        T_est,R_est,P_lst,Q_lst = self.algorithm(x,y,self.n_points)

        t_est = -2*eo|T_est
            
        # Q_est = T_est*R_est*P*~R_est*~T_est
        # q_bar = q.sum()/self.n_points
        # q_bar_est = T_est*R_est*p.sum()*~R_est*~T_est/self.n_points

        # print("Primitives Error:",pyga.mag_sq(P_I(Q_est - Q)).sum())
        # print("Center of Mass of q:",q_bar)
        # print("Center of Mass of q_est:",q_bar_est)
        # print("Center of Mass diff :" , q_bar - q_bar_est)

        # Save the list of eigenmultivectors
        self.P_lst[0] = P_lst
        self.P_lst[1] = Q_lst
        
        # Calculate the estimated point cloud from x and from y
        # y_est = R_est*x*~R_est + t_est
        x_est = ~R_est*(y - t_est)*R_est
        
        print_rigtr_error_metrics(R_est,self.R,T_est,self.T,self.n_points,self.sigma)
        
        x_pts = cga3d_vector_array_to_nparray(x_est)

        self.pcd[2].points = o3d.utility.Vector3dVector(x_pts)
        self.noisy_pcd[2] = copy.deepcopy(self.pcd[2])

    def get_default_algorithm(self):
        return algorithms.estimate_transformation_0

    def compute_primitives(self,j):
        # Do not compute primitives if not drawing
        if self.draw_primitives_bool[0] == False and  self.draw_primitives_bool[1] == False and self.draw_primitives_bool[2] == False:
            return
        x_pts = np.asarray(self.noisy_pcd[j].points)
        
        # Convert numpy array to multivector array
        x = nparray_to_3dvga_vector_array(x_pts)

        # Convert to CGA
        p = eo + x + (1/2)*pyga.mag_sq(x)*einf 

        P_lst,lambda_P = get_3dcga_eigmvs(p,grades=self.eig_grades)
        self.P_lst[j] = P_lst
        

    

    def draw_primitives(self,j):
        self.remove_primitives(j)
        if self.draw_primitives_bool[0] == False and self.draw_primitives_bool[1] == False and self.draw_primitives_bool[2] == False:
            return

        # value = check_orthogonality(self.P_lst[0])
        # value = check_orthogonality(self.P_lst[1])

        for i in range(len(self.P_lst[j])):
            d,l,radius_sq = get_properties(self.P_lst[j][i])
            d_array = np.array(d)
            if((d_array*d_array).sum() < 1E-12): # check if is sphere
                if self.draw_primitives_bool[0]:
                    primitive,_ = self.get_sphere(radius_sq,l)
                    self.scene.scene.add_geometry(str(j)+'primitive'+str(i),
                                                  primitive,
                                                  self.primitive_material[j])
            else:
                if self.draw_primitives_bool[1]:    
                    filled_circle,torus_mesh0,torus_mesh1 = self.create_circle(np.sqrt(abs(radius_sq)),l,d)

                    self.scene.scene.add_geometry(str(j)+'primitive'+str(i),
                                                filled_circle,
                                                self.get_transparent_material(self.pc_color[j]))

                    self.scene.scene.add_geometry(str(j)+'unfilled_primitive0_'+str(i),
                                                torus_mesh0,
                                                self.get_solid_material(self.pc_color[j]))

                    self.scene.scene.add_geometry(str(j)+'unfilled_primitive1_'+str(i),
                                                torus_mesh1,
                                                self.get_solid_material(self.pc_color[j]))

            if self.draw_primitives_bool[2]:
                A,B,C,D = get_coeffs(self.P_lst[j][i])
                A = pyga.normalize_mv(A)
                arrow = self.get_arrow(np.array(A.tolist(1)[0][:3]),scale=0.2)
                self.scene.scene.add_geometry(str(j)+'Arrow'+str(i),
                                                    arrow,
                                                    self.get_solid_material(self.pc_color[j]))
                # print(np.array(A.tolist(1)[0][:3]))
                
        # print()

    def remove_primitives(self,j):
        for i in range(15):
            self.scene.scene.remove_geometry(str(j)+'primitive'+str(i))
            self.scene.scene.remove_geometry(str(j)+'unfilled_primitive1_'+str(i))
            self.scene.scene.remove_geometry(str(j)+'unfilled_primitive0_'+str(i))
            self.scene.scene.remove_geometry(str(j)+'Arrow'+str(i))


if __name__ == '__main__':

    '''
    It seems that usually using the center of mass solution for the translation estimation gives the best results, still need 
    further experimentation...
    -501: 0.69 of overlap: Good registration
    '''
    # pcd = o3d.io.read_point_cloud(f'/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper.ply')
    

    sigma = 0.01
    # sigma = 0

    # Get single bunny 
    # pcd = o3d.io.read_point_cloud(f"/home/francisco/Code/Stanford Dataset/bunny/data/bun000.ply")
    pcd = o3d.io.read_point_cloud(f"/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper_res2.ply")
    

    # base_dir =  f'/home/francisco/3dmatch/'
    # cfg = load_3DMatch_PCs(base_dir)
    # src_pcd,tgt_pcd = get_point_clouds(cfg,base_dir,-1)
    
    # Add outlier
    # outlier = np.ones([1,3])*0.1
    # pts = np.r_[np.asarray(tgt_pcd.points),outlier]
    # tgt_pcd.points = o3d.utility.Vector3dVector(pts)

    # Change color of target point cloud
    pts = np.asarray(pcd.points)
    ceofm = pts.mean(axis=0)
    pcd.points = o3d.utility.Vector3dVector(pts - ceofm) # Put the center of mass of the bunnies at the origin
    pcd_copy = copy.deepcopy(pcd)
    
    
    # draw_primitives: (spheres,circles,vectors)

    viewer = PCViewer3D([pcd,pcd_copy],draw_primitives=[False,True,True],sigma=sigma,rdn_rbm=True,eig_grades=[1,2],compare_primitives=False)
    viewer.algorithm = algorithms.estimate_transformation_16
    viewer.update_model()

    viewer.run()

    '''
    pcd = o3d.io.read_point_cloud(f'/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper.ply')
    pts = np.asarray(pcd.points)
    
    n_points = pts.shape[0]
    color = np.array([[0,0,0.5]]*n_points)
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcd1 = copy.deepcopy(pcd)
    color = np.array([[0.5,0,0]]*n_points)
    pcd1.colors = o3d.utility.Vector3dVector(color)

    # The viewer accepts two aligned pointclouds
    # Applies a rigid transformation to pcd1 and tries to estimate the RBM
    viewer = PCViewer3D([pcd,pcd1],draw_primitives=[True,False],sigma=0.0185,rdn_rbm=True)
    viewer.run()
    '''