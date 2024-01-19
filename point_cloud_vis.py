#!/usr/bin/env python
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
import copy
from scipy.spatial.transform import Rotation
from eig_estimation import *


# Convert Rotor to a rotation matrix.
def rotor_to_rotation_matrix(R):
    axis = np.array(normalize(I*R(2)).list(1)[0][:3]) # compute axis from rotor
    angle = np.arccos(R(0))*2 # Compute angle from rotor
    rotation = Rotation.from_rotvec(angle * axis)

    # Get the rotation matrix
    Rot_matrix = rotation.as_matrix()

    return Rot_matrix


class PCViewer3D:
    def __init__(self,pcds,draw_primitives=[1,1],sigma_iter=0.0005,sigma=0.01,rdn_rbm=False,eig_grades = [1,2]):
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
        self.scene = gui.SceneWidget()
        self.scene.scene = o3d.visualization.rendering.Open3DScene(w.renderer)
        w.add_child(self.scene)
        
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultLit"
        self.scene.scene.add_geometry("Point Cloud0", self.noisy_pcd[0], self.material)
        # self.scene.scene.add_geometry("Point Cloud1", self.noisy_pcd[1], self.material)

        self.scene.setup_camera(60, self.scene.scene.bounding_box, (0, 0, 0))
        self.scene.set_on_key(self.on_key)
         

        self.primitive_material = [0,0,0]

        self.primitive_material[0] = self.get_material([0.0,0.0,0.5])
        self.primitive_material[1] = self.get_material([0.5,0.0,0.0])
        self.primitive_material[2] = self.get_material([0.0,0.5,0.0])
        
        self.transp_mat = self.get_material([0.0,0.5,0.5])
        self.transp_mat_neg = self.get_material([0.5,0.5,0.0])
        self.cylinder_mat = self.get_material([1.0,0.0,0.0])
        self.cylinder_mat_neg = self.get_material([0.0,1.0,0.0])

        self.scene.scene.set_background([0.5,0.5,0.5,1.0])
        self.scene.scene.show_axes(True)
        self.rdn_rbm = rdn_rbm
        
        self.n_points = np.asarray(self.pcd[0].points).shape[0]
        
        self.theta = 0
        self.t = 0
        self.camera_pos = [0,0,0]

        if self.rdn_rbm:
            self.T,self.R,self.t = gen_pseudordn_rbm(100,1)
        else:
            self.theta = 90*np.pi/180
            self.R = np.cos(self.theta/2) + e2*I*np.sin(self.theta/2)
            self.t = 0.0*e1 + 0.3*e2 + 0.3*e3
            self.T = 1 + (1/2)*einf*self.t

        # Initialize the estimated point cloud in green
        self.pcd[2] = copy.deepcopy(self.pcd[0])
        self.noisy_pcd[2] = copy.deepcopy(self.pcd[0])
        color = np.array([[0.0,0.5,0.0]]*pts.shape[0])
        self.pcd[2].colors = o3d.utility.Vector3dVector(color)

        self.update_rbm()

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
            self.T,self.R,self.t = gen_pseudordn_rbm(100,1)
        else:
            self.R = np.cos(self.theta/2) + e2*I*np.sin(self.theta/2)
            self.T = 1 + (1/2)*einf*self.t
        # Applies a transformation to pcd[1] and stores it in noisy_pcd[1]
        pts = transform_numpy_cloud(self.pcd[1],self.R,self.t)
        self.noisy_pcd[1].points = o3d.utility.Vector3dVector(pts)
        self.add_noise(self.noisy_pcd[1],self.noisy_pcd[1])

        self.update_model()

    def update_model(self):
        # make noisy_pcd[0] a noisy version of pcd[0]
        self.add_noise(self.pcd[0],self.noisy_pcd[0])
        
        self.estimate_rbm()

        self.update_pc(0)
        self.update_pc(2)

        self.compute_primitives(2)

        self.draw_primitives(0)
        self.draw_primitives(2)

        # self.update_pc(1)
        # self.draw_primitives(1)

    def get_material(self,color):
        transp_mat = o3d.visualization.rendering.MaterialRecord()
        # transp_mat.shader = 'defaultLitTransparency'
        transp_mat.shader = 'defaultLitSSR'
        transp_mat.base_color = [0.0, 0.467, 0.467, 0.2]
        transp_mat.base_roughness = 0.1
        transp_mat.base_reflectance = 0.0
        transp_mat.base_clearcoat = 1.0
        transp_mat.thickness = 1.0
        transp_mat.transmission = 1.0
        transp_mat.absorption_distance = 10
        transp_mat.absorption_color = color
        return transp_mat


    def update_pc(self,j):
        self.scene.scene.remove_geometry("Point Cloud"+str(j))
        self.scene.scene.add_geometry("Point Cloud"+str(j), self.noisy_pcd[j], self.material)

    def run(self):
        gui.Application.instance.run()

    def on_key(self,e):
        if e.key == gui.KeyName.K:
            if e.type == gui.KeyEvent.UP:  
                self.sigma += self.sigma_iter
                self.update_model()
        if e.key == gui.KeyName.J:
            if e.type == gui.KeyEvent.UP:  
                self.sigma -= self.sigma_iter
                if self.sigma < 0:
                    self.sigma = 0                
                self.update_model()

        if e.key == gui.KeyName.Q:
            gui.Application.instance.quit()
        if e.key == gui.KeyName.SPACE:
            self.update_model()
        if e.key == gui.KeyName.C:
            self.scene.setup_camera(60, self.scene.scene.bounding_box, (0, 0, 0))

        if e.key == gui.KeyName.W:
            if e.type == gui.KeyEvent.UP: 
                self.t += 0.01*e1
                self.update_rbm()
                self.update_model()
        if e.key == gui.KeyName.S:
            if e.type == gui.KeyEvent.UP: 
                self.t -= 0.01*e1
                self.update_rbm()
                self.update_model()

        if e.key == gui.KeyName.A:
            if e.type == gui.KeyEvent.UP:
                self.theta += 1/np.pi
                self.update_rbm()
                self.update_model()
        if e.key == gui.KeyName.D:
            if e.type == gui.KeyEvent.UP:
                self.theta -= 1/np.pi
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

    
    def get_circle(self,radius_sq,position,normal):

        # Create a cylinder along the given normal vector
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=np.sqrt(abs(radius_sq)), height=0.0001,resolution=40)

        # Calculate the rotation matrix to align the cylinder with the given normal
        normal = np.array(normal)
        z_axis = normal / np.linalg.norm(normal)
        x_axis = np.cross([0, 0, 1], z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

        # Apply the rotation matrix to the cylinder
        cylinder.rotate(rotation_matrix)

        # Translate the cylinder to the specified position
        cylinder.translate(position)
        cylinder.compute_vertex_normals()
        
        if(radius_sq < 0):
            material = self.cylinder_mat_neg
        else:
            material = self.cylinder_mat
        return cylinder,material


    def estimate_rbm(self):
        x_pts = np.asarray(self.noisy_pcd[0].points)
        y_pts = np.asarray(self.noisy_pcd[1].points)

        # Convert numpy array to multivector array
        x = nparray_to_mvarray(x_pts)
        y = nparray_to_mvarray(y_pts)

        # Convert to CGA
        p = eo + x + (1/2)*mag_sq(x)*einf 
        q = eo + y + (1/2)*mag_sq(y)*einf

        # Get the eigenbivectors
        P_lst,lambda_P = get_eigmvs(p,grades=self.eig_grades)
        Q_lst,lambda_Q = get_eigmvs(q,grades=self.eig_grades)

        # Transform list of multivectors into an array
        P = mv.concat(P_lst) 
        Q = mv.concat(Q_lst)

        # Orient the eigenbivectors by using the points p and q as a reference
        signs = get_orient_diff(P,Q,p,q)
        P = P*signs
        T_est,R_est = estimate_rbm(P,Q)
        T_est = translation_from_cofm(y,x,R_est,self.n_points)
        t_est = -2*eo|T_est
        

        # Save the list of eigenmultivectors
        self.P_lst[0] = P_lst
        self.P_lst[1] = Q_lst
        
        # Calculate the estimated point cloud from x and from y
        y_est = R_est*x*~R_est + t_est
        x_est = ~R_est*(y - t_est)*R_est
        
        print_metrics(R_est,self.R,T_est,self.T,self.n_points,self.sigma)
        
        x_pts = mvarray_to_nparray(x_est)

        self.pcd[2].points = o3d.utility.Vector3dVector(x_pts)
        self.noisy_pcd[2] = copy.deepcopy(self.pcd[2])

    def compute_primitives(self,j):
        x_pts = np.asarray(self.noisy_pcd[j].points)
        
        # Convert numpy array to multivector array
        x = nparray_to_mvarray(x_pts)

        # Convert to CGA
        p = eo + x + (1/2)*mag_sq(x)*einf 

        P_lst,lambda_P = get_eigmvs(p,grades=self.eig_grades)
        self.P_lst[j] = P_lst
        


    def draw_primitives(self,j):
        self.remove_primitives(j)
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
                    primitive,_ = self.get_circle(radius_sq,l,d)
                    self.scene.scene.add_geometry(str(j)+'primitive'+str(i),
                                                primitive,
                                                self.primitive_material[j])
            
    def remove_primitives(self,j):
        for i in range(15):
            self.scene.scene.remove_geometry(str(j)+'primitive'+str(i))


if __name__ == '__main__':
    # pcd = o3d.io.read_point_cloud(f'/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper.ply')
    # pcd = o3d.io.read_point_cloud(f"/home/francisco/Code/Stanford Dataset/bunny/data/bun000.ply")

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
