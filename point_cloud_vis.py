#!/usr/bin/env python
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
import copy
from geo_algebra import *
from scipy.spatial.transform import Rotation
from eigen_rbm_estimation import *



def rotor_to_rotation_matrix(R):
    """
    Convert Rotor to a rotation matrix.

    Parameters:
    3D Rotor

    Returns:
    numpy.ndarray: A 3D rotation matrix.
    """

    axis = np.array(normalize(I*R(2)).list(1)[0][:3]) # compute axis from rotor
    
    angle = np.arccos(R(0))*2 # Compute angle from rotor
    rotation = Rotation.from_rotvec(angle * axis)

    # Get the rotation matrix
    Rot_matrix = rotation.as_matrix()

    return Rot_matrix



class PCViewer3D:
    def __init__(self,pcds,draw_spheres=True,sigma_iter=0.0005):
        self.draw_spheres = draw_spheres
        self.sigma = 0.01
        self.mu = 0
        self.pcd = pcds
        self.noisy_pcd = [copy.deepcopy(pcds[0]),copy.deepcopy(pcds[1])]
        self.P_lst = [0,0]
        self.n_points = 100
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
        self.scene.scene.add_geometry("Point Cloud1", self.noisy_pcd[1], self.material)

        self.scene.setup_camera(60, self.scene.scene.bounding_box, (0, 0, 0))
        self.scene.set_on_key(self.on_key)
         
        
        self.transp_mat = self.get_material([0.0,0.5,0.5])
        self.transp_mat_neg = self.get_material([0.5,0.5,0.0])
        self.cylinder_mat = self.get_material([1.0,0.0,0.0])
        self.cylinder_mat_neg = self.get_material([0.0,1.0,0.0])

        self.scene.scene.set_background([0.5,0.5,0.5,1.0])
        self.scene.scene.show_axes(True)
        
        self.camera_pos = [0,0,0]

        self.theta = 90*np.pi/180
        print("nbr points:",np.asarray(self.pcd[j].points).shape[0])
        # T,R,t = gen_pseudordn_rbm(100,0)

        self.R = np.cos(self.theta/2) + e2*I*np.sin(self.theta/2)
        self.t = 0.0*e1 + 0.3*e2 + 0.3*e3
        self.T = 1 + (1/2)*einf*self.t


        self.update_rbm()
        self.update_model()

    def update_rbm(self):
        self.R = np.cos(self.theta/2) + e2*I*np.sin(self.theta/2)
        self.T = 1 + (1/2)*einf*self.t
        pts = transform_numpy_cloud(self.pcd[0],self.R,self.t)
        self.noisy_pcd[1].points = o3d.utility.Vector3dVector(pts)
        self.pcd[1].points = o3d.utility.Vector3dVector(pts)
        self.update_model()

    def update_model(self):
        self.update_pc(0)
        self.update_pc(1)
        self.estimate_rbm()
        #self.draw_primitives(0)
        self.draw_primitives(1)

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
        pts = np.copy(np.asarray(self.pcd[j].points))
        noise = np.random.normal(self.mu,self.sigma,size=pts.shape)
        pts += noise
        self.noisy_pcd[j].points = o3d.utility.Vector3dVector(pts)
        self.scene.scene.remove_geometry("Point Cloud"+str(j))
        self.scene.scene.add_geometry("Point Cloud"+str(j), self.noisy_pcd[j], self.material)

    def run(self):
        gui.Application.instance.run()

    def on_key(self,e):
        if e.key == gui.KeyName.K:
            if e.type == gui.KeyEvent.UP:  # check UP so we default to DOWN
                self.sigma += self.sigma_iter
                print('sigma:',self.sigma)
                self.update_model()
        if e.key == gui.KeyName.J:
            if e.type == gui.KeyEvent.UP:  # check UP so we default to DOWN
                self.sigma -= self.sigma_iter
                if self.sigma < 0:
                    self.sigma = 0                
                self.update_model()

                print('sigma:',self.sigma)
        if e.key == gui.KeyName.Q:
            gui.Application.instance.quit()
        if e.key == gui.KeyName.SPACE:
            self.update_model()
        if e.key == gui.KeyName.C:
            self.scene.setup_camera(60, self.scene.scene.bounding_box, (0, 0, 0))

        if e.key == gui.KeyName.W:
            if e.type == gui.KeyEvent.UP:  # check UP so we default to DOWN
                self.t += 0.01*e1
                self.update_rbm()
                self.update_model()
        if e.key == gui.KeyName.S:
            if e.type == gui.KeyEvent.UP:  # check UP so we default to DOWN
                self.t -= 0.01*e1
                self.update_rbm()
                self.update_model()

        if e.key == gui.KeyName.A:
            if e.type == gui.KeyEvent.UP:  # check UP so we default to DOWN
                self.theta += 1/np.pi
                self.update_rbm()
                self.update_model()
        if e.key == gui.KeyName.D:
            if e.type == gui.KeyEvent.UP:  # check UP so we default to DOWN
                self.theta -= 1/np.pi
                self.update_rbm()
                self.update_model()

        


        return gui.Widget.EventCallbackResult.IGNORED
    
    def get_sphere(self,radius_sq,location):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=np.sqrt(abs(radius_sq)))
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
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=np.sqrt(abs(radius_sq)), height=0.0001)

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

    def rotate_pcd(self,j,R):
        self.pcd[j].rotate(R)
        self.noisy_pcd[j].rotate(R)


    def estimate_rbm(self):
        pts = np.asarray(self.noisy_pcd[0].points)
        # x_lst = initialize_vanilla_cloud(pts.tolist()[:self.n_points])
        x_lst = initialize_vanilla_cloud(pts.tolist())
        p_lst = vanilla_to_cga_vecs(x_lst)

        pts = np.asarray(self.noisy_pcd[1].points)
        # y_lst = initialize_vanilla_cloud(pts.tolist()[:self.n_points])
        y_lst = initialize_vanilla_cloud(pts.tolist())
        q_lst = vanilla_to_cga_vecs(y_lst)

        # Get the eigenbivectors
        P_lst,lambda_P = get_eigmvs(p_lst,grades=[1,2])
        Q_lst,lambda_Q = get_eigmvs(q_lst,grades=[1,2])

        orient_lst = get_orient_diff(P_lst,Q_lst,p_lst,q_lst)
        P_lst = apply_orientation(P_lst,orient_lst)
        T_est,R_est = estimate_rbm_1(P_lst,Q_lst)
        t_est = -2*eo|T_est 
        print_metrics(R_est,self.R,T_est,self.T)
        #print(-2*eo|T_est)
        #print(R_est)
        self.P_lst[0] = P_lst
        self.P_lst[1] = Q_lst
        
        
        self.pcd[2] = copy.deepcopy(self.pcd[0])
        pts = transform_numpy_cloud(self.pcd[0],R_est,t_est)
        self.pcd[2].points = o3d.utility.Vector3dVector(pts)
        color = np.array([[0.0,0.5,0.0]]*pts.shape[0])
        self.pcd[2].colors = o3d.utility.Vector3dVector(color)
        self.noisy_pcd[2] = copy.deepcopy(self.pcd[2])

        self.update_pc(2)


    def draw_primitives(self,j):
        self.remove_primitives(j)
        for i in range(len(self.P_lst[j])):
            d,l,radius_sq = get_properties(self.P_lst[j][i])
            d_array = np.array(d)
            if((d_array*d_array).sum() < 1E-12): # check if is sphere
                if self.draw_spheres:
                    primitive,material = self.get_sphere(radius_sq,l)
                    self.scene.scene.add_geometry(str(j)+'primitive'+str(i),primitive,material)
            else:
                primitive,material = self.get_circle(radius_sq,l,d)
                self.scene.scene.add_geometry(str(j)+'primitive'+str(i),primitive,material)
            
    def remove_primitives(self,j):
        for i in range(15):
            self.scene.scene.remove_geometry(str(j)+'primitive'+str(i))


if __name__ == '__main__':
    # pcd = o3d.io.read_point_cloud(f'/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper.ply')
    pcd = o3d.io.read_point_cloud(f'/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper_res2.ply')
    pts = np.asarray(pcd.points)
    # print(pts.tolist())
    # print(pts.shape)
    
    n_points = pts.shape[0]
    color = np.array([[0,0,0.5]]*n_points)
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcd1 = copy.deepcopy(pcd)
    color = np.array([[0.5,0,0]]*n_points)
    pcd1.colors = o3d.utility.Vector3dVector(color)


    viewer = PCViewer3D([pcd,pcd1],draw_spheres=False)
    viewer.run()
