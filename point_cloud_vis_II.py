# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform

from cga3d_estimation import *
import algorithms_eigmvs as algs
import multilinear_algebra as multiga

import sys
import copy
from matplotlib.pyplot import cm


isMacOS = (platform.system() == "Darwin")


class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        # self.bg_color = gui.Color(0.5, 0.5, 0.5)
        self.bg_color = gui.Color(1, 1, 1) # Default to white backgroud
        self.show_skybox = False
        self.show_axes = False
        self.use_ibl = True
        self.use_sun = True
        self.new_ibl_name = None  # clear to None after loading
        self.ibl_intensity = 45000
        self.sun_intensity = 45000
        self.sun_dir = [0.577, -0.577, -0.577]
        self.sun_color = gui.Color(1, 1, 1)

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.LIT]


        self.set_transparent_material([0.9, 0.9, 0.9])

    def set_transparent_material(self,color):
        self.transp_mat = o3d.visualization.rendering.MaterialRecord()

        self.transp_mat.shader = 'defaultLitTransparency'
        # self.transp_mat.shader = 'defaultLitSSR'
        self.transp_mat.base_color = color + [0.5]

        self.transp_mat.base_roughness = 1
        self.transp_mat.base_reflectance = 0.0
        self.transp_mat.base_clearcoat = 0
        self.transp_mat.thickness = 0
        self.transp_mat.transmission = 0
        self.transp_mat.absorption_distance = 0
        self.transp_mat.absorption_color = [0,0,0]


    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        assert (self.material.shader == Settings.LIT)
        prefab = Settings.PREFAB[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)

    def apply_lighting_profile(self, name):
        profile = Settings.LIGHTING_PROFILES[name]
        for key, val in profile.items():
            setattr(self, key, val)


class PointCloudSettings:
    '''Class for a single point cloud'''
    def __init__(self,pcd,id):
        self.translation_axis = np.array([1,2,4])
        self.translation_magnitude = 0
        self.rotation_angle = 0
        self.rotation_axis = np.array([1,0,0])
        self.pcd = copy.deepcopy(pcd)
        self.noisy_pcd = copy.deepcopy(pcd)
        self.rotation_matrix = np.eye(3)
        self.translation_vector = np.zeros(3)
        self.color = [0,0,0,1]
        self.id = id
        self.Rotor = vga3d.multivector([1],basis=['e'])
        self.Motor = vga3d.multivector([1],basis=['e'])
        self._draw_geometries = [False,False,True]
        self.arrow_scale = 1
        self.geometries = []
        self.ground_eigmvs = []
        self.eigmvs = []
        self.eigbivs = []
        self.eigvecs = []
        self.show = True

    def compute_transformation(self): 
        ''' Determines the rotation matrix and translation matrix'''
        translation_vector = self.translation_magnitude*self.translation_axis
        
        # Compute the rotor: Rotor = cos(angle) + I*axis*sin(angle)
        Rotor = axis_angle_to_rotor(self.rotation_axis,self.rotation_angle)
        rotation_matrix = rotor3d_to_matrix(Rotor)
        return translation_vector,rotation_matrix,Rotor

    def compute_motor(self):
        ''' Computes a motor from the axis angle and translation vector''' 
        t_vec = self.translation_magnitude*self.translation_axis
        Rotor = axis_angle_to_rotor(self.rotation_axis,self.rotation_angle)
        t = vga3d.multivector(t_vec.tolist(),grades=1)
        Translator = 1 + (1/2)*einf*t
        return Translator*Rotor

    def get_pcd_as_mvcloud(self):
        ''' Gets the noisy point cloud data as a multivector point cloud'''
        pts = self.get_pcd_as_nparray()
        return nparray_to_3dvga_vector_array(pts)

    def get_pcd_as_nparray(self):
        return np.asarray(self.noisy_pcd.points)
    
    def get_points_nbr(self):
        pts = self.get_pcd_as_nparray()
        return pts.shape[0]

    def update_noisy_pcd(self,scene,material):
        '''Update the point cloud from a given scene'''
        scene.remove_geometry("noisy_pcd_" + str(self.id))
        if self.show:
            material.base_color = self.color
            # self.compute_pcd_normals(i) # Do not update normals 
            scene.add_geometry("noisy_pcd_" + str(self.id), self.noisy_pcd,
                                                            material)
    def compute_pcd_normals(self):
        self.noisy_pcd.estimate_normals()
        self.noisy_pcd.normalize_normals()

    def update_gaussian_noise(self,sigma):
        pts = np.copy(np.asarray(self.pcd.points))
        noise = np.random.normal(0,sigma,size=pts.shape)
        pts += noise
        self.noisy_pcd.points = o3d.utility.Vector3dVector(pts)

    def update_axis_angle(self,trans_axis,trans_mag,rot_angle,rot_axis):
        self.rotation_angle = rot_angle
        self.rotation_axis = rot_axis
        self.translation_axis = trans_axis
        self.translation_magnitude = trans_mag

    


    def update_point_cloud_2(self,scene,material,transp_mat):
        '''Computes the rigid transformation and applies it to the points and the primitives
           Computes a motor and rotation matrix '''

        # t,R,Rotor = self.compute_transformation()
        Motor_new = self.compute_motor()
        t_new,R_new = motor_to_rotation_translation(Motor_new)

        # t_old = vga3d.multivector(self.translation_vector.tolist(),grades=1) # Convert to geometric algebra
        # Motor_old = (1+(1/2)*einf*t_old)*self.Rotor

        Motor_old = self.Motor
        # t_new = vga3d.multivector(t.tolist(),grades=1) # Convert to geometric algebra
        # Motor_new = (1+(1/2)*einf*t_new)*Rotor
        
        Motor = Motor_new*~Motor_old
        t,R = motor_to_rotation_translation(Motor)

        
        # Rotate and translate the point cloud data
        data = [self.pcd,self.noisy_pcd]
        for i in range(len(data)):
            data[i].rotate(R@self.rotation_matrix.T, center=np.array([0, 0, 0]))
            data[i].translate(-self.translation_vector)
            data[i].rotate(R@self.rotation_matrix.T, center=np.array([0, 0, 0]))
            data[i].translate(t)
        
        # # Apply the rigid transformation to the eigenmultivectors using the motor
        # for i in range(len(self.eigmvs)):
        #     self.eigmvs[i] = Motor*self.eigmvs[i]*~Motor

        # Apply the rigid transformation to the eigenbivectors using the motor
        for i in range(len(self.eigbivs)):
            self.eigbivs[i] = Motor*self.eigbivs[i]*~Motor

        # Apply the rigid transformation to the eigenvectors using the motor
        for i in range(len(self.eigvecs)):
            self.eigvecs[i] = Motor*self.eigvecs[i]*~Motor

        self.translation_vector = t_new
        self.rotation_matrix = R_new
        self.Motor = Motor_new
        _,self.Rotor = decompose_motor(Motor)

        self.redraw_geometries(scene,material,transp_mat)

    def update_point_cloud(self):
        '''Computes the rigid transformation (Motor) and applies it to the points and the primitives
           Computes a Motor. Applies the new transformation composed with the adjoint of the old.'''

        Motor = self.compute_motor()
        self.apply_motor(Motor*~self.Motor)


    def redraw_geometries(self,scene,material,transp_mat):
        ''' Redraws the point cloud and its geometries'''
        self.compute_geometry_from_eigmvs() # Recompute geometries
        self.draw_geometries(scene,material,transp_mat)
        self.update_noisy_pcd(scene,material)
    
    def apply_motor(self,Motor):
        ''' Applies a rigid transformation to the point clouds and the eigenmultivectors '''
        t_vec,R_matrix = motor_to_rotation_translation(Motor) # Get rotation matrix and translation

        # Apply the rigid transformation to the eigenbivectors using the motor
        for i in range(len(self.eigbivs)):
            self.eigbivs[i] = Motor*self.eigbivs[i]*~Motor

        # Apply the rigid transformation to the eigenvectors using the motor
        for i in range(len(self.eigvecs)):
            self.eigvecs[i] = Motor*self.eigvecs[i]*~Motor

        data = [self.pcd,self.noisy_pcd]
        for i in range(len(data)):
            data[i].rotate(R_matrix,center=np.array([0,0,0])) # Do not forget the center!!!
            data[i].translate(t_vec)

        # Important to always 'normalise' the motor, otherwise the values tend to explode
        self.Motor = project_motor(Motor*self.Motor) # Project to the motor manifold

    

    def compute_eigmvs(self):
        '''Computes the eigenmultivectors associated to the noisy point cloud'''
        x_pts = np.asarray(self.noisy_pcd.points)
    
        # Convert numpy array to multivector array
        x = nparray_to_3dvga_vector_array(x_pts)

        # Extend the points via the conformal mapping
        p = conformal_mapping(x)

        # Compute the eigenmultivectors associated with the points p
        self.eigmvs,self.eigvalues = get_3dcga_eigmvs(p)

        # disambiguate the sign of the eigenmultivectors 
        self.mvref = compute_reference(p)
        for i in range(len(self.eigmvs)):
            self.eigmvs[i] *= np.sign((self.eigmvs[i]*self.mvref)(0))

        # Separate the eigenmultivectors into bivector and vector
        self.eigbivs,self.eigvecs = separate_grades(self.eigmvs)
            

    def compute_geometry_from_eigmvs(self):
        self.geometries = [0]*3 # [sphere,circle,arrow]
        # The boolean that decides which material are transparent
        self._transp_mat = [0]*3 # [sphere,circle,arrow]
        
        # A different list for each
        for i in range(len(self.geometries)):
            self.geometries[i] = []
            self._transp_mat[i] = []

        # Build the geometries for the spheres
        for i in range(len(self.eigvecs)):
            d,l,radius_sq = get_properties(self.eigvecs[i])
            d_array = np.array(d)
            sphere = self.get_sphere(radius_sq,l)
            self.geometries[0] += [sphere]

        # Build the geometries for the arrows and for the circles
        for i in range(len(self.eigbivs)):
            d,l,radius_sq = get_properties(self.eigbivs[i])
            A,B,C,D = get_coeffs(self.eigbivs[i]) # compute the arrows from the circles
            arrow = self.get_arrow(np.array(A.tolist(1)[0][:3]),scale=self.arrow_scale)
            circle_geometries = self.create_circle(np.sqrt(abs(radius_sq)),l,d)
            
            self.geometries[1] += circle_geometries
            self.geometries[2] += [arrow]
            self._transp_mat[1] += [True,True,False] # Solid 2D torus, transparent circles

        self._transp_mat[0] = [True]*len(self.geometries[0]) # Transparent spheres
        self._transp_mat[2] = [False]*len(self.geometries[2]) # Solid arrows

    def draw_geometries(self,scene,material,transp_mat):
        material.base_color = self.color
        # print(self.color[:2])
        # print(type(self.color))

        transp_mat.base_color = np.r_[self.color[:3],0.5]
        # transp_mat.base_color[3] = 0.5 # Set alpha to 0.5
        self.remove_geometries(scene)
        if self.show:
            for i in range(len(self.geometries)):
                if self._draw_geometries[i]:
                    for j in range(len(self.geometries[i])):
                        _material = transp_mat if self._transp_mat[i][j] else material # check if material is transparent
                        scene.add_geometry("primitive_"+str(self.id)+'_'+str(i)+'_'+str(j),self.geometries[i][j],_material)
        
    def remove_geometries(self,scene):
        for i in range(len(self.geometries)):
            for j in range(len(self.geometries[i])):
                scene.remove_geometry("primitive_"+str(self.id)+'_'+str(i)+'_'+str(j))
    
    def update_axis_angle_from_motor(self):
        t,a,theta = motor_to_axis_angle(self.Motor)
        t_normalized = pyga.normalize_mv(t)

        self.translation_vector = np.array(t.tolist(1)[0][:3])
        self.translation_axis = np.array(t_normalized.tolist(1)[0][:3])
        self.translation_magnitude = (t_normalized|t)(0)
        self.rotation_axis = np.array(a.tolist(1)[0][:3])
        self.rotation_angle = theta*2
        print("trans vector",t)
        print("rot axis",a)
        print("rot angle",theta)

    def update_geometries(self,scene,material,transp_mat):
        self.compute_eigmvs()
        self.compute_geometry_from_eigmvs()
        self.draw_geometries(scene,material,transp_mat)

    def get_sphere(self,radius_sq,location):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=np.sqrt(abs(radius_sq)),resolution=20)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.7, 0.1, 0.1])  # To be changed to the point color.
        sphere = sphere.translate(location)
        
        return sphere

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
        thickness = 0.01
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

class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"

    MATERIAL_NAMES = ["Lit", "Unlit", "Normals", "Depth"]
    MATERIAL_SHADERS = [
        Settings.LIT, Settings.UNLIT, Settings.NORMALS, Settings.DEPTH
    ]

    def __init__(self, width, height):
        self.settings = Settings()
        self.benchmark_init()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        self.window = gui.Application.instance.create_window(
            "Open3D", width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)

        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        self._settings_panel.background_color = gui.Color(0.5, 0.5, 0.5, 0.5) # Set the widget to transparent

        # Create a collapsible vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.
        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        view_ctrls.set_is_open(False)

        self._arcball_button = gui.Button("Arcball")
        self._arcball_button.horizontal_padding_em = 0.5
        self._arcball_button.vertical_padding_em = 0
        self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
        self._fly_button = gui.Button("Fly")
        self._fly_button.horizontal_padding_em = 0.5
        self._fly_button.vertical_padding_em = 0
        self._fly_button.set_on_clicked(self._set_mouse_mode_fly)
        self._model_button = gui.Button("Model")
        self._model_button.horizontal_padding_em = 0.5
        self._model_button.vertical_padding_em = 0
        self._model_button.set_on_clicked(self._set_mouse_mode_model)
        self._sun_button = gui.Button("Sun")
        self._sun_button.horizontal_padding_em = 0.5
        self._sun_button.vertical_padding_em = 0
        self._sun_button.set_on_clicked(self._set_mouse_mode_sun)
        self._ibl_button = gui.Button("Environment")
        self._ibl_button.horizontal_padding_em = 0.5
        self._ibl_button.vertical_padding_em = 0
        self._ibl_button.set_on_clicked(self._set_mouse_mode_ibl)
        view_ctrls.add_child(gui.Label("Mouse controls"))
        # We want two rows of buttons, so make two horizontal layouts. We also
        # want the buttons centered, which we can do be putting a stretch item
        # as the first and last item. Stretch items take up as much space as
        # possible, and since there are two, they will each take half the extra
        # space, thus centering the buttons.
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._arcball_button)
        h.add_child(self._fly_button)
        h.add_child(self._model_button)
        h.add_stretch()
        view_ctrls.add_child(h)
        h = gui.Horiz(0.25 * em)  # row 2
        h.add_stretch()
        h.add_child(self._sun_button)
        h.add_child(self._ibl_button)
        h.add_stretch()
        view_ctrls.add_child(h)

        self._show_skybox = gui.Checkbox("Show skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_skybox)

        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        view_ctrls.add_child(grid)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_axes)

        self._profiles = gui.Combobox()
        for name in sorted(Settings.LIGHTING_PROFILES.keys()):
            self._profiles.add_item(name)
        self._profiles.add_item(Settings.CUSTOM_PROFILE_NAME)
        self._profiles.set_on_selection_changed(self._on_lighting_profile)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(gui.Label("Lighting profiles"))
        view_ctrls.add_child(self._profiles)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)

        advanced = gui.CollapsableVert("Advanced lighting", 0,
                                       gui.Margins(em, 0, 0, 0))
        advanced.set_is_open(False)

        self._use_ibl = gui.Checkbox("HDR map")
        self._use_ibl.set_on_checked(self._on_use_ibl)
        self._use_sun = gui.Checkbox("Sun")
        self._use_sun.set_on_checked(self._on_use_sun)
        advanced.add_child(gui.Label("Light sources"))
        h = gui.Horiz(em)
        h.add_child(self._use_ibl)
        h.add_child(self._use_sun)
        advanced.add_child(h)

        self._ibl_map = gui.Combobox()
        for ibl in glob.glob(gui.Application.instance.resource_path +
                             "/*_ibl.ktx"):
            self._ibl_map.add_item(os.path.basename(ibl[:-8]))
        self._ibl_map.selected_text = AppWindow.DEFAULT_IBL
        self._ibl_map.set_on_selection_changed(self._on_new_ibl)
        self._ibl_intensity = gui.Slider(gui.Slider.INT)
        self._ibl_intensity.set_limits(0, 200000)
        self._ibl_intensity.set_on_value_changed(self._on_ibl_intensity)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("HDR map"))
        grid.add_child(self._ibl_map)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._ibl_intensity)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Environment"))
        advanced.add_child(grid)

        self._sun_intensity = gui.Slider(gui.Slider.INT)
        self._sun_intensity.set_limits(0, 200000)
        self._sun_intensity.set_on_value_changed(self._on_sun_intensity)
        self._sun_dir = gui.VectorEdit()
        self._sun_dir.set_on_value_changed(self._on_sun_dir)
        self._sun_color = gui.ColorEdit()
        self._sun_color.set_on_value_changed(self._on_sun_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._sun_intensity)
        grid.add_child(gui.Label("Direction"))
        grid.add_child(self._sun_dir)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._sun_color)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Sun (Directional light)"))
        advanced.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(advanced)

        material_settings = gui.CollapsableVert("Material settings", 0,
                                                gui.Margins(em, 0, 0, 0))
        material_settings.set_is_open(False)

        self._shader = gui.Combobox()
        self._shader.add_item(AppWindow.MATERIAL_NAMES[0])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[1])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[2])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[3])
        self._shader.set_on_selection_changed(self._on_shader)
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(self._on_material_prefab)
        # self._material_color = gui.ColorEdit()
        # self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(self._shader)
        grid.add_child(gui.Label("Material"))
        grid.add_child(self._material_prefab)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)

        rts_settings = gui.CollapsableVert("Rigid Transformation", 0,
                                                gui.Margins(em, 0, 0, 0))

        self._rot_axis = gui.VectorEdit()
        self._rot_axis.background_color = gui.Color(0.5, 0.5, 0.5, 0.5)
        self._rot_axis.set_on_value_changed(self._on_rot_axis)

        self._rot_angle = gui.Slider(gui.Slider.INT)
        self._rot_angle.set_limits(0,360)
        self._rot_angle.set_on_value_changed(self._on_rot_angle)

        self._trans_axis = gui.VectorEdit()
        self._trans_axis.background_color = gui.Color(0.5, 0.5, 0.5, 0.5)
        self._trans_axis.set_on_value_changed(self._on_trans_axis)

        self._trans_mag = gui.Slider(gui.Slider.DOUBLE)
        self._trans_mag.set_limits(0.0,1.0)
        self._trans_mag.background_color = gui.Color(0.5, 0.5, 0.5, 0.5)
        self._trans_mag.set_on_value_changed(self._on_trans_mag)

        self._gaussian_noise = gui.Slider(gui.Slider.DOUBLE)
        self._gaussian_noise.set_limits(0.0,0.05)
        self._gaussian_noise.background_color = gui.Color(0.5, 0.5, 0.5, 0.5)
        self._gaussian_noise.set_on_value_changed(self._on_gaussian_noise)
        self._gaussian_noise.double_value = self.sigma
        
        self._point_cloud = gui.Combobox()
        self._point_cloud.set_on_selection_changed(self._on_point_cloud)

        self._point_cloud_color = gui.ColorEdit()
        self._point_cloud_color.set_on_value_changed(self._on_point_cloud_color)

        self._increase_arrows = gui.Button("Increase")
        self._decrease_arrows = gui.Button("Decrease")
        self._increase_arrows.set_on_clicked(self._on_increase_arrows)
        self._decrease_arrows.set_on_clicked(self._on_decrease_arrows)

        self._update_primitives = gui.Button("Update Primitives")
        self._update_primitives.set_on_clicked(self._on_update_primitives)
        
        self._show_point_cloud = gui.Checkbox("Point Cloud")
        self._show_point_cloud.set_on_checked(self._on_show_point_cloud)

        self._est_transformation = gui.Button("Est. Motor")
        self._est_transformation.set_on_clicked(self._on_est_transformation)

        self._choose_algorithm = gui.Combobox()
        self.alg_list,self.alg_names = algs.get_algorithms()
        for i in range(len(self.alg_names)):
            self._choose_algorithm.add_item(self.alg_names[i])
        self._choose_algorithm.set_on_selection_changed(self._on_choose_algorithm)
        self.algorithm = self.alg_list[0]

        self._draw_spheres = gui.Button("Spheres")
        self._draw_circles = gui.Button("Circles")
        self._draw_arrows = gui.Button("Arrows")
        self._draw_spheres.set_on_clicked(self._on_draw_spheres)
        self._draw_circles.set_on_clicked(self._on_draw_circles)
        self._draw_arrows.set_on_clicked(self._on_draw_arrows)
        self._draw_spheres.toggleable = True
        self._draw_circles.toggleable = True
        self._draw_arrows.toggleable = True
        self._draw_spheres.is_on = self._draw_primitives[0]
        self._draw_circles.is_on = self._draw_primitives[1]
        self._draw_arrows.is_on = self._draw_primitives[2]

        # Use tab control as a ghost widget
        # grid.add_child(gui.TabControl())
        
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Point Cloud"))
        grid.add_child(self._point_cloud)
        grid.add_child(gui.Label("Point Cloud Color"))
        grid.add_child(self._point_cloud_color)
        grid.add_child(gui.Label("Rotation Axis"))
        grid.add_child(self._rot_axis)
        grid.add_child(gui.Label("Translation Axis"))
        grid.add_child(self._trans_axis)
        grid.add_child(gui.Label("Rotation Angle"))
        grid.add_child(self._rot_angle)
        grid.add_child(gui.Label("Translation Mag."))
        grid.add_child(self._trans_mag)
        grid.add_child(gui.Label("Gaussian Noise"))
        grid.add_child(self._gaussian_noise)
        grid.add_child(self._update_primitives)
        grid.add_child(self._est_transformation)
        grid.add_child(self._show_point_cloud)
        grid.add_child(gui.Label("   ")) # ghost widget
        grid.add_child(gui.Label("Algorithms"))
        grid.add_child(self._choose_algorithm)



        rts_settings.add_child(grid)

        grid = gui.VGrid(3,0.25*em)
        grid.add_child(self._draw_spheres)
        grid.add_child(self._draw_circles)
        grid.add_child(self._draw_arrows)
        grid.add_child(gui.Label("Arrows"))
        grid.add_child(self._increase_arrows)
        grid.add_child(self._decrease_arrows)

        rts_settings.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)
        self._settings_panel.add_child(rts_settings)

        

        # ----

        # Normally our user interface can be children of all one layout (usually
        # a vertical layout), which is then the only child of the window. In our
        # case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout
        # callback. The on_layout callback should set the frame
        # (position + size) of every child correctly. After the callback is
        # done the window will layout the grandchildren.
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current Image...", AppWindow.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Lighting & Materials",
                                   AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

        self._apply_settings()
        
    

    def benchmark_init(self):
        self.sigma = 0
        self.translation_axis = np.zeros(3)
        self.translation_magnitude = 0
        self.rotation_angle = 0
        self.rotation_axis = np.array([1,0,0])
        self.point_clouds = []
        self.pcd_index = 0
        self._draw_primitives = [True,False,True] # Default values (spheres,circles,arrows)
        self.target_idx = 0
        self.source_idx = 1

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(
                self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity)
        sun_color = [
            self.settings.sun_color.red, self.settings.sun_color.green,
            self.settings.sun_color.blue
        ]
        self._scene.scene.scene.set_sun_light(self.settings.sun_dir, sun_color,
                                              self.settings.sun_intensity)
        self._scene.scene.scene.enable_sun_light(self.settings.use_sun)

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._use_ibl.checked = self.settings.use_ibl
        self._use_sun.checked = self.settings.use_sun
        self._ibl_intensity.int_value = self.settings.ibl_intensity
        self._sun_intensity.int_value = self.settings.sun_intensity
        self._sun_dir.vector_value = self.settings.sun_dir
        self._sun_color.color_value = self.settings.sun_color
        self._material_prefab.enabled = (
            self.settings.material.shader == Settings.LIT)
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])
        # self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_fly(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _set_mouse_mode_sun(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_use_ibl(self, use):
        self.settings.use_ibl = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_use_sun(self, use):
        self.settings.use_sun = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_lighting_profile(self, name, index):
        if name != Settings.CUSTOM_PROFILE_NAME:
            self.settings.apply_lighting_profile(name)
            self._apply_settings()

    def _on_new_ibl(self, name, index):
        self.settings.new_ibl_name = gui.Application.instance.resource_path + "/" + name
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_ibl_intensity(self, intensity):
        self.settings.ibl_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_intensity(self, intensity):
        self.settings.sun_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_dir(self, sun_dir):
        self.settings.sun_dir = sun_dir
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_color(self, color):
        self.settings.sun_color = color
        self._apply_settings()

    def _on_shader(self, name, index):
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_material_prefab(self, name, index):
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    # def _on_material_color(self, color):
    #     self.settings.material.base_color = [
    #         color.red, color.green, color.blue, color.alpha
    #     ]
    #     self.settings.apply_material = True
    #     self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()
    


    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Open3D GUI Example"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    
    def _on_rot_axis(self, vec):
        if (vec*vec).sum() == 0:
            self.rotation_axis = np.array([1,0,0])
        else:
            self.rotation_axis = vec/np.sqrt((vec*vec).sum()) # Normalize to unity
        self.point_clouds[self.pcd_index].update_axis_angle(self.translation_axis,self.translation_magnitude,self.rotation_angle,self.rotation_axis)
        self.point_clouds[self.pcd_index].update_point_cloud()
        self.point_clouds[self.pcd_index].redraw_geometries(self._scene.scene,self.settings.material,self.settings.transp_mat)
        
    def _on_trans_axis(self, vec):
        self.translation_axis = vec
        self.point_clouds[self.pcd_index].update_axis_angle(self.translation_axis,self.translation_magnitude,self.rotation_angle,self.rotation_axis)
        self.point_clouds[self.pcd_index].update_point_cloud()
        self.point_clouds[self.pcd_index].redraw_geometries(self._scene.scene,self.settings.material,self.settings.transp_mat)

    def _on_rot_angle(self,value):
        self.rotation_angle = value/180*np.pi # convert to radians
        self.point_clouds[self.pcd_index].update_axis_angle(self.translation_axis,self.translation_magnitude,self.rotation_angle,self.rotation_axis)
        self.point_clouds[self.pcd_index].update_point_cloud()
        self.point_clouds[self.pcd_index].redraw_geometries(self._scene.scene,self.settings.material,self.settings.transp_mat)
    
    def _on_trans_mag(self,value):
        self.translation_magnitude = value
        self.point_clouds[self.pcd_index].update_axis_angle(self.translation_axis,self.translation_magnitude,self.rotation_angle,self.rotation_axis)
        self.point_clouds[self.pcd_index].update_point_cloud()
        self.point_clouds[self.pcd_index].redraw_geometries(self._scene.scene,self.settings.material,self.settings.transp_mat)

    def _on_gaussian_noise(self,value):
        self.sigma = value
        for i in range(len(self.point_clouds)):
            self.point_clouds[i].update_gaussian_noise(self.sigma)
            self.point_clouds[i].update_noisy_pcd(self._scene.scene,self.settings.material)

    def _on_update_primitives(self):
        for i in range(len(self.point_clouds)):
            self.point_clouds[i].update_geometries(self._scene.scene,self.settings.material,self.settings.transp_mat)

        # for i in range(len(self.point_clouds[0].eigmvs)):
        #     print(pyga.numpy_max(self.point_clouds[0].eigmvs[i] - self.point_clouds[1].eigmvs[i]))

    def update_primitives_toggle(self):
        for i in range(len(self.point_clouds)):
            self.point_clouds[i]._draw_geometries = self._draw_primitives
            self.point_clouds[i].draw_geometries(self._scene.scene,self.settings.material,self.settings.transp_mat)

    def _on_draw_spheres(self):
        self._draw_primitives[0] = self._draw_spheres.is_on
        self.update_primitives_toggle()

    def _on_draw_circles(self):
        self._draw_primitives[1] = self._draw_circles.is_on
        self.update_primitives_toggle()

    def _on_draw_arrows(self):
        self._draw_primitives[2] = self._draw_arrows.is_on
        self.update_primitives_toggle()

    def update_gui_variables(self,index):
        # Update the values of the variables
        self.rotation_axis = self.point_clouds[index].rotation_axis
        self.rotation_angle = self.point_clouds[index].rotation_angle
        self.translation_magnitude = self.point_clouds[index].translation_magnitude
        self.translation_axis = self.point_clouds[index].translation_axis
        
        # Update the values of the gui elements
        self._rot_axis.vector_value = self.point_clouds[index].rotation_axis
        self._rot_angle.double_value = self.point_clouds[index].rotation_angle/np.pi*180
        self._trans_mag.double_value = self.point_clouds[index].translation_magnitude
        self._trans_axis.vector_value = self.point_clouds[index].translation_axis 
        color = self.point_clouds[index].color
        # Set the gui color to the color of the selected point cloud
        self._point_cloud_color.color_value = gui.Color(color[0],color[1],color[2],color[3])
        # set the checkbox to the value of the selected point cloud
        self._show_point_cloud.checked = self.point_clouds[index].show

    def update_gui(self):
        ''' Updates the variables and the values of the gui, with respect to the selected point cloud'''
        self.update_gui_variables(self.pcd_index)

    def _on_point_cloud(self,name,index):
        self.pcd_index = index
        self.update_gui()

    def _on_show_point_cloud(self,checked):
        '''Redraw the point cloud and its corresponding geometries'''
        if self.pcd_index < len(self.point_clouds):
            self.point_clouds[self.pcd_index].show = checked
            self.point_clouds[self.pcd_index].update_noisy_pcd(self._scene.scene,self.settings.material)
            self.point_clouds[self.pcd_index].draw_geometries(self._scene.scene,self.settings.material,self.settings.transp_mat)
    
    def _on_est_transformation(self):
        if len(self.point_clouds) >= 2:
            for i in [self.source_idx,self.target_idx]: # update the eigenmultivectors of the source and target point clouds
                self.point_clouds[i].update_geometries(self._scene.scene,self.settings.material,self.settings.transp_mat)
            M_est = self.algorithm(self.point_clouds[self.source_idx],self.point_clouds[self.target_idx]) # Use the selected algorithm to determine the motor
            
            self.point_clouds[self.source_idx].apply_motor(M_est) # Apply transformation to the source point cloud
            self.point_clouds[self.source_idx].update_axis_angle_from_motor()
            if self.source_idx == self.pcd_index: # Update gui when the source point cloud is selected 
                self.update_gui_variables(self.source_idx)
            self.point_clouds[self.source_idx].redraw_geometries(self._scene.scene,self.settings.material,self.settings.transp_mat)

    def _on_choose_algorithm(self,name,index):
        self.algorithm = self.alg_list[index] 

    def _on_point_cloud_color(self,color):
        self.point_clouds[self.pcd_index].color = [ 
            color.red, color.green, color.blue, color.alpha ]
        self.point_clouds[self.pcd_index].update_noisy_pcd(self._scene.scene,self.settings.material)
        self.point_clouds[self.pcd_index].draw_geometries(self._scene.scene,self.settings.material,self.settings.transp_mat)

    def _on_increase_arrows(self):
        for i in range(len(self.point_clouds)):
            self.point_clouds[i].arrow_scale *= 1.1
            self.point_clouds[i].redraw_geometries(self._scene.scene,self.settings.material,self.settings.transp_mat)

    def _on_decrease_arrows(self):
        for i in range(len(self.point_clouds)):
            self.point_clouds[i].arrow_scale /= 1.1
            self.point_clouds[i].redraw_geometries(self._scene.scene,self.settings.material,self.settings.transp_mat)
    


    def load(self, path):
        ''' Reads a point cloud from path and creates two PointCloudSettings classes'''

        self._scene.scene.clear_geometry()
        pcd = o3d.io.read_point_cloud(path)
        
        # Put the center of mass of the dataset at the origin
        pts = np.asarray(pcd.points)
        ceofm = pts.mean(axis=0)
        pcd.points = o3d.utility.Vector3dVector(pts - ceofm) 

        n_pcds = 2
        self.point_clouds = []

        # Default colors for the point clouds
        # color = cm.rainbow(np.linspace(0, 1, n_pcds+1))
        color = [np.array([0,139,29,255])/255, np.array([0,39,255,255])/255]

        self._point_cloud.clear_items()
        # Add gaussian noise and update the point clouds
        for i in range(n_pcds):
            point_cloud = PointCloudSettings(pcd,i)
            point_cloud.compute_pcd_normals()
            point_cloud.update_gaussian_noise(self.sigma)
            point_cloud.color = color[i]
            point_cloud._draw_geometries = self._draw_primitives
            point_cloud.show = True
            point_cloud.update_noisy_pcd(self._scene.scene,self.settings.material)
            self.point_clouds += [point_cloud]
            self._point_cloud.add_item(str(i))
        
        self.update_gui()

        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())


    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)


def main():
    # We need to initialize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = AppWindow(1024, 768)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()