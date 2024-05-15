from cga3d_estimation import *

import pasta3d
import open3d as o3d

import sdrsac
from dcp_model import *

from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE;

import scipy.special as sp

import teaser

# import numpy as np

'''
This script is used to declare multiple algorithms used to estimate rigid transformations between point clouds x and y. Note that we have a similar script in algorithms_pcviz.py
though that script is to be used in the context of the visualizer.

In the algorithms for estimating transformations we use the docstring to name
the function when printing results of the algorithms. Thus the docstring has to be small
yet descriptive enough regarding wht the algorithm does. 
Also it must not have newline characters, that is it must be written on a single line.
Use the first line of the docstring to get the name of the algorithm. The rest of the algorithm description can 
stay a line bellow of the name.

# Can also use lambda expression to declare functions
F = lambda X: ((x-x_bar)*X*(x-x_bar)).sum()
G = lambda X: ((y-y_bar)*X*(y-y_bar)).sum()

'''


def get_algorithm_name(algorithm):
    return algorithm.__doc__.split("\n")[0]

def compute_references(p,q):
    p_ref = p.sum()
    p_ref /= (p_ref|einf)(0)

    q_ref = q.sum()
    q_ref /= (q_ref|einf)(0)

    P_ref = (1 + ii)*einf^(1+ p_ref)
    Q_ref = (1 + ii)*einf^(1+ q_ref)

    return P_ref,Q_ref


def estimate_transformation_pasta(x,y,npoints):
    '''PASTA-3D'''
    x_array = cga3d_vector_array_to_nparray(x)
    y_array = cga3d_vector_array_to_nparray(y)

    R_matrix, t_vec = pasta3d.pasta3d_rototranslation(y_array, x_array, 'max')
    T_est, R_est = rotation_translation_to_translator_rotator(R_matrix,t_vec)

    return (T_est,R_est)

eye4 = np.eye(4)
icp_dist_threshold = 100

def __estimate_transformation_ICP__(x_array,y_array):

    x_pcd = o3d.geometry.PointCloud()
    x_pcd.points = o3d.utility.Vector3dVector(x_array)
    y_pcd = o3d.geometry.PointCloud()
    y_pcd.points = o3d.utility.Vector3dVector(y_array)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        x_pcd, y_pcd, icp_dist_threshold, eye4,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    H = reg_p2p.transformation
    t_vec = H[0:3,3]
    R_matrix = H[0:3,0:3]

    return rotation_translation_to_translator_rotator(R_matrix,t_vec)

def estimate_transformation_ICP(x,y,npoints):
    '''ICP'''
    x_array = cga3d_vector_array_to_nparray(x)
    y_array = cga3d_vector_array_to_nparray(y)

    return __estimate_transformation_ICP__(x_array,y_array)


def __estimate_transformation_CGA__(P_biv,P_vec,P_ref,Q_biv,Q_vec,Q_ref):

    s = 2 # Consider the two eigenbivectors with the smallest eigenvalue

    P_biv = mv.concat(P_biv[s:])
    Q_biv = mv.concat(Q_biv[s:])

    # Scale the vectors 
    Q_vec[0] /= (Q_vec[0]*Q_ref)(0)
    P_vec[0] /= (P_vec[0]*P_ref)(0)

    # Computes the optimal rotor from the first coefficients of P and of Q
    P1,P2,P3,P4 = get_coeffs(P_biv)
    Q1,Q2,Q3,Q4 = get_coeffs(Q_biv)
    # define the rotor valued function
    def Func(Rotor):
        return (Q1*Rotor*~P1 + ~Q1*Rotor*P1).sum()

    basis,rec_basis = get_3dvga_rotor_basis()
    R_lst,lambda_R = multiga.symmetric_eigen_decomp(Func,basis,rec_basis)
    R_est = R_lst[3] # Chose the eigenrotor with the biggest eigenvalue

    # Use the first eigenvector to determine the translation
    T_est = exact_translation(R_est*P_vec[0]*~R_est,Q_vec[0])

    return (T_est,R_est)



def estimate_transformation_CGA(x,y,npoints):
    '''CGA-EVD
        Estimates the rigid body motion between two point clouds using the eigenmultivector.
        From the eigenmultivectors we estimate the rotation and translation. 
        To estimate the sign we use references from the PCs directly.
    '''
    s = 2 # Consider the two eigenbivectors with the smallest eigenvalue
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenvectors    
    P_vec,lambda_Pvec = get_3dcga_eigmvs(p,grades=1)
    Q_vec,lambda_Qvec = get_3dcga_eigmvs(q,grades=1)
    
    # Get the eigenbivectors and respective eigenvalues. The eigenvalues are ordered from smallest to biggest
    P_biv,lambda_Pbiv = get_3dcga_eigmvs(p,grades=2)
    Q_biv,lambda_Qbiv = get_3dcga_eigmvs(q,grades=2)

    P_ref = compute_reference(p)
    Q_ref = compute_reference(q)

    for i in range(len(P_biv)):
        P_biv[i] *= np.sign((P_biv[i]*P_ref)(0))
        Q_biv[i] *= np.sign((Q_biv[i]*Q_ref)(0))

    T_est,R_est = __estimate_transformation_CGA__(P_biv,P_vec,P_ref,Q_biv,Q_vec,Q_ref)

    return (T_est,R_est)

eps = 1e-12
def estimate_transformation_VGA(x,y,npoints):
    '''VGA-EVD
        Estimates the translation using the center of mass of the point clouds.
        Solves an eigenvalue problem to extract rotation invariant eigenvectors from each point cloud.
    '''

    basis,rec_basis = get_3dvga_basis(1)
    
    # Determine centers of mass
    x_bar = x.sum()/npoints
    y_bar = y.sum()/npoints
    
    # Compute the eigendecomposition
    F = multiga.get_reflections_function(x-x_bar)
    G = multiga.get_reflections_function(y-y_bar)
    P_lst,lambda_P = multiga.symmetric_eigen_decomp(F,basis,rec_basis)
    Q_lst,lambda_Q = multiga.symmetric_eigen_decomp(G,basis,rec_basis)

    # print("Eigenvalue Check (VGA) G:",multiga.check_eigenvalues(G,Q_lst,lambda_Q))
    # print("Eigenvalue Check (VGA) F:",multiga.check_eigenvalues(F,P_lst,lambda_P))
    
    # Correct the sign of the eigenvectors
    for i in range(len(P_lst)):
        P_lst[i] = P_lst[i]*correct_sign_2(P_lst[i],x-x_bar)
        Q_lst[i] = Q_lst[i]*correct_sign_2(Q_lst[i],y-y_bar)

    # If the eigenvectors are equal or almost equal
    max_val0 = max(pyga.numpy_max(Q_lst[0]),pyga.numpy_max(P_lst[0]))
    max_val1 = max(pyga.numpy_max(Q_lst[1]),pyga.numpy_max(P_lst[1]))
    diff0 = pyga.numpy_max(Q_lst[0]- P_lst[0])
    diff1 = pyga.numpy_max(Q_lst[1]- P_lst[1])

    if diff0 < max_val0*1e-20 or diff1 < max_val1*1e-20:
        R_est = vga3d.mvarray([1],basis=[1]) 
    else:
        R_est = ~exact_rotation(Q_lst[0],Q_lst[1],P_lst[0],P_lst[1])
    T_est = translation_from_cofm(y,x,R_est,npoints)

    return (T_est,R_est)

def estimate_transformation_sdrsac(x,y,npoints):
    '''SDRSAC'''

    x_array = cga3d_vector_array_to_nparray(x)
    y_array = cga3d_vector_array_to_nparray(y)

    R_matrix, t_vec = sdrsac.sdrsac(x_array, y_array)

    T_est, R_est = rotation_translation_to_translator_rotator(R_matrix,t_vec)

    return (T_est,R_est)


def __estimate_transformation_dcp__(x_nparray,y_nparray):
    x_torch = torch.from_numpy(x_nparray).unsqueeze(2).T.float()
    y_torch = torch.from_numpy(y_nparray).unsqueeze(2).T.float()

    rotation_ab_pred, translation_ab_pred, _, _ = net(x_torch, y_torch)

    R_matrix = rotation_ab_pred.detach().numpy()[0]
    t_vec = translation_ab_pred.detach().numpy()[0]


    T_est,R_est = rotation_translation_to_translator_rotator(R_matrix,t_vec)

    return (T_est,R_est)


def estimate_transformation_dcp(x,y,npoints):
    '''DCP'''

    every_k_points = 10

    x_nparray = cga3d_vector_array_to_nparray(x)
    y_nparray = cga3d_vector_array_to_nparray(y)

    x_pcd = o3d.geometry.PointCloud()
    x_pcd.points = o3d.utility.Vector3dVector(x_nparray)

    y_pcd = o3d.geometry.PointCloud()
    y_pcd.points = o3d.utility.Vector3dVector(y_nparray)

    x_pcd = x_pcd.uniform_down_sample(every_k_points)
    y_pcd = y_pcd.uniform_down_sample(every_k_points)

    x_nparray = np.asarray(x_pcd.points)
    y_nparray = np.asarray(y_pcd.points)

    return __estimate_transformation_dcp__(x_nparray,y_nparray)

def numpy_to_icp_point3D(array):
    plist = array.tolist();
    p3dlist = [];
    for x,y,z in plist:
        pt = POINT3D(x,y,z);
        p3dlist.append(pt);
    return p3dlist;

def __estimate_transformation_GOICP__(y_array,x_array):
    goicp = GoICP();
    
    goicp.loadModelAndData(len(x_array), numpy_to_icp_point3D(x_array), len(y_array), numpy_to_icp_point3D(y_array));
    goicp.setDTSizeAndFactor(300, 2.0);
    goicp.BuildDT();
    goicp.Register();
    R_matrix = np.array(goicp.optimalRotation())
    t_vec = np.array(goicp.optimalTranslation())


    T_est,R_est = rotation_translation_to_translator_rotator(R_matrix,t_vec)

    return (T_est,R_est)


def estimate_transformation_GOICP(x,y,npoints):
    '''Go-ICP'''

    x_nparray = cga3d_vector_array_to_nparray(x)
    y_nparray = cga3d_vector_array_to_nparray(y)

    return __estimate_transformation_GOICP__(x_nparray,y_nparray)

def convert_to_list_of_matrices(array):
    lst = []
    for i in range(len(array)):
        lst += [np.asmatrix(array[i,:,np.newaxis])]
    return lst

def __estimate_transformation_FPFH__(x_array,y_array):

    et = 0.1
    div = 2
    nneighbors = 8
    rad = 0.01

    Icp = FPFH(et, div, nneighbors, rad)   # Fast PFH
    transformed_source = Icp.solve(convert_to_list_of_matrices(x_array), convert_to_list_of_matrices(y_array))
    R_list = Icp._Rlist
    t_list = Icp._tlist

    # Calculate the final R and t which were applied to the source cloud
    R_final = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    for R in R_list:
        R_final = R.dot(R_final)
    i = 0
    t_final = t_list[0]
    for i in range(1, len(t_list)):
        t_final = R_list[i]*t_final + t_list[i]

    T_est,R_est = rotation_translation_to_translator_rotator(R_final,t_final.T[0])
    
    return (T_est,R_est)

def estimate_transformation_TEASER(x,y,npoints):
    """TEASER++"""

    x_nparray = cga3d_vector_array_to_nparray(x)
    y_nparray = cga3d_vector_array_to_nparray(y)

    x_pcd = o3d.geometry.PointCloud()
    y_pcd = o3d.geometry.PointCloud()

    x_pcd.points = o3d.utility.Vector3dVector(x_nparray)
    y_pcd.points = o3d.utility.Vector3dVector(y_nparray)
    
    return __estimate_transformation_TEASER__(x_pcd,y_pcd)

def __estimate_transformation_TEASER__(x_pcd,y_pcd):

    VOXEL_SIZE = 0.001
    # x_pcd = x_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    # y_pcd = x_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

    x_array = teaser.pcd2xyz(x_pcd) # np array of size 3 by N
    y_array = teaser.pcd2xyz(y_pcd) # np array of size 3 by M
    
    radius_feat = 0.1
    x_feats = teaser.extract_fpfh(x_pcd,radius_feat)
    y_feats = teaser.extract_fpfh(y_pcd,radius_feat)

    # establish correspondences by nearest neighbour search in feature space
    corrs_x, corrs_y = teaser.find_correspondences(x_feats, y_feats, mutual_filter=True)
    x_corr = x_array[:,corrs_x] # np array of size 3 by num_corrs
    y_corr = y_array[:,corrs_y] # np array of size 3 by num_corrs

    # robust global registration using TEASER++
    NOISE_BOUND = 0.002
    teaser_solver = teaser.get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(x_corr,y_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation

    T_est,R_est = rotation_translation_to_translator_rotator(R_teaser,t_teaser)
    return T_est,R_est


# Estimate with know correspondences
def estimate_transformation_corr_VGA(x,y,npoints):
    '''VGA RBM Known Corrs'''
    x_bar = x.sum()/npoints
    y_bar = y.sum()/npoints
    R_est = estimate_rot_3dvga(x-x_bar,y-y_bar)
    T_est = translation_from_cofm(y,x,R_est,npoints)
    return (T_est,R_est)

algorithms_list = [estimate_transformation_pasta,
                   estimate_transformation_ICP,
                   estimate_transformation_CGA,
                   estimate_transformation_VGA,
                   estimate_transformation_dcp,
                   estimate_transformation_corr_VGA,
                   estimate_transformation_TEASER]

def get_alg_name_list(algorithms):
    names = []
    for i in range(len(algorithms)):
        names += [get_algorithm_name(algorithms[i])]

    return names