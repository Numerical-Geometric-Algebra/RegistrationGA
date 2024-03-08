from cga3d_estimation import *

import pasta3d
import open3d as o3d
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
    '''PASTA 3D'''
    x_array = cga3d_vector_array_to_nparray(x)
    y_array = cga3d_vector_array_to_nparray(y)

    R_matrix, t_vec = pasta3d.pasta3d_rototranslation(y_array, x_array, 'max')

    R_est = rotmatrix_to_3drotor(R_matrix)
    t_est = nparray_to_3dvga_vector_array(t_vec)
    T_est = 1 + (1/2)*einf*t_est

    return (T_est,R_est,None,None)

eye4 = np.eye(4)
icp_dist_threshold = 100

def estimate_transformation_ICP(x,y,npoints):
    '''ICP'''
    x_array = cga3d_vector_array_to_nparray(x)
    y_array = cga3d_vector_array_to_nparray(y)

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

    R_est = rotmatrix_to_3drotor(R_matrix)
    t_est = nparray_to_3dvga_vector_array(t_vec)
    T_est = 1 + (1/2)*einf*t_est

    return (T_est,R_est,None,None)


def estimate_transformation_CGA(x,y,npoints):
    '''CGA eigmvs
        Estimates the rigid body motion between two point clouds using the eigenmultivector.
        From the eigenmultivectors we estimate the rotation and translation. 
        To estimate the sign we use references from the PCs directly.
    '''
    s = 2 # Consider the two eigenbivectors with the smallest eigenvalue
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenvectors    
    P_vec_lst,lambda_Pvec = get_3dcga_eigmvs(p,grades=1)
    Q_vec_lst,lambda_Qvec = get_3dcga_eigmvs(q,grades=1)
    
    # Get the eigenbivectors and respective eigenvalues. The eigenvalues are ordered from smallest to biggest
    P_biv_lst,lambda_Pbiv = get_3dcga_eigmvs(p,grades=2)
    Q_biv_lst,lambda_Qbiv = get_3dcga_eigmvs(q,grades=2)

    P_vec = mv.concat(P_vec_lst)
    Q_vec = mv.concat(Q_vec_lst)

    # Transform list of multivectors into an array
    P_biv = mv.concat(P_biv_lst[s:])
    Q_biv = mv.concat(Q_biv_lst[s:])

    P_ref = compute_reference(p)
    Q_ref = compute_reference(q)
    
    P_biv *= mv.sign((P_biv*P_ref)(0))
    Q_biv *= mv.sign((Q_biv*Q_ref)(0))

    # Computes the optimal rotor from the first coefficients of P and of Q
    P1,P2,P3,P4 = get_coeffs(P_biv)
    Q1,Q2,Q3,Q4 = get_coeffs(Q_biv)
    # define the rotor valued function
    def Func(Rotor):
        return (Q1*Rotor*~P1 + ~Q1*Rotor*P1).sum()

    basis,rec_basis = get_3dvga_rotor_basis()
    R_lst,lambda_R = multiga.symmetric_eigen_decomp(Func,basis,rec_basis)
    R_est = R_lst[3] # Chose the eigenrotor with the biggest eigenvalue
    
    for i in range(len(Q_vec_lst)):
        Q_vec_lst[i] /= (Q_vec_lst[i]*Q_ref)(0)
        P_vec_lst[i] /= (P_vec_lst[i]*P_ref)(0)

    for i in range(len(Q_biv_lst)):
        Q_biv_lst[i] /= (Q_biv_lst[i]*Q_ref)(0)
        P_biv_lst[i] /= (P_biv_lst[i]*P_ref)(0)

    P_lst = P_biv_lst + P_vec_lst
    Q_lst = Q_biv_lst + Q_vec_lst
    
    # Use the first eigenvector to estimate the translation
    T_est = exact_translation(R_est*P_vec_lst[0]*~R_est,Q_vec_lst[0])

    return (T_est,R_est,P_lst,Q_lst)

eps = 1e-12
def estimate_transformation_VGA(x,y,npoints):
    '''VGA CeofMass
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

    R_est = ~exact_rotation(Q_lst[0],Q_lst[1],P_lst[0],P_lst[1])
    T_est = translation_from_cofm(y,x,R_est,npoints)

    return (T_est,R_est,P_lst,Q_lst)

# Estimate with know correspondences
def estimate_transformation_corr_VGA(x,y,npoints):
    '''VGA RBM Known Corrs'''
    x_bar = x.sum()/npoints
    y_bar = y.sum()/npoints
    R_est = estimate_rot_3dvga(x-x_bar,y-y_bar)
    T_est = translation_from_cofm(y,x,R_est,npoints)
    return (T_est,R_est,None,None)

algorithms_list = [estimate_transformation_pasta,estimate_transformation_ICP,estimate_transformation_CGA,estimate_transformation_VGA,estimate_transformation_corr_VGA]

def get_alg_name_list(algorithms):
    names = []
    for i in range(len(algorithms)):
        names += [get_algorithm_name(algorithms[i])]

    return names