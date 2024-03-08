from cga3d_estimation import *
# from point_cloud_vis_II import PointCloudSettings
import pasta3d
import open3d as o3d

''' This script is aimed to define different algorithms to determine the rigid transformation between point cloud
    To get the noisy point cloud data use the class functions 
        source.get_pcd_as_nparray() # to get as a numpy array
        source.get_pcd_as_mvcloud() # to get as multivector array

    To get the eigenmultivectors access the class variables:
        source.eigbivs,source.eigvecs # access the eigenbivectors and eigenvectors respectively
    To get the reference use 
        source.mvref

    The output must be a motor. If computing with numpy use:
        M_est = rotation_translation_to_motor(R_matrix,t_vec)
  '''


def estimate_transformation_CGA(source,target):
    '''CGA ExactTRS'''
    s = 2 # Consider the two eigenbivectors with the smallest eigenvalue
    
    # Get the eigenmultivectors and the reference
    P_biv,P_vec,P_ref = source.eigbivs,source.eigvecs,source.mvref
    Q_biv,Q_vec,Q_ref = target.eigbivs,target.eigvecs,target.mvref

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

    return T_est*R_est


def estimate_transformation_pasta(source,target):
    '''PASTA 3D'''
    x_array = source.get_pcd_as_nparray()
    y_array = target.get_pcd_as_nparray()

    R_matrix, t_vec = pasta3d.pasta3d_rototranslation(y_array, x_array, 'max')

    M_est = rotation_translation_to_motor(R_matrix,t_vec)

    return M_est

def estimate_transformation_VGA(source,target):
    '''VGA CeofMass
        Estimates the translation using the center of mass of the point clouds.
        Solves an eigenvalue problem to extract rotation invariant eigenvectors from each point cloud.
    '''
    
    x = source.get_pcd_as_mvcloud()
    y = target.get_pcd_as_mvcloud()

    basis,rec_basis = get_3dvga_basis(1)
    
    # Determine centers of mass
    x_bar = x.sum()/source.get_points_nbr()
    y_bar = y.sum()/target.get_points_nbr()
    
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
    
    # Use center of mass to determine translation
    z_est = (R_est*x*~R_est).sum()/source.get_points_nbr()
    w = y.sum()/target.get_points_nbr()
    t_est = w - z_est
    T_est = 1 + (1/2)*einf*t_est

    return T_est*R_est

eye4 = np.eye(4)
icp_dist_threshold = 100
def estimate_transformation_ICP(source,target):
    '''ICP'''
    x_array = source.get_pcd_as_nparray()
    y_array = target.get_pcd_as_nparray()

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

    M_est = rotation_translation_to_motor(R_matrix,t_vec)

    return M_est


alg_lst = [estimate_transformation_CGA,
                 estimate_transformation_VGA,
                 estimate_transformation_pasta,
                 estimate_transformation_ICP]

def get_algorithm_name(algorithm):
    return algorithm.__doc__.split("\n")[0]

def get_algorithms():
    '''Get the algorithms and the respective names.
       Names are determined as the first line of the docstring of each function.'''
    names = []
    for i in range(len(alg_lst)):
        names += [get_algorithm_name(alg_lst[i])]

    return alg_lst,names