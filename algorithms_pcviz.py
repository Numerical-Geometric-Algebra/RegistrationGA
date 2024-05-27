from cga3d_estimation import *
import open3d as o3d
from algorithms import __estimate_transformation_DCP__, __estimate_transformation_ICP__, __estimate_transformation_CGA__, __estimate_transformation_GOICP__, __estimate_transformation_TEASER__
from algorithms import *

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


 

def estimate_transformation_CGA__(source,target):
    '''CGA-EVD'''
    
    # Get the eigenmultivectors and the reference
    P_biv,P_vec,P_ref = source.eigbivs,source.eigvecs,source.mvref
    Q_biv,Q_vec,Q_ref = target.eigbivs,target.eigvecs,target.mvref

    T_est,R_est = __estimate_transformation_CGA__(P_biv,P_vec,P_ref,Q_biv,Q_vec,Q_ref)

    return T_est*R_est


def estimate_transformation_PASTA__(source,target):
    '''PASTA-3D'''
    import pasta3d
    x_array = source.get_pcd_as_nparray()
    y_array = target.get_pcd_as_nparray()

    R_matrix, t_vec = pasta3d.pasta3d_rototranslation(y_array, x_array, 'max')

    M_est = rotation_translation_to_motor(R_matrix,t_vec)

    return M_est


def estimate_transformation_VGA__(source,target):
    '''VGA-EVD
        Estimates the translation using the center of mass of the point clouds.
        Solves an eigenvalue problem to extract rotation invariant eigenvectors from each point cloud.
    '''
    
    x = source.get_pcd_as_mvcloud()
    y = target.get_pcd_as_mvcloud()

    T_est,R_est = estimate_transformation_VGA(x,y,source.get_points_nbr())

    return T_est*R_est

eye4 = np.eye(4)
icp_dist_threshold = 100
def estimate_transformation_ICP__(source,target):
    '''ICP'''
    x_array = source.get_pcd_as_nparray()
    y_array = target.get_pcd_as_nparray()

    T_est,R_est = __estimate_transformation_ICP__(x_array,y_array)

    return T_est*R_est

def estimate_transformation_sdrsac__(source,target):
    '''SDRSAC'''
    x_array = source.get_pcd_as_nparray()
    y_array = target.get_pcd_as_nparray()

    R_matrix, t_vec = sdrsac.sdrsac(x_array.T, y_array.T,max_itr=1)

    M_est = rotation_translation_to_motor(R_matrix,t_vec)

    return M_est


def estimate_transformation_DCP__(source,target):
    '''DCP'''
    x_nparray = source.get_pcd_as_nparray()
    y_nparray = target.get_pcd_as_nparray()

    T_est,R_est = __estimate_transformation_DCP__(x_nparray,y_nparray)

    return T_est*R_est

def estimate_transformation_GOICP__(source,target):
    '''Go-ICP'''

    x_array = source.get_pcd_as_nparray()
    y_array = target.get_pcd_as_nparray()

    T_est,R_est = __estimate_transformation_GOICP__(x_array,y_array)

    return T_est*R_est

def estimate_transformation_TEASER__(source,target):
    '''TEASER++'''

    T_est,R_est = __estimate_transformation_TEASER__(source.noisy_pcd,target.noisy_pcd)

    return T_est*R_est

alg_lst = [estimate_transformation_CGA__,
           estimate_transformation_VGA__,
           estimate_transformation_PASTA__,
           estimate_transformation_ICP__,
           estimate_transformation_sdrsac__,
           estimate_transformation_DCP__,
           estimate_transformation_GOICP__,
           estimate_transformation_TEASER__]

def get_algorithm_name(algorithm):
    return algorithm.__doc__.split("\n")[0]

def get_algorithms():
    '''Get the algorithms and the respective names.
       Names are determined as the first line of the docstring of each function.'''
    names = []
    for i in range(len(alg_lst)):
        names += [get_algorithm_name(alg_lst[i])]

    return alg_lst,names