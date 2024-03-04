from cga3d_estimation import *
# from point_cloud_vis_II import PointCloudSettings
import pasta3d
import open3d as o3d

def get_algorithm_name(algorithm):
    return algorithm.__doc__.split("\n")[0]

def estimate_transformation_0(source,target):
    '''CGA ExactTRS '''
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