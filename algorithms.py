from cga3d_estimation import *

import pasta3d
import open3d as o3d
# import numpy as np

'''
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

def estimate_transformation_0(x,y,npoints):
    '''CGA RBM Null Reflection
        Estimates the rigid body motion between two point clouds using the eigenmultivector.
        From the eigenmultivectors we estimate the rotation and translation. 
        To estimate the sign we use references from the PCs themselves.
    '''
    eig_grades = [1,2]
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenbivectors
    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Rescale Q to the appropriate scale factor
    P_ref,Q_ref = compute_references(p,q)
    Q /= (Q*Q_ref)(0)
    P /= (P*P_ref)(0)
    
    T_est,R_est = estimate_rigtr(P,Q)

    return (T_est,R_est,P_lst,Q_lst)

def estimate_transformation_12(x,y,npoints):
    '''CGA RBM signed
        Estimates the rigid body motion between two point clouds using the eigenmultivector.
        From the eigenmultivectors we estimate the rotation and translation. 
        To estimate the sign we use references from the PCs directly.
    '''
    eig_grades = [1,2]
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenbivectors
    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Rescale Q to the appropriate sign factor
    P_ref,Q_ref = compute_references(p,q)

    Q *= mv.sign((Q*Q_ref)(0))
    P *= mv.sign((P*P_ref)(0))
    
    T_est,R_est = estimate_rigtr(P,Q)

    return (T_est,R_est,P_lst,Q_lst)


def estimate_transformation_13(x,y,npoints):
    '''CGA Exact Translation V2
        Uses the first eigenvector to estimate the tranlation vector. (Uses the exact translation formula)
        Estimates the rigid transformation between two point clouds using the eigenmultivector.
        From the eigenmultivectors we estimate the rotation and translation. 
        To estimate the sign we use references from the PCs directly.
    '''
    eig_grades = [1,2]
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenbivectors
    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Resign Q to the appropriate sign factor
    P_ref,Q_ref = compute_references(p,q)
    
    Q *= mv.sign((Q*Q_ref)(0))
    P *= mv.sign((P*P_ref)(0))

    _,R_est = estimate_rigtr(P,Q)

    # Use the first eigenmultivector to estimate the translation exactly
    Q_lst[0] *= np.sign((Q_lst[0]*Q_ref)(0))
    P_lst[0] *= np.sign((P_lst[0]*P_ref)(0))
    T_est = exact_translation(R_est*P_lst[0]*~R_est,Q_lst[0])

    return (T_est,R_est,P_lst,Q_lst)


def estimate_transformation_14(x,y,npoints):
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

def order_by_abs_eigvalues(P,lambda_P):
    indices = np.argsort(abs(lambda_P))
    P = [P[i] for i in indices]
    lambda_P = lambda_P[indices]
    return P,lambda_P

def estimate_transformation_16(x,y,npoints):
    '''CGA ExactTrs II
        Estimates the rigid body motion between two point clouds using the eigenmultivector.
        From the eigenmultivectors we estimate the rotation and translation. 
        To estimate the sign we use references from the PCs directly.
    '''
    s = 2 # Consider only the first two eigenbivectors
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenvectors    
    P_vec_lst,lambda_Pvec = get_3dcga_eigmvs(p,grades=1)
    Q_vec_lst,lambda_Qvec = get_3dcga_eigmvs(q,grades=1)
    # P_vec_lst,lambda_Pvec = order_by_abs_eigvalues(P_vec_lst,lambda_Pvec) 
    # Q_vec_lst,lambda_Qvec = order_by_abs_eigvalues(Q_vec_lst,lambda_Qvec)

    P_vec = mv.concat(P_vec_lst)
    Q_vec = mv.concat(Q_vec_lst)

    # Get the eigenbivectors
    P_biv_lst,lambda_Pbiv = get_3dcga_eigmvs(p,grades=2)
    Q_biv_lst,lambda_Qbiv = get_3dcga_eigmvs(q,grades=2)
    P_biv_lst,lambda_Pbiv = order_by_abs_eigvalues(P_biv_lst,lambda_Pbiv) # Reorder the eigenbivectors
    Q_biv_lst,lambda_Qbiv = order_by_abs_eigvalues(Q_biv_lst,lambda_Qbiv)


    # Transform list of multivectors into an array
    P_biv = mv.concat(P_biv_lst[:s]) 
    Q_biv = mv.concat(Q_biv_lst[:s])

    P_ref = compute_reference(p)
    Q_ref = compute_reference(q)
    
    P_biv *= mv.sign((P_biv*P_ref)(0))
    Q_biv *= mv.sign((Q_biv*Q_ref)(0))

    # P_biv /= (P_biv*P_ref)(0)
    # Q_biv /= (Q_biv*Q_ref)(0)

    P1,P2,P3,P4 = get_coeffs(P_biv)
    Q1,Q2,Q3,Q4 = get_coeffs(Q_biv)

    # lambda_P = vga3d.multivector(np.expand_dims(lambda_Pbiv,axis=1).tolist(),basis=['e'])(0)
    # lambda_Q = vga3d.multivector(np.expand_dims(lambda_Qbiv,axis=1).tolist(),basis=['e'])(0)

    # Weight by the absolute value of the eigenvalues
    # P1 = P1*(abs(lambda_P) + abs(lambda_Q))
    # Q1 = Q1*(abs(lambda_P) + abs(lambda_Q))

    # define the rotor valued function
    def Func(Rotor):
        return (Q1*Rotor*~P1 + ~Q1*Rotor*P1).sum()

    basis,rec_basis = get_3dvga_rotor_basis()
    R_lst,lambda_R = multiga.symmetric_eigen_decomp(Func,basis,rec_basis)
    R_est = R_lst[3] # Chose the eigenrotor with the biggest eigenvalue
    
    for i in range(len(Q_vec_lst)):
        Q_vec_lst[i] /= (Q_vec_lst[i]*Q_ref)(0)
        P_vec_lst[i] /= (P_vec_lst[i]*P_ref)(0)
    
    # Use the first eigenvector to estimate the translation
    T_est = exact_translation(R_est*P_vec_lst[0]*~R_est,Q_vec_lst[0])

    return (T_est,R_est,P_biv_lst,Q_biv_lst)


# Use the center of mass to estimate the translation
def estimate_transformation_1(x,y,npoints):
    '''CGA RBM CeOM'''
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenbivectors
    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Orient the eigenbivectors by using the points p and q as a reference
    signs = get_orient_diff(P,Q,p,q)
    P = P*signs
    T_est,R_est = estimate_rigtr(P,Q)

    # Determine the translation from the center of mass
    T_est = translation_from_cofm(y,x,R_est,npoints)
    # Need to calculate number of points, for that need to implement the len(x) method
    return (T_est,R_est,P_lst,Q_lst)

def estimate_transformation_2(x,y,npoints):
    '''CGA RBM Refine Rot'''
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenbivectors
    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Orient the eigenbivectors by using the points p and q as a reference
    signs = get_orient_diff(P,Q,p,q)
    P = P*signs
    T_est,R_est = estimate_rigtr(P,Q)
    
    # Determine the translation from the center of mass
    T_est = translation_from_cofm(y,x,R_est,npoints)
    S = ~T_est*Q*T_est

    # Refine the rotation
    R_est = estimate_rot_3dcga(P,S)

    # Need to calculate number of points, for that need to implement the len(x) method
    return (T_est,R_est,P_lst,Q_lst)

def get_CGA_rot_func(x,npoints):
    x_bar = x.sum()/npoints
    x_prime = x - x_bar
    p = eo + x_prime + (1/2)*pyga.mag_sq(x_prime)*einf

    def F(X):
        return (p*X*p).sum()

    return F

def estimate_transformation_3(x,y,npoints):
    '''CGA RBM Centered'''
    basis,rec_basis = get_3dcga_basis(eig_grades)
    F = get_CGA_rot_func(x,npoints)
    G = get_CGA_rot_func(y,npoints)
    P_lst,lambda_P = multiga.symmetric_eigen_decomp(F,basis,rec_basis)
    Q_lst,lambda_Q = multiga.symmetric_eigen_decomp(G,basis,rec_basis)
    
    x_bar = x.sum()/npoints
    y_bar = y.sum()/npoints

    # Convert to CGA 
    p = eo + x-x_bar + (1/2)*pyga.mag_sq(x-x_bar)*einf
    q = eo + y-y_bar + (1/2)*pyga.mag_sq(y-y_bar)*einf

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Correct sign
    signs = get_orient_diff(P,Q,p,q)
    P = P*signs

    R_est = estimate_rot_3dcga(P,Q)
    T_est = translation_from_cofm(y,x,R_est,npoints)

    return (T_est,R_est,P_lst,Q_lst)


def estimate_transformation_7(x,y,npoints):
    '''CGA RBM Centered Brute Force'''

    basis,rec_basis = get_3dcga_basis(2)
    F = get_CGA_rot_func(x,npoints)
    G = get_CGA_rot_func(y,npoints)
    P_lst,lambda_P = multiga.symmetric_eigen_decomp(F,basis,rec_basis)
    Q_lst,lambda_Q = multiga.symmetric_eigen_decomp(G,basis,rec_basis)
    
    x_bar = x.sum()/npoints
    y_bar = y.sum()/npoints

    # Convert to CGA 
    p = eo + x-x_bar + (1/2)*pyga.mag_sq(x-x_bar)*einf
    q = eo + y-y_bar + (1/2)*pyga.mag_sq(y-y_bar)*einf

    R_est = brute_force_estimate_CGA(P_lst,Q_lst)
    T_est = translation_from_cofm(y,x,R_est,npoints)

    return (T_est,R_est,P_lst,Q_lst)


def get_VGA_rot_func(x,npoints):
    x_bar = x.sum()/npoints
    def F(X):
        return ((x-x_bar)*X*(x-x_bar)).sum()

    return F

eps = 1e-12
def estimate_transformation_4(x,y,npoints):
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
def estimate_transformation_5(x,y,npoints):
    '''VGA RBM Known Corrs'''
    x_bar = x.sum()/npoints
    y_bar = y.sum()/npoints
    R_est = estimate_rot_3dvga(x-x_bar,y-y_bar)
    T_est = translation_from_cofm(y,x,R_est,npoints)
    return (T_est,R_est,None,None)

def get_proj_func(P,npoints):
    P_bar = P.sum()/npoints
    def F(X):
        return ((X|(P - P_bar))*(P - P_bar)).sum()

    return F

def get_rej_func(P,npoints):
    P_bar = P.sum()/npoints
    def F(X):
        return ((X^(P - P_bar))*(P - P_bar)).sum()

    return F

def get_non_linear_func(P,npoints):
    def F(X):
        scale = (P*~P)(0)
        scale = scale*scale
        return ((X^P)*P*scale).sum()
    return F





def estimate_transformation_6(x,y,npoints):
    '''VGA RBM Nonlinear Func'''
    basis,rec_basis = get_3dvga_basis(1)

    x_bar = x.sum()/npoints
    y_bar = y.sum()/npoints

    F1 = get_proj_func(x-x_bar,npoints)
    G1 = get_proj_func(y-y_bar,npoints)
    F2 = get_non_linear_func(x-x_bar,npoints)
    G2 = get_non_linear_func(y-y_bar,npoints)

    P_lst1,lambda_P1 = multiga.symmetric_eigen_decomp(F1,basis,rec_basis)
    Q_lst1,lambda_Q1 = multiga.symmetric_eigen_decomp(G1,basis,rec_basis)

    P_lst2,lambda_P2 = multiga.symmetric_eigen_decomp(F2,basis,rec_basis)
    Q_lst2,lambda_Q2 = multiga.symmetric_eigen_decomp(G2,basis,rec_basis)

    R_est = brute_force_estimate_VGA(P_lst1,P_lst2,Q_lst1,Q_lst2)

    T_est = translation_from_cofm(y,x,R_est,npoints)

    return (T_est,R_est,P_lst1,P_lst2)




def estimate_transformation_8(x,y,npoints):
    '''CGA RBM H_matrix
        Computes a matrix which relates the eigenmultivectors P_lst and Q_lst.
        We take certain coefficents of the matrix to determine the translation
        and the rotation. 
        The default sign atribution is done with get_3dcga_eigmvs, where it uses the point
        at infinity to switch signs.
    '''

    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=1)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=1)

    H_diff,H_adj = multiga.get_orthogonal_func(P_lst,Q_lst)

    basis = [eo,e1,e2,e3,einf]
    rec_basis = [-einf,e1,e2,e3,-eo]

    # # Verify that H is in fact magnitude preserving
    # for i in range(len(basis)):
    #     print(pyga.mag_sq(H_diff(basis[i])))
    # print()

    H_matrix = multiga.get_matrix(H_diff,basis,rec_basis).T

    t_vec = H_matrix[1:4,0]
    R_matrix = H_matrix[1:4,1:4]

    R_est = rotmatrix_to_3drotor(R_matrix)
    t_est = nparray_to_3dvga_vector_array(t_vec)

    T_est = 1 + (1/2)*einf*t_est

    return (T_est,R_est,P_lst,Q_lst)

def get_versor_from_function(H,basis,rec_basis):
    ''' Computes the versor of a special orthogonal function in 3D CGA. 
        It uses the bivector decomposition of d_x^H(x) to determine the versor U of H.
    '''
    B = -multiga.compute_bivector_from_skew(H,basis,rec_basis)

    alpha1 = -(1/2)*(pyga.mag_sq(B) - np.sqrt(pyga.mag_sq(B)**2 - pyga.mag_sq(B^B)))
    alpha2 = -(1/2)*(pyga.mag_sq(B) + np.sqrt(pyga.mag_sq(B)**2 - pyga.mag_sq(B^B)))

    if abs(alpha1) < 1e-4: # parabolic rotation
        B1 = -(1/2)*(B*(B^B))/pyga.mag_sq(B)
        R1 = 1 + (1/2)*B1
    else:
        A = (1/2)*(B^B)/alpha1
        B1 = B*(1-A)/(1-pyga.mag_sq(A))
        v1 = (rdn_3dcga_vector()|B1)*pyga.inv(B1)
        R1_sq = H(v1)*pyga.inv(v1)
        R1 = pyga.rotor_sqrt(R1_sq)
    B2 = B - B1
    
    v2 = (rdn_3dcga_vector()|B2)*pyga.inv(B2)
    R2_sq = H(v2)*pyga.inv(v2)
    U = R1*pyga.rotor_sqrt(R2_sq)
    return U

def estimate_transformation_9(x,y,npoints):
    '''CGA Motor Estimation
        From the versor U of H it estimates the best motor Ue = Re*Te
    '''
    basis,rec_basis = get_3dcga_basis([1])

    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=1)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=1)

    # Form the orthogonal function H(x) = sum_i (x|P[i])*inv(Q[i])
    H_diff,H_adj = multiga.get_orthogonal_func(P_lst,Q_lst)    

    U = get_versor_from_function(H_diff,basis,rec_basis)

    T_est,R_est = best_motor_estimation(U)

    return (T_est,R_est,P_lst,P_lst)

def estimate_transformation_15(x,y,npoints):
    '''CGA Motor Projection
        Pseudo projects to the space of motors.
        Projects the estimated versor U of H to the space of rotations and translations directly. R = P_I(U), t = -2*P_I(((eo|U)*~R)(1))
    '''
    basis,rec_basis = get_3dcga_basis([1])

    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=1)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=1)

    # Form the orthogonal function H(x) = sum_i (x|P[i])*inv(Q[i])
    H_diff,H_adj = multiga.get_orthogonal_func(P_lst,Q_lst)
    U = get_versor_from_function(H_diff,basis,rec_basis)

    # Pseudo project to the space of motors
    T_est,R_est = decompose_motor(U) # Get the rotation and translation from U

    return (T_est,R_est,P_lst,P_lst)


def estimate_transformation_10(x,y,npoints):
    '''CGA exact translation
    '''
    eig_grades = [1,2]
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenbivectors
    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Rescale Q to the appropriate scale factor
    P_ref,Q_ref = compute_references(p,q)
    Q /= (Q*Q_ref)(0)
    P /= (P*P_ref)(0)

    # print((Q*Q_ref)(0))
    # print(P_ref)

    # Q *= (Q*Q_ref)(0)/(P*P_ref)(0)
    Q_lst[0] /= (Q_lst[0]*Q_ref)(0)
    P_lst[0] /= (P_lst[0]*P_ref)(0)


    # P_lst[1] *= np.sign((P_lst[1]*P_ref)(0)*(Q_lst[1]*Q_ref)(0))
    # print("P_lst =",P_lst)
    # print("Q_lst =",Q_lst)
    # signs = get_orient_diff(P,Q,p,q)
    # P = P*signs
    T_est,R_est = estimate_rigtr(P,Q)
    # print("algorithm 10")
    T_est = exact_translation(R_est*P_lst[0]*~R_est,Q_lst[0])

    # print(pyga.numpy_max((T_est*R_est*einf*~R_est*~T_est - einf)))
    # print(pyga.numpy_max((T_est*R_est*p*~R_est*~T_est - q)))


    # T_est1 = exact_translation(R_est*P_lst[1]*~R_est,Q_lst[1])
    # t_est = -(eo|(T_est0 + T_est1))
    # T_est = 1 + (1/2)*einf*t_est
    # print()


    return (T_est,R_est,P_lst,Q_lst)

def estimate_transformation_11(x,y,npoints):
    ''' CGA exact translation 2
    '''
    eig_grades = [1]
    s = eo + x + (1/2)*(x*x)(0)*einf
    q = eo + y + (1/2)*(y*y)(0)*einf

    S_lst,lambda_P = get_3dcga_eigmvs(s,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    S = mv.concat(S_lst)
    Q = mv.concat(Q_lst)

    S_ref = einf^(1+ s.sum()/npoints)
    Q_ref = einf^(1+ q.sum()/npoints)

    S *= mv.sign((S*S_ref)(0)*(Q*Q_ref)(0))
    S_lst[0] *= np.sign((S_lst[0]*S_ref)(0)*(Q_lst[0]*Q_ref)(0))

    # print("algorithm 11")
    T_est = exact_translation(S_lst[0],Q_lst[0])
    
    # print("Check Translation:",pyga.numpy_max(Proj_I(T_est*S*~T_est - Q)))
    # print("Check Translation:",(Proj_I(T_est*S*~T_est - Q)))

    R_est = vga3d.multivector([1],basis=['e'])

    return (T_est,R_est,S_lst,Q_lst)


eps = 1e-12
def estimate_rotation_0(x,y,npoints):
    '''VGA ROT Unit Reflection
        pyga.normalizes the points before estimating the rotation.
        Should be robust for outliers far away from the origin. 
    '''

    # pyga.normalize points
    x = pyga.normalize(x)
    y = pyga.normalize(y)

    # Determine the eigenvectors
    basis,rec_basis = get_3dvga_basis(1)
    F = multiga.get_reflections_function(x)
    G = multiga.get_reflections_function(y)
    P_lst,lambda_P = multiga.symmetric_eigen_decomp(F,basis,rec_basis)
    Q_lst,lambda_Q = multiga.symmetric_eigen_decomp(G,basis,rec_basis)

    # Correct the sign of the eigenvectors
    for i in range(len(P_lst)):
        P_lst[i] = P_lst[i]*correct_sign_2(P_lst[i],x)
        Q_lst[i] = Q_lst[i]*correct_sign_2(Q_lst[i],y)

    R_est = ~exact_rotation(Q_lst[0],Q_lst[1],P_lst[0],P_lst[1])

    return (1,R_est,P_lst,Q_lst)

def estimate_rotation_1(x,y,npoints):
    '''VGA ROT Scaled Reflection
        Uses the points directly. It should illustrate better the advantage of VGA ROT Unit Reflection 
        when dealing with outliers.
    '''

    # Determine the eigenvectors
    basis,rec_basis = get_3dvga_basis(1)
    F = multiga.get_reflections_function(x)
    G = multiga.get_reflections_function(y)
    P_lst,lambda_P = multiga.symmetric_eigen_decomp(F,basis,rec_basis)
    Q_lst,lambda_Q = multiga.symmetric_eigen_decomp(G,basis,rec_basis)

    # Correct the sign of the eigenvectors
    for i in range(len(P_lst)):
        P_lst[i] = P_lst[i]*correct_sign_2(P_lst[i],x)
        Q_lst[i] = Q_lst[i]*correct_sign_2(Q_lst[i],y)

    R_est = ~exact_rotation(Q_lst[0],Q_lst[1],P_lst[0],P_lst[1])

    return (1,R_est,P_lst,Q_lst)

eps = 1e-12

eig_grades = [1]
def estimate_rotation_2(x,y,npoints):
    '''CGA ROT Null Reflection
        Converts the points to Conformal Geometric Algebra. 
        Then estimates the rotation using the eigenvectors.
        Only uses vectors to estimate rotation.
    '''
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenbivectors
    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Orient the eigenbivectors by using the points p and q as a reference
    signs = get_orient_diff(P,Q,p,q)
    P = P*signs

    R_est = estimate_rot_3dcga(P,Q)

    return (1,R_est,P_lst,Q_lst)

eig_grades = [1,2]
def estimate_rotation_3(x,y,npoints):
    '''CGA ROT pyga.normalized Null Reflection
        Same as CGA ROT Null Reflection, but it pyga.normalizes the points before.
    '''
    # pyga.normalize points
    x = pyga.normalize(x)
    y = pyga.normalize(y)

    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenbivectors
    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Orient the eigenbivectors by using the points p and q as a reference
    signs = get_orient_diff(P,Q,p,q)
    P = P*signs

    R_est = estimate_rot_3dcga(P,Q)

    return (1,R_est,P_lst,Q_lst)


def correct_sign_vec_rot_CGA(P):
    return mv.sign((P|einf)(0))*P

eig_grades = [1]
def estimate_rotation_4(x,y,npoints):
    '''CGA ROT Null Reflection Scaled
        Scales the eigenvectors in order for the eo component to be equal on both
        eigenvectors
    '''
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenbivectors
    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Correctly scale the eigenvector using the point at infinity
    # Only use this when P is of grade one 
    P = P/(P*einf)(0)
    Q = Q/(Q*einf)(0)

    # Estimate the rotation using the euclidean components of P and Q
    R_est = estimate_rot_3dcga(P,Q)

    return (1,R_est,P_lst,Q_lst)


eig_grades = [1]
def estimate_rotation_5(x,y,npoints):
    '''CGA ROT H_matrix
        Estimate the rotor by determining the eigendecomposition of H_diff.
        Even though H_plus is a symmetric transformation, the matrix of H_plus is not.
        So since the multiga.symmetric_eigen_decomp does not deal well with non symetric matrices, 
        this will eventually provide poor results.
    '''
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=1)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=1)

    H_diff,H_adj = multiga.get_orthogonal_func(P_lst,Q_lst)

    H_plus = lambda X: (H_diff(X) + H_adj(X))/2

    basis,rec_basis = get_3dcga_basis(1)
    W,lambda_W = multiga.symmetric_eigen_decomp(H_plus,basis,rec_basis)

    R_est = ~pyga.rotor_sqrt(Proj_I(W[0]*H_diff(W[0])))

    return (1,R_est,P_lst,Q_lst)


def estimate_rotation_6(x,y,npoints):
    '''VGA ROT H_matrix SD
        Estimate a rotation via relating the eigenvectors of F and G
        Because of the sign this relation will not hold to be correct.
        If the determinant of H is -1 then we will have a reflection instead of
        a rotation.
        If the sign changes provide two reflections then we will go back to 
        having a rotation.
        Use the spectral decomposition of the rotation matrix to estimate the
        rotation.
    '''
    # Determine the eigenvectors
    basis,rec_basis = get_3dvga_basis(1)
    F = multiga.get_reflections_function(x)
    G = multiga.get_reflections_function(y)
    P_lst,lambda_P = multiga.symmetric_eigen_decomp(F,basis,rec_basis)
    Q_lst,lambda_Q = multiga.symmetric_eigen_decomp(G,basis,rec_basis)

    # Correct the sign of the eigenvectors
    for i in range(len(P_lst)):
        P_lst[i] = P_lst[i]*correct_sign_2(P_lst[i],x)
        Q_lst[i] = Q_lst[i]*correct_sign_2(Q_lst[i],y)

    H_diff,H_adj = multiga.get_orthogonal_func(P_lst,Q_lst)
    H_plus = lambda X: (H_diff(X) + H_adj(X))/2
    V_lst,lambda_V = multiga.symmetric_eigen_decomp(H_plus,basis,rec_basis)

    R_est = ~pyga.rotor_sqrt(V_lst[0]*H_diff(V_lst[0]))

    return (1,R_est,P_lst,Q_lst)


RBM_algs_list = [estimate_transformation_0,estimate_transformation_1,estimate_transformation_2,estimate_transformation_3,estimate_transformation_4,
                 estimate_transformation_5,estimate_transformation_6,estimate_transformation_7,estimate_transformation_8]
ROT_algs_list = [estimate_rotation_0,estimate_rotation_1,estimate_rotation_2,estimate_rotation_3,estimate_rotation_4,estimate_rotation_5,estimate_rotation_6]

algorithms_list = RBM_algs_list + ROT_algs_list

def get_alg_name_list(algorithms):
    names = []
    for i in range(len(algorithms)):
        names += [get_algorithm_name(algorithms[i])]

    return names