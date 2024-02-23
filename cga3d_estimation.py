#!/usr/bin/env python3
import gasparsegen
import geo_algebra as pyga
import multilinear_algebra as multiga
from gasparsegen import multivector as mv
import numpy as np
import math

vga3d = gasparsegen.GA(3,compute_mode='generated')
cga3d = gasparsegen.GA(4,1,compute_mode='generated') # Use 3D conformal geometric algebra
basis = cga3d.basis()
locals().update(basis) # Update all of the basis blades variables into the environment

# compute the cga null basis
einf = (1.0*e5 - 1.0*e4)*(1/np.sqrt(2))
eo = (1.0*e5 + 1.0*e4)*(1/np.sqrt(2))

I = e1*e2*e3
ii = I*(eo^einf)

def grade_involution(X):
    return X(0) - X(1) + X(2) - X(3) + X(4) - X(5)

def grade_involution_3dvga(X):
    return X(0) - X(1) + X(2) - X(3)

def reciprocal_blades_3dcga(basis):
    # Assumes that the basis are orthogonal
    rec_basis = [0]*len(basis) # reciprocal basis blades
    for i in range(len(basis)):
        # Reflect by the negative squaring vector
        rec_basis[i] = -e5*~grade_involution(basis[i])*e5
    return rec_basis

def get_3dvga_rotor_basis():
    vga_rotor_basis =  list(vga3d.basis(grades=[0,2]).values())
    vga_rec_rotor_basis = pyga.reciprocal_blades(vga_rotor_basis)
    return (vga_rotor_basis,vga_rec_rotor_basis)

def get_3dcga_basis(grades):
    cga_basis = list(cga3d.basis(grades=grades).values())
    cga_rec_basis = reciprocal_blades_3dcga(cga_basis)    
    return (cga_basis,cga_rec_basis)

def get_3dvga_basis(grades):
    vga_basis = list(vga3d.basis(grades=grades).values())
    vga_rec_basis = pyga.reciprocal_blades(vga_basis)
    return (vga_basis,vga_rec_basis)

'''Functions to generate random multivectors'''


def rdn_gaussian_3dcga_mvarray(mu,sigma,size):
    grades = [0,1,2,3,4,5]
    return pyga.rdn_gaussian_multivector_array(mu,sigma,cga3d,grades,size)

def rdn_gaussian_3dvga_vecarray(mu,sigma,size):
    return pyga.rdn_gaussian_multivector_array(mu,sigma,vga3d,1,size)

def rdn_3dcga_bivector_array(size):
    return pyga.rdn_multivector_array(cga3d,2,size)

def rdn_3dvga_vector_array(size):
    return pyga.rdn_multivector_array(vga3d,1,size)

def rdn_3dcga_mvarray(size):
    grades = [0,1,2,3,4,5]
    return pyga.rdn_multivector_array(cga3d,grades,size)

def rdn_3dvga_mvarray(size):
    grades = [0,1,2,3]
    return pyga.rdn_multivector_array(vga3d,grades,size)

def rdn_3dcga_biv():
    return pyga.rdn_multivector_array(cga3d,grades=2,size=1)

def rdn_3dvga_biv():
    return pyga.rdn_multivector_array(vga3d,grades=2,size=1)

def rdn_3dvga_vector():
    return pyga.rdn_multivector_array(vga3d,grades=1,size=1)

def rdn_3dcga_vector():
    return pyga.rdn_multivector_array(cga3d,grades=1,size=1)

def rdn_3dcga_multivector():
    grades = [0,1,2,3,4,5]
    return pyga.rdn_multivector_array(cga3d,grades,size=1)

def rdn_3dvga_rotor():
    a = rdn_3dvga_vector()
    b = rdn_3dvga_vector()
    return pyga.normalize_mv(a*b)

def gen_rdn_3dcga_rotor():
    a = rdn_3dcga_vector()
    b = rdn_3dcga_vector()
    return pyga.normalize_mv(a*b)

def rdn_3dcga_translator(scale=10):
    t = scale*rdn_3dvga_vector()
    return 1 + (1/2)*einf*t

def gen_pseudordn_3dvga_rotor(angle):
    theta = angle*np.pi/180
    u = pyga.normalize_mv(rdn_3dvga_vector())
    R = np.cos(theta/2) + I*u*np.sin(theta/2)
    return R

def gen_pseudordn_rigtr(angle,mag):
    ''' generate a random rigid transformation (rigtr)
        generate a random rotation with a given angle
        and a translation with a given magnitude'''
    R = gen_pseudordn_3dvga_rotor(angle)
    t = mag*pyga.normalize_mv(rdn_3dvga_vector())
    T = 1 + (1/2)*einf*t
    return (T,R)


def generate_rdn_3dcga_mvclouds(m,sigma=0.01,mu=0):
    '''Generate two random 3dcga multivector clouds, which relate via a rigid transformation plus noise.'''
    
    R = rdn_3dvga_rotor()
    t = 10*rdn_3dvga_vector()
    T = 1 + (1/2)*einf*t
    U = T*R

    P = rdn_3dcga_mvarray(m)
    N = rdn_gaussian_3dcga_mvarray(m)

    Q = U*P*~U + N

    return (P,Q)


def Proj_I(X):
    '''Projects to 3D VGA, I is the unit pss of VGA'''
    return X(0) + ((X(1) + X(2) + X(3))|I)*~I

def get_coeffs(X):
    ''' Extract the coefficints of a multivector 
        The multivector can be written in the form 
        X = eo*A + einf*B + (eo^einf)*C + D'''
    A = -Proj_I(einf|X)
    B = -Proj_I(eo|X)
    C = Proj_I(X|(eo^einf))
    D = Proj_I(X)
    
    # Check if the extraction was correct
    #print(eo*A + einf*B + (eo^einf)*C + D -X)
    
    return (A,B,C,D)


def partial_3dcga_norm(X):
    A,B,C,D = get_coeffs(X)
    return mv.sqrt(mag_sq(A) + mag_sq(C) + mag_sq(D))

def partial_3dcga_norm_sq(X):
    A,B,C,D = get_coeffs(X)
    return mag_sq(A) + mag_sq(C) + mag_sq(D)

def plus_norm(X):
    A,B,C,D = get_coeffs(X)
    return mv.sqrt(mag_sq(A) +  mag_sq(B) + mag_sq(C) + mag_sq(D))

def plus_norm_sq(X):
    A,B,C,D = get_coeffs(X)
    return mag_sq(A) +  mag_sq(B) + mag_sq(C) + mag_sq(D)


def decompose_motor(U):
    ''' Decomposes a motor as the composition of a rotation and translation
        Also serves to pseudo project to the space of motors U = T*R
    '''
    # Pseudo project to the space of motors
    R_est = pyga.normalize_mv(Proj_I(U))
    t_est = -2*Proj_I(((eo|U)*~R_est)(1))
    T_est = 1 + (1/2)*einf*t_est
    return T_est,R_est

eps = 0.001
def normalize_null_mv(X):
    '''Normalizes all types of multivectors
       Ignores almost null multivectors
       Only works for Multivector arrays with only a single element'''

    magnitude = np.sqrt(abs((X*~X)(0)))
    if(magnitude == 0.0):
        return X

    relative_mag = abs(magnitude/np.sqrt(plus_norm_sq(X)))
    if relative_mag < eps:
        return X
    else: 
        return X/magnitude

def normalize_null_mvlist(X_lst):
    ''' Normalizes a list of multivectors'''
    for i in range(len(X_lst)):
        X_lst[i] = normalize_null_mv(X_lst[i])
    return X_lst


def conformal_mapping(x):
    return eo + x + (1/2)*pyga.mag_sq(x)*einf

def translation_from_cofm(y,x,R_est,n_points):
    '''Estimate the translation using the center of mass'''
    z_est = R_est*x*~R_est
    t_est = (y.sum() - z_est.sum())/n_points
    T_est = 1 + (1/2)*einf*t_est

    return T_est

def exact_translation(S,Q):

    Q1,Q2,Q3,Q4 = get_coeffs(Q)
    S1,S2,S3,S4 = get_coeffs(S)

    # print("mag_sq(Q1)=",pyga.mag_sq(Q1))

    t_est = ((Q3 + Q4 - (S3 + S4))*pyga.inv(Q1))(1)
    T_est = 1 + (1/2)*einf*t_est

    return T_est

def estimate_translation(P,Q):
    '''Given two 3D CGA multivector arrays estimate the tranlsation
       This is part of the coefficients method
    '''

    A,B,C,D = get_coeffs(P)
    E,F,G,H = get_coeffs(Q)
    
    v = ((A + E)*~(G+H-C-D))(1).sum()
    s = ((A*~A)(0) + (E*~E)(0)).sum()

    t_est = v/s
    T_est = 1 + (1/2)*einf*t_est
    return T_est


def estimate_rigtr(P,Q):
    '''Estimate the rigid transformation U=TR using the eigendecomposition function'''
    basis,rec_basis = get_3dvga_rotor_basis()
    A,B,C,D = get_coeffs(P)
    E,F,G,H = get_coeffs(Q)

    # define the rotor valued function
    def Func(Rotor):
        return (E(1)*Rotor*A(1) - E(2)*Rotor*A(2)).sum() 
        
    R_lst,lambda_R = multiga.symmetric_eigen_decomp(Func,basis,rec_basis)
    R_est = R_lst[3] # Chose the eigenrotor with the biggest eigenvalue
    T_est = estimate_translation(R_est*P*~R_est,Q)
    return (T_est,R_est)

def estimate_rigtr_2(P,Q):
    '''Estimate the rigid transformation U=TR using the eigendecomposition function'''
    basis,rec_basis = get_3dvga_rotor_basis()
    A,B,C,D = get_coeffs(P)
    E,F,G,H = get_coeffs(Q)

    # define the rotor valued function
    def Func(Rotor):
        return (E(1)*Rotor*A(1) - E(2)*Rotor*A(2)).sum() 
        
    R_lst,lambda_R = multiga.symmetric_eigen_decomp(Func,basis,rec_basis)
    R_est = R_lst[3] # Chose the eigenrotor with the biggest eigenvalue
    T_est = estimate_translation(R_est*P(1)*~R_est,Q(1))
    return (T_est,R_est)



def estimate_rot_3dcga(P,S):
    '''Estimate the rotation from two multivector 3D CGA arrays'''

    basis,rec_basis = get_3dvga_rotor_basis()
    A,B,C,D = get_coeffs(P)
    J,K,L,M = get_coeffs(S)
    
    # Define the rotor valued function
    def Func(R):
        return (J*R*~A + ~J*R*A + K*R*~B + ~K*R*B + L*R*~C + ~L*R*C + M*R*~D + ~M*R*D).sum()
    
    R_lst,lambda_R = multiga.symmetric_eigen_decomp(Func,basis,rec_basis)
    R_est = R_lst[3] # Chose the eigenrotor with the biggest eigenvalue
    return R_est

def estimate_rot_3dvga(X,Y):
    '''Estimate the rotation from two multivector 3D VGA arrays'''
    basis,rec_basis = get_3dvga_rotor_basis()

    def Func(R):
        return (~Y*R*X + Y*R*~X).sum()

    R_lst,lambda_R = multiga.symmetric_eigen_decomp(Func,basis,rec_basis)
    R_est = R_lst[3] # Chose the eigenrotor with the biggest eigenvalue
    return R_est


def exact_rotation(a1,a2,b1,b2):
    n = pyga.normalize_mv(I*((a1-b1)^(a2-b2)))
    Pperp_a1 = (a1|n)*n
    P_a1 = (a1^n)*n
    R_sq = (b1 - Pperp_a1)*pyga.inv(P_a1)
    return pyga.rotor_sqrt(R_sq)


def rotmatrix_to_3drotor(rot_matrix):
    '''Takes a 3D rotation matrix and computes the 3D rotor'''
    eigenvalues, eigenvectors = np.linalg.eig((rot_matrix + rot_matrix.T)/2)

    # The eigenvector with eigenvalue of smallest magnitude 
    u = np.real(eigenvectors[:,np.argmin(abs(eigenvalues))])
    v = rot_matrix@u
    u = vga3d.multivector(list(u),grades=1) # Convert to VGA
    v = vga3d.multivector(list(v),grades=1) # Convert to VGA

    rotor = pyga.normalize_mv(v)*pyga.normalize_mv(u)

    return pyga.rotor_sqrt(rotor)

def estimate_rot_SVD(p,q):
    basis = [e1,e2,e3]

    def f(x):
        return (q*(p|x)).sum()

    f_matrix = get_matrix(f,basis,basis)

    U, S, V = np.linalg.svd(f_matrix, full_matrices=True)
    M = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,np.linalg.det(U)*np.linalg.det(V)]])
    rot_matrix = U@M@V

    R = rotmatrix_to_3drotor(rot_matrix)
    return R


def best_motor_estimation(V):
    M1 = Proj_I(V)
    M2 = -Proj_I(eo|V)
    M = M1 + einf*M2
    
    W = M*~(I*M2) + I*M2*~M
    v = (I*(eo|W))(0)
    u = (I*(eo|(M*~M)))(0)
    s = u/v
    
    Ue = M - s*I*M2
    Re = Proj_I(Ue)
    Te = 1 - einf*((eo|Ue)*~Re)(1)
    
    return Te,Re

def get_3dcga_eigmvs(p,grades=[1,2]):
    '''Computes the eigenmultivectors from the points p in 3D cga.
       The Function for which the eigenmultvectors are computed is F(X) = sum_i p[i]*X*p[i].
    '''
    basis,rec_basis = get_3dcga_basis(grades)
    F = multiga.get_reflections_function(p)
    P_lst,lambda_P = multiga.symmetric_eigen_decomp(F,basis,rec_basis)

    # print("Eigenvalue Check (CGA):",multiga.check_eigenvalues(F,P_lst,lambda_P))
    # Use the point at infinity to define a sign for the multivectors
    # If the scalar product with the point at infinity is zero then the sign of that zero is used
    for i in range(len(P_lst)):
        P_lst[i] = pyga.normalize_mv(P_lst[i])
        P_lst[i] = P_lst[i]*math.copysign(1,(P_lst[i]*einf)(0))
    
    return (P_lst,lambda_P)


''' Sign estimation functions. These are used to estimate the sign of ambiguous eigenmultivectors '''

def get_reference(p):
    p_bar = p.sum()
    p_ref = (p*p_bar*p).sum()
    P_ref = (p_bar^p_ref) + p_bar
    return P_ref

def get_orient_array(X,X_ref):
    return mv.sign((X*X_ref)(0))

def get_orient_diff(P,Q,p,q):
    P_ref = get_reference(p)
    Q_ref = get_reference(q)
    sign_p = get_orient_array(P,P_ref)
    sign_q = get_orient_array(Q,Q_ref)
    return sign_p*sign_q

# It would have to be for each of the P's
def get_orient_diff_2(P,Q,p,q):
    P_ref = get_reference(p)
    Q_ref = get_reference(q)
    sign_P = ((P_ref|P)*(P_ref|P)*(P_ref|P)).sum()(0)
    sign_Q = ((Q_ref|Q)*(Q_ref|Q)*(Q_ref|Q)).sum()(0)
    return sign_P*sign_Q

def correct_sign_1(v,x,x_bar):
    a = np.array((v|(x-x_bar)).tolist()[0])
    if abs(np.max(a)) > abs(np.min(a)):
        return 1
    else:
        return -1

def correct_sign_2(v,x):
    a = np.array((v|x).tolist(0)[0])
    alpha = (a**3).sum()
    
    if alpha > 0:
        return 1
    else:
        return -1

def correct_sign_3(P1,P2):
    return np.sign((P1*P2)(0))

def correct_sign_CGA(P,p):
    A = ((P(2)|p).sum())|p
    a = ((A + (P*p)).tolist(0)[0])
    
    a = np.array(a)
    alpha = (a**3).sum()

    if alpha > 0:
        return 1
    else:
        return -1

def orient_multivectors(X,A):
    scalar = mv.sign((X*A)(0))
    return X*scalar


def get_rigtr_error_metrics(R,R_est,T,T_est):
    '''Computes error metrics between the rotation and translation and the ground truth'''
    t = -eo|T*2
    t_est = -eo|T_est*2
    R_est = pyga.normalize_mv(R_est)
    costheta = (R_est*~R)(0)
    if abs(costheta) > 1:
        costheta = 1
    ang_error = np.arccos(costheta)/np.pi*360 # gets two times theta
    if ang_error > 180:
        ang_error = 360 - ang_error

    # Compute the magnitude of tranlation error
    t_mag_error = pyga.mag_mv(t - t_est)

    # Compute the error between the planes of rotation
    cosphi = (pyga.normalize_mv(R(2))*~pyga.normalize_mv(R_est(2)))(0)
    if abs(cosphi) > 1:
        cosphi = 1
    phi = np.arccos(cosphi)/np.pi*180
    if(phi > 180):
        phi = 360 - phi
    
    # When the translation vector t is zero or t_est is zero then cos_trans will be nan
    cos_trans = (pyga.normalize_mv(t)|pyga.normalize_mv(t_est))(0)
    if abs(cos_trans) > 1:
        cos_trans = 1
    t_angle_error = np.arccos(cos_trans)/np.pi*180

    return ang_error,t_mag_error,t_angle_error,phi

def print_rigtr_error_metrics(R,R_est,T,T_est,m_points=-1,sigma=-1):
    if(m_points > 0):
        print("Nbr points:",m_points)
    if(sigma >= 0):
        print("Sigma:", sigma)
    ang_error,t_mag_error,t_angle_error,plane_angle = get_rigtr_error_metrics(R,R_est,T,T_est)
    print("Angle between planes of rotation:",plane_angle)
    print("Angle Error:",ang_error)
    print("Translation Error:", t_mag_error)
    print("arccos(t_hat|t_est_hat):",t_angle_error)

# Still need to divide all of these by the length of the array
def compute_PC_error(t,R,x,y):
    y_est = R*x*~R + t
    return mag_sq(y_est - y).sum()

def compute_error(T,R,P,Q):
    P_est = ~R*~T*Q*T*R
    return partial_3dcga_norm(P_est-P).sum()

def compute_error_euclidean(R,P,Q):
    Q_est = R*P*~R
    return mag_sq(Q_est-Q).sum()





'''Specific for the visualization tool (point_cloud_vis.py)'''

def get_properties(X):
    # if X is a vector then the following should holds
    # X = -(eo + l +(1/2)*rho_sq*einf)*(d|eo)

    d = (-einf|X)^einf
    scalar = 1/((einf|X)*(einf|X))(0)
    l = -(1/2)*scalar*X*einf*X
    rho_sq = (scalar*X*grade_involution(X))(0)
    return ((-d|eo).tolist(1)[0][:3],Proj_I(l(1)).tolist(1)[0][:3],rho_sq)
    #return (-d|eo,Proj_I(l),rho_sq)

def nparray_to_3dvga_vector_array(x_array):
    '''Converts a numpy array to a an array of multivectors of grade one'''
    return pyga.nparray_to_vecarray(vga3d,x_array)

def cga3d_vector_array_to_nparray(x):
    ''' Takes a multivector and extracts the grade one elements into a list
        Consider only the first 3 basis elements '''
    return np.array(x.tolist(1)[0])[:,:3]

def transform_numpy_cloud(pcd,R,t):
    ''' Rotates and translates a numpy point clou
        R and t might be in a 3d cga thus the operation R*x*~R + t might
        take the points from 3d vga to 3d cga'''
    pts = np.asarray(pcd.points)
    x = pyga.nparray_to_vecarray(vga3d,pts)
    y = R*x*~R + t
    return cga3d_vector_array_to_nparray(y)

