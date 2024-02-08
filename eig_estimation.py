#!/usr/bin/env python
from geo_algebra import *
import math

'''
The functions to help estimate the rigid body motion between two point clouds.
'''


def reflection(X,Y):
    return (X*Y*X).sum()

def orient_multivectors(X,A):
    scalar = mv.sign((X*A)(0))
    return X*scalar

def get_func(X):
    def F(Y):
        return (X*Y*X).sum()
    return F 

def get_ang_error(R_est,R):
    costheta = (R_est*~R)(0)
    if abs(costheta) > 1:
        return 0
    ang_error = np.arccos(costheta)/np.pi*360
    if ang_error > 180:
        ang_error = 360 - ang_error
    return abs(ang_error)

def print_metrics(R_est,R,T_est,T,m_points=-1,sigma=-1):
    t = -eo|T*2
    t_est = -eo|T_est*2
    costheta = (R_est*~R)(0)
    if abs(costheta) > 1:
        # print("Cos Angle Error:",costheta)
        costheta = 1
    ang_error = np.arccos(costheta)/np.pi*360 # gets two times theta
    if ang_error > 180:
        ang_error = 360 - ang_error
    if(sigma >= 0):
        print("Sigma:", sigma)

    if(m_points > 0):
        print("Nbr points:",m_points)
    print("Angle Error:",ang_error)
    print("Translation Error:", mag_mv(t - t_est))

    cosphi = (normalize_mv(R(2))*~normalize_mv(R_est(2)))(0)
    if abs(cosphi) > 1:
        cosphi = 1
    phi = np.arccos(cosphi)/np.pi*180
    if(phi > 180):
        phi = 360 - phi
    print("Angle between planes of rotation:",phi)
    print()

def gen_pseudordn_rbm(angle,mag):
    theta = angle*np.pi/180
    u = normalize_mv(rdn_vanilla_vec())
    R = np.cos(theta/2) + I*u*np.sin(theta/2)
    t = mag*rdn_vanilla_vec()
    T = 1 + (1/2)*einf*t
    return (T,R)

def check_eigenvalues(F,V,lambda_V):
    values = []
    for i in range(len(V)):
        values += [(F(V[i]) - lambda_V[i]*V[i]).tolist(1)[0]]
    arr = np.array(values)
    # print(arr)
    return abs(arr).max()
    # print("Eigen Relation: Worst Value:",)

def get_eigmvs(p,grades=[1,2]):
    '''Computes the eigenmultivectors for the conformal geometric algebra.
        The Function for which the eigenmultvectors are computed is F(X) = sum_i p[i]*X*p[i].
    '''
    basis,rec_basis = get_cga_basis(grades)
    F = get_func(p)
    P_lst,lambda_P = eigen_decomp(F,basis,rec_basis)

    
    # If the scalar product with the point at infinity is zero then the sign of that zero is used
    for i in range(len(P_lst)):
        P_lst[i] = P_lst[i]*math.copysign(1,(P_lst[i]*einf)(0))
    
    return (P_lst,lambda_P)


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

def get_array(x_lst):
    array = np.zeros((len(x_lst),vga.size(1)))
    for i in range(len(x_lst)):
        array[i] = np.array(x_lst[i].list(1)[0][:3])
    return array

def mvarray_to_nparray(x):
    return np.array(x.tolist(1)[0])[:,:3]

def transform_numpy_cloud(pcd,R,t):
    pts = np.asarray(pcd.points)
    x = nparray_to_vga_vecarray(pts)
    y = R*x*~R + t
    return mvarray_to_nparray(y)

def get_properties(X):
    # if X is a vector then the following should holds
    # X = -(eo + l +(1/2)*rho_sq*einf)*(d|eo)

    d = (-einf|X)^einf
    scalar = 1/((einf|X)*(einf|X))(0)
    l = -(1/2)*scalar*X*einf*X
    rho_sq = (scalar*X*grade_involution(X))(0)
    return ((-d|eo).tolist(1)[0][:3],P_I(l(1)).tolist(1)[0][:3],rho_sq)
    #return (-d|eo,P_I(l),rho_sq)
    
# Sanity check if eigenmultivectors are orthogonal
def check_orthogonality(P_lst):
    matrix = np.zeros([len(P_lst),len(P_lst)])
    for i in range(len(P_lst)):
        for j in range(len(P_lst)):
            matrix[i][j] = (P_lst[i]*P_lst[j])(0)
    
    return np.max(abs(abs(matrix) - np.eye(len(P_lst))))

def get_binary_list(n):
    lst = [0]*(2**n)
    for i in range(2**n):
        lst[i] = [2*int(bit)-1 for bit in bin(i)[2:].zfill(n)]
    return lst

def brute_force_estimate_VGA(P1_lst,P2_lst,Q1_lst,Q2_lst):
    n = 2
    sign_list = get_binary_list(n)
    best = 100000
    best_idx = 0
    R_best = 1

    P_lst_s = [0]*n

    for i in range(len(sign_list)):
        for j in range(n): # iterate over two of the eigenvectors 
            P_lst_s[j] = P1_lst[j]*sign_list[i][j] # Change the sign
        
        R_est = ~exact_rotation(Q1_lst[0],Q1_lst[1],P_lst_s[0],P_lst_s[1])
        Q2_est = R_est*P2_lst[0]*~R_est
        
        # If the third eigenvector is aligned than it is the right rotation
        if mag_mv(Q2_est - Q2_lst[0]) < best or mag_mv(Q2_est + Q2_lst[0]) < best:
            best_idx = i
            best = min(mag_mv(Q2_est - Q2_lst[0]),mag_mv(Q2_est + Q2_lst[0]))
            R_best = R_est

    return R_best


# Gets the index of a vector which is the somewhat collinear to A
def get_idx_col(P,P_lst,eps):
    best = 0
    best_idx = -1
    for i in range(len(P_lst)):
        A,B,C,D = get_coeffs(P)
        A1,B1,C1,D1 = get_coeffs(P_lst[i])
        if abs((A*A1)(0)) > eps:
            return i
    return None
    
# Get index of a vector somewhat orthogonal to
def get_idx_ortho(P,P_lst,eps):
    best = 0
    best_idx = -1
    for i in range(len(P_lst)):
        A,B,C,D = get_coeffs(P)
        A1,B1,C1,D1 = get_coeffs(P_lst[i])
        if abs((A*A1)(0)) < eps:
            return i
    return None

def swap_entries(P_lst,idx1,idx2):
    P = P_lst[idx2]
    P_lst[idx2] = P_lst[idx1]
    P_lst[idx1] = P

def copy_lst(P_lst):
    P_copy = [0]*len(P_lst)
    for i in range(len(P_lst)):
        P_copy[i] = 1*P_lst[i]
    return P_copy

def get_H_funcs(A,B):
    def H_diff(x):
        out = 0
        for i in range(len(A)):
            out += (x*A[i])(0)*inv(B[i])
        return out

    def H_adj(x):
        out = 0
        for i in range(len(B)):
            out += (x*inv(B[i]))(0)*A[i]
        return out

    return H_diff,H_adj

def eigen_decomp_H_plus(H_plus):

    basis,rec_basis = get_cga_basis([1])
    # H_plus = lambda X: (H_diff(X) + H_adj(X))/2

    V,lambda_V  = eigen_decomp(H_plus,basis,rec_basis)
    return V,lambda_V

def check_compare_funcs(F,G):
    basis,_ = get_cga_basis([1])
    values = []
    for i in range(len(basis)):
        values += [(F(basis[i])(1) - G(basis[i])(1)).tolist(1)[0]]
    arr = abs(np.array(values))
    return arr.max()

def numpy_max(X):
    '''Converts to numpy and then computes the max'''
    arr = np.array(X.tolist()[0])
    return abs(arr).max()

def compute_eigvalues_from_H_plus(V,H):
    eigvalues = []
    for i in range(len(V)):
        eigvalues += [H(V[i])*inv(V[i])]
    def H_check(X):
        out = 0
        for i in range(len(V)):
            out += eigvalues[i]*(X|V[i])*inv(V[i])
        return out
    return H_check,eigvalues


def compute_ortho_matrix(a):
    matrix = np.zeros([len(a),len(a)])
    for i in range(len(a)):
        for j in range(len(a)):
            matrix[i][j] = (a[i]|a[j])(0)

    matrix[matrix < 0.00001] = 0
    return matrix

def compute_versor_decomp_CGA(H_diff,H_adj):
    '''Computes the versor of an orthogonal transformation H
    which does not include parabolic rotations
    each simple reflection contributes with a minus sign
    each reflection which squares to -1 contribute with a minus sign
    H(x) = (-1)**k*U*x*inv(U) where U is a k-versor.
    
    The eigenvectors of H_plus might not be orthogonal when the eigenvalue is either one or minus one.
    When the eigenvectors of H_plus are either one or minus one we take the 
    wedge product in this way guaranteeng that the non-orthogonal eigenvectors create a blade 
    instead of a rotor.

    In some cases the a's are repeated!!!! In this situation do not take the wedge product.
    We only take the wedge product when the vector is linear independent
    Note that when the eigenvalue is dfferent from plus or minus one
    then the real part of the eigevalues of H must be distinct.
    '''

    basis,rec_basis = get_cga_basis([1])

    H_plus = lambda X: (H_diff(X) + H_adj(X))/2
    a,lambda_plus  = eigen_decomp(H_plus,basis,rec_basis)

    # Put the ones or minus ones as the first elements  
    indices = np.argsort(abs(abs(lambda_plus) - 1))
    lambda_plus = lambda_plus[indices]
    a = [a[i] for i in indices]

    sign = 1
    U = 1 + 0*e12
    i = 0

    # Find reflections
    while abs(abs(lambda_plus[i]) - 1) < 0.0001:
        # While eigenvalue is equal to one or minus one
        if lambda_plus[i] < -0.9:    
            # Take the wedge product because the a's might not be unique
            U_test = U^a[i]
        
            # Check if U_test is non-zero
            if numpy_max(U_test) > 0.000001:
                # Only take the wedge product when the a[i] is linearly independent
                U = U_test
            sign *= -1
        i += 1
        if i >= len(a):
            break 

    # Find rotations
    while i < len(a):
        U *= the_other_rotor_sqrt(H_diff(a[i])*inv(a[i]))
        i += 2

    # print(U)
    # print(numpy_max(U*~U))
    U = normalize_mv(U)

    # print("Lambda:",lambda_plus)
    # print("a:",compute_ortho_matrix(a))
    return U,sign*(U*~U)(0)


# def compute_decomp_H(H_diff,H_adj):
#     H_plus = lambda X: (H_diff(X) + H_adj(X))/2
#     a,lambda_plus  = eigen_decomp(H_plus,basis,rec_basis)



def best_motor_estimation(V):
    M1 = P_I(V)
    M2 = -P_I(eo|V)
    M = M1 + einf*M2
    
    W = M*~(I*M2) + I*M2*~M
    v = (I*(eo|W))(0)
    u = (I*(eo|(M*~M)))(0)
    s = u/v
    
    Ue = M - s*I*M2
    Re = P_I(Ue)
    Te = 1 - (1/2)*einf*((eo|Ue)*~Re)(1)
    
    return Te,Re

def brute_force_estimate_CGA(P_lst,Q_lst):
    n = 3
    sign_list = get_binary_list(n)
    best = 100000
    best_idx = 0
    R_best = 1
    eps_col = 1e-5
    eps_ortho = 1e-5
    P_lst = copy_lst(P_lst)
    Q_lst = copy_lst(Q_lst)
    P_lst_s = [0]*n

    idx_ortho = get_idx_ortho(P_lst[0],P_lst[1:],eps_ortho) + 1
    swap_entries(P_lst,1,idx_ortho)
    swap_entries(Q_lst,1,idx_ortho)

    idx_col = get_idx_col(P_lst[0],P_lst[2:],eps_col) + 2
    swap_entries(P_lst,2,idx_col)
    swap_entries(Q_lst,2,idx_col)

    for i in range(len(sign_list)):
        for j in range(n): # iterate over two of the eigenvectors 
            P_lst_s[j] = P_lst[j]*sign_list[i][j] # Change the sign
        
        A1,B1,C1,D1 = get_coeffs(P_lst_s[0])
        A2,B2,C2,D2 = get_coeffs(P_lst_s[1])
        A3,B3,C3,D3 = get_coeffs(P_lst_s[2])

        E1,F1,G1,H1 = get_coeffs(Q_lst[0])
        E2,F2,G2,H2 = get_coeffs(Q_lst[1])
        E3,F3,G3,H3 = get_coeffs(Q_lst[2])

        R_est = ~exact_rotation(normalize_mv(E1),normalize_mv(E2),normalize_mv(A1),normalize_mv(A2))
        E3_est = R_est*A3*~R_est
        
        # If the third eigenvector is aligned than it is the right rotation
        if mag_mv(E3_est - E3) < best:
            best_idx = i
            best = mag_mv(E3_est - E3)
            R_best = R_est

    return R_best

def get_metrics(R,R_est,T,T_est):
    t = -eo|T*2
    t_est = -eo|T_est*2
    R_est = normalize_mv(R_est)
    costheta = (R_est*~R)(0)
    if abs(costheta) > 1:
        costheta = 1
    ang_error = np.arccos(costheta)/np.pi*360 # gets two times theta
    if ang_error > 180:
        ang_error = 360 - ang_error

    # Compute the magnitude of tranlation error
    t_error = mag_mv(t - t_est)

    # Compute the error between the planes of rotation
    cosphi = (normalize_mv(R(2))*~normalize_mv(R_est(2)))(0)
    if abs(cosphi) > 1:
        cosphi = 1
    phi = np.arccos(cosphi)/np.pi*180
    if(phi > 180):
        phi = 360 - phi

    return ang_error,t_error,phi