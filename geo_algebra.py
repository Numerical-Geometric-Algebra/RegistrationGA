#!/usr/bin/env python3
import gasparsegen
import numpy as np
from gasparsegen import multivector as mv
from sympy import poly, nroots
from sympy.abc import x

vga = gasparsegen.GA(3,compute_mode='generated')
cga = gasparsegen.GA(4,1,compute_mode='generated') # Use 3D conformal geometric algebra
basis = cga.basis()
locals().update(basis) # Update all of the basis blades variables into the environment

# compute the cga null basis
einf = (1.0*e5 - 1.0*e4)*(1/np.sqrt(2))
eo = (1.0*e5 + 1.0*e4)*(1/np.sqrt(2))

I = e1*e2*e3
ii = I*(eo^einf)

def grade_involution(X):
    return X(0) - X(1) + X(2) - X(3) + X(4) - X(5)

def grade_involution_vga(X):
    return X(0) - X(1) + X(2) - X(3)

def reciprocal_blades_cga(basis):
    rec_basis = [0]*len(basis) # reciprocal basis blades
    sigma = e5
    for i in range(len(basis)):
        rec_basis[i] = -sigma*~grade_involution(basis[i])*sigma
    return rec_basis

def reciprocal_blades_vga(basis):
    rec_basis = [0]*len(basis) # reciprocal basis blades
    for i in range(len(basis)):
        rec_basis[i] = ~basis[i]
    return rec_basis

def get_vga_rotor_basis():
    vga_rotor_basis =  list(vga.basis(grades=[0,2]).values())
    vga_rec_rotor_basis = reciprocal_blades_vga(vga_rotor_basis)
    return (vga_rotor_basis,vga_rec_rotor_basis)

def get_cga_basis(grades):
    cga_basis = list(cga.basis(grades=grades).values())
    cga_rec_basis = reciprocal_blades_cga(cga_basis)    
    return (cga_basis,cga_rec_basis)

def nparray_to_mvarray(x_array):
    return vga.multivector(x_array.tolist(),grades=1)

def mag_mv(X):
    return np.sqrt(abs((X*~X)(0)))

def normalize_mv(X):
    return X/mag_mv(X)

def mag_sq(X):
    return (X*~X)(0)

def mag(X):
    return mv.sqrt(abs((X*~X)(0)))

def inv(X): # Element wise inversion
    return ~X/mag_sq(X)

def normalize(X): # Element wise normalization
    return X/mag(X)

def rdn_kvector_array(ga,grade,size):
    mvsize = ga.size(grade)
    ones = 0.5*np.ones([size,mvsize])
    x_array = np.random.rand(size,mvsize) - ones
    return nparray_to_mvarray(ga,grade,x_array)

def rdn_gaussian_kvector_array(mu,sigma,ga,grade,size):
    mvsize = ga.size(grade)
    x_array = np.random.normal(mu,sigma,[size,mvsize])
    return nparray_to_mvarray(ga,grade,x_array)

def rdn_gaussian_cga_mvarray(mu,sigma,size):
    mvsize = ga.size()
    x_array = np.random.normal(mu,sigma,[size,mvsize])
    return nparray_to_mvarray(ga,grade,x_array)

def rdn_gaussian_vga_array(mu,sigma,size):
    return rdn_gaussian_kvector_array(mu,sigma,vga,1,size)

def rdn_cga_bivector_array(size):
    return rdn_kvector_array(cga,2,size)

def rdn_vga_vector_array(size):
    return rdn_kvector_array(vga,1,size)

def rdn_cga_mvarray(size):
    mvsize = cga.size()
    ones = 0.5*np.ones([size,mvsize])
    Xlst = (np.random.rand(size,mvsize) - ones).tolist()
    return cga.multivector(Xlst)

def rdn_vga_mvarray(size):
    mvsize = vga.size()
    ones = 0.5*np.ones([size,mvsize])
    Xlst = (np.random.rand(size,mvsize) - ones).tolist()
    return vga.multivector(Xlst)


def rdn_kvector(ga,grade):
    size = ga.size(grade)
    ones = 0.5*np.ones([size])
    return ga.multivector(list(np.random.rand(size) - ones),grades=grade)

# generate random cga bivector
def rdn_biv():
    return cga.multivector(list(np.random.rand(cga.size(2)) - 0.5*np.ones([cga.size(2)])),grades=2)

# generate random vga vector
def rdn_vanilla_vec():
    return vga.multivector(list(np.random.rand(vga.size(2))-0.5*np.ones([vga.size(2)])),grades=1)

# generate random cga vector
def rdn_cga_multivector():
    return cga.multivector(list(np.random.rand(cga.size()) -0.5*np.ones([cga.size()])))

def rdn_rotor():
    a = normalize(rdn_vanilla_vec())
    b = normalize(rdn_vanilla_vec())
    return a*b

def rdn_translator():
    t = 10*rdn_vanilla_vec()
    return 1 + (1/2)*einf*t

def P_I(X):
    return X(0) + ((X(1) + X(2) + X(3))|I)*~I

# Extract the coefficints of a multivector 
# The multivector can be written in the form 
# X = eo*A + einf*B + (eo^einf)*C + D
def get_coeffs(X):
    A = -P_I(einf|X)
    B = -P_I(eo|X)
    C = P_I(X|(eo^einf))
    D = P_I(X)
    
    #print(eo*A + einf*B + (eo^einf)*C + D -X)
    
    return [A,B,C,D]


def sqrt_rotor(e,u):
    e = normalize(e)
    u = normalize(u)
    scalar = 1/np.sqrt((e|u)(0) + 1)
    return (1/np.sqrt(2))*scalar*(e*u + 1)


def dist(X,Y):
    A,B,C,D = get_coeffs(X-Y)
    return mv.sqrt(mag_sq(A) + mag_sq(C) + mag_sq(D))

def dist_sq(X,Y):
    A,B,C,D = get_coeffs(X-Y)
    return mv.sqrt(mag_sq(A) + mag_sq(C) + mag_sq(D))

def pos_dist(X,Y):
    A,B,C,D = get_coeffs(X-Y)
    return mv.sqrt(mag_sq(A) +  mag_sq(B) + mag_sq(C) + mag_sq(D))

def pos_dist_sq(X,Y):
    A,B,C,D = get_coeffs(X-Y)
    return mag_sq(A) +  mag_sq(B) + mag_sq(C) + mag_sq(D)

def proj(B,X):
    return (X|B)*inv(B)

def proj_perp(B,X):
    return X - proj(B,X)

# Normalizes all types of multivectors
# Ignores almost null multivectors
# Only works for Multivectors with only a single element
eps = 1E-10
def normalize_null(X):
    magnitude = np.sqrt(abs((X*~X)(0)))
    scalar = 1
    if magnitude < eps:
        if abs(pos_cga_dist(X,X) - magnitude) < eps/2:
            scalar = 1/magnitude
    else: 
        scalar = 1/magnitude
    return X*scalar


def vga_to_cga(x):
    p = eo + x + (1/2)*mag_sq(x)*einf
    return p


# Generate random multivector arrays
def generate_rdn_mvarrayss(m,sigma=0.01,mu=0):
    
    R = rdn_rotor()
    t = 10*rdn_vanilla_vec()
    T = 1 + (1/2)*einf*t

    P = rdn_cga_mvarray(m)
    N = rdn_gaussian_cga_mvarray(m)

    Q = T*R*P*~R*~T + N

    return (P,Q)



def mean(X):
    # Still need way to get the length of multivector array
    # Computes the mean of a list of multivectors
    X_bar = X.sum()/len(X)
    return X_bar
    

def normalize_null_mvs(X_lst):
    for i in range(len(X_lst)):
        X_lst[i] = normalize_null(X_lst[i])
    return X_lst


def eigen_decomp(F,basis,rec_basis):
    # Solves the eigendecomposition of a multilinear transformation F
    beta = np.zeros([len(basis),len(basis)])
    mv_lst = [0]*len(basis)

    for i in range(len(basis)):
        mv_lst[i] = F(basis[i])

    for i in range(len(basis)):
        for j in range(len(basis)):
            beta[i][j] += (mv_lst[i]*(rec_basis[j]))(0)

    eigenvalues, eigenvectors = np.linalg.eig(beta.T)
    Y = [0]*len(eigenvalues)
    # Convert the eigenvectors to eigenmultivectors
    for i in range(len(eigenvalues)):
        u = np.real(eigenvectors[:,i])
        for j in range(len(basis)):
            Y[i] += u[j]*basis[j]

    #Order eigenmultivectors and eigenvalues by the absolute value of the eigenvalues
    indices = np.argsort(abs(eigenvalues))
    Y_ordered = [Y[i] for i in indices]
    eigenvalues_ordered = eigenvalues[indices]
    
    return Y_ordered,np.real(eigenvalues_ordered)

def translation_from_cofm(y,x,R_est,n_points):
    # Estimate the translation using the center of mass
    z_est = R_est*x*~R_est
    t_est = (y.sum() - z_est.sum())/n_points
    T_est = 1 + (1/2)*einf*t_est

    return T_est

# From a estimated rotation estimate the translation
def estimate_translation(R,P,Q):
    def Rot(X):
        return R*X*~R
    
    A,B,C,D = get_coeffs(P)
    E,F,G,H = get_coeffs(Q)
    
    # Use only grade two elements
    # v = ((Rot(A) + E)(1)*~(G+H-Rot(C+D))(2))(1).sum()
    v = ((Rot(A) + E)*~(G+H-Rot(C+D)))(1).sum()
    scalar = (mag_sq(A) + mag_sq(E)).sum()

    t_est = v/scalar
    T_est = 1 + (1/2)*einf*t_est
    return T_est

# Estimate the rigid body motion using the eigendecomposition function
def estimate_rbm(P,Q):
    basis,rec_basis = get_vga_rotor_basis()
    A,B,C,D = get_coeffs(P)
    E,F,G,H = get_coeffs(Q)

    # define the rotor valued function
    def Func(Rotor):
        return (E(1)*Rotor*A(1) - E(2)*Rotor*A(2)).sum() 
        
    R_lst,lambda_R = eigen_decomp(Func,basis,rec_basis)
    R_est = R_lst[3] # Chose the eigenrotor with the biggest eigenvalue
    T_est = estimate_translation(R_est,P,Q)
    return (T_est,R_est)

# Estimate the rotation from two multivector CGA arrays
def estimate_rot_CGA(P,S):
    basis,rec_basis = get_vga_rotor_basis()
    A,B,C,D = get_coeffs(P)
    J,K,L,M = get_coeffs(S)
    
    # Define the rotor valued function
    def Func(R):
        return (J*R*~A + ~J*R*A + K*R*~B + ~K*R*B + L*R*~C + ~L*R*C + M*R*~D + ~M*R*D).sum()
    
    R_lst,lambda_R = eigen_decomp(Func,basis,rec_basis)
    R_est = R_lst[3] # Chose the eigenrotor with the biggest eigenvalue
    return R_est

def estimate_rot_vga(X,Y):
    basis,rec_basis = get_vga_rotor_basis()

    def Func(R):
        return (~Y*R*X + Y*R*~X).sum()

    R_lst,lambda_R = eigen_decomp(Func,basis,rec_basis)
    R_est = R_lst[3] # Chose the eigenrotor with the biggest eigenvalue
    return R_est
    

def rotor_sqrt_mv(R):
    theta = np.arccos(R(0))
    B = normalize_mv(R(2))
    return np.cos(theta/2) + B*np.sin(theta/2)

def rotor_sqrt(R):
    theta = mv.arccos(R(0))
    B = normalize(R(2))
    return mv.cos(theta/2) + B*mv.sin(theta/2)

def rotmatrix_to_rotor(rot_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(rot_matrix)

    # The axis of rotation
    u = np.real(eigenvectors[:,np.argmax(eigenvalues)])
    Kn = np.array([[0,-u[2],u[1]],
                   [u[2],0,-u[0]],
                   [-u[1],u[0],0]])

    cos_theta = (np.trace(rot_matrix) - 1)/2
    sin_theta = -np.trace(Kn@rot_matrix)/2

    u = vga.multivector(list(u),grades=1) # Convert to VGA
    rotor = cos_theta + I*u*sin_theta

    return rotor_sqrt_mv(rotor)


def estimate_rot_SVD(p,q):
    matrix = np.zeros([3,3])
    basis_vecs = [e1,e2,e3]

    def f(x):
        return (q*(p|x)).sum()

    for j in range(len(basis_vecs)):
        for k in range(len(basis_vecs)):
            matrix[j][k] += (f(basis_vecs[j])|basis_vecs[k])(0)

    U, S, V = np.linalg.svd(matrix, full_matrices=True)
    M = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,np.linalg.det(U)*np.linalg.det(V)]])
    rot_matrix = U@M@V

    R = rotmatrix_to_rotor(rot_matrix)
    return R



# Still need to divide all of these by the length of the array
def compute_PC_error(t,R,x,y):
    y_est = R*x*~R + t
    return mag_sq(y_est - y).sum()

def compute_error(T,R,P,Q):
    P_est = ~R*~T*Q*T*R
    return dist(P_est,P).sum()

def compute_error_euclidean(R,P,Q):
    Q_est = R*P*~R
    return mag_sq(Q_est-Q).sum()



