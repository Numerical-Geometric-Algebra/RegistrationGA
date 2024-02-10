#!/usr/bin/env python3
import gasparsegen
import numpy as np
from gasparsegen import multivector as mv
# from sympy import poly, nroots
# from sympy.abc import x


def reciprocal_blades(basis):
    '''Reciprocal blades for positive squaring basis vectors'''
    rec_basis = [0]*len(basis) # reciprocal basis blades
    for i in range(len(basis)):
        rec_basis[i] = ~basis[i]
    return rec_basis

def get_ga_basis(ga,grades):
    basis = list(ga.basis(grades=grades).values())
    rec_basis = reciprocal_blades(basis)
    return (basis,rec_basis)

def nparray_to_mvarray(ga,grade,x_array):
    return ga.multivector(x_array.tolist(),grades=grade)

def nparray_to_vecarray(ga,x_array):
    return ga.multivector(x_array.tolist(),grades=1)

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

def rdn_multivector_array(ga,grades,size):
    mvsize = ga.size(grades)
    ones = 0.5*np.ones([size,mvsize])
    x_array = np.random.rand(size,mvsize) - ones
    return nparray_to_mvarray(ga,grades,x_array)

def rdn_gaussian_multivector_array(mu,sigma,ga,grades,size):
    mvsize = ga.size(grades)
    x_array = np.random.normal(mu,sigma,[size,mvsize])
    return nparray_to_mvarray(ga,grades,x_array)


def rotor_sqrt(R):
    B = normalize_mv(R(2))
    s = (R(2)*~B)(0)*(B*~B)(0)
    c = R(0)
    t = s/(1+c)

    return normalize_mv(1+B*t)

def rotor_sqrt_from_vectors(e,u):
    e = normalize_mv(e)
    u = normalize_mv(u)
    scalar = 1/np.sqrt((e|u)(0) + 1)
    return (1/np.sqrt(2))*scalar*(e*u + 1)

def proj(B,X):
    return (X|B)*inv(B)

def proj_perp(B,X):
    return X - proj(B,X)

def mean(X):
    '''Computes the mean of an array of multivectors
       Still need way to get the length of multivector array
       Do not use!!! Will give error'''
    X_bar = X.sum()/len(X)
    return X_bar
    
def numpy_max(X):
    '''Converts to numpy and then computes the max'''
    arr = np.array(X.tolist()[0])
    return abs(arr).max()


# vga = gasparsegen.GA(3,compute_mode='generated')
# cga = gasparsegen.GA(4,1,compute_mode='generated') # Use 3D conformal geometric algebra
# basis = cga.basis()
# locals().update(basis) # Update all of the basis blades variables into the environment

# # compute the cga null basis
# einf = (1.0*e5 - 1.0*e4)*(1/np.sqrt(2))
# eo = (1.0*e5 + 1.0*e4)*(1/np.sqrt(2))

# I = e1*e2*e3
# ii = I*(eo^einf)

# def grade_involution(X):
#     return X(0) - X(1) + X(2) - X(3) + X(4) - X(5)

# def grade_involution_vga(X):
#     return X(0) - X(1) + X(2) - X(3)

# def reciprocal_blades_cga(basis):
#     rec_basis = [0]*len(basis) # reciprocal basis blades
#     sigma = e5
#     for i in range(len(basis)):
#         rec_basis[i] = -sigma*~grade_involution(basis[i])*sigma
#     return rec_basis



# def get_3dvga_rotor_basis():
#     vga_rotor_basis =  list(vga.basis(grades=[0,2]).values())
#     vga_rec_rotor_basis = reciprocal_blades_vga(vga_rotor_basis)
#     return (vga_rotor_basis,vga_rec_rotor_basis)

# def get_3dcga_basis(grades):
#     cga_basis = list(cga.basis(grades=grades).values())
#     cga_rec_basis = reciprocal_blades_cga(cga_basis)    
#     return (cga_basis,cga_rec_basis)

# def get_vga_basis(grades):
#     vga_basis = list(vga.basis(grades=grades).values())
#     vga_rec_basis = reciprocal_blades_vga(vga_basis)
#     return (vga_basis,vga_rec_basis)

# def nparray_to_vga_vecarray(x_array):
#     return vga.multivector(x_array.tolist(),grades=1)


# def rdn_gaussian_cga_mvarray(mu,sigma,size):
#     mvsize = ga.size()
#     x_array = np.random.normal(mu,sigma,[size,mvsize])
#     return nparray_to_mvarray(ga,grade,x_array)

# def rdn_gaussian_vga_vecarray(mu,sigma,size):
#     return rdn_gaussian_kvector_array(mu,sigma,vga,1,size)

# def rdn_cga_bivector_array(size):
#     return rdn_kvector_array(cga,2,size)

# def rdn_vga_vector_array(size):
#     return rdn_kvector_array(vga,1,size)

# def rdn_cga_mvarray(size):
#     mvsize = cga.size()
#     ones = 0.5*np.ones([size,mvsize])
#     Xlst = (np.random.rand(size,mvsize) - ones).tolist()
#     return cga.multivector(Xlst)

# def rdn_vga_mvarray(size):
#     mvsize = vga.size()
#     ones = 0.5*np.ones([size,mvsize])
#     Xlst = (np.random.rand(size,mvsize) - ones).tolist()
#     return vga.multivector(Xlst)


# def rdn_kvector(ga,grade):
#     size = ga.size(grade)
#     ones = 0.5*np.ones([size])
#     return ga.multivector(list(np.random.rand(size) - ones),grades=grade)

# generate random cga bivector
# def rdn_biv():
#     return cga.multivector(list(np.random.rand(cga.size(2)) - 0.5*np.ones([cga.size(2)])),grades=2)

# generate random vga vector
# def rdn_vanilla_vec():
#     return vga.multivector(list(np.random.rand(vga.size(2))-0.5*np.ones([vga.size(2)])),grades=1)

# def rdn_cga_vec():
#     return rdn_kvector(cga,1)

# generate random cga vector
# def rdn_cga_multivector():
#     return cga.multivector(list(np.random.rand(cga.size()) -0.5*np.ones([cga.size()])))

# def rdn_rotor():
#     a = normalize(rdn_vanilla_vec())
#     b = normalize(rdn_vanilla_vec())
#     return a*b

# def rdn_translator():
#     t = 10*rdn_vanilla_vec()
#     return 1 + (1/2)*einf*t

# def P_I(X):
#     return X(0) + ((X(1) + X(2) + X(3))|I)*~I

# # Extract the coefficints of a multivector 
# # The multivector can be written in the form 
# # X = eo*A + einf*B + (eo^einf)*C + D
# def get_coeffs(X):
#     A = -P_I(einf|X)
#     B = -P_I(eo|X)
#     C = P_I(X|(eo^einf))
#     D = P_I(X)
    
#     #print(eo*A + einf*B + (eo^einf)*C + D -X)
    
#     return [A,B,C,D]



# def gen_rdn_CGA_rotor():
#     a = rdn_cga_vec()
#     b = rdn_cga_vec()
#     B = normalize_mv(a^b)
    
#     if (B*~B)(0) < 0:
#         gamma = 10*np.random.rand()
#         return np.cosh(gamma) + B*np.sinh(gamma)
#     else:
#         theta = np.random.rand()*np.pi
#         return np.cos(theta) + B*np.sin(theta)
    
    

# def dist(X,Y):
#     A,B,C,D = get_coeffs(X-Y)
#     return mv.sqrt(mag_sq(A) + mag_sq(C) + mag_sq(D))

# def dist_sq(X,Y):
#     A,B,C,D = get_coeffs(X-Y)
#     return mv.sqrt(mag_sq(A) + mag_sq(C) + mag_sq(D))

# def pos_dist(X,Y):
#     A,B,C,D = get_coeffs(X-Y)
#     return mv.sqrt(mag_sq(A) +  mag_sq(B) + mag_sq(C) + mag_sq(D))

# def pos_dist_sq(X,Y):
#     A,B,C,D = get_coeffs(X-Y)
#     return mag_sq(A) +  mag_sq(B) + mag_sq(C) + mag_sq(D)

# def plus_norm_sq(X):
#     A,B,C,D = get_coeffs(X)
#     return mag_sq(A) +  mag_sq(B) + mag_sq(C) + mag_sq(D)

# # Normalizes all types of multivectors
# # Ignores almost null multivectors
# # Only works for Multivectors with only a single element
# eps = 0.001
# def normalize_null_mv(X):
#     magnitude = np.sqrt(abs((X*~X)(0)))
#     if(magnitude == 0.0):
#         return X

#     relative_mag = abs(magnitude/np.sqrt(plus_norm_sq(X)))
#     if relative_mag < eps:
#         return X
#     else: 
#         return X/magnitude


# def vga_to_cga(x):
#     p = eo + x + (1/2)*mag_sq(x)*einf
#     return p


# # Generate random multivector arrays
# def generate_rdn_mvarrayss(m,sigma=0.01,mu=0):
    
#     R = rdn_rotor()
#     t = 10*rdn_vanilla_vec()
#     T = 1 + (1/2)*einf*t

#     P = rdn_cga_mvarray(m)
#     N = rdn_gaussian_cga_mvarray(m)

#     Q = T*R*P*~R*~T + N

#     return (P,Q)





# def normalize_null_mvs(X_lst):
#     for i in range(len(X_lst)):
#         X_lst[i] = normalize_null(X_lst[i])
#     return X_lst


# def get_matrix(F,basis,rec_basis):
#     beta = np.zeros([len(basis),len(basis)])
#     mv_lst = [0]*len(basis)

#     for i in range(len(basis)):
#         mv_lst[i] = F(basis[i])

#     for i in range(len(basis)):
#         for j in range(len(basis)):
#             beta[i][j] += (mv_lst[i]*(rec_basis[j]))(0)
    
#     return beta

# This function computes a function from a matrix 
# We assume that the matrix was obtained via the above get_matrix function
# def get_func_from_matrix(beta,basis,rec_basis):
#     def F(X):
#         out = 0
#         for i in range(len(matrix)):
#             for j in range(len(matrix)):
#               out += beta[i][j]*(X*rec_basis[i])(0)*basis[j]
#         return out
#     return F

# def convert_numpyeigvecs_to_eigmvs(eigenvalues, eigenvectors,basis,rec_basis):
#     Y = [0]*len(eigenvalues)
#     # Convert the eigenvectors to eigenmultivectors
#     for i in range(len(eigenvalues)):
#         u = np.real(eigenvectors[:,i])
#         for j in range(len(basis)):
#             Y[i] += u[j]*basis[j]

#     #Order eigenmultivectors and eigenvalues by the absolute value of the eigenvalues
#     indices = np.argsort(abs(eigenvalues))
#     Y_ordered = [Y[i] for i in indices]
#     eigenvalues_ordered = eigenvalues[indices]
#     for i in range(len(Y_ordered)):
#         Y_ordered[i] = Y_ordered[i]
#         # Y_ordered[i] = normalize_mv(Y_ordered[i])
#         # Y_ordered[i] = normalize_null(Y_ordered[i])
#     return Y_ordered,np.real(eigenvalues_ordered)

# def eigen_decomp(F,basis,rec_basis):
#     # Solves the eigendecomposition of a multilinear transformation F
#     beta = get_matrix(F,basis,rec_basis)
#     eigenvalues, eigenvectors = np.linalg.eig(beta.T)

    # return convert_numpyeigvecs_to_eigmvs(eigenvalues,eigenvectors,basis,rec_basis)

# def translation_from_cofm(y,x,R_est,n_points):
#     # Estimate the translation using the center of mass
#     z_est = R_est*x*~R_est
#     t_est = (y.sum() - z_est.sum())/n_points
#     T_est = 1 + (1/2)*einf*t_est

#     return T_est

# # From a estimated rotation estimate the translation
# def estimate_translation(R,P,Q):
#     def Rot(X):
#         return R*X*~R
    
#     A,B,C,D = get_coeffs(P)
#     E,F,G,H = get_coeffs(Q)
    
#     # Use only grade two elements
#     # v = ((Rot(A) + E)(1)*~(G+H-Rot(C+D))(2))(1).sum()
#     v = ((Rot(A) + E)*~(G+H-Rot(C+D)))(1).sum()
#     scalar = (mag_sq(A) + mag_sq(E)).sum()

#     t_est = v/scalar
#     T_est = 1 + (1/2)*einf*t_est
#     return T_est

# # From a estimated rotation estimate the translation
# def estimate_translation_2(R,P,Q):

#     A,B,C,D = get_coeffs(P)
#     E,F,G,H = get_coeffs(Q)
    
#     v = ((A + E)*~(G+H-C-D))(1).sum()
#     s = ((A*~A)(0) + (E*~E)(0)).sum()

#     t_est = v/s
#     T_est = 1 + (1/2)*einf*t_est
#     return T_est


# # Estimate the rigid body motion using the eigendecomposition function
# def estimate_rbm(P,Q):
#     basis,rec_basis = get_vga_rotor_basis()
#     A,B,C,D = get_coeffs(P)
#     E,F,G,H = get_coeffs(Q)

#     # define the rotor valued function
#     def Func(Rotor):
#         return (E(1)*Rotor*A(1) - E(2)*Rotor*A(2)).sum() 
        
#     R_lst,lambda_R = eigen_decomp(Func,basis,rec_basis)
#     R_est = R_lst[3] # Chose the eigenrotor with the biggest eigenvalue
#     T_est = estimate_translation(R_est,P,Q)
#     return (T_est,R_est)

# # Estimate the rotation from two multivector CGA arrays
# def estimate_rot_CGA(P,S):
#     basis,rec_basis = get_vga_rotor_basis()
#     A,B,C,D = get_coeffs(P)
#     J,K,L,M = get_coeffs(S)
    
#     # Define the rotor valued function
#     def Func(R):
#         return (J*R*~A + ~J*R*A + K*R*~B + ~K*R*B + L*R*~C + ~L*R*C + M*R*~D + ~M*R*D).sum()
    
#     R_lst,lambda_R = eigen_decomp(Func,basis,rec_basis)
#     R_est = R_lst[3] # Chose the eigenrotor with the biggest eigenvalue
#     return R_est

# def estimate_rot_VGA(X,Y):
#     basis,rec_basis = get_vga_rotor_basis()

#     def Func(R):
#         return (~Y*R*X + Y*R*~X).sum()

#     R_lst,lambda_R = eigen_decomp(Func,basis,rec_basis)
#     R_est = R_lst[3] # Chose the eigenrotor with the biggest eigenvalue
#     return R_est

# def check_compare_funcs(F,G,basis):
#     values = []
#     for i in range(len(basis)):
#         values += [(F(basis[i])(1) - G(basis[i])(1)).tolist(1)[0]]
#     arr = abs(np.array(values))
#     return arr.max()

# def rotor_sqrt_mv(R):
#     if(R(0) > 1):
#         theta = 0
#     else:
#         theta = np.arccos(R(0))    
#     B = normalize_mv(R(2))
#     return np.cos(theta/2) + B*np.sin(theta/2)

# def rotor_sqrt(R):
#     theta = mv.arccos(R(0))
#     B = normalize(R(2))
#     return mv.cos(theta/2) + B*mv.sin(theta/2)

# def the_other_rotor_sqrt(R):
#     B = normalize_mv(R(2))
#     s = (R(2)*~B)(0)*(B*~B)(0)
#     c = R(0)
#     t = s/(1+c)

#     return normalize_mv(1+B*t)

# def exact_rotation(a1,a2,b1,b2):
#     n = normalize_mv(I*((a1-b1)^(a2-b2)))
#     Pperp_a1 = (a1|n)*n
#     P_a1 = (a1^n)*n
#     R_sq = (b1 - Pperp_a1)*inv(P_a1)
#     return rotor_sqrt_mv(R_sq)

# def rotmatrix_to_rotor(rot_matrix):
#     eigenvalues, eigenvectors = np.linalg.eig(rot_matrix)

#     # The axis of rotation
#     u = np.real(eigenvectors[:,np.argmax(eigenvalues)])
#     Kn = np.array([[0,-u[2],u[1]],
#                    [u[2],0,-u[0]],
#                    [-u[1],u[0],0]])

#     cos_theta = (np.trace(rot_matrix) - 1)/2
#     sin_theta = -np.trace(Kn@rot_matrix)/2

#     u = vga.multivector(list(u),grades=1) # Convert to VGA
#     rotor = cos_theta - I*u*sin_theta

#     return the_other_rotor_sqrt(rotor)

# def the_other_rotmatrix_to_rotor(rot_matrix):
#     eigenvalues, eigenvectors = np.linalg.eig((rot_matrix + rot_matrix.T)/2)

#     # The eigenvector with eigenvalue of smallest magnitude 
#     u = np.real(eigenvectors[:,np.argmin(abs(eigenvalues))])
#     v = rot_matrix@u
#     u = vga.multivector(list(u),grades=1) # Convert to VGA
#     v = vga.multivector(list(v),grades=1) # Convert to VGA

#     rotor = normalize_mv(u)*normalize_mv(v)

#     return ~the_other_rotor_sqrt(rotor)

# def estimate_rot_SVD(p,q):
#     matrix = np.zeros([3,3])
#     basis_vecs = [e1,e2,e3]

#     def f(x):
#         return (q*(p|x)).sum()

#     for j in range(len(basis_vecs)):
#         for k in range(len(basis_vecs)):
#             matrix[j][k] += (f(basis_vecs[j])|basis_vecs[k])(0)

#     U, S, V = np.linalg.svd(matrix, full_matrices=True)
#     M = np.array([[1,0,0],
#                   [0,1,0],
#                   [0,0,np.linalg.det(U)*np.linalg.det(V)]])
#     rot_matrix = U@M@V

#     R = rotmatrix_to_rotor(rot_matrix)
#     return R



# # Still need to divide all of these by the length of the array
# def compute_PC_error(t,R,x,y):
#     y_est = R*x*~R + t
#     return mag_sq(y_est - y).sum()

# def compute_error(T,R,P,Q):
#     P_est = ~R*~T*Q*T*R
#     return dist(P_est,P).sum()

# def compute_error_euclidean(R,P,Q):
#     Q_est = R*P*~R
#     return mag_sq(Q_est-Q).sum()



