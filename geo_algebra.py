#!/usr/bin/env python3
import gasparse
import numpy as np
from gasparse import multivector
from sympy import poly, nroots
from sympy.abc import x

vga = gasparse.GA(3)
cga = gasparse.GA(4,1) # Use 3D conformal geometric algebra
basis = cga.basis()
locals().update(basis) # Update all of the basis blades variables into the environment


# compute the cga null basis
einf = (1.0*e5 - 1.0*e4)*(1/np.sqrt(2))
eo = (1.0*e5 + 1.0*e4)*(1/np.sqrt(2))

I = e1*e2*e3
ii = I*(eo^einf)

def inv(X):
    scalar = 1/(X*~X)(0)
    return X*scalar

def normalize(X):
    scalar = 1/np.sqrt((X*~X)(0))
    return X*scalar

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
def rdn_multivector():
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


def mag(X):
    return (X*~X)(0)

def dist(X,Y):
    A,B,C,D = get_coeffs(X-Y)
    return np.sqrt(mag(A) + mag(C) + mag(D))


def pos_cga_dist(X,Y):
    A,B,C,D = get_coeffs(X-Y)
    return np.sqrt(mag(A) +  mag(B) + mag(C) + mag(D))


def comp_dist_matrix(X_list,Y_list):
    matrix = np.zeros([len(X_list),len(Y_list)])
    for i in range(len(X_list)):
        for j in range(len(Y_list)):
            #matrix[i][j] = abs(get_float(X_list[i]|Y_list[j]))
            #matrix[i][j] = dist(X_list[i],Y_list[j])
            matrix[i][j] = pos_cga_dist(X_list[i],Y_list[j])

    return matrix

def trans_list(X_list,U):
    new_list = []
    for i in range(len(X_list)):
        new_list += [U*X_list[i]*~U] 
    return new_list

def proj(B,X):
    return (X|B)*inv(B)

def proj_perp(B,X):
    return X - proj(B,X)


def rotor_sqrt(R):
    theta = np.arccos(R(0))
    B = normalize(R(2))
    return np.cos(theta/2) + B*np.sin(theta/2)


eps = 1E-10
# Normalizes all types of multivectors
# Ignores almost null multivectors
def normalize_null(X):
    magnitude = np.sqrt(abs((X*~X)(0)))
    scalar = 1
    if magnitude < eps:
        if abs(pos_cga_dist(X,X) - magnitude) < eps/2:
            # check if the multivector is null
            scalar = 1/magnitude
    else: 
        scalar = 1/magnitude
    return X*scalar


# Estimate correspondences
def compute_magnitudes(X_list):
    mag_vector = np.zeros([len(X_list),1])
    for i in range(len(X_list)):
        mag_vector[i] = mag(X_list[i])
    return mag_vector

def estimate_corrs_index(X_maglist,Y_maglist):
    Matrix = np.abs(X_maglist - Y_maglist.T)
    return np.argmin(Matrix,axis=1)

def get_corr(X_list,Y_list):
    X_magarray = compute_magnitudes(X_list)
    Y_magarray = compute_magnitudes(Y_list)
    corr_index = estimate_corrs_index(X_magarray,Y_magarray)
    return corr_index

def point_convolution(X_lst,s,gamma,f):
    # convolution for each point
    out_scalars = 0j*np.zeros([len(X_lst),1])

    for i in range(len(X_lst)):
        vec_i = normalize(X_lst[i])
        for j in range(len(X_lst)):
            if i != j:
                scalar = (vec_i|normalize(X_lst[j]))(0)
                theta = np.arccos(scalar)
                magnitude = np.sqrt(mag(X_lst[i] - X_lst[j]))
                out_scalars[i] += np.exp(-s*magnitude**gamma)*np.exp(f*1j*theta)
    return out_scalars

def exp_corrs(X_lst,Y_lst,s,gamma,f):
    alphas = point_convolution(X_lst,s,gamma,f)
    betas = point_convolution(Y_lst,s,gamma,f)
    return estimate_corrs_index(alphas,betas)


def rdn_gaussian_vec(mu,sigma):
    s = np.random.normal(mu, sigma, 3)
    return vga.multivector(list(s),grades=1)


def generate_rdn_PC(m):
    x_lst = []
    for i in range(m):
        x_lst += [10*np.random.rand()*normalize(rdn_vanilla_vec())]
    return x_lst

def generate_unitcube_rdn_PC(m):
    x_lst = []
    for i in range(m):
        x_lst += [rdn_vanilla_vec()]
    return x_lst

def vanilla_to_cga_vecs(x_lst):
    p_lst = []
    for x in x_lst:
        p_lst += [ eo + x + (1/2)*mag(x)*einf] 

    return p_lst

def apply_vec_RBM(x_lst,R,t):
    y_lst = []
    for x in x_lst:
        y_lst += [R*x*~R + t]
    return y_lst

def gen_gaussian_noise_list(m,mu,sigma):
    noise = []
    for i in range(m):
        noise += [rdn_gaussian_vec(mu,sigma)]
    return noise

def add_noise(x_lst,noise):
    y_lst = []
    for i in range(len(x_lst)):
        y_lst += [x_lst[i] + noise[i]]
    return y_lst

def apply_rotation(x_lst,R):
    y_lst = []
    for x in x_lst:
        y_lst += [R*x*~R]
    return y_lst


def convoution(x_lst):
    X_lst = [0]*len(x_lst)
    mean = 0
    for i in range(len(x_lst)):
        mean += x_lst[i]

    for i in range(len(x_lst)):
        biv = x_lst[i]^mean
        theta = np.sqrt(mag(biv))
        U = np.cos(theta) + biv*np.sin(theta)
        X_lst[i] = U*x_lst[i]*~U

    return X_lst


def transform(f_list,p_lst):
    prods = []

    for i in range(len(p_lst)):
        for j in range(i+1,len(p_lst)):
            prods += [p_lst[i]*p_lst[j]]

    F_transform = [0]*len(f_list)

    for k in range(len(f_list)):
        F_transform[k] = 0
        for i in range(len(prods)):
            scalar = (prods[i](0))*2*np.pi*f_list[k]
            F_transform[k] += (prods[i](2))*(np.cos(scalar) + ii*np.sin(scalar))

    return F_transform

def exp(X):
    # To do this computation we are assuming 
    # that X is simple and of unique grade
    s = (X|X)(0)
    if abs(s) < eps:
        # consider a null multivector
        return 1 + X
    elif s < 0:
        theta = np.sqrt(abs(s))
        return np.cos(theta) + X*np.sin(theta)
    elif s > 0:
        gamma = np.sqrt(s)
        return np.cosh(gamma) + X*np.sinh(gamma)

epsilon = 0.00001

def simple_transform(f_list,x_lst):
    F_transform = [0]*len(f_list)
    mean = 0
    for i in range(len(x_lst)):
        mean += x_lst[i]
    mean *= (1/len(x_lst))

    for i in range(len(f_list)):
        #sum_exp = 0
        for j in range(len(x_lst)):
            #exp_value = np.exp(-abs(f_list[i]-np.sqrt(mag(x_lst[j]))))
            #exp_value = np.exp(-f_list[i]*np.sqrt(mag(x_lst[j])))
            #sum_exp += exp_value
            #*np.sqrt(mag(x_lst[j]))
            #angle = 2*np.pi*f_list[i]*np.sqrt(mag(x_lst[j]))
            #F_transform[i] += mean*exp_imag

            angle = np.pi*f_list[i]
            exp_imag = np.cos(angle) + I*np.sin(angle)
            U = np.cos(angle) + x_lst[j]*I*np.sin(angle)
            F_transform[i] += U*mean*~U*exp_imag

        #mag_value = np.sqrt(mag(F_transform[i]))
        #if mag_value > epsilon:
        #   F_transform[i] = F_transform[i]*(1/(mag_value))
    
    return F_transform



# Generate random multivector clouds
def generate_rdn_MCs(m,noise=0):
    Q_lst = []
    P_lst = []

    R = rdn_rotor()
    t = 10*rdn_vanilla_vec()
    T = 1 + (1/2)*einf*t

    for i in range(m):
        Q = 10*np.random.rand()*normalize_null(rdn_multivector())
        P = ~R*~T*Q*T*R + noise*rdn_multivector() # Don't forget to allways add a bunch of noise
        #Q_lst = [Q(1)] + [Q(2)] + [Q(3)] + [Q(4)] + Q_lst
        #P_lst = [P(1)] + [P(2)] + [P(3)] + [P(4)] + P_lst
        Q_lst = [Q(1) + Q(2) + Q(3) + Q(4)] + Q_lst
        P_lst = [P(1) + P(2) + P(3) + P(4)] + P_lst

    return (P_lst,Q_lst)

def estimate_rot(p_lst,q_lst):
    beta_matrix = np.zeros([4,4])
    basis_rotor = [blades['e'],e12,e13,e23]

    for k in range(len(p_lst)):
        p = p_lst[k]
        q = q_lst[k]

        def Func(Rotor):
            #return (~q*Rotor*p + q*Rotor*~p)*(1/(1E-12 + np.sqrt(mag(p) + mag(q))))
            return ~q*Rotor*p + q*Rotor*~p
            
        for i in range(4):
            for j in range(4):
                beta_matrix[i][j] += (Func(basis_rotor[i])*(~basis_rotor[j]))(0)

    eigenvalues, eigenvectors = np.linalg.eig(beta_matrix)
    u = eigenvectors[:,np.argmax(eigenvalues)]# eigenvectors are column vectors
    R_est = 0
    

    for i in range(4):
        R_est += np.real(u[i])*basis_rotor[i]

    R_est = normalize(R_est)
    
    return R_est
    


'''
Try to replace this function by the multilinear eigenvalue estimator function
I still don't compreedn well how can this work since it does not use the reciprocal basis
to form the matrix of F. 
'''

def estimate_rbm(P_lst,Q_lst):
    # The optimal rotation and translation
    beta_matrix = np.zeros([4,4])

    basis_rotor = [e,e12,e13,e23]

    for k in range(len(P_lst)):
        A,B,C,D = get_coeffs(P_lst[k])
        E,F,G,H = get_coeffs(Q_lst[k])

        def F(Rotor):
            return E(1)*Rotor*A(1) - E(2)*Rotor*A(2)
            
        for i in range(4):
            for j in range(4):
                beta_matrix[i][j] += (F(basis_rotor[i])*(~basis_rotor[j]))(0)
        
    eigenvalues, eigenvectors = np.linalg.eig(beta_matrix)
    R_est = 0

    u = eigenvectors[:,np.argmax(eigenvalues)]
    for i in range(4):
        R_est += u[i]*basis_rotor[i]

    R_est = normalize(R_est)
    def Rot(X):
        return R_est*X*~R_est

    scalar = 0
    v = 0
    for i in range(len(P_lst)):
        A,B,C,D = get_coeffs(P_lst[i])
        E,F,G,H = get_coeffs(Q_lst[i])
        
        v += ((Rot(A) + E)*~(G+H-Rot(C+D)))(1)
        scalar += mag(A) + mag(E)
        #print(scalar)

    t_est = v*(1/scalar)
    T_est = 1 + (1/2)*einf*t_est

    return (T_est,R_est)

# From a estimated rotation estimate the translation
def estimate_translation(R_est,P_lst,Q_lst):
    def Rot(X):
        return R_est*X*~R_est

    scalar = 0
    v = 0
    for i in range(len(P_lst)):
        A,B,C,D = get_coeffs(P_lst[i])
        E,F,G,H = get_coeffs(Q_lst[i])
        
        v += ((Rot(A) + E)*~(G+H-Rot(C+D)))(1)
        scalar += mag(A) + mag(E)
        #print(scalar)

    t_est = v*(1/scalar)
    T_est = 1 + (1/2)*einf*t_est
    return T_est

# Estimate the rigid body motion using the eigendecomposition function
def estimate_rbm_1(P_lst,Q_lst):
    vga_rotor_basis = list(vga.basis(grades=[0,2]).values())
    vga_rec_rotor_basis = reciprocal_blades_vga(vga_rotor_basis)

    # define the rotor valued function
    def Func(Rotor):
        out = 0
        for k in range(len(P_lst)):
            A,B,C,D = get_coeffs(P_lst[k])
            E,F,G,H = get_coeffs(Q_lst[k])
            out += E(1)*Rotor*A(1) - E(2)*Rotor*A(2)
        return out
    R_lst,lambda_R = eigen_decomp(Func,vga_rotor_basis,vga_rec_rotor_basis)
    R_est = R_lst[3] # Chose the eigenrotor with the biggest eigenvalue
    T_est = estimate_translation(R_est,P_lst,Q_lst)
    return (T_est,R_est)

def grade_involution(X):
    return X(0) - X(1) + X(2) - X(3) + X(4) - X(5)

def grade_involution_vga(X):
    return X(0) - X(1) + X(2) - X(3)

def grade_project_lst(X_lst,grades):
    Y_lst = [0]*len(X_lst)
    for i in range(len(X_lst)):
        for j in range(len(grades)):
            Y_lst[i] += X_lst[i](grades[j])
    return Y_lst

def mv_list_mean(X_lst):
    # Computes the mean of a list of multivectors
    X_bar = 0
    for i in range(len(X_lst)):
        X_bar += X_lst[i]
    return (1/len(X_lst))*X_bar

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

def normalize_null_mvs(X_lst):
    for i in range(len(X_lst)):
        X_lst[i] = normalize_null(X_lst[i])
    return X_lst


def eigen_decomp(F,basis,rec_basis):
    # Solves the eigendecomposition of a multilinear transformation F
    beta = np.zeros([len(basis),len(basis)])

    for i in range(len(basis)):
        for j in range(len(basis)):
            beta[i][j] += (F(basis[i])*(rec_basis[j]))(0)

    eigenvalues, eigenvectors = np.linalg.eig(beta.T)
    Y = [0]*len(eigenvalues)
    # Convert the eigenvectors to eigenmultivectors
    for i in range(len(eigenvalues)):
        u = np.real(eigenvectors[:,i])
        for j in range(len(basis)):
            Y[i] += u[j]*basis[j]

    #Order eigenmultivectors and eigenvalues by the eigenvalues
    indices = np.argsort(eigenvalues)
    Y_ordered = [Y[i] for i in indices]
    eigenvalues_ordered = eigenvalues[indices]
    
    return Y_ordered,np.real(eigenvalues_ordered)

def estimate_rot_SVD(p_lst,q_lst):
    matrix = np.zeros([3,3])
    basis_vecs = [e1,e2,e3]
    for i in range(len(p_lst)):
        q = q_lst[i]
        p = p_lst[i]
        def f(x):
            return q*(p|x)

        for j in range(len(basis_vecs)):
            for k in range(len(basis_vecs)):
                matrix[j][k] += (f(basis_vecs[j])|basis_vecs[k])(0)

    U, S, V = np.linalg.svd(matrix, full_matrices=True)
    M = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,np.linalg.det(U)*np.linalg.det(V)]])
    rot_matrix = U@M@V

    eigenvalues, eigenvectors = np.linalg.eig(rot_matrix)

    # The axis of rotation
    u = np.real(eigenvectors[:,np.argmax(eigenvalues)])
    Kn = np.array([[0,-u[2],u[1]],
                   [u[2],0,-u[0]],
                   [-u[1],u[0],0]])

    cos_theta = (np.trace(rot_matrix) - 1)/2
    sin_theta = -np.trace(Kn@rot_matrix)/2

    ga_vec = vga.multivector(list(u),grade=1)
    rotor = cos_theta + I*ga_vec*sin_theta

    return rotor_sqrt(rotor)

    #q_numpy = np.zeros([3,len(p_lst)])
    #for i in range(len(p_lst)):
    #    q = q_lst[i]
    #    p = p_lst[i]




def compute_PC_error(t,R,x_lst,y_lst):
    error = 0
    for i in range(len(x_lst)):
        y_est = R*x_lst[i]*~R + t
        error += mag(y_est-y_lst[i])
    return error/len(x_lst)


def compute_error(T_est,R_est,P_lst,Q_lst):
    error = 0
    for i in range(len(P_lst)):
        P = P_lst[i]
        Q = Q_lst[i]
        P_est = ~R_est*~T_est*Q*T_est*R_est
        error += dist(P_est,P)
    return error/len(P_lst)

def compute_error_euclidean(R_est,P_lst,Q_lst):
    error = 0
    for i in range(len(P_lst)):
        P = P_lst[i]
        Q = Q_lst[i]
        Q_est = R_est*P*~R_est
        error += mag(Q_est-Q)
    return error/len(P_lst)