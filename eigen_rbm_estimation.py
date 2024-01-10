from geo_algebra import *

'''
This snippet describes code to determine the rigid body motion between two noisy Point Clouds
To be robust to noise it first solves two eigenvalue problems stated in a multivector space
that is it finds the eigenmultivector of F(X) = pi*X*pi and G(X) = qi*X*qi. Then it orders the 
eigenmultivectors by magnitude of the absolute value. Before using the eigenmultivectors we correct 
the orientation by using the mean and pi*p_bar*pi. Then since both eigenmultivectors are ordered in the 
same manner we do not need to estimate correspondences. The last is to use the rigid body estimator to 
find the rotation and the translation. 
'''


def reflect_list(X_lst,X):
    Y = 0
    for i in range(len(X_lst)):
        Y += X_lst[i]*X*X_lst[i]
    return Y

def orient_multivectors(X_lst,Xv,Xb):
    Y_lst = [0]*len(X_lst)

    for i in range(len(X_lst)):
        X = X_lst[i]
        scalar = (X(1)*Xv)(0) + (X(2)*Xb)(0)
        if scalar == 0:
            scalar = 1
        else:
            scalar = np.sign(scalar)
        Y_lst[i] = scalar*X
    return Y_lst

m_points = 1000
mu = 0
sigma = 0.1

x_lst = generate_rdn_PC(m_points)
theta = 100*np.pi/180
u = normalize(rdn_vanilla_vec())
R = np.cos(theta/2) + I*u*np.sin(theta/2)
t = 10*rdn_vanilla_vec()
T = 1 + (1/2)*einf*t
y_lst = apply_vec_RBM(x_lst,R,t,mu,sigma)

p_lst = vanilla_to_cga_vecs(x_lst)
q_lst = vanilla_to_cga_vecs(y_lst)

q_reorder = q_lst[0]
q_lst[0] = q_lst[1]
q_lst[1] = q_reorder

cga_basis = list(cga.basis(grades=[1,2]).values())

def get_func(X_lst):    
    def F(Y):
        out = 0
        for i in range(len(X_lst)):
            out += X_lst[i]*Y*X_lst[i]
        return out
    return F

cga_rec_basis = reciprocal_blades_cga(cga_basis)
P_lst,lambda_P = eigen_decomp(get_func(p_lst),cga_basis,cga_rec_basis)
Q_lst,lambda_Q = eigen_decomp(get_func(q_lst),cga_basis,cga_rec_basis)

# Normalize the multivectors
P_lst = normalize_null_mvs(P_lst)
Q_lst = normalize_null_mvs(Q_lst)

# Compute the normalized means of the eigenmvs and vectors of the PCs
P_bar = normalize_null(mv_list_mean(P_lst))
Q_bar = normalize_null(mv_list_mean(Q_lst))

# Compute the components used to orient the eigenmultivectors
p_bar = normalize_null(mv_list_mean(p_lst))
q_bar = normalize_null(mv_list_mean(q_lst))
p_ref = normalize_null(reflect_list(p_lst,p_bar))
q_ref = normalize_null(reflect_list(q_lst,q_bar))

p_biv = p_bar^p_ref
q_biv = q_bar^q_ref

P_oriented = orient_multivectors(P_lst,p_bar,p_biv)
Q_oriented = orient_multivectors(Q_lst,q_bar,q_biv)

Q_est_oriented = trans_list(P_oriented,T*R)

T_est,R_est = estimate_rbm(P_oriented,Q_oriented)

print("Angle Error")
print(np.arccos((R_est*~R)(0))/np.pi*360)

'''
# Check that the eigenmultivectors are orthogonal and they are also blades
matrix = np.zeros([len(P_lst),len(P_lst)])
self_prod = [0]*len(P_lst)
for i in range(len(P_lst)):
    self_prod[i] = P_lst[i]*P_lst[i]
    for j in range(len(P_lst)):
        matrix[i][j] = get_float(P_lst[i]*P_lst[j])


# Sanity check if the P's and Q's are eigenmultivectors
Func = get_func(p_lst)
for i in range(len(P_lst)):
    print(P_lst[i]*lambda_P[i] - Func(P_lst[i]))
'''

'''
TODO:
    - For large amounts of noise it is difficult to determine the orientation of 
    the eigenmultivectors, as such by increasing the noise above a certain threshold we are 
    not able to determine the right orientation resulting in a very bad rotation accuracy
    - [ ] Find a more noise robust approach to estimate orientation of multivectors 
    - [ ] Study the algorithm under the influence of outliers
    - [x] Study the solution for the eigenmultivectors, it seems that even though we find the
    eigenvectors of the matrix of F, the corresponding multivectors are not eigenmultivectors
'''
