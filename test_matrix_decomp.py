from benchmark_rot import *

def rotor_sqrt_choose(R):
    B = normalize_mv(R(2))
    if (B*B)(0) > 0:
        if R(0) < 1:
            gamma = 0
        else:
            gamma = np.arccosh(abs(R(0)))
        return np.cosh(gamma/2) + B*np.sinh(gamma/2)
    else:
        return the_other_rotor_sqrt(R)

eps = 1e-10
def compute_rotor(V,lambda_V,F_diff):
    A = 1
    i = 0
    while i < len(V):
        if i < len(V)-1:
            if abs(lambda_V[i+1]-lambda_V[i]) < eps:
                R = V[i]*F_diff(V[i])
                A *= rotor_sqrt_choose(R)
                i += 1
            elif lambda_V[i] < 0:
                A *= V[i]
        i += 1
    return ~A

# Computes the spectral decomposition when the matrix P_lst is the rotation matrix
def spectral_decomposition(P_lst):
    F_diff,F_adj = get_rot_func(P_lst)

    # Create a symmetric transformation
    def F_plus(x):
        return (F_diff(x) + F_adj(x))/2
    
    basis,rec_basis = get_cga_basis(1)
    V,lambda_V = eigen_decomp(F_plus,basis,rec_basis)

    # Determine the last eigenvector
    # Apparently it is somehow missing
    # V[4] = (V[0]^V[1]^V[2]^V[3])*ii

    R = compute_rotor(V,lambda_V,F_diff)
    return R

    # lambda_V = [0]*len(V)

    # for i in range(len(V)):
    #     lambda_V[i] = F_adj(V[i])*V[i]

    # return V,lambda_V

def spectral_decomposition_xy(x,y,npoints):
    p = eo + x + (1/2)*mag_sq(x)*einf
    q = eo + y + (1/2)*mag_sq(y)*einf

    P_lst,lambda_P = get_eigmvs(p,grades=1)
    Q_lst,lambda_Q = get_eigmvs(q,grades=1)

    for i in range(len(P_lst)):
        P_lst[i] = P_lst[i]*np.sign((P_lst[i]*einf)(0))
        Q_lst[i] = Q_lst[i]*np.sign((Q_lst[i]*einf)(0))

    F_diff,F_adj = get_rot_func(P_lst)
    G_diff,G_adj = get_rot_func(Q_lst)

    def H_diff(X):
        return G_adj(F_diff(X))

    def H_adj(X):
        return F_adj(G_diff(X))

    def H_plus(X):
        return (G_adj(F_diff(X)) + F_adj(G_diff(X)))/2
    
    def H_minus(X):
        return (G_adj(F_diff(X)) - F_adj(G_diff(X)))/2

    basis,rec_basis = get_cga_basis(1)
    W,lambda_W = eigen_decomp(H_plus,basis,rec_basis)
    
    # W[4] = (W[0]^W[1]^W[2]^W[3])*ii

    V = W
    V[4] = (W[0]^W[1]^W[2]^W[3])*ii

    # print(H_plus(W[4]) - W[4])
    # lambda_U = []
    # for i in range(len(W)):
    #     lambda_U += [~normalize_mv(W[i]*H_minus(W[i]) + lambda_W[i])]

    return W,lambda_W


def check_orthogonal(W):
    gamma = np.zeros([len(W),len(W)])
    for i in range(len(W)):
        for j in range(len(W)):
            gamma[i][j] = (W[i]|W[j])(0)
    return 1*(abs(gamma) > 1e-5)
    
# Remove linear dependent eigen vectors
def remove_LDE(V,lambda_V):
    # Create a copy of V
    W = [0]*len(V)
    lambda_W = [0]*len(lambda_W)

    for i in range(len(V)):
        W[i] = 1*V[i]
        lambda_W[i] = 1*lambda_V[i]
    
    # Remove linearly dependent vectors
    i = 0
    while i < len(W):
        j = 0
        while j < len(W):
            if i != j:
                if (W[i]|inv(W[j]))(0) > 1e-5:
                    del W[i]
                    del lambda_W[i]
                    print("Deleting repeated eigenvectors and eigenvalues")
                    break
            j += 1
        i += 1

    return W,lambda_W


def compute_complex_eigenvalues(H_diff,W):
    # Note that it is also possible to compute the eigenvalues 
    # Using the skew symmetric part of H
    # for i in range(len(W)):
    #     lambda_H += [inv(W[i])*H_minus(W[i]) + lambda_W[i]]

    # Here we use H directly
    lambda_H = []
    for i in range(len(W)):
        l = inv(W[i])*H_diff(W[i])
        lambda_H += [l]
    
    return W,lambda_H

eps = 1e-12
def scaled_RBM_versors(W):
    V = -eo|(W*einf)
    gamma = np.log((V*~V)(0))

    S = np.cosh(gamma/2) + (eo^einf)*np.sinh(gamma/2)
    R = V*np.exp(-gamma/2)
    T = V*~S*~R
    return (T,R,S)

def RBM_versors(W):
    # W = T*R
    R = -eo|(W*einf)
    T = W*~R
    # t = P_I(-2*eo|T)
    # t = -2*(eo|W)*~R
    # T = 1 + (1/2)*einf*t

    return (T,normalize_mv(P_I(R)))



def get_H_funcs(P_lst,Q_lst):
    F_diff,F_adj = get_rot_func(P_lst)
    G_diff,G_adj = get_rot_func(Q_lst)

    def H_diff(X):
        return G_adj(F_diff(X))

    def H_adj(X):
        return F_adj(G_diff(X))

    def H_plus(X):
        return (G_adj(F_diff(X)) + F_adj(G_diff(X)))/2
    
    def H_minus(X):
        return (G_adj(F_diff(X)) - F_adj(G_diff(X)))/2

    return H_diff,H_adj,H_plus,H_minus

def check_square_root(R,R_sq):
    print("Type of rotor:",(R(2)*~R(2))(0))
    print("Normalized R:",(R*~R)(0))
    print("Normalized R_sq:",(R_sq*~R_sq)(0))
    if R(0) < 1:
        print("theta:",np.arccos(R(0))/np.pi*180)
    else:
        print("gamma:",np.arccosh(R(0)))

    print(abs(np.array((R_sq*R_sq - R).tolist()[0])).max())
    print()

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(f"/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper_res2.ply")
    pts = np.asarray(pcd.points)
    sigma = 0
    npoints = pts.shape[0]

    T,R = gen_pseudordn_rbm(100,10)
    t = -2*eo|T
    noise = rdn_gaussian_vga_vecarray(0,sigma,npoints)

    x = nparray_to_vga_vecarray(pts)
    y = R*x*~R + t + noise
    # y = x + t

    p = eo + x + (1/2)*mag_sq(x)*einf
    q = eo + y + (1/2)*mag_sq(y)*einf

    P_lst,lambda_P = get_eigmvs(p,grades=1)
    Q_lst,lambda_Q = get_eigmvs(q,grades=1)

    H_diff,H_adj,H_plus,H_minus = get_H_funcs(P_lst,Q_lst)

    basis = [eo,e1,e2,e3,einf]
    rec_basis = [-einf,e1,e2,e3,-eo]

    H_matrix = get_matrix(H_diff,basis,rec_basis).T

    t_vec = H_matrix[1:4,0]
    R_matrix = H_matrix[1:4,1:4]

    R_est = rotmatrix_to_rotor(R_matrix)

    # basis,rec_basis = get_cga_basis(1)
    # W,lambda_W = eigen_decomp(H_plus,basis,rec_basis)

    # V,lambda_V = compute_complex_eigenvalues(H_diff,W)

    # for i in range(len(lambda_V)):
    #     # print(lambda_V[i]*~lambda_V[i])
    #     L_sq = the_other_rotor_sqrt(lambda_V[i])
    #     check_square_root(lambda_V[i],L_sq)

    # print("Checking Complex Eigenvalues") 
    # for i in range(len(V)):
    #     print(inv(V[i])*H_diff(V[i]) - lambda_V[i])

    # print("Checking if real eigenvectors are correct:")
    # for i in range(len(W)):
    #     print(H_plus(W[i])*inv(W[i]))

    # print("The eigenvalues:")
    # print(lambda_W)

    # beta = get_matrix(H_plus,basis,rec_basis)
    # print(beta - beta.T)

    # gamma = np.zeros([len(W),len(W)])
    # for i in range(len(W)):
    #     for j in range(len(W)):
    #         gamma[i][j] = (W[i]|inv(W[j]))(0)

    # gamma = (gamma > 1e-5)*1 - np.eye(len(W))
    # print(gamma.sum() > 1.999999)

    # i = 0
    # while i < len(W):
    #     j = 0
    #     while j < len(W):
    #         if i != j:
    #             if (W[i]|inv(W[j]))(0) > 1e-5:
    #                 del W[i]
    #                 break
    #         j += 1
    #     i += 1

    # if len(W) == len(lambda_W)-1:
    #     A = 1
    #     for i in range(len(W)):
    #         A *= W[i]
    #     W += [A*ii]

    # for i in range(len(W)):
    #     print(W[i])

    

    # A = 0
    # if(len(W) < len(lambda_W)):
    #     A = 1
    #     for i in range(len(W)):
    #         A ^= W[i]
    #     A = normalize_mv(A(len(W))|ii)

    # A = normalize_mv((W[0]^W[1])|ii)
    # W = [W[0],W[1]]

    # U = 1
    # for i in range(len(lambda_V)):
    #     U *= lambda_V[i]

    # U_lst = []
    # for i in range(len(lambda_V)):
    #     U_lst += [the_other_rotor_sqrt(normalize_mv(lambda_V[i]))]
    
    # print("Confirming the square root:\n")
    # for i in range(len(lambda_V)):
    #     print("Normalized:",lambda_V[i]*~lambda_V[i])
    #     print(abs(np.array((U_lst[i]*U_lst[i] - lambda_V[i]).tolist()[0])).max())
    #     print()


    # def H_check(X):
    #     # return (~U*X*U)(1)
    #     out = 0
    #     for i in range(len(V)):
    #         rr = rotor_sqrt_choose(lambda_V[i])
    #         out += (~rr*((X|V[i])*inv(V[i]))(1)*rr)(1)
    #         # out += (~rr*X*rr)(1)

    #     # if(abs(mag_sq(A)) > 0):
    #     #     out += ((X|A)*inv(A))(1)
    #     return out

    # print("Check for W")
    # for i in range(len(W)):
    #     print(H_diff(W[i]) - H_check(W[i]))

    # print(H_diff(W[0] + 5*W[1] + 7*W[2] + 9*W[3]) - H_check(W[0] + 5*W[1] + 7*W[2]+9*W[3]))
    # print(H_diff(W[0] + 5*W[1] + 7*W[2] + 9*W[3]+W[4]) - H_check(W[0] + 5*W[1] + 7*W[2]+9*W[3]+W[4]))
    # print(H_diff(W[0] + 5*W[1] + 7*W[2]) - H_check(W[0] + 5*W[1] + 7*W[2]))

    # print(lambda_W)
    # print()
    # print(H_diff(e1) - H_check(e1))
    # print(H_diff(e2) - H_check(e2))


    # print(lambda_W)

    

    # for i in range(len(gamma)):
    #     if gamma[i].sum() > 0:
    #         j = gamma[i].argmax()
    
    
    

    # print()
    # for i in range(len(W)):
    #     print(inv(W[i])*H_diff(W[i]))
    #     print(inv(W[i])*H_minus(W[i]) + lambda_W[i])
    #     print()


    # W,lambda_W = spectral_decomposition_xy(x,y,npoints)

    

    # lambda_sqrt = []
    # for i in range(len(lambda_U)):
    #     lambda_sqrt += [rotor_sqrt_choose(lambda_U[i])]

    # p = eo + x + (1/2)*mag_sq(x)*einf
    # q = eo + y + (1/2)*mag_sq(y)*einf

    # P_lst,lambda_P = get_eigmvs(p,grades=1)
    # Q_lst,lambda_Q = get_eigmvs(q,grades=1)

    # Realign the eigenvcectors
    # for i in range(len(Q_lst)):
    #     Q_lst[i] = ~R*Q_lst[i]*R

    # for i in range(len(P_lst)):
    #     P_lst[i] = P_lst[i]*np.sign((P_lst[i]*einf)(0))
    #     Q_lst[i] = Q_lst[i]*np.sign((Q_lst[i]*einf)(0))

    # F_diff,F_adj = get_rot_func(P_lst)
    # G_diff,G_adj = get_rot_func(Q_lst)

    # def H_diff(X):
    #     return G_adj(F_diff(X))

    # def H_adj(X):
    #     return F_adj(G_diff(X))

    # def H_plus(X):
    #     return (G_adj(F_diff(X)) + F_adj(G_diff(X)))/2
    
    # def H_minus(X):
    #     return (G_adj(F_diff(X)) - F_adj(G_diff(X)))/2

    # def J_diff(X):
    #     return G_diff(F_adj(X))

    # basis,rec_basis = get_cga_basis(1)

    # beta = get_matrix(J_diff,basis,rec_basis)

    # U,S,Vh = np.linalg.svd(beta)
    # Q,R = np.linalg.qr(beta)

    # W,lambda_W = eigen_decomp(H_plus,basis,rec_basis)
    # W,lambda_W = eigen_decomp(J_diff,basis,rec_basis)
    
    # U_lst = []
    # for i in range(len(W)):
    #     U_lst += [~normalize_mv(W[i]*H_minus(W[i]) + lambda_W[i])]

    # sign = [0]*len(P_lst)
    # for i in range(len(P_lst)):
    #     sign[i] = (H_diff(P_lst[i])*P_lst[i])(0)*(P_lst[i]*P_lst[i])(0)

    # R_est = ~rotor_sqrt_mv(P_I(W[0]*H_diff(W[0])))

    # U = rotor_sqrt_choose(W[0]*H_diff(W[0]))
    # U *= rotor_sqrt_choose(W[3]*H_diff(W[3]))
    # if lambda_W[2] < 1:
    #     U *= W[2]
    # U = ~U

    # print(lambda_W)


    # T_est,R_est,S_est = scaled_RBM_versors(U)

    # print(lambda_W)
    # print(np.arccos((R*~R_est)(0))/np.pi*360)

    # A = spectral_decomposition(P_lst)
    # B = spectral_decomposition(Q_lst)

    # A = ~(rotor_sqrt_choose(lambda_V[0])*rotor_sqrt_choose(lambda_V[4])*V[3])
    # B = ~(rotor_sqrt_choose(lambda_U[0])*rotor_sqrt_choose(lambda_U[4])*U[3])

    # The function F differential will be expressed as -R*x*~R because it is an orthogonal transformation

    # def F_check(X):
    #     out = 0
    #     for i in range(len(V)):
    #         out += (((X|V[i])*~V[i])*lambda_V[i])(1)
    #     return out

    # def G_check(X):
    #     out = 0
    #     for i in range(len(U)):
    #         out += (((X|U[i])*~U[i])*lambda_U[i])(1)
    #     return out