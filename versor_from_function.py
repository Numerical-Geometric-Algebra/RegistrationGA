from eig_estimation import *
import open3d as o3d
import matplotlib.pyplot as plt
from algorithms import *

basis,rec_basis = get_cga_basis([1])

basis = [eo,e1,e2,e3,einf]
rec_basis = [-einf,e1,e2,e3,-eo]

M = 1
s = 10
iters = 1

for i in range(s):

    V = 1
    # create random versor
    for i in range(M):
        V *= rdn_cga_vec()

    # Compute random rotation
    # for i in range(2):
    #     V*= rdn_vanilla_vec()

    # theta = 45/180*np.pi
    # V *= 1 + 5*einf*e2
    # V = 1 + 5*einf*(e1+e2)
    
    # V = 0.5893417311209419*e1 + 0.5187890655206802*e2 + -0.5767754814365584*e3 + -0.3217394783600013*e4 + -0.2294602764449405*e5
    V = normalize_mv(V)

    for j in range(iters):

        V = normalize_mv(V)

        # V = e1*e2*e3*(10+e4*e5)
        # V = e1*e2*e5*(10+e3*e4)
        # V = 0.4452503179085513*e1 + 0.03105614900183773*e2 + 0.162515026225041*e3 + -0.4990915508881664*e4 + 0.3299636239013176*e5
        # V = 1.001367886134567*e1 + -0.2200979366441679*e2 + 0.6603288120695339*e3 + 0.1909250601574395*e4 + 0.7236485775166798*e5
        # V  = -0.8867112692248958*e1 + 0.004440092206331109*e2 + 0.3865781043214471*e3 + 0.3969997326949899*e4 + 0.3054963304033023*e5
        # print("j:",j)
        # print("V:",V)

        def H_diff(x):
            return V*x*~V

        def H_adj(x):
            return ~V*x*V

        H_plus = lambda X: (1/2)*(H_diff(X) + H_adj(X))

        # beta = get_matrix(H_diff,basis,rec_basis)

        # eigenvalues, eigenvectors = np.linalg.eig(beta + beta.T)
        # print(eigenvalues)
        # print(eigenvectors.T)

        U,sign = compute_versor_decomp_CGA(H_diff,H_adj)
        # a,lambda_a,B = compute_smth(H_diff,H_adj)

        # print("sign=",sign)
        # print("U=",U)

        def H_check(X):
            return sign*(U*X*~U)(1)

        value = check_compare_funcs(H_check,H_diff)
        # print("eigvalues H_plus:",lambda_plus)
        
        # if value > 1e-7:
        #     V += 0.00000001*e1
        print("Compare_funcs:",value)
        print()
        # V += 0.00000001*e1


