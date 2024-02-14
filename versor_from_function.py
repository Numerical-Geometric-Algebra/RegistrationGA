import geo_algebra as pyga
import open3d as o3d
import matplotlib.pyplot as plt
from algorithms import *
import multilinear_algebra as multiga

np.set_printoptions(linewidth=np.inf)

ga = gasparsegen.GA(4,1)
print("Algebra initialized")
basis = ga.basis()
locals().update(basis)

n_samples = 10
n_iters = 1

basis,rec_basis = pyga.get_ga_basis_vectors(ga)

M = 2
s = 10
iters = 1

V = 0.5260392003277276 + -0.2563066616685884*e12 + 0.2652767324823975*e13 + -0.04248061958180708*e23 + -2.02987165574332*e14 + -0.6658737273723848*e24 + 1.025611314740798*e34 + -0.8977341698961641*e15 + -0.5855029280659305*e25 + 0.7547857168975942*e35 + -2.304732137244466*e45
V = -1.514983649197955 + 0.3098262339114742*e12 + -0.3722549845254647*e13 + 0.230683476046965*e23 + -0.2285044285660838*e14 + 0.2996675368209393*e24 + -0.1899146423640989*e34 + -0.7319618590281312*e15 + -0.5897105894151736*e25 + 1.253522684017213*e35 + 1.142888012177476*e45
V = -0.6508619669808327 + 0.02235302186009213*e12 + 0.02717525631608767*e13 + 0.001522526315597824*e23 + -0.05398721643466731*e14 + -0.1797134479365891*e24 + -0.2148059480928383*e34 + 0.1449105272701016*e15 + 0.7019075296685562*e25 + 0.8434603185093174*e35 + -0.5302040725186802*e45
V = 0.1568934850921305 + -0.465127078825784*e12 + 1.220900582313039*e13 + 1.182035271041489*e23 + 0.1800901260380679*e14 + 0.252160871029149*e24 + 0.1522601900802229*e34 + -1.056836813006311*e1234 + 0.1266603721488664*e15 + 0.5749757776904081*e25 + -0.9328612937973194*e35 + -0.754474832857539*e1235 + -0.298261285865189*e45 + 0.42781081747732*e1245 + -1.127280314892461*e1345 + -0.1897996100053409*e2345

for i in range(s):

    # create random versor
    V = 1
    for i in range(M):
        V *= pyga.rdn_multivector_array(ga,grades=1,size=1)

    # Compute random rotation
    # for i in range(2):
    #     V*= rdn_vanilla_vec()

    # theta = 45/180*np.pi
    # V *= 1 + 5*einf*e2
    # V = 1 + 5*einf*(e1+e2)
    # t = e3 + 2*e2 + 3*e1

    # V = (1 + 5*(e4-e5)*t)*(1 + e123*t)
    # V = 0.5893417311209419*e1 + 0.5187890655206802*e2 + -0.5767754814365584*e3 + -0.3217394783600013*e4 + -0.2294602764449405*e5
    V = pyga.normalize_mv(V)

    for j in range(iters):
        # V = pyga.normalize_mv(V)

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

        
        # Check the decomposition of the bivector of H
        B = multiga.compute_bivector_from_skew(H_diff,basis,rec_basis)
        B_lst,a = multiga.biv_decomp(B,basis,rec_basis)
        B_check = 0
        for k in range(len(B_lst)):
            B_check += B_lst[k]

        # print("B diff:",pyga.numpy_max(B-B_check))
        


        # beta = get_matrix(H_diff,basis,rec_basis)

        # eigenvalues, eigenvectors = np.linalg.eig(beta + beta.T)
        # print(eigenvalues)
        # print(eigenvectors.T)

        # U,sign2,b = multiga.compute_versor_decomp(H_diff,H_adj,basis,rec_basis)
        W,A = multiga.compute_ortho_decomposition_skew(H_diff,basis,rec_basis)

        def F_diff(x):
            # return H_diff(pyga.inv(W)*x*W)*pyga.inv(A)
            # return H_diff((x^A)*pyga.inv(A))
            # return W*pyga.proj(A,x)*~W - H_diff(x)
            return W*(x|A)*pyga.inv(A)*pyga.inv(W) - H_diff(x)

        def F_adj(x):
            # return W*H_adj(x)*pyga.inv(W)
            # return (H_adj(x)^A)*pyga.inv(A)
            return ((pyga.inv(W)*x*W)|A)*pyga.inv(A) - H_adj(x)

        U,sign,b = multiga.compute_versor_decomp(F_diff,F_adj,basis,rec_basis)
        
        B2 = multiga.compute_bivector_from_skew(F_diff,basis,rec_basis)
        print("Bivector:",B2)
        print("W=",W)
        print("V=",V)
        print("U=",U)
        print("A=",A)

        W *= U
        # sign *= sign0
        # print("sign=",sign)
        # print("U=",U)

        # print("V=",V)
        # print("W=",W)
        
        def H_check(X):
            return sign*(W*X*pyga.inv(W))(1)
        value = multiga.check_compare_funcs(H_check,H_diff,basis)
        print("Compare_funcs:",value)

        # print("eigvalues H_plus:",lambda_plus)
        
        # if value > 1e-7:
        #     V += 0.0000001*e12
        
        # V += 0.00000001*e1
        print()


