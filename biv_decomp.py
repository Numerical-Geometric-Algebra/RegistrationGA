import geo_algebra as pyga
import numpy as np
import gasparsegen
from gasparse import multivector as mv
# from ga_decomposition import *
import multilinear_algebra as multiga

np.set_printoptions(linewidth=np.inf)

ga = gasparsegen.GA(4,1)
basis = ga.basis()
locals().update(basis)

n_samples = 1
n_iters = 1

basis,rec_basis = pyga.get_ga_basis_vectors(ga)

def check_ortho_bivs(B_lst):
    matrix = np.zeros([len(B_lst),len(B_lst)])
    for i in range(len(B_lst)):
        for j in range(len(B_lst)):
            matrix[i][j] = (B_lst[i]|B_lst[j])(0)
    return matrix

v = e3+2*e2+5*e1
B = 5*(e4-e5)*v + e123*v
# B = 5*(e4-e5)*v


for i in range(n_samples):
    # B = pyga.rdn_multivector_array(ga,2,1)
    # B = e12 + e34 + e56
    for j in range(n_iters):
        # B += 0.0001*pyga.rdn_multivector_array(ga,2,1)

        # B = (e1-e5)*e2 + 3*(e2 + e3)*e4 + 5*((e2+e4)^e6)

        # def F(x):
        #     return (x|B)(1)

        # basis = list(ga.basis(grades=1).values())
        # F_matrix = pyga.get_matrix(F,basis,basis)

        # G_matrix = np.matmul(F_matrix.T,F_matrix)
        # eigenvalues, eigenvectors = np.linalg.eig(G_matrix.T)
        # a,lambda_a = pyga.convert_numpyeigvecs_to_eigmvs(eigenvalues,eigenvectors,basis,basis)

        # B1 = -(F(a[1])*pyga.inv(a[1]))(2)
        # B2 = -(F(a[3])*pyga.inv(a[3]))(2)
        B_lst = multiga.biv_decomp(B,basis,rec_basis)
        B_check = 0
        for i in range(len(B_lst)):
            B_check += B_lst[i]
    
        # print(lambda_a)
        print("j=",j)

        # for k in range(len(B_lst)):
        #     print(pyga.numpy_max((a[i]|B_lst[i]) - (a[i]|B)))
        # print(check_ortho_bivs(B_lst))
        # print(B)
        # for k in range(len(B_lst)):
            # print(B_lst[k])
        print("B diff:",pyga.numpy_max(B-B_check))
        # print(pyga.numpy_max(B-B_check))
        # print()
        # def F_check(x):
        #     return (x|B_check)(1)
        # np.sqrt(lambda_a[1])*
    print()
# value = pyga.check_compare_funcs(F,F_check,basis)

# print(value)