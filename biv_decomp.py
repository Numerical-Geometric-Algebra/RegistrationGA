import geo_algebra as pyga
import numpy as np
import gasparse
from gasparse import multivector as mv
from ga_decomposition import *

ga = gasparse.GA(4,2)
basis = ga.basis()
locals().update(basis)

n_samples = 10

for i in range(n_samples):
    # B = pyga.rdn_kvector(ga,2)

    B = (e1-e5)*e2 + 3*(e2 + e3)*e4 + 5*((e2+e4)^e6)

    # def F(x):
    #     return (x|B)(1)

    # basis = list(ga.basis(grades=1).values())
    basis = [e1,e2,e3,e4,e5,e6]
    rec_basis = [e1,e2,e3,e4,-e5,-e6]
    # F_matrix = pyga.get_matrix(F,basis,basis)

    # G_matrix = np.matmul(F_matrix.T,F_matrix)
    # eigenvalues, eigenvectors = np.linalg.eig(G_matrix.T)
    # a,lambda_a = pyga.convert_numpyeigvecs_to_eigmvs(eigenvalues,eigenvectors,basis,basis)

    # B1 = -(F(a[1])*pyga.inv(a[1]))(2)
    # B2 = -(F(a[3])*pyga.inv(a[3]))(2)
    B_lst,a = biv_decomp(B,basis,rec_basis)
    B_check = 0
    for i in range(len(B_lst)):
        B_check += B_lst[i]
    B_check = B_check/2
    # print(lambda_a)
    print(pyga.numpy_max(B-B_check))
    # print(pyga.numpy_max(B-B_check))
    # print()
    # def F_check(x):
    #     return (x|B_check)(1)
    # np.sqrt(lambda_a[1])*

# value = pyga.check_compare_funcs(F,F_check,basis)

# print(value)