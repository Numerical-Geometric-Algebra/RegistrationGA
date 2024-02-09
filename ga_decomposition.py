import geo_algebra as pyga
import numpy as np


def biv_decomp(B,basis,rec_basis):
    '''
    Computes the bivector decomposition from a bivector B. Uses the composition of 
    a function with its adjoint that is G(x) = (x|B)|~B in the euclidean case.
    Does not work with negative signature. It might have to do with the sign of the eigenvalues
    '''
    def F(x):
        return (x|B)(1)

    def G(x):
        return ((x|B)|~B)(1)

    G_matrix = pyga.get_matrix(G,basis,rec_basis)
    eigenvalues, eigenvectors = np.linalg.eig(G_matrix.T)
    a,lambda_a = pyga.convert_numpyeigvecs_to_eigmvs(eigenvalues,eigenvectors,basis,rec_basis)

    # print(lambda_a)
    B_lst = [0]*len(a)
    for i in range(len(a)):
        B_lst[i] = -(F(a[i])*pyga.inv(a[i]))(2)
    
    return B_lst,a

    # B1 = -(F(a[1])*pyga.inv(a[1]))(2)
    # B2 = -(F(a[3])*pyga.inv(a[3]))(2)
