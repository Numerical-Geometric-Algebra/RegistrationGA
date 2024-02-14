
# Since this is a subfolder need to import from parrent folder...
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import geo_algebra as pyga
import open3d as o3d
import matplotlib.pyplot as plt
from algorithms import *
import multilinear_algebra as multiga

np.set_printoptions(linewidth=np.inf)

p,q = 4,2
ga = gasparsegen.GA(p,q)
basis = ga.basis()
locals().update(basis)


n_samples = 20
n_mvs = 5

# Get the basis vectors and their reciprocals
basis,rec_basis = pyga.get_ga_basis_vectors(ga)

pos_basis = basis[:p]
neg_basis = basis[p:]

# Generate random linear function
A = [0]*n_mvs
for j in range(n_mvs):
    A[j] = pyga.rdn_multivector_array(ga,grades=1,size=1) # Generate a bunch of multivectors

# Define the symmetric function f
def f(x):
    out = 0 
    for j in range(n_mvs):
        out += (A[j]*x*A[j])(1)
    return out

a = []
for i in range(len(basis)):
    a += [f(basis[i])]

f_matrix = np.zeros([len(basis),len(basis)],dtype=complex)
for i in range(len(basis)):
    for j in range(p):
        f_matrix[i][j] = (a[i]|basis[j])(0)
    for j in range(p,p+q,1):
        f_matrix[i][j] = -1j*(a[i]|basis[j])(0)

eigenvalues, eigenvectors = np.linalg.eigh(f_matrix)


# v,lambda_f = multiga.convert_numpyeigvecs_to_eigmvs(eigenvalues,eigenvectors,basis,rec_basis)


# n_reflections = 2

# for i in range(n_samples):
#     # create random versor
#     V = 1
#     for i in range(n_reflections):
#         V *= pyga.rdn_multivector_array(ga,grades=1,size=1)
#     V = pyga.normalize_mv(V)

#     # Define the differential the adjoint and the symmetric part of H 
#     H_diff = lambda X: (-1)**n_reflections*V*X*pyga.inv(V)
#     H_adj = lambda X: (-1)**n_reflections*pyga.inv(V)*X*V

#     U,sign = multiga.compute_versor_symmetric(H_diff,H_adj,basis,rec_basis)

#     def H_check(X):
#         return sign*(U*X*pyga.inv(U))(1)

#     # Determines if H_check is equal to H_diff    
#     value = multiga.check_compare_funcs(H_check,H_diff,basis)
#     print("Compare Functions:",value)
#     print(U*~V)



