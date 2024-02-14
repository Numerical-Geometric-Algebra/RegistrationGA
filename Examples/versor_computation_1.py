
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

ga = gasparsegen.GA(16,compute_mode='large')
print("Geometric Algebra Initialized!!")
basis = ga.basis()
locals().update(basis)

n_samples = 20

# Get the basis vectors and their reciprocals
basis,rec_basis = pyga.get_ga_basis_vectors(ga)
n_reflections = 2

for i in range(n_samples):
    # create random versor
    V = 1
    for i in range(n_reflections):
        V *= pyga.rdn_multivector_array(ga,grades=1,size=1)
    V = pyga.normalize_mv(V)

    # Define the differential the adjoint and the symmetric part of H 
    H_diff = lambda X: (-1)**n_reflections*V*X*pyga.inv(V)
    H_adj = lambda X: (-1)**n_reflections*pyga.inv(V)*X*V

    U,sign = multiga.compute_versor_symmetric(H_diff,H_adj,basis,rec_basis)

    def H_check(X):
        return sign*(U*X*pyga.inv(U))(1)

    # Determines if H_check is equal to H_diff    
    value = multiga.check_compare_funcs(H_check,H_diff,basis)
    print("Compare Functions:",value)
    print(U*~V)



