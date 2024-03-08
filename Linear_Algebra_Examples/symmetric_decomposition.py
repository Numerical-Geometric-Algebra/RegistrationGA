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
p,q = 6,0
ga = gasparsegen.GA(p,q)
print("Geometric Algebra Initialized!!")
basis = ga.basis()
locals().update(basis)

'''
    This script tests the decomposition of a symmetric linear transformation. The algorithm takes a linear
    function defined in a geometric algebra, then converts it to a matrix and uses numpy to find the decomposition.
    
    IMPORTANT: When the multiplicity is greater than one we form blades A = a1^a2^...^ak from the eigenvectors with
               the same eigenvalues. This provides a unique decomposition!!!

For algebras with negative signature sometimes the decomposition cannot find all the eigenblades associated with
the symmetric function. We presume that happens when the eigenblades are null blades.  
Null blades act as projections to zero. Only reciprocal vectors are affected by these blades. 

Even for positive signature geometric algebras, sometimes, not all the eigenvectors are found. 
'''

n_samples = 20
n_mvs = 1

# Get the basis vectors and their reciprocals
basis,rec_basis = pyga.get_ga_basis_vectors(ga)
pss = pyga.get_pseudoscalar(ga)
print(pss)

for i in range(n_samples):
    # Generate random linear function
    A = [0]*n_mvs
    for j in range(n_mvs):
        A[j] = pyga.rdn_multivector_array(ga,grades=[2],size=1) # Generate a bunch of multivectors

    # Define the differential of H 
    def H_diff(x):
        out = 0 
        for j in range(n_mvs):
            out += (A[j]*x*A[j])(1)
        return out
    

    B,lambda_H = multiga.symmetric_eigen_blades_decomp(H_diff,basis,rec_basis)

    def H_check(x):
        out = 0
        for j in range(len(lambda_H)):
            out += lambda_H[j]*(x|B[j])*pyga.inv(B[j])
        return out

    # Determines if H_check is equal to H_diff    
    value = multiga.check_compare_funcs(H_check,H_diff,basis)
    print("Compare Functions:",value)
    if value > 0.001:
        print("The eigendecomposition is not correct!!")        
        break
    print()
