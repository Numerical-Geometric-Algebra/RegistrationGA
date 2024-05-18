# from cga3d_estimation import *
# b = pyga.rdn_multivector_array(vga3d,1,1)
# a = conformal_mapping(b)

# def F(x):
#     return ((x|a)*a).sum()

# basis,rec_basis = pyga.get_basis(cga3d,grades=1)
# P,lamba = multiga.symmetric_eigen_decomp_1(F,basis,rec_basis,ii)

# print(lamba[0])




# Since this is a subfolder need to import from parrent folder...
# import sys
# import os
# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)

import multilinear_algebra as multiga
import numpy as np
import gasparse
import geo_algebra as pyga

np.set_printoptions(linewidth=np.inf)
p,q = 4,2
ga = gasparse.GA(p,q)
print("Geometric Algebra Initialized!!")
basis = ga.basis()
locals().update(basis)

n_samples = 20
n_mvs = 1
grades = [2]

# Get the basis vectors and their reciprocals
basis,rec_basis = pyga.get_ga_basis_vectors(ga)
pss = pyga.get_pseudoscalar(ga)
print(pss)

for i in range(n_samples):
    # Generate random linear function
    # A = [0]*n_mvs
    # for j in range(n_mvs):
    A = pyga.rdn_multivector_array(ga,grades=grades,size=n_mvs) # Generate a bunch of multivectors

    # Define the differential of H 
    def H_diff(x):
        return (A*x*A)(1).sum()
        
    B,lambda_H = multiga.symmetric_eigen_decomp_1(H_diff,basis,rec_basis,pss)

    B = gasparse.mvarray.concat(B)
    lambda_H = gasparse.mvarray.concat(lambda_H)

    def H_check(x):
        return (lambda_H*(x|B)*pyga.inv(B)).sum()



    # Determines if H_check is equal to H_diff    
    value = multiga.check_compare_funcs(H_check,H_diff,basis)
    print("Compare Functions:",value)
    if value > 0.001:
        print("The eigendecomposition is not correct!!")        
    #     break
    print()
