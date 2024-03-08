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

ga = gasparsegen.GA(8)
print("Geometric Algebra Initialized!!")
basis = ga.basis()
locals().update(basis)

'''
This script tests the decomposition of a bivector. 
Since the decomposition of bivectors is equivalent to the decomposition of skew transformations 
we can also use a similar approach to compute the decomposition of a skew transformation 
It works for any geometric algebra with positive signature!!
'''

n_samples = 10

# Get the basis vectors and their reciprocals
basis,rec_basis = pyga.get_ga_basis_vectors(ga)


for i in range(n_samples):
    # generate random bivector
    B = pyga.rdn_multivector_array(ga,grades=2,size=1)
    B_lst = multiga.biv_decomp(B,basis,rec_basis)
    
    B_check = 0
    for i in range(len(B_lst)):
        B_check += B_lst[i]

    # Confirms if the obtained bivector is the same as the one given
    print("B diff:",pyga.numpy_max(B-B_check))
    