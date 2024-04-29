import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import geo_algebra as pyga
import open3d as o3d
import matplotlib.pyplot as plt
import multilinear_algebra as multiga
from cga3d_estimation import *

np.set_printoptions(linewidth=np.inf)

N = 1000 # Number of points to be fitted
r = 4 # The dimension of the subspace
A = 1

# Form an r-blade
a = pyga.rdn_multivector_array(cga3d,grades=[1],size=r)
for i in range(r):
    A ^= a[i]
A = pyga.normalize(A)

# Project the points to the linear subspace A
y = (pyga.rdn_multivector_array(cga3d,grades=[1],size=N)|A)*pyga.inv(A)
y = pyga.normalize(y(1))

# Add gaussian noise to the points in CGA
noise = pyga.rdn_gaussian_multivector_array(0,0.1,cga3d,grades=[1],size=N)
x = y + noise
# x = conformal_mapping(pyga.rdn_multivector_array(vga3d,grades=[1],size=4)) # Generate two random vectors in CGA

# Get the basis vectors and their reciprocals
basis,rec_basis = pyga.get_ga_basis_vectors(cga3d)

def F(Z):
    return ((Z|x)*x).sum()

eigenvecs,eigvals = multiga.symmetric_eigen_decomp(F,basis,rec_basis)

P = eigenvecs
l = eigvals

B = pyga.normalize(P[0]^P[2]^P[3]^P[4])


print(get_properties(B*ii))
print(get_properties(A*ii))

print()
print(eigenvecs)
print(eigvals)


