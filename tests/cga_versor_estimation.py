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
import math

np.set_printoptions(linewidth=np.inf)
p,q = 4,1
ga = gasparsegen.GA(p,q)
print("Geometric Algebra Initialized!!")
basis = ga.basis()
locals().update(basis)

# Get the basis vectors and their reciprocals
basis,rec_basis = pyga.get_ga_basis_vectors(ga)
upss = pyga.get_pseudoscalar(ga) # get the unit pseudoscalar of this algebra

# compute the cga null basis
einf = (1.0*e5 - 1.0*e4)*(1/np.sqrt(2))
eo = (1.0*e5 + 1.0*e4)*(1/np.sqrt(2))

def multilinear_project(X,basis,rec_basis):
    out = 0
    for i in range(len(basis)):
        out += (X*rec_basis[i])(0)*basis[i]

def normalize_eigenvectors(P):
    for i in range(len(P)):
        P[i] = pyga.normalize_mv(P[i])
        P[i] *= math.copysign(1,(P[i]*einf)(0))
    return P

'''
This script describes an approach to estimate special orthogonal transformations between two multivector clouds in 3D CGA. 
The relationship between the multivector clouds is of a special orthogonal transformation plus additive gaussian noise.
Outline:
    - Generate random multivectors X
    - Apply random (constrained) orthogonal transformation Y = U*X*~U + N
    - Form the multilinear transformations F and G
    - Find the eigenvectors P and Q of F and G
    - From the eigenvectors P and Q Form the orthogonal transformation H(x) = sum_i (x|P[i])*Q[i]
    - Compute the versor V of H
'''

mu,sigma,n_mvs = 0,0.0,100
angle = 100/180*np.pi

n_samples = 10

# Define the basis for which we are taking the orthogonal transformation
I = e123
M_basis = [1,I*e1,I*e2,I*e3,einf*e1,einf*e2,einf*e3,einf*I]
M_rec_basis = [1,-I*e1,-I*e2,-I*e3,-eo*e1,-eo*e2,-eo*e3,eo*I]

# M_rec_basis = [0]*len(M_basis)
# for i in range(len(M_basis)):
#     M_rec_basis[i] = ~M_basis[i]
vga_vecbasis = [e1,e2,e3]

for i in range(n_samples):

    # Generate the translation vector
    t = pyga.rdn_mvarray_from_basis(ga,vga_vecbasis,1)*0
    T = 1 + (1/2)*einf*t

    # Generate the rotator
    b1 = pyga.rdn_mvarray_from_basis(ga,vga_vecbasis,1)
    b2 = pyga.rdn_mvarray_from_basis(ga,vga_vecbasis,1)
    R = pyga.normalize_mv(b1*b2)

    # Define the motor
    U = T*R
    
    # Apply the transformation plus some noise
    n = pyga.rdn_gaussian_mvarray_from_basis(mu,sigma,ga,vga_vecbasis,n_mvs)
    x = pyga.rdn_mvarray_from_basis(ga,vga_vecbasis,n_mvs)
    y = R*x*~R + t + n

    # convert to CGA
    p = eo + x + (1/2)*(x*x)(0)*einf
    q = eo + y + (1/2)*(y*y)(0)*einf

    F = lambda Z: (p*Z*p).sum()
    G = lambda Z: (q*Z*q).sum()

    P,lambda_P = multiga.symmetric_eigen_decomp(F,basis,rec_basis)
    Q,lambda_Q = multiga.symmetric_eigen_decomp(G,basis,rec_basis)

    P = normalize_eigenvectors(P)
    Q = normalize_eigenvectors(Q)

    def H_diff(z):
        out = 0
        for i in range(len(P)):
            out += (z|P[i])*pyga.inv(Q[i])
        return out

    def H_adj(z):
        out = 0
        for i in range(len(P)):
            out += (z|pyga.inv(Q[i]))*P[i]
        return out

    V,sign = multiga.compute_versor_symmetric(H_diff,H_adj,basis,rec_basis)

    if sign < -0.9999: # If sign is negative then it is a reflection (Euclidean GA)
        V = upss*V
        sign *= (upss*~upss)(0)
        print("Changing to a Rotation")

    def H_check(X):
        return sign*(V*X*pyga.inv(V))(1)

    # Determines if H_check is equal to H_diff
    value = multiga.check_compare_funcs(H_check,H_diff,basis)
    print("Compare Functions:",value)
    print("(Compare versors) U*~V=",U*~V)
    print("V=",V)
    print()
