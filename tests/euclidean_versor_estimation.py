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
p,q = 5,0
ga = gasparsegen.GA(p,q)
print("Geometric Algebra Initialized!!")
basis = ga.basis()
locals().update(basis)

# Get the basis vectors and their reciprocals
basis,rec_basis = pyga.get_ga_basis_vectors(ga)
upss = pyga.get_pseudoscalar(ga) # get the unit pseudoscalar of this algebra

def get_sign(X,P):
    if ((P*X)(0)*(P*X)(0)*(P*X)(0)).sum()(0) < 0:
        return -1
    else:
        return 1

def disambiguate_sign(X,Y,P,Q):
    for i in range(len(P)):
        P[i] *= get_sign(X,P[i])*get_sign(Y,Q[i])

def multilinear_project(X,basis,rec_basis):
    out = 0
    for i in range(len(basis)):
        out += (X*rec_basis[i])(0)*basis[i]

'''
This script describes an approach to estimate special orthogonal transformations between two multivector clouds. 
In particular we consider a space of positive signature.

The relationship between the multivector clouds is of a special orthogonal transformation plus additive gaussian noise.
Outline:
    - Generate random multivectors X
    - Apply random (constrained) orthogonal transformation Y = U*X*~U + N
    - Form the multilinear transformations F and G
    - Find the eigenvectors P and Q of F and G
    - From the eigenvectors P and Q Form the orthogonal transformation H(x) = sum_i (x|P[i])*Q[i]
    - Compute the versor V of H
'''


mu,sigma,n_mvs = 0,0.1,100
angle = 100/180*np.pi

n_samples = 10

# Define the basis for which we are taking the orthogonal transformation
v = (e4 - e5)/np.sqrt(2)
I = e123
M_basis = [1,I*e1,I*e2,I*e3,v*e1,v*e2,v*e3,v*I]
M_rec_basis = [0]*len(M_basis)
for i in range(len(M_basis)):
    M_rec_basis[i] = ~M_basis[i]

for i in range(n_samples):

    a = pyga.rdn_mvarray_from_basis(ga,[e1,e2,e3],1)
    b1 = pyga.rdn_mvarray_from_basis(ga,[e1,e2,e3],1)
    b2 = pyga.rdn_mvarray_from_basis(ga,[e1,e2,e3],1)


    # Generate the versor in some multilinear space
    B = pyga.normalize_mv(v^a)
    R = pyga.normalize_mv(b1*b2)
    U = (np.cos(angle) + B*np.sin(angle))*R

    # Apply the transformation plus some noise
    N = pyga.rdn_gaussian_multivector_array(mu,sigma,ga,None,n_mvs)
    X = pyga.rdn_multivector_array(ga,None,n_mvs)
    Y = U*X*~U + N


    F = lambda Z: (X*Z*X).sum()
    G = lambda Z: (Y*Z*Y).sum()

    P,lambda_P = multiga.symmetric_eigen_decomp(F,basis,rec_basis)
    Q,lambda_Q = multiga.symmetric_eigen_decomp(G,basis,rec_basis)

    disambiguate_sign(X,Y,P,Q)

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
    print()
