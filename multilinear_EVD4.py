import gasparse
import numpy as np
from multilinear_algebra import *
from gasparse import mvarray as mv
# ga = gasparse.GA(3)
# ga = gasparse.GA(4,1) # Use 3D conformal geometric algebra
# basis = ga.basis()
# locals().update(basis) # Update all of the basis blades variables into the environment
# compute the ga null basis
# einf = (1.0*e5 - 1.0*e4)*(1/np.sqrt(2))
# eo = (1.0*e5 + 1.0*e4)*(1/np.sqrt(2))

ga = gasparse.GA(4)
locals().update(ga.basis())


nvecs = 50
mu = 0
sigma = 0

basis,rbasis = pyga.get_basis(ga,1)
# basis = [e1,e2,e3,einf]
# rbasis = [e1,e2,e3,-eo]

# S1 = e1^e2^e3^eo
# S2 = -e1^e2^e3^einf

# def ProjB(X):
#     return (X(1,2,3,4)|S1)*S2

# a = np.random.rand(nvecs,ga.size(1,2,3))
# a = ga.mvarray(a.tolist(),grades=[1,2,3])
# A = np.random.rand(nvecs,3,ga.size(1))
A = np.random.rand(nvecs,ga.size(1))
A = ga.mvarray(A.tolist(),grades=[1])
# lst = []
# for i in range(len(A)):
#     lst += [A[i].prod()]
# A = mv.concat(lst)

# Crete a random motor
# t = np.random.rand(3)
# t = ga.mvarray(t.tolist(),grades=1)
# T = 1 + (1/2)*einf*t

# W = np.random.rand(2,ga.size(1))
# W = ga.mvarray(W.tolist(),grades=1)
# R = pyga.normalize(W.prod())
# R = T*R


W = np.random.rand(4,ga.size(1))
W = ga.mvarray(W.tolist(),grades=1)
R = pyga.normalize(W.prod())


N = np.random.normal(mu,sigma,[nvecs,ga.size(1,3)])
N = ga.mvarray(N.tolist(),grades=[1,3])

# a = pyga.normalize(a)
B = R*A*~R + N

def F(X):
    return (~B*X*A + B*X*~A).sum()

def F4(X):
    return F(X)

p4,eigvalues4 = symmetric_eigen_decomp(F4,basis,rbasis)
r4  = pyga.normalize(p4[-1])

def F3(X):
    return F(X*r4)*pyga.inv(r4)
    
p3,eigvalues3 = symmetric_eigen_decomp(F3,basis,rbasis)
r3  = pyga.normalize(p3[-1])

def F2(X):
    return F(X*r3*r4)*pyga.inv(r4)*pyga.inv(r3)
    
p2,eigvalues2 = symmetric_eigen_decomp(F2,basis,rbasis)
r2  = pyga.normalize(p2[-1])

def F1(X):
    return F(X*r2*r3*r4)*pyga.inv(r4)*pyga.inv(r3)*pyga.inv(r2)
    
p1,eigvalues1 = symmetric_eigen_decomp(F1,basis,rbasis)

r1 = pyga.normalize(p1[-1])

R_est = r1*r2*r3*r4
B_est = R_est*A*~R_est
# print("b=",b,sep='')
# print("b_est=",b_est,sep='')
# print(pyga.normalize(b)*pyga.normalize(b_est))
# print(R*~R_est)
print((pyga.normalize(B)|pyga.normalize(B_est))(0))
print((R*~R_est)(0))
