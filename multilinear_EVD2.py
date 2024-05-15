import gasparse
import numpy as np
from multilinear_algebra import *

def right_contraction(F,basis,rbasis):
    # take derivative with respect to x1: F1(X) = d_x1 F(x1*X)
    def F1(X):
        out = 0
        for i in range(len(basis)):
            out += rbasis[i]*F(basis[i]*X)
        return out
    return F1


vga = gasparse.GA(3)
locals().update(vga.basis())
nvecs = 2
mu = 0
sigma = 0

basis,recbasis = pyga.get_basis(vga,1)

a = np.random.rand(nvecs,vga.size(1))
a = vga.mvarray(a.tolist(),grades=1)

R = np.random.rand(vga.size(0,2))
R = vga.mvarray(R.tolist(),grades=[0,2])
R = pyga.normalize(R)

n = np.random.normal(mu,sigma,[nvecs,vga.size(1)])
n = vga.mvarray(n.tolist(),grades=1)

# a = pyga.normalize(a)
b = R*a*~R + n

def F(X):
    return (b*X*a).sum()

F_right = right_contraction(F,basis,recbasis)
p_right,right_eigvals = symmetric_eigen_decomp(F_right,basis,recbasis)

r2  = pyga.normalize(p_right[-1])

def G(r1):
    return F(r1*r2)*pyga.inv(r2)

p_left,left_eigvals = symmetric_eigen_decomp(G,basis,recbasis)

r1 = pyga.normalize(p_left[-1])

R_est = r1*r2
b_est = R_est*a*~R_est
print("b=",b,sep='')
print("b_est=",b_est,sep='')
print(pyga.normalize(b)*pyga.normalize(b_est))
print(R*~R_est)
