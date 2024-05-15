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

def left_contraction(F,basis,rbasis):
    # take derivative with respect to x1: F1(X) = d_x1 F(x1*X)
    def F1(X):
        out = 0
        for i in range(len(basis)):
            out += rbasis[i]*F(X*basis[i])
        return out
    return F1

vga = gasparse.GA(3)
locals().update(vga.basis())
nvecs = 1

basis,recbasis = pyga.get_basis(vga,1)

a = np.random.rand(nvecs,vga.size(1))
a = vga.mvarray(a.tolist(),grades=1)

b = np.random.rand(nvecs,vga.size(1))
b = vga.mvarray(b.tolist(),grades=1)

a = pyga.normalize(a)
b = pyga.normalize(b)


def F(X):
    return (b*X*a).sum()

# The plane of rotation ??
B = pyga.normalize((a^b).sum())

a1 = pyga.normalize(((e1|B)*pyga.inv(B))(1))
a2 = pyga.normalize((a1|B)(1))

# recbasis = basis = [a1,a2]

F_left = left_contraction(F,basis,recbasis)
F_right = right_contraction(F,basis,recbasis)
    
p_left,left_eigvals = symmetric_eigen_decomp(F_left,basis,recbasis)
p_right,right_eigvals = symmetric_eigen_decomp(F_right,basis,recbasis)


rotorbasis = [1,e12,e13,e23]
recrotorbasis = [1,-e12,-e13,-e23]

R,eigvals = symmetric_eigen_decomp(F,rotorbasis,recrotorbasis)

a = a[0]
b = b[0]

n = 3
rho = (n-4)/n
gamma = (a|b)(0)
phi = -(n-2)/n
# lambda1 = (-gamma*(1-rho) + np.sqrt(gamma**2*(1-rho)**2 + 4*rho))/2
# r1 = a + (1/lambda1)*rho*b

# Insanity checks
r1 = p_left[1]
lmbda = left_eigvals[1]
t = (b|r1)*a + rho*((a|r1)*b - gamma*r1)
lambda1 = t*pyga.inv(r1)

# print(r1^B)
# print(lambda1)
# print(lmbda*phi)
# print(F_left(r1)*pyga.inv(r1))

def F_left_test(x):
    # return (b|x)*a + rho*((a|x)*b - gamma*x)
    return n*(b|x)*a + (n-4)*(b^x)*a

def F_right_test(x):
    return -(n-2)*b*x*a

R = pyga.normalize(1+b*a)
x = (e1|B)*pyga.inv(B)
y = pyga.inv(x)*R

# R*a*~R = b
print(R*a*~R - b)

# x*y = R
print(R-x*y)