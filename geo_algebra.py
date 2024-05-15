#!/usr/bin/env python3
import gasparse
import numpy as np
from gasparse import mvarray as mv


def nparray_to_mvarray(ga,grade,x_array):
    if grade is None:
        return ga.mvarray(x_array.tolist())
    return ga.mvarray(x_array.tolist(),grades=grade)

def nparray_to_mvarray_from_basis(ga,basis,x_array):
    return ga.mvarray(x_array.tolist(),basis=basis)

def nparray_to_vecarray(ga,x_array):
    return ga.mvarray(x_array.tolist(),grades=1)

def mag_sq(X):
    return (X*~X)(0)

def mag(X):
    return mv.sqrt(abs((X*~X)(0)))

def inv(X): # Element wise inversion
    return ~X/mag_sq(X)

def normalize(X): # Element wise normalization
    return X/mag(X)


eps = 0.001
def check_null_mv(x):
    '''Checks if a multivector is null'''
    magnitude = mag(x)
    if magnitude == 0.0:
        return True
    x_array = np.array(x.tolist()[0])
    pos_mag = np.sqrt((x_array*x_array).sum())

    rel_mag = magnitude/pos_mag
    if rel_mag < eps:
        return True
    else:
        return False

def get_basis(ga,grades):
    basis = list(ga.basis(grades=grades).values())
    basis = gasparse.mvarray.concat(basis)
    recbasis = inv(basis)
    return basis,recbasis

def random_blade(ga,grade):
    A_list = np.random.rand(grade,ga.size(1)).tolist()
    A = ga.mvarray(A_list,grades=1)
    A = A.prod()(grade)
    return A

def get_pseudoscalar(ga):
    grade = len(ga.metric())
    basis = list(ga.basis(grades=grade).values())
    return basis[0]


def reciprocal_blades(basis):
    '''Reciprocal blades for positive squaring basis vectors'''
    rec_basis = [0]*len(basis) # reciprocal basis blades
    for i in range(len(basis)):
        rec_basis[i] = ~basis[i]
    return rec_basis

def get_ga_basis(ga,grades):
    basis = list(ga.basis(grades=grades).values())
    rec_basis = reciprocal_blades(basis)
    return (basis,rec_basis)

def get_negative_signature_pss(ga):
    basis = list(ga.basis(grades=1).values())
    I_neg = ga.mvarray([1],basis=['e'])
    for i in range(len(basis)):
        if (basis[i]|basis[i])(0) == -1.0:
            I_neg ^= basis[i]
    return I_neg

def get_ga_basis_vectors(ga):
    ''' Gets a basis for the algebra, then computes a reciprocal basis ''' 
    basis = list(ga.basis(grades=1).values())
    I_neg = get_negative_signature_pss(ga)
    rec_basis = [0]*len(basis)
    sign = (-1)**mv.grade(I_neg)

    for i in range(len(basis)):
        rec_basis[i] = sign*I_neg*basis[i]*inv(I_neg)
    
    return (basis,rec_basis)

def rdn_mvarray_from_basis(ga,basis,size):
    mvsize = len(basis)
    ones = 0.5*np.ones([size,mvsize])
    x_array = np.random.rand(size,mvsize) - ones
    return nparray_to_mvarray_from_basis(ga,basis,x_array)

def rdn_multivector_array(ga,grades,size):
    if grades is None:
        mvsize = ga.size()
        ones = 0.5*np.ones([size,mvsize])
        x_array = np.random.rand(size,mvsize) - ones
        return nparray_to_mvarray(ga,grades,x_array)
    
    mvsize = ga.size(grades)
    ones = 0.5*np.ones([size,mvsize])
    x_array = np.random.rand(size,mvsize) - ones
    return nparray_to_mvarray(ga,grades,x_array)

def rdn_gaussian_mvarray_from_basis(mu,sigma,ga,basis,size):
    mvsize = len(basis)
    x_array = np.random.normal(mu,sigma,[size,mvsize])
    return nparray_to_mvarray_from_basis(ga,basis,x_array)

def rdn_gaussian_multivector_array(mu,sigma,ga,grades,size):
    if grades is None:
        mvsize = ga.size()
        x_array = np.random.normal(mu,sigma,[size,mvsize])
        return nparray_to_mvarray(ga,grades,x_array)

    mvsize = ga.size(grades)
    x_array = np.random.normal(mu,sigma,[size,mvsize])
    return nparray_to_mvarray(ga,grades,x_array)


def rotor_sqrt(R):
    B = normalize(R(2))
    s = (R(2)*~B)(0)*(B*~B)(0)
    c = R(0)
    t = s/(1+c)

    return normalize(1+B*t)

def rotor_sqrt_secure(R):
    if(numpy_max(R(2)) == 0):
        return R
    else:
        return rotor_sqrt(R)

def rotor_sqrt_from_vectors(e,u):
    e = normalize(e)
    u = normalize(u)
    scalar = 1/np.sqrt((e|u)(0) + 1)
    return (1/np.sqrt(2))*scalar*(e*u + 1)

def proj(B,X):
    return (X|B)*inv(B)

def proj_perp(B,X):
    return X - proj(B,X)

def mean(X):
    '''Computes the mean of an array of multivectors
       Still need way to get the length of multivector array
       Do not use!!! Will give error'''
    X_bar = X.sum()/len(X)
    return X_bar
    
def numpy_max(X):
    '''Converts to numpy and then computes the max'''
    arr = np.array(X.tolist()[0])
    return abs(arr).max()
