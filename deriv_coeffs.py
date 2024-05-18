import math
import numpy as np

def compute_gamma(r,s,n):
    '''
       Computes the coefficients obtained when taking the multivector derivative
       ds Ar Xs = gamma_rs Ar 
    '''

    K = int(1/2*(r+s-abs(r-s)))
    out = 0
    for k in range(K+1):
        out += (-1)**(r*s-k)*math.comb(r,k)*math.comb(n-r,s-k)
    return out

def compute_sum_gammas(r,n):
    out = 0
    for s in range(n+1):
        out += compute_gamma(r,s,n)
    return out

def output_array(n):
    a = np.zeros([n+1])
    for r in range(n+1):
        a[r] = compute_sum_gammas(r,n)
    return a

def output_table(n):
    a = np.zeros([n+1,n+1],dtype=int)
    for r in range(n+1):
        for s in range(n+1):
            a[r][s] = compute_gamma(r,s,n)
    return a

def gen_latex_table(n):
    a  = output_table(n)
    string = ''
    for s in range(n+1):
        string += ('&s=%i'% s)

    string += '\\\ \hline\n'
    for r in range(n+1):
        string += ('r=%i'% r)
        for s in range(n+1):
            string += ('&%i'%a[r][s])
        string += '\\\ \hline\n'

    return string