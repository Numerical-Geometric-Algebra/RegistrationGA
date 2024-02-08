from eig_estimation import *
import open3d as o3d
import matplotlib.pyplot as plt
from algorithms import *

basis,rec_basis = get_cga_basis([1])

M = 2
s = 20

for i in range(s):

    V = 1
    # create random versor
    for i in range(M):
        V *= rdn_cga_vec()

    # V = e1*e2*e3*(10+e4*e5)
    # V = e1*e2*e5*(10+e3*e4)

    V = normalize_mv(V)

    def H_diff(x):
        return V*x*~V

    def H_adj(x):
        return ~V*x*V

    U,sign = compute_versor_decomp_CGA(H_diff,H_adj)

    # print("sign=",sign)
    # print("U=",U)

    def H_check(X):
        return sign*(U*X*~U)(1)

    value = check_compare_funcs(H_check,H_diff)
    # print("eigvalues H_plus:",lambda_plus)
    print("Compare_funcs:",value)
    # print()

