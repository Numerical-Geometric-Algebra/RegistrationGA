from geo_algebra import *

x_lst = rdn_cga_kveclist(100,grade=[1])
X_lst,lambda_X = get_eigmvs(x_lst,grades=[2,3])


'''
    How do i find a general unique decomposition for multilinear transformations when the multiplicity 
    of the eigenvalues is not equal to one??? For the case of a linear transfomation it is 
    quite trivial, yet for a general multilinear transformation it is way harder.
    Start by studying what is being done with complex matrices...
    Consider grade preserving transformations. Bivector valued of a bivetor variable.
    Then start mixing grades a little bit.
    Relate with the linear transformation counterpart approach.
'''