import numpy as np
import geo_algebra as pyga

def get_matrix(F,basis,rec_basis):
    F_matrix = np.zeros([len(basis),len(basis)])
    mv_lst = [0]*len(basis)

    for i in range(len(basis)):
        mv_lst[i] = F(basis[i])

    for i in range(len(basis)):
        for j in range(len(basis)):
            F_matrix[i][j] += (mv_lst[i]*(rec_basis[j]))(0)
    
    return F_matrix


def get_func_from_matrix(F_matrix,basis,rec_basis):
    '''This function computes a function from a matrix 
       We assume that the matrix was obtained via the above get_matrix function
    '''
    def F(X):
        out = 0
        for i in range(len(matrix)):
            for j in range(len(matrix)):
              out += F_matrix[i][j]*(X*rec_basis[i])(0)*basis[j]
        return out
    return F

def convert_numpyeigvecs_to_eigmvs(eigenvalues, eigenvectors,basis,rec_basis):
    '''From a set of eigenvectors computes the eigenmultivectors and orders the result by the absolute value of the eigenvalues
    '''
    Y = [0]*len(eigenvalues)
    # Convert the eigenvectors to eigenmultivectors
    for i in range(len(eigenvalues)):
        u = np.real(eigenvectors[:,i])
        for j in range(len(basis)):
            Y[i] += u[j]*basis[j]

    #Order eigenmultivectors and eigenvalues by the absolute value of the eigenvalues
    indices = np.argsort(abs(eigenvalues))
    Y_ordered = [Y[i] for i in indices]
    eigenvalues_ordered = eigenvalues[indices]
    return Y_ordered,np.real(eigenvalues_ordered)

def symmetric_eigen_decomp(F,basis,rec_basis):
    '''Solves the eigendecomposition of a multilinear symmetric function F'''
    F_matrix = get_matrix(F,basis,rec_basis)
    eigenvalues, eigenvectors = np.linalg.eig(F_matrix .T)
    return convert_numpyeigvecs_to_eigmvs(eigenvalues, eigenvectors,basis,rec_basis)

def biv_decomp(B,basis,rec_basis):
    '''
    Computes the bivector decomposition from a bivector B. Uses the composition of 
    a function with its adjoint that is G(x) = (x|B)|~B in the euclidean case.
    For negative signature: 
        - It might not allways work.
        - Does not allways retrieve a set of orthogonal vectors
        - When the ai's do not form a set of orthogonal vectors then the decomposition does not work.
    '''
    def F(x):
        return (x|B)(1)

    def G(x):
        return ((x|B)|~B)(1)

    G_matrix = get_matrix(G,basis,rec_basis)
    eigenvalues, eigenvectors = np.linalg.eig(G_matrix.T)
    a,lambda_a = convert_numpyeigvecs_to_eigmvs(eigenvalues,eigenvectors,basis,rec_basis)

    B_lst = [0]*len(a)
    for i in range(len(a)):
        B_lst[i] = -(F(a[i])*pyga.inv(a[i]))(2)
    
    A = 1
    for i in range(len(a)):
        A ^= a[i]

    print("a[1]^...^a[n]=",A)

    return B_lst,a


def compute_versor_decomp_CGA(H_diff,H_adj,basis,rec_basis):
    '''Computes the versor of an orthogonal transformation H
    which does not include parabolic rotations
    each simple reflection contributes with a minus sign
    each reflection which squares to -1 contribute with a minus sign
    H(x) = (-1)**k*U*x*pyga.inv(U) where U is a k-versor.'''
    '''
    The eigenvectors of H_plus might not be orthogonal when the eigenvalue is either one or minus one.
    When the eigenvectors of H_plus are either one or minus one we take the 
    wedge product in this way guaranteeng that the non-orthogonal eigenvectors create a blade 
    instead of a rotor.

    In some cases the a's are repeated!!!! In this situation do not take the wedge product.
    We only take the wedge product when the vector is linear independent
    Note that when the eigenvalue is dfferent from plus or minus one
    then the real part of the eigevalues of H must be distinct.
    In the case of CGA the versor is not unique we can always multiply V by rho = alpha + beta*I, 
    with rho**2 = 1 or rho**2 = -1, rho commutes with all multivector elements. 
    
    Sometimes it does not work yet if I disturb H a little then somehow the solution is slightly different.
    '''

    # basis,rec_basis = get_cga_basis([1])

    H_plus = lambda X: (H_diff(X) + H_adj(X))/2
    a,lambda_plus  = symmetric_eigen_decomp(H_plus,basis,rec_basis)

    # Put the ones or minus ones as the first elements  
    indices = np.argsort(abs(abs(lambda_plus) - 1))
    lambda_plus = lambda_plus[indices]
    a = [a[i] for i in indices]

    sign = 1
    ga = basis[0].GA()
    U = ga.multivector([0],basis=['e'])
    i = 0

    # Find reflections
    while abs(abs(lambda_plus[i]) - 1) < 0.0001:
        # While eigenvalue is equal to one or minus one
        if lambda_plus[i] < -0.9:    
            # Take the wedge product because the a's might not be unique
            U_test = U^a[i]
        
            # Check if U_test is non-zero
            if pyga.numpy_max(U_test) > 0.000001:
                # Only take the wedge product when the a[i] is linearly independent
                U = U_test
                sign *= -1
        i += 1
        if i >= len(a):
            break 

    # Find rotations
    while i < len(a):
        U *= pyga.rotor_sqrt(H_diff(a[i])*pyga.inv(a[i]))
        i += 2

    # print(U)
    # print(numpy_max(U*~U))
    U = pyga.normalize_mv(U)
    # print(numpy_max(U))

    # print("Lambda:",lambda_plus)
    # print("a:",compute_ortho_matrix(a))
    return U,sign*(U*~U)(0)


def compute_eigvalues_from_eigvecs(V,H):
    eigvalues = []
    for i in range(len(V)):
        eigvalues += [H(V[i])*pyga.inv(V[i])]
    def H_check(X):
        out = 0
        for i in range(len(V)):
            out += eigvalues[i]*(X*V[i])(0)*pyga.inv(V[i])
        return out
    return H_check,eigvalues

def compute_ortho_matrix_check(a):
    matrix = np.zeros([len(a),len(a)])
    for i in range(len(a)):
        for j in range(len(a)):
            matrix[i][j] = (a[i]|a[j])(0)

    matrix[matrix < 0.00001] = 0
    return matrix

def check_compare_funcs(F,G,basis):
    values = []
    for i in range(len(basis)):
        values += [(F(basis[i])(1) - G(basis[i])(1)).tolist(1)[0]]
    arr = abs(np.array(values))
    return arr.max()

def check_eigenvalues(F,V,lambda_V):
    values = []
    for i in range(len(V)):
        values += [(F(V[i]) - lambda_V[i]*V[i]).tolist(1)[0]]
    arr = np.array(values)
    # print(arr)
    return abs(arr).max()

def get_orthogonal_func(A,B):
    def H_diff(x):
        out = 0
        for i in range(len(A)):
            out += (x*A[i])(0)*pyga.inv(B[i])
        return out

    def H_adj(x):
        out = 0
        for i in range(len(B)):
            out += (x*pyga.inv(B[i]))(0)*A[i]
        return out

    return H_diff,H_adj


def get_reflections_function(X):
    def F(Y):
        return (X*Y*X).sum()
    return F 


def get_proj_func(P,npoints):
    def F(X):
        return ((X|P)*P).sum()
    return F

def get_rej_func(P,npoints):
    def F(X):
        return ((X^P)*P).sum()
    return F

