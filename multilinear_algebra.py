import numpy as np
import geo_algebra as pyga
from gasparse import mvarray as mv


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

def compute_bivector_from_skew(F,basis,rec_basis):
    ''' Computes the bivector of the skew symmetric part of F'''

    out = 0
    for i in range(len(basis)):
        out += basis[i]^F(rec_basis[i])
    return out/2

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
    indices = np.argsort(eigenvalues)
    Y_ordered = [Y[i] for i in indices]
    eigenvalues_ordered = eigenvalues[indices]
    return Y_ordered,np.real(eigenvalues_ordered)

def convert_numpyeigvecs_to_eigmvs_1(eigenvalues, eigenvectors,basis,rec_basis,upss):
    '''From a set of eigenvectors computes the eigenmultivectors and orders the result by the absolute value of the eigenvalues
    upss: The unit pseudoscalar. It should behave as the unit imaginary, that is, it must square to minus one and commutes with all elements
    '''
    Y = [0]*len(eigenvalues)
    # Convert the eigenvectors to eigenmultivectors
    for i in range(len(eigenvalues)):
        u = np.real(eigenvectors[:,i])
        v = np.imag(eigenvectors[:,i])
        for j in range(len(basis)):
            Y[i] += u[j]*basis[j] + v[j]*basis[j]*upss

    #Order eigenmultivectors and eigenvalues by the absolute value of the eigenvalues
    indices = np.argsort(np.abs(eigenvalues))
    Y_ordered = [Y[i] for i in indices]
    eigenvalues_ordered = eigenvalues[indices]

    # Convert complex eigenvalues to scalar + pseudoscalar 
    eigvalsga = [0]*len(eigenvalues_ordered)
    for i in range(len(eigenvalues_ordered)):
        eigvalsga[i] = np.real(eigenvalues_ordered[i]) + upss*np.imag(eigenvalues_ordered[i])


    return Y_ordered,eigvalsga

def compute_ortho_matrix_check(a):
    matrix = np.zeros([len(a),len(a)])
    for i in range(len(a)):
        for j in range(len(a)):
            matrix[i][j] = (a[i]|a[j])(0)

    matrix[abs(matrix) < 0.00001] = 0
    return matrix

def form_eigenblades_from_eigvectors(a,lambda_a):
    ''' Takes a set of eigenvectors and forms blades when the multiplicity of the eigenvalues is 
        greater then one  '''
    # Join the elements which have multiplicity greater than one
    B_lst = []
    lambda_lst = []
    j = 0
    while j < len(lambda_a):
        value = lambda_a[j]
        B = 1
        while j < len(lambda_a) and abs(value - lambda_a[j]) < 0.0001:
            B_test = B^a[j]
            if pyga.numpy_max(B_test) > 0.0001:
                B = B_test
            j += 1

        lambda_lst += [value]
        B_lst += [B]

    return B_lst,lambda_lst

def symmetric_eigen_decomp(F,basis,rec_basis):
    '''Solves the eigendecomposition of a multilinear symmetric function F'''
    F_matrix = get_matrix(F,basis,rec_basis)
    eigenvalues, eigenvectors = np.linalg.eig(F_matrix.T)
    print(eigenvalues)
    return convert_numpyeigvecs_to_eigmvs(eigenvalues, eigenvectors,basis,rec_basis)

def symmetric_eigen_decomp_1(F,basis,rec_basis,upss):
    '''Solves the eigendecomposition of a multilinear symmetric function F'''
    F_matrix = get_matrix(F,basis,rec_basis)
    eigenvalues, eigenvectors = np.linalg.eig(F_matrix.T)
    print(eigenvalues)
    return convert_numpyeigvecs_to_eigmvs_1(eigenvalues, eigenvectors,basis,rec_basis,upss)

def symmetric_eigen_blades_decomp(F,basis,rec_basis):
    a,lambda_a = symmetric_eigen_decomp(F,basis,rec_basis)
    print(lambda_a)
    # for i in range(len(a)):
    #     print(a[i])
    return form_eigenblades_from_eigvectors(a,lambda_a)

def biv_decomp(B,basis,rec_basis):
    '''
    Computes the bivector decomposition from a bivector B. Uses the composition of 
    a function with its adjoint that is G(x) = (x|B)|~B.
    '''
    '''
    For negative signature: 
        - It might not allways work.
        - Does not allways retrieve a set of orthogonal vectors
        - When the ai's do not form a set of orthogonal vectors then the decomposition does not work.
    '''
    def F(x):
        return (x|B)(1)

    def G(x):
        return ((x|B)|~B)(1)

    a,lambda_a = symmetric_eigen_decomp(G,basis,rec_basis)

    B_lst = []

    i = 0
    while i < len(a):
        Bi = -(F(a[i])*pyga.inv(a[i]))(2)
        B_lst += [Bi]
        i += 2
    
    A = 1
    for i in range(len(a)):
        A ^= a[i]

    return B_lst


def compute_versor_symmetric(H_diff,H_adj,basis,rec_basis):
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
    
    Sometimes it does not work yet if I disturb H a little then somehow the solution is very different.
    '''

    H_plus = lambda X: (H_diff(X) + H_adj(X))/2
    a,lambda_plus  = symmetric_eigen_decomp(H_plus,basis,rec_basis)

    # Put the ones or minus ones as the first elements  
    indices = np.argsort(abs(abs(lambda_plus) - 1))
    lambda_plus = lambda_plus[indices]
    a = [a[i] for i in indices]

    sign = 1
    ga = basis[0].GA()
    U = ga.mvarray([1.0],basis=['e'])
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

    U = pyga.normalize(U)
    return U,sign*(U*~U)(0)


def compute_versor_skew(F,basis,rec_basis):
    ''' Computes the versor of an orthogonal transformation using the skew symmetric part of F.
        This decomposition only works for special orthogonal transformations.
        Since reflections are symmetric they get anhialated when computing the bivector.
        This algorithm works mostly for 'positive' geometric algebras.  
    '''
    '''
        For reflections it forms a blade A for each linearly indepedent vector that is found A ^= a[i],
        then it multiplies by the versor.
        Not allways the eigenvectors of the decomposition form an orthogonal basis for the entire space.
    '''
    ga = basis[0].GA()
    B = compute_bivector_from_skew(F,basis,rec_basis)

    # If it is symmetric, return the identity.
    if pyga.numpy_max(B) < 0.00001:
        return ga.mvarray([1],basis=['e']) # It must be a multivector

    pss = list(ga.basis(grades=len(ga.metric())).values())[0]
    # Get the blades and the eigenvectors of x|B
    def G(x):
        return ((x|B)|~B)(1)

    a,lambda_G = symmetric_eigen_decomp(G,basis,rec_basis)

    V = 1
    i = 0
    A = ga.mvarray([1],basis=['e'])

    while i < len(a):
        if abs(lambda_G[i]) > 1e-10: # Found a vector in F_minus
            Vi_sq = F(a[i])*pyga.inv(a[i])
            V *= pyga.rotor_sqrt(Vi_sq)
            i += 1 # skip next iteration
        i += 1
    return V

    
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


def check_compare_funcs(F,G,basis):
    ''' Computes the difference between the matrices associated to the functions F and G.
        Using some basis bi, compute max_i,j (F(bi) - G(bi))|bj
    '''
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

