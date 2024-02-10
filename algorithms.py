from cga3d_estimation import *

'''
In the algorithms for estimating transformations we use the docstring to name
the function when printing results of the algorithms. Thus the docstring has to be small
yet descriptive enough regarding wht the algorithm does. 
Also it must not have newline characters, that is it must be written on a single line.
Use the first line of the docstring to get the name of the algorithm. The rest of the algorithm description can 
stay a line bellow of the name.

# Can also use lambda expression to declare functions
F = lambda X: ((x-x_bar)*X*(x-x_bar)).sum()
G = lambda X: ((y-y_bar)*X*(y-y_bar)).sum()

TODO: 
    - Restrict the maximum number of character for the docstring
'''


def get_algorithm_name(algorithm):
    return algorithm.__doc__.split("\n")[0]


def estimate_transformation_0(x,y,npoints):
    '''CGA RBM Null Reflection
        Estimates the rigid body motion between two point clouds using the eigenmultivector.
        From the eigenmultivectors we estimate the rotation and translation. 
        To estimate the sign we use references from the PCs themselves.
    '''
    eig_grades = [1,2]
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenbivectors
    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Orient the eigenbivectors by using the points p and q as a reference
    signs = get_orient_diff(P,Q,p,q)
    P = P*signs
    T_est,R_est = estimate_rigtr(P,Q)
    return (T_est,R_est,P_lst,Q_lst)

# Use the center of mass to estimate the translation
def estimate_transformation_1(x,y,npoints):
    '''CGA RBM CeOM'''
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenbivectors
    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Orient the eigenbivectors by using the points p and q as a reference
    signs = get_orient_diff(P,Q,p,q)
    P = P*signs
    T_est,R_est = estimate_rigtr(P,Q)

    # Determine the translation from the center of mass
    T_est = translation_from_cofm(y,x,R_est,npoints)
    # Need to calculate number of points, for that need to implement the len(x) method
    return (T_est,R_est,P_lst,Q_lst)

def estimate_transformation_2(x,y,npoints):
    '''CGA RBM Refine Rot'''
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenbivectors
    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Orient the eigenbivectors by using the points p and q as a reference
    signs = get_orient_diff(P,Q,p,q)
    P = P*signs
    T_est,R_est = estimate_rigtr(P,Q)
    
    # Determine the translation from the center of mass
    T_est = translation_from_cofm(y,x,R_est,npoints)
    S = ~T_est*Q*T_est

    # Refine the rotation
    R_est = estimate_rot_3dcga(P,S)

    # Need to calculate number of points, for that need to implement the len(x) method
    return (T_est,R_est,P_lst,Q_lst)

def get_CGA_rot_func(x,npoints):
    x_bar = x.sum()/npoints
    x_prime = x - x_bar
    p = eo + x_prime + (1/2)*pyga.mag_sq(x_prime)*einf

    def F(X):
        return (p*X*p).sum()

    return F

def estimate_transformation_3(x,y,npoints):
    '''CGA RBM Centered'''
    basis,rec_basis = get_3dcga_basis(eig_grades)
    F = get_CGA_rot_func(x,npoints)
    G = get_CGA_rot_func(y,npoints)
    P_lst,lambda_P = multiga.symmetric_eigen_decomp(F,basis,rec_basis)
    Q_lst,lambda_Q = multiga.symmetric_eigen_decomp(G,basis,rec_basis)
    
    x_bar = x.sum()/npoints
    y_bar = y.sum()/npoints

    # Convert to CGA 
    p = eo + x-x_bar + (1/2)*pyga.mag_sq(x-x_bar)*einf
    q = eo + y-y_bar + (1/2)*pyga.mag_sq(y-y_bar)*einf

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Correct sign
    signs = get_orient_diff(P,Q,p,q)
    P = P*signs

    R_est = estimate_rot_3dcga(P,Q)
    T_est = translation_from_cofm(y,x,R_est,npoints)

    return (T_est,R_est,P_lst,Q_lst)


def estimate_transformation_7(x,y,npoints):
    '''CGA RBM Centered Brute Force'''

    basis,rec_basis = get_3dcga_basis(2)
    F = get_CGA_rot_func(x,npoints)
    G = get_CGA_rot_func(y,npoints)
    P_lst,lambda_P = multiga.symmetric_eigen_decomp(F,basis,rec_basis)
    Q_lst,lambda_Q = multiga.symmetric_eigen_decomp(G,basis,rec_basis)
    
    x_bar = x.sum()/npoints
    y_bar = y.sum()/npoints

    # Convert to CGA 
    p = eo + x-x_bar + (1/2)*pyga.mag_sq(x-x_bar)*einf
    q = eo + y-y_bar + (1/2)*pyga.mag_sq(y-y_bar)*einf

    R_est = brute_force_estimate_CGA(P_lst,Q_lst)
    T_est = translation_from_cofm(y,x,R_est,npoints)

    return (T_est,R_est,P_lst,Q_lst)


def get_VGA_rot_func(x,npoints):
    x_bar = x.sum()/npoints
    def F(X):
        return ((x-x_bar)*X*(x-x_bar)).sum()

    return F

eps = 1e-12
def estimate_transformation_4(x,y,npoints):
    '''VGA RBM Centered'''

    basis,rec_basis = get_3dvga_basis(1)
    
    # Determine centers of mass
    x_bar = x.sum()/npoints
    y_bar = y.sum()/npoints
    
    # Compute the eigendecomposition
    F = multiga.get_reflections_function(x-x_bar)
    G = multiga.get_reflections_function(y-y_bar)
    P_lst,lambda_P = multiga.symmetric_eigen_decomp(F,basis,rec_basis)
    Q_lst,lambda_Q = multiga.symmetric_eigen_decomp(G,basis,rec_basis)

    # Correct the sign of the eigenvectors
    for i in range(len(P_lst)):
        P_lst[i] = P_lst[i]*correct_sign_2(P_lst[i],x-x_bar)
        Q_lst[i] = Q_lst[i]*correct_sign_2(Q_lst[i],y-y_bar)

    R_est = ~exact_rotation(Q_lst[0],Q_lst[1],P_lst[0],P_lst[1])
    T_est = translation_from_cofm(y,x,R_est,npoints)

    return (T_est,R_est,P_lst,Q_lst)

# Estimate with know correspondences
def estimate_transformation_5(x,y,npoints):
    '''VGA RBM Known Corrs'''
    x_bar = x.sum()/npoints
    y_bar = y.sum()/npoints
    R_est = estimate_rot_3dvga(x-x_bar,y-y_bar)
    T_est = translation_from_cofm(y,x,R_est,npoints)
    return (T_est,R_est,None,None)

def get_proj_func(P,npoints):
    P_bar = P.sum()/npoints
    def F(X):
        return ((X|(P - P_bar))*(P - P_bar)).sum()

    return F

def get_rej_func(P,npoints):
    P_bar = P.sum()/npoints
    def F(X):
        return ((X^(P - P_bar))*(P - P_bar)).sum()

    return F

def get_non_linear_func(P,npoints):
    def F(X):
        scale = (P*~P)(0)
        scale = scale*scale
        return ((X^P)*P*scale).sum()
    return F





def estimate_transformation_6(x,y,npoints):
    '''VGA RBM Nonlinear Func'''
    basis,rec_basis = get_3dvga_basis(1)

    x_bar = x.sum()/npoints
    y_bar = y.sum()/npoints

    F1 = get_proj_func(x-x_bar,npoints)
    G1 = get_proj_func(y-y_bar,npoints)
    F2 = get_non_linear_func(x-x_bar,npoints)
    G2 = get_non_linear_func(y-y_bar,npoints)

    P_lst1,lambda_P1 = multiga.symmetric_eigen_decomp(F1,basis,rec_basis)
    Q_lst1,lambda_Q1 = multiga.symmetric_eigen_decomp(G1,basis,rec_basis)

    P_lst2,lambda_P2 = multiga.symmetric_eigen_decomp(F2,basis,rec_basis)
    Q_lst2,lambda_Q2 = multiga.symmetric_eigen_decomp(G2,basis,rec_basis)

    R_est = brute_force_estimate_VGA(P_lst1,P_lst2,Q_lst1,Q_lst2)

    T_est = translation_from_cofm(y,x,R_est,npoints)

    return (T_est,R_est,P_lst1,P_lst2)




def estimate_transformation_8(x,y,npoints):
    '''CGA RBM H_matrix
        Computes a matrix which relates the eigenmultivectors P_lst and Q_lst.
        We take certain coefficents of the matrix to determine the translation
        and the rotation. 
        The default sign atribution is done with get_3dcga_eigmvs, where it uses the point
        at infinity to switch signs.
    '''

    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=1)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=1)

    H_diff,H_adj = multiga.get_orthogonal_func(P_lst,Q_lst)

    basis = [eo,e1,e2,e3,einf]
    rec_basis = [-einf,e1,e2,e3,-eo]

    # # Verify that H is in fact magnitude preserving
    # for i in range(len(basis)):
    #     print(pyga.mag_sq(H_diff(basis[i])))
    # print()

    H_matrix = multiga.get_matrix(H_diff,basis,rec_basis).T

    t_vec = H_matrix[1:4,0]
    R_matrix = H_matrix[1:4,1:4]

    R_est = rotmatrix_to_3drotor(R_matrix)
    t_est = nparray_to_3dvga_vector_array(t_vec)

    T_est = 1 + (1/2)*einf*t_est

    return (T_est,R_est,P_lst,P_lst)

def estimate_transformation_9(x,y,npoints):
    '''CGA test eig-values
        sometimes the eigen-decomposition of H_plus is not correct. 
        I verified if the eigenvector satisfy the eigenvalue equation, which is 
        not allways satisfied. After some experimentation with various levels of noise 
        the above problem does not seem to occur as often as I thought it would.
        This function takes the eigenvectors P and Q of F and G respectively 
        and computes the eigendecomposition of the 
        transformation H(x) = sum_i (x|P_i)*inv(Q_i)
        The main idea is that because of noise H is going to be a general orthogonal transformation in A(p,q)
        Then by taking the spectral decomposition I was wondering If I can extract the rotation and translation
        from H.
    '''
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=1)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=1)

    H_diff,H_adj = multiga.get_orthogonal_func(P_lst,Q_lst)

    basis,rec_basis = get_3dcga_basis([1])
    H_plus = lambda X: (H_diff(X) + H_adj(X))/2

    V,lambda_V  = multiga.symmetric_eigen_decomp(H_plus,basis,rec_basis)
    multiga.check_eigenvalues(H_plus,V,lambda_V)
    
    print("lambda:",lambda_V)
    # Verify if the 'complex' eigenvalues of H_diff are unitary
    # print()
    values = []
    for i in range(len(V)):
        lambda_i = H_diff(V[i])*V[i]
        # print(lambda_i)
        # print()
        values += [(lambda_i*~lambda_i)(0)]
    arr = np.array(values)
    # print()
    # print("Unitary Eigenvalues",abs(abs(arr) - 1).max())
    # print("Test eigmvs:")

    eigvalues = []
    for i in range(len(V)):
        eigvalues += [H_diff(V[i])*pyga.inv(V[i])]

    def H_check(X):
        out = 0
        for i in range(len(V)):
            out += eigvalues[i]*(x|V[i])*pyga.inv(V[i])
        return out

    # Check if H_check is equal to H
    values = []
    for i in range(len(basis)):
        values += [(H_check(basis[i])(1) - H_diff(basis[i])(1)).tolist(1)[0]]
    arr = abs(np.array(values))
    print("Max complex eigenvalue Error:", arr.max())

    # Verify that H is in fact magnitude preserving
    values = []
    for i in range(len(basis)):
        values += [pyga.mag_sq(H_diff(basis[i]))]
    arr = abs(abs(np.array(values)) - 1)
    # print("Unitary Transformation",arr.max())
    
    Versor = 1
    sign = 1
    Versor *= pyga.rotor_sqrt(H_diff(V[0])*pyga.inv(V[0]))
    # Versor *= the_other_pyga.rotor_sqrt(H_diff(V[2])*pyga.inv(V[2]))
    Versor *= pyga.rotor_sqrt(H_diff(V[4])*pyga.inv(V[4]))
    if lambda_V[2] < -0.9:
        Versor *= V[2]
        sign = -1

    U = Versor

    # Check if U is the versor of H
    values = []
    for i in range(len(basis)):
        values += [((sign*U*basis[i]*~U)(1) - H_diff(basis[i])).tolist(1)[0]]
    arr = abs(np.array(values))
    print("Max diff Versor:", arr.max())
    T_est,R_est = best_motor_estimation(U)
    # R_est = pyga.normalize_mv(Proj_I(U))
    # t_est = -2*Proj_I(((eo|U)*~R_est)(1))
    # print(R_est)
    # R_est = 1 + 0*e12
    # t_est = 0
    # T_est = 1 + (1/2)*einf*t_est

    return (T_est,R_est,P_lst,P_lst)


eps = 1e-12
def estimate_rotation_0(x,y,npoints):
    '''VGA ROT Unit Reflection
        pyga.normalizes the points before estimating the rotation.
        Should be robust for outliers far away from the origin. 
    '''

    # pyga.normalize points
    x = pyga.normalize(x)
    y = pyga.normalize(y)

    # Determine the eigenvectors
    basis,rec_basis = get_3dvga_basis(1)
    F = multiga.get_reflections_function(x)
    G = multiga.get_reflections_function(y)
    P_lst,lambda_P = multiga.symmetric_eigen_decomp(F,basis,rec_basis)
    Q_lst,lambda_Q = multiga.symmetric_eigen_decomp(G,basis,rec_basis)

    # Correct the sign of the eigenvectors
    for i in range(len(P_lst)):
        P_lst[i] = P_lst[i]*correct_sign_2(P_lst[i],x)
        Q_lst[i] = Q_lst[i]*correct_sign_2(Q_lst[i],y)

    R_est = ~exact_rotation(Q_lst[0],Q_lst[1],P_lst[0],P_lst[1])

    return (1,R_est,P_lst,Q_lst)

def estimate_rotation_1(x,y,npoints):
    '''VGA ROT Scaled Reflection
        Uses the points directly. It should illustrate better the advantage of VGA ROT Unit Reflection 
        when dealing with outliers.
    '''

    # Determine the eigenvectors
    basis,rec_basis = get_3dvga_basis(1)
    F = multiga.get_reflections_function(x)
    G = multiga.get_reflections_function(y)
    P_lst,lambda_P = multiga.symmetric_eigen_decomp(F,basis,rec_basis)
    Q_lst,lambda_Q = multiga.symmetric_eigen_decomp(G,basis,rec_basis)

    # Correct the sign of the eigenvectors
    for i in range(len(P_lst)):
        P_lst[i] = P_lst[i]*correct_sign_2(P_lst[i],x)
        Q_lst[i] = Q_lst[i]*correct_sign_2(Q_lst[i],y)

    R_est = ~exact_rotation(Q_lst[0],Q_lst[1],P_lst[0],P_lst[1])

    return (1,R_est,P_lst,Q_lst)

eps = 1e-12

eig_grades = [1]
def estimate_rotation_2(x,y,npoints):
    '''CGA ROT Null Reflection
        Converts the points to Conformal Geometric Algebra. 
        Then estimates the rotation using the eigenvectors.
        Only uses vectors to estimate rotation.
    '''
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenbivectors
    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Orient the eigenbivectors by using the points p and q as a reference
    signs = get_orient_diff(P,Q,p,q)
    P = P*signs

    R_est = estimate_rot_3dcga(P,Q)

    return (1,R_est,P_lst,Q_lst)

eig_grades = [1,2]
def estimate_rotation_3(x,y,npoints):
    '''CGA ROT pyga.normalized Null Reflection
        Same as CGA ROT Null Reflection, but it pyga.normalizes the points before.
    '''
    # pyga.normalize points
    x = pyga.normalize(x)
    y = pyga.normalize(y)

    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenbivectors
    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Orient the eigenbivectors by using the points p and q as a reference
    signs = get_orient_diff(P,Q,p,q)
    P = P*signs

    R_est = estimate_rot_3dcga(P,Q)

    return (1,R_est,P_lst,Q_lst)


def correct_sign_vec_rot_CGA(P):
    return mv.sign((P|einf)(0))*P

eig_grades = [1]
def estimate_rotation_4(x,y,npoints):
    '''CGA ROT Null Reflection Scaled
        Scales the eigenvectors in order for the eo component to be equal on both
        eigenvectors
    '''
    # Convert to CGA
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    # Get the eigenbivectors
    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    # Transform list of multivectors into an array
    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # Correctly scale the eigenvector using the point at infinity
    # Only use this when P is of grade one 
    P = P/(P*einf)(0)
    Q = Q/(Q*einf)(0)

    # Estimate the rotation using the euclidean components of P and Q
    R_est = estimate_rot_3dcga(P,Q)

    return (1,R_est,P_lst,Q_lst)


eig_grades = [1]
def estimate_rotation_5(x,y,npoints):
    '''CGA ROT H_matrix
        Estimate the rotor by determining the eigendecomposition of H_diff.
        Even though H_plus is a symmetric transformation, the matrix of H_plus is not.
        So since the multiga.symmetric_eigen_decomp does not deal well with non symetric matrices, 
        this will eventually provide poor results.
    '''
    p = eo + x + (1/2)*pyga.mag_sq(x)*einf
    q = eo + y + (1/2)*pyga.mag_sq(y)*einf

    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=1)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=1)

    H_diff,H_adj = multiga.get_orthogonal_func(P_lst,Q_lst)

    H_plus = lambda X: (H_diff(X) + H_adj(X))/2

    basis,rec_basis = get_3dcga_basis(1)
    W,lambda_W = multiga.symmetric_eigen_decomp(H_plus,basis,rec_basis)

    R_est = ~pyga.rotor_sqrt(Proj_I(W[0]*H_diff(W[0])))

    return (1,R_est,P_lst,Q_lst)


def estimate_rotation_6(x,y,npoints):
    '''VGA ROT H_matrix SD
        Estimate a rotation via relating the eigenvectors of F and G
        Because of the sign this relation will not hold to be correct.
        If the determinant of H is -1 then we will have a reflection instead of
        a rotation.
        If the sign changes provide two reflections then we will go back to 
        having a rotation.
        Use the spectral decomposition of the rotation matrix to estimate the
        rotation.
    '''
    # Determine the eigenvectors
    basis,rec_basis = get_3dvga_basis(1)
    F = multiga.get_reflections_function(x)
    G = multiga.get_reflections_function(y)
    P_lst,lambda_P = multiga.symmetric_eigen_decomp(F,basis,rec_basis)
    Q_lst,lambda_Q = multiga.symmetric_eigen_decomp(G,basis,rec_basis)

    # Correct the sign of the eigenvectors
    for i in range(len(P_lst)):
        P_lst[i] = P_lst[i]*correct_sign_2(P_lst[i],x)
        Q_lst[i] = Q_lst[i]*correct_sign_2(Q_lst[i],y)

    H_diff,H_adj = multiga.get_orthogonal_func(P_lst,Q_lst)
    H_plus = lambda X: (H_diff(X) + H_adj(X))/2
    V_lst,lambda_V = multiga.symmetric_eigen_decomp(H_plus,basis,rec_basis)

    R_est = ~pyga.rotor_sqrt(V_lst[0]*H_diff(V_lst[0]))

    return (1,R_est,P_lst,Q_lst)


RBM_algs_list = [estimate_transformation_0,estimate_transformation_1,estimate_transformation_2,estimate_transformation_3,estimate_transformation_4,
                 estimate_transformation_5,estimate_transformation_6,estimate_transformation_7,estimate_transformation_8]
ROT_algs_list = [estimate_rotation_0,estimate_rotation_1,estimate_rotation_2,estimate_rotation_3,estimate_rotation_4,estimate_rotation_5,estimate_rotation_6]

algorithms_list = RBM_algs_list + ROT_algs_list

def get_alg_name_list(algorithms):
    names = []
    for i in range(len(algorithms)):
        names += [get_algorithm_name(algorithms[i])]

    return names