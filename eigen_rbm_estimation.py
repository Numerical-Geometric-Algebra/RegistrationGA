from geo_algebra import *

'''
This snippet describes code to determine the rigid body motion between two noisy Point Clouds
To be robust to noise it first solves two eigenvalue problems stated in a multivector space
that is it finds the eigenmultivector of F(X) = pi*X*pi and G(X) = qi*X*qi. Then it orders the 
eigenmultivectors by magnitude of the absolute value. Before using the eigenmultivectors we correct 
the orientation by using the mean and pi*p_bar*pi. Then since both eigenmultivectors are ordered in the 
same manner we do not need to estimate correspondences. The last is to use the rigid body estimator to 
find the rotation and the translation. 

Comment:
    No noise:
        - If both point clouds have five or more points then our method is able to allways estimate the rotation and 
        and translation
        - If the number of points is equal to four then we are only able to estimate the rotation correctly
        - If the number of points is less than four then it is not possible to estimate the RBM
        - We only need three eigenbivectors to estimate the rotation and translation correctly
'''



def reflect_list(X_lst,X):
    Y = 0
    for i in range(len(X_lst)):
        Y += X_lst[i]*X*X_lst[i]
    return Y

def orient_multivectors(X_lst,X_ref):
    Y_lst = [0]*len(X_lst)

    for i in range(len(X_lst)):
        X = X_lst[i]
        scalar = (X*X_ref)(0)
        if scalar == 0:
            scalar = 1
        else:
            scalar = np.sign(scalar)
        Y_lst[i] = scalar*X
    return Y_lst

def get_orient_list(X_lst,X_ref):
    lst = [0]*len(X_lst)
    for i in range(len(X_lst)):
        X = X_lst[i]
        scalar = (X*X_ref)(0)
        if scalar == 0:
            scalar = 1
        else:
            scalar = np.sign(scalar)
        lst[i] = scalar

    return lst

def compute_sign_change(lst1,lst2):
    lst = [0]*len(lst1)
    for i in range(len(lst1)):
        lst[i] = lst1[i]*lst2[i]
    return lst

def get_func(X_lst):
    def F(Y):
        out = 0
        for i in range(len(X_lst)):
            out += X_lst[i]*Y*X_lst[i]
        return out
    return F 

def get_ang_error(R_est,R):
    costheta = (R_est*~R)(0)
    if abs(costheta) > 1:
        return 0
    ang_error = np.arccos(costheta)/np.pi*360
    if ang_error > 180:
        ang_error = 360 - ang_error
    return abs(ang_error)

def print_metrics(R_est,R,T_est,T,m_points=-1):
    t = -eo|T*2
    t_est = -eo|T_est*2
    costheta = (R_est*~R)(0)
    if abs(costheta) > 1:
        # print("Cos Angle Error:",costheta)
        costheta = 1
    ang_error = np.arccos(costheta)/np.pi*360 # gets two times theta
    if ang_error > 180:
        ang_error = 360 - ang_error
    if(m_points > 0):
        print("Nbr points:",m_points)
    print("Angle Error:",ang_error)
    print("Translation Error:", mag(t - t_est))

    cosphi = (normalize(R(2))*~normalize(R_est(2)))(0)
    if abs(cosphi) > 1:
        cosphi = 1
    phi = np.arccos(cosphi)/np.pi*180
    if(phi > 180):
        phi = 360 - phi
    print("Angle between planes of rotation:",phi)
    print()

def gen_pseudordn_rbm(angle,mag):
    theta = angle*np.pi/180
    u = normalize(rdn_vanilla_vec())
    R = np.cos(theta/2) + I*u*np.sin(theta/2)
    t = mag*rdn_vanilla_vec()
    T = 1 + (1/2)*einf*t
    return (T,R,t)

def get_eigmvs(p_lst,grades=[1,2]):
    cga_basis = list(cga.basis(grades=grades).values())
    cga_rec_basis = reciprocal_blades_cga(cga_basis)

    P_lst,lambda_P = eigen_decomp(get_func(p_lst),cga_basis,cga_rec_basis)
    P_lst = normalize_null_mvs(P_lst)
    
    return (P_lst,lambda_P)

def get_reference(p_lst):
    p_bar = mv_list_mean(p_lst)
    p_ref = reflect_list(p_lst,p_bar)
    p_ref = (p_bar^p_ref) + p_bar
    return p_ref

def get_orient_diff(P_lst,Q_lst,p_lst,q_lst):
    p_ref = get_reference(p_lst)
    q_ref = get_reference(q_lst)
    lstp = get_orient_list(P_lst,p_ref)
    lstq = get_orient_list(Q_lst,q_ref)
    return compute_sign_change(lstp,lstq)

def apply_orientation(P_lst,lst):
    X_lst = []
    for i in range(len(P_lst)):
        X_lst += [lst[i]*P_lst[i]]
    return X_lst

def try_toorient(P_lst,p_lst):
    p_bar = mv_list_mean(p_lst)
    p_ref = reflect_list(p_lst,p_bar)
    p_ref = (p_bar^p_ref) + p_bar

    P_oriented = orient_multivectors(P_lst,p_ref)
    return P_oriented

# Computes the plus distance between two multivector arrays
def plus_dist_list(X_lst,Y_lst):
    lst = []
    for i in range(len(X_lst)):
        lst += [pos_cga_dist(X_lst[i],Y_lst[i])]
    return lst

def filter_bivectors(X_lst):
    Y_lst = []
    for i in range(len(X_lst)):
        if(X_lst[i].grade() == 2):
            Y_lst += [X_lst[i]]
    return Y_lst

def get_binary_list(n):
    lst = [0]*(2**n)
    for i in range(2**n):
        lst[i] = [2*int(bit)-1 for bit in bin(i)[2:].zfill(n)]
    return lst

def cga_dist_lst(P_lst,Q_lst):
    dist = 0 
    for i in range(len(P_lst)):
        dist += pos_cga_dist(P_lst[i],Q_lst[i])
    return dist

def dist_lst(P_lst,Q_lst):
    dist = 0 
    for i in range(len(P_lst)):
        dist += abs(mag(P_lst[i] - Q_lst[i]))
    return dist

def brute_force_estimate(P_lst,Q_lst,n=3):
    '''
        We need three bivectors to estimate a rotation and a translation
        There are 10 eigenbivectors in total in 3D CGA we could do the brute force method for 
        multiple sets of three
    '''
    P_lst = P_lst[-n:]
    Q_lst = Q_lst[-n:]

    sign_list = get_binary_list(n)
    max_dist = np.inf
    best_index = -1

    P_lst_s = [0]*n

    for i in range(len(sign_list)):
        for j in range(n): # iterate over the bivectors 
            P_lst_s[j] = P_lst[j]*sign_list[i][j] # Change the sign

        T_est,R_est = estimate_rbm_1(P_lst_s,Q_lst)
        Q_lst_s = trans_list(P_lst_s,T_est*R_est)
        dist = dist_lst(Q_lst,Q_lst_s)
        
        if(dist < max_dist):
            max_dist = dist
            best_index = i
    
    print(sign_list[best_index])
    print(max_dist)
    # Compute the rotation for the best signs
    for j in range(n): # iterate over the multivectors 
        P_lst_s[j] = P_lst[j]*sign_list[best_index][j] # Change the sign
    
    T_est,R_est = estimate_rbm_1(P_lst_s,Q_lst)
    return (T_est,R_est)


def dist_based_orient(P_lst,Q_lst,T,R):
    P_oriented = []
    Q_lst_ = trans_list(P_lst,T*R)

    for i in range(len(P_lst)):
        dist_pos = pos_cga_dist(Q_lst[i],Q_lst_[i])
        dist_neg = pos_cga_dist(Q_lst[i],-Q_lst_[i])
        print(dist_neg,dist_pos)
        if dist_neg < dist_pos:
            P_lst[i] *= -1

'''
Tests to show:
    Varying the number of points with no noise m_points->(2,3,...,100)
    Varying the number of points for different values of noise m_points->(5,6,...,100)
'''

mu = 0
sigma = 0
m_points = 10

max_points = 10
min_points = 10
for m_points in range(max_points,min_points-1,-1):

    # x_lst = generate_rdn_PC(m_points)
    x_lst = generate_unitcube_rdn_PC(m_points)
    T,R,t = gen_pseudordn_rbm(100,10)

    y_lst_ = apply_vec_RBM(x_lst,R,t)
    noise  = gen_gaussian_noise_list(m_points,mu,sigma)
    y_lst = add_noise(y_lst_,noise)

    # Convert to CGA
    p_lst = vanilla_to_cga_vecs(x_lst)
    q_lst = vanilla_to_cga_vecs(y_lst)

    # Get the eigenbivectors
    P_lst,lambda_P = get_eigmvs(p_lst,grades=2)
    Q_lst,lambda_Q = get_eigmvs(q_lst,grades=2)

    T_est,R_est = brute_force_estimate(P_lst,Q_lst,10)
    print("Brute Force:")
    print_metrics(R_est,R,T_est,T,m_points)


    orient_lst = get_orient_diff(P_lst,Q_lst,p_lst,q_lst)
    P_lst = apply_orientation(P_lst,orient_lst)
    print(orient_lst)
    T_est,R_est = estimate_rbm_1(P_lst,Q_lst)
    print("Easy Orient:")
    print_metrics(R_est,R,T_est,T,m_points)


'''
# Check that the eigenmultivectors are orthogonal and they are also blades
matrix = np.zeros([len(P_lst),len(P_lst)])
self_prod = [0]*len(P_lst)
for i in range(len(P_lst)):
    self_prod[i] = P_lst[i]*P_lst[i]
    for j in range(len(P_lst)):
        matrix[i][j] = get_float(P_lst[i]*P_lst[j])


# Sanity check if the P's and Q's are eigenmultivectors
Func = get_func(p_lst)
for i in range(len(P_lst)):
    print(P_lst[i]*lambda_P[i] - Func(P_lst[i]))
'''

'''
TODO:
    - For large amounts of noise it is difficult to determine the orientation of 
    the eigenmultivectors, as such by increasing the noise above a certain threshold we are 
    not able to determine the right orientation resulting in a very bad rotation accuracy
    - [ ] Find a more noise robust approach to estimate orientation of multivectors 
    - [ ] Study the algorithm under the influence of outliers
    - [x] Study the solution for the eigenmultivectors, it seems that even though we find the
    eigenvectors of the matrix of F, the corresponding multivectors are not eigenmultivectors
'''
