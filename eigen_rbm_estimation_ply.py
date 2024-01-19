from geo_algebra import *
import open3d as o3d

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

def reflection(X,Y):
    return (X*Y*X).sum()

def orient_multivectors(X,A):
    scalar = mv.sign((X*A)(0))
    return X*scalar

def get_orient_array(X,X_ref):
    return mv.sign((X*X_ref)(0))

def compute_sign_change(lst1,lst2):
    lst = [0]*len(lst1)
    for i in range(len(lst1)):
        lst[i] = int(lst1[i]*lst2[i])
    return lst

def get_func(X):
    def F(Y):
        return (X*Y*X).sum()
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
    print("Translation Error:", mag_mv(t - t_est))

    cosphi = (normalize_mv(R(2))*~normalize_mv(R_est(2)))(0)
    if abs(cosphi) > 1:
        cosphi = 1
    phi = np.arccos(cosphi)/np.pi*180
    if(phi > 180):
        phi = 360 - phi
    print("Angle between planes of rotation:",phi)
    print()

def gen_pseudordn_rbm(angle,mag):
    theta = angle*np.pi/180
    u = normalize_mv(rdn_vanilla_vec())
    R = np.cos(theta/2) + I*u*np.sin(theta/2)
    t = mag*rdn_vanilla_vec()
    T = 1 + (1/2)*einf*t
    return (T,R,t)

def get_eigmvs(p,grades=[1,2]):
    cga_basis = list(cga.basis(grades=grades).values())
    cga_rec_basis = reciprocal_blades_cga(cga_basis)

    P_lst,lambda_P = eigen_decomp(get_func(p),cga_basis,cga_rec_basis)
    for i in range(len(P_lst)):
        P_lst[i] = normalize(P_lst[i])
    
    return (P_lst,lambda_P)

def get_reference(p):
    p_bar = p.sum()
    p_ref = reflection(p,p_bar)
    P_ref = (p_bar^p_ref) + p_bar
    return P_ref

def get_orient_diff(P,Q,p,q):
    P_ref = get_reference(p)
    Q_ref = get_reference(q)
    sign_p = get_orient_array(P,P_ref)
    sign_q = get_orient_array(Q,Q_ref)
    return sign_p*sign_q

def apply_orientation(P,sign):
    return P*sign

def try_toorient(P,p):
    p_bar = p.sum()
    p_ref = reflection(p,p_bar)
    P_ref = (p_bar^p_ref) + p_bar

    P_oriented = orient_multivectors(P,P_ref)
    return P_oriented

def get_binary_list(n):
    lst = [0]*(2**n)
    for i in range(2**n):
        lst[i] = [2*int(bit)-1 for bit in bin(i)[2:].zfill(n)]
    return lst

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
    # print(max_dist)
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


def initialize_vanilla_cloud(lst):
    x_lst = [0]*len(lst)
    for i in range(len(lst)):
        x_lst[i] = vga.multivector(lst[i],grades=1)
    return x_lst
    
def get_array(x_lst):
    array = np.zeros((len(x_lst),vga.size(1)))
    for i in range(len(x_lst)):
        array[i] = np.array(x_lst[i].list(1)[0][:3])
    return array

def mvarray_to_nparray(x):
    return np.array(x.tolist(1)[0])

def transform_numpy_cloud(pcd,R,t):
    pts = np.asarray(pcd.points).tolist()
    x = vga.multivector(pts,grades=1)
    y = R*x*~R + t
    return mvarray_to_nparray(y)

'''
Tests to show:
    Varying the number of points with no noise m_points->(2,3,...,100)
    Varying the number of points for different values of noise m_points->(5,6,...,100)
'''
if __name__ == "__main__":
    mu = 0
    sigma = 0.02
    m_points = 100

    # pcd = o3d.io.read_point_cloud(f'/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper.ply')
    # pcd = o3d.io.read_point_cloud(f'/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper_res1.ply')
    # pcd = o3d.io.read_point_cloud(f'/home/francisco/Code/Stanford Dataset/bunny/data/bun000.ply')
    # pcd = o3d.io.read_point_cloud(f'/home/francisco/3dmatch/test/7-scenes-redkitchen/fragments/cloud_bin_0.ply')
    pcd = o3d.io.read_point_cloud(f'/home/francisco/3dmatch/test/sun3d-home_at-home_at_scan1_2013_jan_1/fragments/cloud_bin_0.ply')
    pts = np.asarray(pcd.points)
    
    # Convert to vga vectors
    x = vga.multivector(pts.tolist(),grades=1)
    
    # Define the rotation and the translation
    theta = 90*np.pi/180
    R = np.cos(theta/2) + e2*I*np.sin(theta/2)
    t = 0.0*e1 + 0.3*e2 + 0.3*e3
    T = 1 + (1/2)*einf*t
    # T,R,t = gen_pseudordn_rbm(100,10)
    m_points = len(pts)
    max_points = 10
    min_points = 1
    i = 0
    for m_points_ in range(max_points,min_points-1,-1):
        print("Experiment:",i)
        i += 1
        
        # Add gaussian noise
        noise = rdn_gaussian_vga_array(mu,sigma,m_points)
        y = R*x*~R + t + noise

        # Convert points to CGA
        p = eo + x + (1/2)*mag_sq(x)*einf
        q = eo + y + (1/2)*mag_sq(y)*einf
        
        # Get the eigenbivectors
        P_lst,lambda_P = get_eigmvs(p,grades=[1,2])
        Q_lst,lambda_Q = get_eigmvs(q,grades=[1,2])

        # Transform list of multivectors into an array
        P = mv.concat(P_lst) 
        Q = mv.concat(Q_lst)

        # Orient the eigenbivectors from using the points p and q as a reference
        orient_array = get_orient_diff(P,Q,p,q)
        P = P*orient_array
        T_est,R_est = estimate_rbm_1(P,Q)
        
        # Print stuff
        #print(orient_array)
        #print("Easy Orient:")
        print_metrics(R_est,R,T_est,T,m_points)
        #print("t_est:",-2*eo|T_est)
        #print("R_est:",R_est)

'''
    Note that the function F `get_func(p)` and G `get_func(q)` are grade preserving by which it makes sense that the 
    eigenmultivectors to be of unique grade. Furthermore since they are orthogonal and they span 
    the entire fifteen dimensional space then they sould form an orthonormal basis for the bivectors and the vectors.
    Thus concluding that they in fact are blades.
'''

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
