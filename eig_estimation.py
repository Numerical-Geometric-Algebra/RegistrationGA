#!/usr/bin/env python
from geo_algebra import *

'''
The functions to help estimate the rigid body motion between two point clouds.
'''


def reflection(X,Y):
    return (X*Y*X).sum()

def orient_multivectors(X,A):
    scalar = mv.sign((X*A)(0))
    return X*scalar

def get_orient_array(X,X_ref):
    return mv.sign((X*X_ref)(0))

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

def print_metrics(R_est,R,T_est,T,m_points=-1,sigma=-1):
    t = -eo|T*2
    t_est = -eo|T_est*2
    costheta = (R_est*~R)(0)
    if abs(costheta) > 1:
        # print("Cos Angle Error:",costheta)
        costheta = 1
    ang_error = np.arccos(costheta)/np.pi*360 # gets two times theta
    if ang_error > 180:
        ang_error = 360 - ang_error
    if(sigma >= 0):
        print("Sigma:", sigma)

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
    basis,rec_basis = get_cga_basis(grades)
    P_lst,lambda_P = eigen_decomp(get_func(p),basis,rec_basis)
    for i in range(len(P_lst)):
        P_lst[i] = normalize_mv(P_lst[i])
    
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

def get_array(x_lst):
    array = np.zeros((len(x_lst),vga.size(1)))
    for i in range(len(x_lst)):
        array[i] = np.array(x_lst[i].list(1)[0][:3])
    return array

def mvarray_to_nparray(x):
    return np.array(x.tolist(1)[0])[:,:3]

def transform_numpy_cloud(pcd,R,t):
    pts = np.asarray(pcd.points)
    x = nparray_to_mvarray(pts)
    y = R*x*~R + t
    return mvarray_to_nparray(y)

def get_properties(X):
    d = (-einf|X)^einf
    scalar = 1/((einf|X)*(einf|X))(0)
    l = -(1/2)*scalar*X*einf*X
    rho_sq = (scalar*X*grade_involution(X))(0)
    # if X is a vector then the following holds
    # X = -(eo + l +(1/2)*rho_sq*einf)*(d|eo)
    return ((-d|eo).tolist(1)[0][:3],P_I(l(1)).tolist(1)[0][:3],rho_sq)
    #return (-d|eo,P_I(l),rho_sq)
