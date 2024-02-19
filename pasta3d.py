import numpy as np
from scipy.spatial import ConvexHull


def shape_moments_3d(tris):
    # tris : resulting shape = (nb_simplices, 4, 3), for each simplex you have a 4 by 3 matrix containing the vertex + the next vertex + the next vertex + the hull centroid

    # Now building V
    a = tris[:,1,:] - tris[:,0,:]  # v1 - v0  # shape = (nb_simplices, 3)
    b = tris[:,2,:] - tris[:,0,:]  # v2 - v0  # shape = (nb_simplices, 3)
    c = tris[:,3,:] - tris[:,0,:]  # v3 - v0  # shape = (nb_simplices, 3)

    V = np.stack((a,b,c), axis=2) # [v1-v0  v2-v0  v3-v0]  # shape = (nb_simplices, 3, 3)

    # Compute the First moment (cent : centroid)
    volumes = np.absolute(np.linalg.det(V))  # shape = (nb_simplices,)
    centroid_simplices = np.mean(tris, axis=1)  # shape = (nb_simplices, 3)
    cent = np.average(centroid_simplices, axis=0, weights=volumes)  # first_moment = sum of vertices average of each simplex, weighted by volume

    # Compute the covariance...
    K = np.array([
                [1,   1/4,  1/4,  1/4],
                [1/4,  1/10,  1/20,  1/20],
                [1/4,  1/20,  1/10,  1/20],
                [1/4,  1/20,  1/20,  1/10]
                ])

    d = tris[:,0,:] - cent[None,:]  # v0 - q
    V_tilde = np.stack((d,a,b,c), axis=2)  # shape = (nb_simplices, 3, 4)
    temp = V_tilde @ K  # shape = (nb_simplices, 3, 4)
    covs = np.einsum("ijl, ikl -> ijk", temp, V_tilde)  # shape = (nb_simplices, 3, 3)
    cov = np.average(covs, axis=0, weights=volumes)  # shape = (3, 3)

    return cent, cov


def choose_direction_3d(mu, eigvec, cloud, dir_mode='max'):
    "pick the direction in which the perturbation is maximum and hope it is preserved under the transformation"
                            
    if dir_mode == 'max':
        dots_1 = np.einsum("ij,j->i", cloud - mu[None,:], eigvec[:,2])
        dots_2 = np.einsum("ij,j->i", cloud - mu[None,:], eigvec[:,1])
        M1 = np.max(dots_1)
        m1 = np.min(dots_1)

        M2 = np.max(dots_2)
        m2 = np.min(dots_2)
        if abs(M1) < abs(m1):
            eigvec[:,2] = - eigvec[:,2]
        
        if abs(M2) < abs(m2):
            eigvec[:,1] = - eigvec[:,1]

        if np.linalg.det(eigvec) < 0:
            eigvec[:,0] = - eigvec[:,0]

        return eigvec

    if dir_mode == 'mean':
        dots_1 = np.einsum("ij,j->i", cloud - mu[None,:], eigvec[:,2])
        dots_2 = np.einsum("ij,j->i", cloud - mu[None,:], eigvec[:,1])

        if np.mean(dots_1) < 0:
            eigvec[:,2] = - eigvec[:,2]
        
        if np.mean(dots_2) < 0:
            eigvec[:,1] = - eigvec[:,1]

        if np.linalg.det(eigvec) < 0:
            eigvec[:,0] = - eigvec[:,0]

        return eigvec


def cloud_moments_3d(cloud, dir_mode='max'):
    hull = ConvexHull(cloud)
    verts = cloud[hull.vertices,:]
    inner = np.mean(verts, axis=0)     
    simplices = cloud[hull.simplices, :]
    nb_simplices = np.shape(simplices)[0]
    inner_temp = np.broadcast_to(inner, shape = (nb_simplices, 1, 3))
    # add the inner vertex
    tris = np.concatenate((simplices, inner_temp), axis = 1) # shape = (nb_simplices, 4, 3)

    mu, sigma = shape_moments_3d(tris)
    eigval, eigvec = np.linalg.eigh(sigma)

    eigvec = choose_direction_3d(mu, eigvec, cloud, dir_mode=dir_mode)

    return {
        'mu': mu,
        'sigma': sigma,
        'eigval': eigval,
        'eigvec': eigvec,
        'hull': hull,
        'vert': verts,
        'tris': tris,
        'cloud': cloud
    }


def pasta3d_rototranslation(cloud_A, cloud_B, dir_mode='max'):
    '''Note that the transformation  returned is for a sensor pose. This means it returns (R, p) such that: cloud_A = R @ cloud_B + p'''
    A, B = [cloud_moments_3d(cloud, dir_mode) for cloud in [cloud_A, cloud_B]]

    R = A['eigvec'] @ B['eigvec'].T
    p = A['mu'] - R @ B['mu']

    return R, p

