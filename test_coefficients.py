from cga3d_estimation import *
import open3d as o3d
from algorithms import *
from benchmark import benchmark

np.set_printoptions(linewidth=np.inf)

nsamples = 10
npoints = 200
eig_grades = [1]

pcd = o3d.io.read_point_cloud(f"/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper_res2.ply")
pts = np.asarray(pcd.points)
npoints = pts.shape[0]

x = nparray_to_3dvga_vector_array(pts)



# algorithms = [estimate_transformation_10,estimate_transformation_11]

# for i in range(nsamples):
#     benchmark(pts,sigma=0,algorithms=algorithms,rotangle=0,tscaling=10,niters=1,noutliers=0,maxoutlier=0)

for i in range(nsamples):
    # x = pyga.rdn_mvarray_from_basis(cga3d,[e1,e2,e3],size=1)
    T = rdn_3dcga_translator(scale=0.1)
    t = -2*(eo|T)
    R = rdn_3dvga_rotor()

    # Do not define versors as python scalars!! The operation ~ on python scalars are binary inversions!!
    # R = cga3d.multivector([1],grades=0)
    
    y = R*x*~R + t

    p = eo + x + (1/2)*(x*x)(0)*einf
    q = eo + y + (1/2)*(y*y)(0)*einf

    P_ref,Q_ref = compute_references(p,q)

    P_lst,lambda_P = get_3dcga_eigmvs(p,grades=eig_grades)
    Q_lst,lambda_Q = get_3dcga_eigmvs(q,grades=eig_grades)

    P = mv.concat(P_lst)
    Q = mv.concat(Q_lst)

    # P /= (P*P_ref)(0)
    # Q /= (Q*Q_ref)(0)

    Q *= (Q*Q_ref)(0)/(P*P_ref)(0)

    # print((T*P*~T)*Q)
    print(pyga.numpy_max(T*R*P*~R*~T-Q))
    

    # print(pyga.numpy_max((T*p*~T) - q))
    # print(pyga.numpy_max(y-x - t))
    # print(t)



    # # T,R = gen_pseudordn_rigtr(0,10)

    
    # print("t=",t)
    # y = x + t

    # # T_est,R_est,S_lst,Q_lst = estimate_transformation_10(x,y,npoints)
    # # print("Translation error (10):",pyga.numpy_max(T-T_est))
    # T_est,R_est,S_lst,Q_lst = estimate_transformation_11(x,y,npoints)
    # print("Translation error (11):",pyga.numpy_max(T-T_est))

#     print()


