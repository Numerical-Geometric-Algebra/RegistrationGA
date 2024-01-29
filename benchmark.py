from eig_estimation import *
import open3d as o3d
import matplotlib.pyplot as plt


# def get_metrics(R,R_est,T,T_est):
#     t = -eo|T*2
#     t_est = -eo|T_est*2
#     costheta = (R_est*~R)(0)
#     if abs(costheta) > 1:
#         costheta = 1
#     ang_error = np.arccos(costheta)/np.pi*360 # gets two times theta
#     if ang_error > 180:
#         ang_error = 360 - ang_error

#     # Compute the magnitude of tranlation error
#     t_error = mag_mv(t - t_est)

#     # Compute the error between the planes of rotation
#     cosphi = (normalize_mv(R(2))*~normalize_mv(R_est(2)))(0)
#     if abs(cosphi) > 1:
#         cosphi = 1
#     phi = np.arccos(cosphi)/np.pi*180
#     if(phi > 180):
#         phi = 360 - phi

#     return ang_error,t_error,phi

def benchmark_with_outliers(pts,R,T,sigma,algorithms,niters,noutliners,maxoutlier):
    # get the translation vector
    t = -eo|T*2
    nalgorithms = len(algorithms)

    ang_error = np.zeros([nalgorithms,niters])
    pos_error = np.zeros([nalgorithms,niters])
    plane_error = np.zeros([nalgorithms,niters])
    
    for i in range(niters):
        rot_angle = np.random.rand()*360
        T,R = gen_pseudordn_rbm(rot_angle,10)
        t = -eo|T*2
        outliers = maxoutlier*np.random.rand(noutliners,3)
        pts_with_outliers = np.r_[outliers,pts]
        npoints = pts.shape[0]
        
        x = nparray_to_vga_vecarray(pts)
        noise = rdn_gaussian_vga_vecarray(0,sigma,npoints)
        y = R*x*~R + t + noise
        x = nparray_to_vga_vecarray(pts_with_outliers)
        for j in range(nalgorithms):
            T_est,R_est,_,_ = algorithms[j](x,y,npoints)
            ang_error[j][i],pos_error[j][i],plane_error[j][i] = get_metrics(R,R_est,T,T_est)

    return ang_error,pos_error,plane_error

def benchmark_algorithms(x,R,T,npoints,sigma,algorithms,nalgorithms,niters):
    
    # get the translation vector
    t = -eo|T*2

    ang_error = np.zeros([nalgorithms,niters])
    pos_error = np.zeros([nalgorithms,niters])
    plane_error = np.zeros([nalgorithms,niters])
    
    for i in range(niters):
        noise = rdn_gaussian_vga_vecarray(0,sigma,npoints)
        y = R*x*~R + t + noise
        for j in range(nalgorithms):
            T_est,R_est,_,_ = algorithms[j](x,y,npoints)
            ang_error[j][i],pos_error[j][i],plane_error[j][i] = get_metrics(R,R_est,T,T_est)

    return ang_error,pos_error,plane_error


if __name__ == '__main__':
    
    # pcd = o3d.io.read_point_cloud(f"/home/francisco/Code/Stanford Dataset/bunny/data/bun000.ply")
    pcd = o3d.io.read_point_cloud(f"/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper_res2.ply")
    pts = np.asarray(pcd.points)
    # algorithms = [estimate_transformation_0,estimate_transformation_1,estimate_transformation_4]
    algorithms = [estimate_transformation_1,estimate_transformation_6,estimate_transformation_4]
    alg_name = ["CGA CeOM","VGA 2F_CoSign","VGA CoSign_2"]
    # algorithms = [estimate_transformation_7]
    # alg_name = ["CGA BruteForce"]
    niters = 1
    sigma = 0
    bench_sigma = True
    bench_outliers = False

    if bench_sigma:

        npoints = pts.shape[0]

        x = nparray_to_vga_vecarray(pts)
        T,R = gen_pseudordn_rbm(100,10)
        
        sigmas = np.arange(0.000,0.041,0.002)
        # sigmas = np.arange(0.000,0.003,0.001)
        
        benchmark_angle_worst_array = np.zeros([len(sigmas),len(algorithms)])
        benchmark_pos_worst_array = np.zeros([len(sigmas),len(algorithms)])

        benchmark_angle_best_array = np.zeros([len(sigmas),len(algorithms)])
        benchmark_pos_best_array = np.zeros([len(sigmas),len(algorithms)])

        benchmark_angle_mean_array = np.zeros([len(sigmas),len(algorithms)])
        benchmark_pos_mean_array = np.zeros([len(sigmas),len(algorithms)])

        i = 0
        for sigma in sigmas:
            T,R = gen_pseudordn_rbm(100,10)
            print("Sigma:",sigma)
            benchmark_values = benchmark_algorithms(x,R,T,npoints,sigma,algorithms,len(algorithms),niters)
            
            benchmark_angle_worst_array[i] =  benchmark_values[0].max(axis=1)
            benchmark_angle_best_array[i] =  benchmark_values[0].min(axis=1)
            benchmark_angle_mean_array[i] =  benchmark_values[0].sum(axis=1)/len(benchmark_values[0])

            benchmark_pos_worst_array[i] =  benchmark_values[1].max(axis=1)
            benchmark_pos_best_array[i] =  benchmark_values[1].min(axis=1)
            benchmark_pos_mean_array[i] =  benchmark_values[1].sum(axis=1)/len(benchmark_values[1])
            i += 1
        
        x_axis = sigmas
    


    elif bench_outliers:
        noutliers = np.arange(0,51,1)
        benchmark_angle_worst_array = np.zeros([len(noutliers),len(algorithms)])
        benchmark_pos_worst_array = np.zeros([len(noutliers),len(algorithms)])

        benchmark_angle_best_array = np.zeros([len(noutliers),len(algorithms)])
        benchmark_pos_best_array = np.zeros([len(noutliers),len(algorithms)])

        benchmark_angle_mean_array = np.zeros([len(noutliers),len(algorithms)])
        benchmark_pos_mean_array = np.zeros([len(noutliers),len(algorithms)])

        maxoutlier = 1
        i = 0
        for noutlier in noutliers:
            T,R = gen_pseudordn_rbm(100,10)
            print("N Outliers:",noutlier)
            benchmark_values = benchmark_with_outliers(pts,R,T,sigma,algorithms,niters,noutlier,maxoutlier)
            
            benchmark_angle_worst_array[i] =  benchmark_values[0].max(axis=1)
            benchmark_angle_best_array[i] =  benchmark_values[0].min(axis=1)
            benchmark_angle_mean_array[i] =  benchmark_values[0].sum(axis=1)/len(benchmark_values[0])

            benchmark_pos_worst_array[i] =  benchmark_values[1].max(axis=1)
            benchmark_pos_best_array[i] =  benchmark_values[1].min(axis=1)
            benchmark_pos_mean_array[i] =  benchmark_values[1].sum(axis=1)/len(benchmark_values[1])
            i += 1

        x_axis = noutliers

    else:
        maxoutliers = np.arange(0,2,0.05)
        benchmark_angle_worst_array = np.zeros([len(maxoutliers),len(algorithms)])
        benchmark_pos_worst_array = np.zeros([len(maxoutliers),len(algorithms)])

        benchmark_angle_best_array = np.zeros([len(maxoutliers),len(algorithms)])
        benchmark_pos_best_array = np.zeros([len(maxoutliers),len(algorithms)])

        benchmark_angle_mean_array = np.zeros([len(maxoutliers),len(algorithms)])
        benchmark_pos_mean_array = np.zeros([len(maxoutliers),len(algorithms)])

        noutlier = 1 # Only a single outlier
        i = 0
        for maxoutlier in maxoutliers:
            T,R = gen_pseudordn_rbm(100,10)
            print("Max dist outlier:",maxoutlier)
            benchmark_values = benchmark_with_outliers(pts,R,T,sigma,algorithms,niters,noutlier,maxoutlier)
            
            benchmark_angle_worst_array[i] =  benchmark_values[0].max(axis=1)
            benchmark_angle_best_array[i] =  benchmark_values[0].min(axis=1)
            benchmark_angle_mean_array[i] =  benchmark_values[0].sum(axis=1)/len(benchmark_values[0])

            benchmark_pos_worst_array[i] =  benchmark_values[1].max(axis=1)
            benchmark_pos_best_array[i] =  benchmark_values[1].min(axis=1)
            benchmark_pos_mean_array[i] =  benchmark_values[1].sum(axis=1)/len(benchmark_values[1])
            i += 1

        x_axis = maxoutliers


    plt.style.use('dark_background')
    fig, (ax1,ax2) = plt.subplots(1, 2) 
    ax1.plot(x_axis,benchmark_angle_worst_array)
    for i in range(len(ax1.lines)):
        ax1.lines[i].set_label(alg_name[i])
    ax1.legend()
    ax1.set_title("Angle Error (Worst)")

    ax2.plot(x_axis,benchmark_pos_worst_array)
    for i in range(len(ax2.lines)):
        ax2.lines[i].set_label(alg_name[i])
    ax2.legend()
    ax2.set_title("Position Error (Worst)")
    

    fig, (ax1,ax2) = plt.subplots(1, 2) 
    ax1.plot(x_axis,benchmark_angle_best_array)
    for i in range(len(ax1.lines)):
        ax1.lines[i].set_label(alg_name[i])
    ax1.legend()
    ax1.set_title("Angle Error (Best)")

    ax2.plot(x_axis,benchmark_pos_best_array)
    for i in range(len(ax2.lines)):
        ax2.lines[i].set_label(alg_name[i])
    ax2.legend()
    ax2.set_title("Position Error (Best)")
    plt.show()



'''
Algorithms:
    - Rotation estimation with eigenvalues of 
    F(X) = sum_i (x_i-x_bar)X(x_i-x_bar)
    G(X) = sum_i (y_i-y_bar)X(y_i-y_bar)

    - Rotations estimation with eigenvalues of
    F(X) = sum_i up(x_i-x_bar)X up(x_i-x_bar)
    G(X) = sum_i up(y_i-y_bar)X up(y_i-y_bar)

    where up(.) converts the elements to CGA.


    - Algorithm 5 can estimate the rotation with only three points. 
      Two are not enough for the algorithm to converge to the solution.
    
    - Solving the problem using only two sets of points and determining the optimal 

'''


'''
To write code that is independent of the quantities being change need to define:
 - Experiment srtucture:
    - Each field of this structure would have:
        - Number of outliers
        - The maximum distance of an outlier
        - The added gaussian noise
        - Or more...
    for noutlier,maxoutlier,sigma in experiments:
        benchmark_values = benchmark_with_outliers(pts,R,T,sigma,algorithms,niters,noutlier,maxoutlier)

The question is how would I visualize the data if more than one field varies in the loop? 
Experiment 'i' in the x axis????

'''