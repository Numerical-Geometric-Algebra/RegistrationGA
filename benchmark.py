from cga3d_estimation import *
import open3d as o3d
import matplotlib.pyplot as plt
from algorithms import *

def benchmark(pts,sigma,algorithms,rotangle,tscaling,niters,noutliners,maxoutlier):
    nalgorithms = len(algorithms)

    ang_error = np.zeros([nalgorithms,niters])
    pos_error = np.zeros([nalgorithms,niters])
    plane_error = np.zeros([nalgorithms,niters])
    
    for i in range(niters):
        T,R = gen_pseudordn_rigtr(rotangle,tscaling)
        t = -eo|T*2 # get the translation vector

        # Add random outliers
        outliers = maxoutlier*np.random.rand(noutliners,3)
        pts_with_outliers = np.r_[outliers,pts]
        npoints = pts.shape[0]
        
        x = nparray_to_3dvga_vector_array(pts)
        noise = rdn_gaussian_3dvga_vecarray(0,sigma,npoints)
        y = R*x*~R + t + noise
        x = nparray_to_3dvga_vector_array(pts_with_outliers)
        for j in range(nalgorithms):
            T_est,R_est,_,_ = algorithms[j](x,y,npoints)
            ang_error[j][i],pos_error[j][i],plane_error[j][i] = get_rigtr_error_metrics(R,R_est,T,T_est)

    return ang_error,pos_error,plane_error

def run_experiments(pts,experiments,algorithms):
    benchmark_angle_worst_array = np.zeros([len(experiments),len(algorithms)])
    benchmark_pos_worst_array = np.zeros([len(experiments),len(algorithms)])

    benchmark_angle_best_array = np.zeros([len(experiments),len(algorithms)])
    benchmark_pos_best_array = np.zeros([len(experiments),len(algorithms)])

    benchmark_angle_mean_array = np.zeros([len(experiments),len(algorithms)])
    benchmark_pos_mean_array = np.zeros([len(experiments),len(algorithms)])

    i = 0
    for exp in experiments:
        print("Experiment:",exp)
        benchmark_values = benchmark(pts,exp['sigma'],algorithms,exp['rot_angle'],exp['trans_scaling'],exp['n_iters'],exp['n_outliers'],exp["max_dist_outlier"])

        benchmark_angle_worst_array[i] =  benchmark_values[0].max(axis=1)
        benchmark_angle_best_array[i] =  benchmark_values[0].min(axis=1)
        benchmark_angle_mean_array[i] =  benchmark_values[0].sum(axis=1)/len(benchmark_values[0])

        benchmark_pos_worst_array[i] =  benchmark_values[1].max(axis=1)
        benchmark_pos_best_array[i] =  benchmark_values[1].min(axis=1)
        benchmark_pos_mean_array[i] =  benchmark_values[1].sum(axis=1)/len(benchmark_values[1])
        i += 1
    bench_rot = [benchmark_angle_worst_array,benchmark_angle_best_array,benchmark_angle_mean_array]
    bench_pos = [benchmark_pos_worst_array,benchmark_pos_best_array,benchmark_pos_mean_array]
    return (bench_rot,bench_pos)


def get_outliers_experiments(n_outliers,sigma,rot_angle,trans_scaling,niters,max_dist_outlier):
    lst = [0]*len(n_outliers)
    for i in range(len(n_outliers)):
        dct = {}
        dct['sigma'] = sigma
        dct['rot_angle'] = rot_angle
        dct['trans_scaling'] = trans_scaling
        dct['n_iters'] = niters
        dct['max_dist_outlier'] = max_dist_outlier
        dct['n_outliers'] = n_outliers[i]
        lst[i] = dct

    return lst

def get_max_dist_outliers_experiments(n_outliers,sigma,rot_angle,trans_scaling,niters,max_dist_outlier):
    lst = [0]*len(max_dist_outlier)
    for i in range(len(max_dist_outlier)):
        dct = {}
        dct['sigma'] = sigma
        dct['rot_angle'] = rot_angle
        dct['trans_scaling'] = trans_scaling
        dct['n_iters'] = niters
        dct['n_outliers'] = n_outliers
        dct['max_dist_outlier'] = max_dist_outlier[i]
        lst[i] = dct

    return lst

def get_noisy_experiments(n_outliers,sigma,rot_angle,trans_scaling,niters,max_dist_outlier):
    lst = [0]*len(sigma)
    for i in range(len(sigma)):
        dct = {}
        dct['sigma'] = sigma[i]
        dct['rot_angle'] = rot_angle
        dct['trans_scaling'] = trans_scaling
        dct['n_iters'] = niters
        dct['n_outliers'] = n_outliers
        dct['max_dist_outlier'] = max_dist_outlier
        lst[i] = dct

    return lst

def get_single_experiment(n_outliers,sigma,rot_angle,trans_scaling,niters,max_dist_outlier):
    dct = {}
    dct['sigma'] = sigma
    dct['rot_angle'] = rot_angle
    dct['trans_scaling'] = trans_scaling
    dct['n_iters'] = niters
    dct['n_outliers'] = n_outliers
    dct['max_dist_outlier'] = max_dist_outlier

    return [dct]

def plot_experiments(bench_rot,bench_pos,x_axis):
    bench_rot_worst,bench_rot_best,bench_rot_mean = bench_rot
    bench_pos_worst,bench_pos_best,bench_pos_mean = bench_pos

    # plt.style.use('dark_background')
    fig, (ax1,ax2) = plt.subplots(1, 2)

    ax1.plot(x_axis,bench_rot_worst)
    for i in range(len(ax1.lines)):
        ax1.lines[i].set_label(get_algorithm_name(algorithms[i]))
    ax1.legend()
    ax1.set_title("Angle Error (Worst)")

    ax2.plot(x_axis,bench_rot_best)
    for i in range(len(ax2.lines)):
        ax2.lines[i].set_label(get_algorithm_name(algorithms[i]))
    ax2.legend()
    ax2.set_title("Angle Error (Best)")
    

    fig, (ax1,ax2) = plt.subplots(1, 2) 
    ax1.plot(x_axis,bench_pos_worst)
    for i in range(len(ax1.lines)):
        ax1.lines[i].set_label(get_algorithm_name(algorithms[i]))
    ax1.legend()
    ax1.set_title("Position Error (Worst)")

    ax2.plot(x_axis,bench_pos_best)
    for i in range(len(ax2.lines)):
        ax2.lines[i].set_label(get_algorithm_name(algorithms[i]))
    ax2.legend()
    ax2.set_title("Position Error (Best)")
    plt.show()


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud(f"/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper_res2.ply")
    pts = np.asarray(pcd.points)

    trans_scaling = 0
    niters = 1
    # algorithms = algorithms_list
    # algorithms = ROT_algs_list
    # estimate_rotation_6
    # algorithms = [estimate_rotation_1,estimate_transformation_8,estimate_transformation_0,estimate_transformation_9]
    # algorithms = [estimate_transformation_9]
    algorithms = [estimate_transformation_0,estimate_transformation_1,estimate_transformation_2,estimate_transformation_3,estimate_transformation_4,estimate_transformation_5,estimate_transformation_8,estimate_transformation_9]
    algorithms += [estimate_rotation_0,estimate_rotation_1,estimate_rotation_2,estimate_rotation_3,estimate_rotation_4,estimate_rotation_5,estimate_rotation_6]
    '''
    Usage of get experiments functions:
        exps = get_exp(n_outliers,sigma,rot_angle,trans_scaling,niters,max_dist_outlier)
    '''

    # Create dictionary for increasing number of outliers
    noutliers = np.arange(0,51,1)
    experiments = get_outliers_experiments(noutliers,0,100,trans_scaling,niters,0.1)
    x_axis = noutliers

    # Create dictionary for increasing the distance of an outlier
    max_dist_outlier = np.arange(0.0001,3,0.05)
    experiments = get_max_dist_outliers_experiments(5,0,100,trans_scaling,niters,max_dist_outlier)
    x_axis = max_dist_outlier
    
    # Create dictionary for increasing the noise of the point clouds
    sigmas = np.arange(0.000,0.051,0.002)
    experiments = get_noisy_experiments(0,sigmas,100,trans_scaling,niters,0.1)
    x_axis = sigmas

    # Dummy experiments
    sigmas = np.zeros(10)
    x_axis = np.arange(len(sigmas))
    experiments = get_noisy_experiments(0,sigmas,100,trans_scaling,niters,0)
    # experiments = get_single_experiment(0,0,100,0,niters,0) + get_single_experiment(0,0.05,100,0,niters,0)
    # experiments = get_single_experiment(0,0,100,0,niters,0)
    # x_axis = [0,1]

    bench_rot,bench_pos = run_experiments(pts,experiments,algorithms)

    plot_experiments(bench_rot,bench_pos,x_axis)