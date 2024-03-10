from cga3d_estimation import *
import open3d as o3d
import pickle
from plot_benchmarks import plot_benchmarks,plot_histogram
import plot_benchmarks as plt_bench
from datetime import datetime
from algorithms import *
import os

def benchmark_iter(pts,sigma,algorithms,T,R,noutliers,maxoutlier):
    ''' Benchmarks a single iteration for each algorithm '''
    
    nalgorithms = len(algorithms)
    ang_error = np.zeros([nalgorithms])
    pos_error = np.zeros([nalgorithms])
    plane_error = np.zeros([nalgorithms])
    trans_angle_error = np.zeros([nalgorithms])

    # Add random outliers
    outliers = maxoutlier*np.random.rand(noutliers,3)
    pts_with_outliers = np.r_[outliers,pts]
    npoints = pts.shape[0]
    
    t = -2*(eo|T) # Compute the translation vector from T
    x = nparray_to_3dvga_vector_array(pts)
    noise = rdn_gaussian_3dvga_vecarray(0,sigma,npoints)
    y = R*x*~R + t + noise

    noise = rdn_gaussian_3dvga_vecarray(0,sigma,npoints)

    x = nparray_to_3dvga_vector_array(pts_with_outliers) + noise
    for j in range(nalgorithms):
        T_est,R_est,_,_ = algorithms[j](x,y,npoints)
        ang_error[j],pos_error[j],plane_error[j],trans_angle_error[j] = get_rigtr_error_metrics(R,R_est,T,T_est)
    
    return ang_error,pos_error,plane_error,trans_angle_error

def benchmark(pts,sigma,algorithms,U,niters,noutliers,maxoutlier):
    ''' Generates random rotation and translation'''

    nalgorithms = len(algorithms)
    ang_error = np.zeros([niters,nalgorithms])
    pos_error = np.zeros([niters,nalgorithms])
    plane_error = np.zeros([niters,nalgorithms])
    trans_angle_error = np.zeros([niters,nalgorithms])
    
    T,R = decompose_motor(U)
    for i in range(niters):
        ang_error[i],pos_error[i],plane_error[i],trans_angle_error[i] = benchmark_iter(pts,sigma,algorithms,T,R,noutliers,maxoutlier)

    return ang_error.T,pos_error.T,plane_error.T,trans_angle_error.T


def filter_experiments(benchmark_values):
    bench_rot = benchmark_values[0]
    # Ignore all benchmarks with angle bigger than 150
    mask = bench_rot > 170
    bench_values_list = [0]*len(benchmark_values)
    for i in range(len(benchmark_values)):
        bench_values_list[i] = np.ma.masked_where(mask,benchmark_values[i])

    return bench_values_list

def run_experiments(pts,experiments,algorithms):

    benchmark_worst_array = np.zeros([3,len(experiments),len(algorithms)])
    benchmark_best_array = np.zeros([3,len(experiments),len(algorithms)])
    benchmark_mean_array = np.zeros([3,len(experiments),len(algorithms)])
    benchmark_std_array = np.zeros([3,len(experiments),len(algorithms)])

    i = 0
    for exp in experiments:
        print("Experiment",i,"of",len(experiments),":",exp)
        benchmark_values = benchmark(pts,exp['sigma'],algorithms,exp['motor'],exp['n_iters'],exp['n_outliers'],exp["max_dist_outlier"])

        # Saves the worst and the best without filtering results
        for j in range(len(benchmark_worst_array)):
            benchmark_worst_array[j][i] =  benchmark_values[j].max(axis=1)
            benchmark_best_array[j][i] =  benchmark_values[j].min(axis=1)

        benchmark_values = filter_experiments(benchmark_values)

        for j in range(len(benchmark_worst_array)):
            benchmark_std_array[j][i] =  benchmark_values[j].std(axis=1)
            benchmark_mean_array[j][i] =  benchmark_values[j].mean(axis=1)
            
        i += 1

    bench_rot = [benchmark_worst_array[0],benchmark_best_array[0],benchmark_mean_array[0],benchmark_std_array[0]]
    bench_pos = [benchmark_worst_array[1],benchmark_best_array[1],benchmark_mean_array[1],benchmark_std_array[1]]
    bench_posangle = [benchmark_worst_array[2],benchmark_best_array[2],benchmark_mean_array[2],benchmark_std_array[2]]
    
    return (bench_rot,bench_pos,bench_posangle)

def run_single_experiment(pts,exp,algorithms):
    benchmark_values = benchmark(pts,exp['sigma'],algorithms,exp['motor'],exp['n_iters'],exp['n_outliers'],exp["max_dist_outlier"])
    return benchmark_values


def get_value(element,i):
    ''' Checks if it is scalar, if not consider that is an array'''
    if np.isscalar(element):
        return element
    else:
        return element[i]

def get_experiments(n_exps,niters,sigma,tscaling,rotangle,noutliers,maxoutlier,trajectory=False):
    U = 1
    lst = [0]*n_exps
    for i in range(n_exps):
        dct = {}
        # Generate the random rigid transformations
        if trajectory:
            R,T = gen_pseudordn_rigtr(rotangle,tscaling)
            U *= T*R
        else:
            if rotangle is not None:
                rotanglei = get_value(rotangle,i)
            else:
                rotanglei = (np.random.rand() - 0.5)*360

            tscalingi = get_value(tscaling,i)

            R,T = gen_pseudordn_rigtr(rotanglei,tscalingi)
            U = T*R

        dct['motor'] = U
        dct['sigma'] = get_value(sigma,i)
        dct['n_iters'] = get_value(niters,i)
        dct['n_outliers'] = get_value(noutliers,i)
        dct['max_dist_outlier'] = get_value(maxoutlier,i)
        lst[i] = dct
        
    return lst


def multiple_experiments_to_dct(benchmark_values,algorithms,x_axis,xlabel_str,fig_name_end,filename):
    bench_rot,bench_pos,bench_posangle = benchmark_values
    now = datetime.now()
    dt_string = now.strftime('_%d_%m_%Y_%H_%M_%S')
    fig_name_end += get_name_filepath(filename)

    # Save the needed data into a dictionary
    dct = {}
    dct["bench_rot"] = bench_rot
    dct["bench_posangle"] = bench_posangle
    dct["bench_pos"] = bench_pos
    dct["x_axis"] = x_axis
    dct["algorithms"] = algorithms
    dct["fig_name_end"] = fig_name_end
    dct["xlabel_str"] = xlabel_str
    dct["dt_string"] = dt_string
    dct["filename"] = filename # Important!! Do not lose the filename!!


    return dct,fig_name_end+dt_string

def save_multiple_experiments(benchmark_values,algorithms,x_axis,xlabel_str,fig_name_end,filename):
    
    dct,fig_name_end = multiple_experiments_to_dct(benchmark_values,algorithms,x_axis,xlabel_str,fig_name_end,filename)

    filename = "Benchmarks/benchmark_" + fig_name_end + ".pickle"
    with open(filename,"wb") as f:
        pickle.dump(dct,f)

    return filename

def single_experiment_to_dct(benchmark_values,algorithms,fig_name_end,filename,sigma):
    bench_rot,bench_pos,_,_ = benchmark_values
    now = datetime.now()
    dt_string = now.strftime('_%d_%m_%Y_%H_%M_%S')
    fig_name_end += get_name_filepath(filename)
    dct = {}
    dct["sigma"] = sigma
    dct["bench_rot"] = bench_rot
    dct["bench_pos"] = bench_pos
    dct["algorithms"] = algorithms
    dct["fig_name_end"] = fig_name_end
    dct["dt_string"] = dt_string
    dct["filename"] = filename # Important!! Do not lose the filename!!

    return dct,fig_name_end + dt_string

def save_single_experiment(benchmark_values,algorithms,fig_name_end,filename,sigma):
    dct,fig_name_end = single_experiment_to_dct(benchmark_values,algorithms,fig_name_end,filename,sigma)

    filename = "Benchmarks/benchmark_single_" + fig_name_end + ".pickle"
    with open(filename,"wb") as f:
        pickle.dump(dct,f)
    
    return filename


def get_name_filepath(filepath):
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]

if __name__ == '__main__':

    show_plot = True # Set to true to show the plots imediatly 
    save_data = False # Set to true to save a .pickle file with the data, saves in the Benchmark folder
    save_plot = False # Set to true to save the plot (needs show_plot = True), saves in the Plots folder
    
    # uncomment/comment if prefer dark background
    plt_bench.set_dark_background()

    # Chose the point cloud that is being benchmarked
    # filename = f"/home/francisco/Code/Stanford Dataset/dragon_fillers/dragonMouth5_0.ply"
    # filename = f"/home/francisco/Code/Stanford Dataset/Armadillo_scans/ArmadilloBack_0.ply"
    filename = f"/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper_res2.ply" 

    pcd = o3d.io.read_point_cloud(filename)
    pts = np.asarray(pcd.points)

    algorithms = []
    algorithms += [estimate_transformation_VGA] # Baseline, works well most of the time
    algorithms += [estimate_transformation_CGA] # Propose algorithm
    algorithms += [estimate_transformation_ICP] # Iterative Closest points
    algorithms += [estimate_transformation_pasta] # PASTA
    

    ''' Defining a random trajectory'''
    # trajectory = True
    # niters = 1
    # ###### Rotation angle and translation increments for each experience
    # rotangle = 10
    # tscaling = 0.01
    # n_exps = 1
    # sigmas = 0.01
    # x_axis = np.arange(n_exps)
    # xlabel_str = "Iteration"
    # fig_name_end = "sigma_" + str(sigmas).replace('.','_') + "_trajectory_"

    '''DUMMY experiments'''
    # niters = 1
    # n_exps = 100
    # rotangle = None
    # tscaling = 1
    # sigmas = 0.025
    # x_axis = np.arange(n_exps)
    # fig_name_end = "dummy"
    # xlabel_str = "Iterations"
    # trajectory = False

    '''Increasing the NOISE of the point clouds'''
    trajectory = False # The rotation and translations are not cummulative
    niters = 10 # Number of iterations for each experiment
    tscaling = 1 # The magnitude of the translation
    rotangle = None # Set to none for random rotation angle
    sigmas = np.arange(0.000,0.0300001,0.001) # The different levels of noise to test against
    n_exps = len(sigmas) # Number of experiments

    x_axis = sigmas
    xlabel_str = r'Noise ($\sigma$)'
    fig_name_end = "magpos_" + str(tscaling).replace('.','_') + "_varsigma_"

    experiments = get_experiments(n_exps,niters,sigmas,tscaling,rotangle,noutliers=0,maxoutlier=0,trajectory=trajectory)
    benchmark_values = run_experiments(pts,experiments,algorithms)
    
    if save_data:
        filename_bench = save_multiple_experiments(benchmark_values,algorithms,x_axis,xlabel_str,fig_name_end,filename)
    if show_plot:
        dct,_ = multiple_experiments_to_dct(benchmark_values,algorithms,x_axis,xlabel_str,fig_name_end,filename)
        plot_benchmarks(dct,save_plot)


    '''Generate data for multiple histogram plots '''
    # sigmas_lst = [0,0.001,0.005,0.01,0.02,0.03] # Chose multiple levels of noise to compare
    # niters = 10
    # for i in range(len(sigmas_lst)):
    #     print("Experiment",i,"for histogram")
    #     sigmas = sigmas_lst[i]
    #     print("sigma =",sigmas)
    #     fig_name_end = "sigma_" + str(sigmas).replace('.','_') + "_"
        
    #     experiments = get_experiments(n_exps=1,niters=niters,sigma=sigmas,tscaling=1,rotangle=None,noutliers=0,maxoutlier=0,trajectory=False)
        
    #     benchmark_values = run_single_experiment(pts,experiments[0],algorithms)
    #     if save_data:
    #         filename_pickle = save_single_experiment(benchmark_values,algorithms,fig_name_end,filename,sigmas)
    #     if show_plot:
    #         dct,_ = single_experiment_to_dct(benchmark_values,algorithms,fig_name_end,filename,sigmas)
    #         plot_histogram(dct,save_plot=save_plot)
        
    if show_plot:
        plt_bench.show_plots()

