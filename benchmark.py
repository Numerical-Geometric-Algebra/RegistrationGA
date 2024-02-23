from cga3d_estimation import *
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm
import matplotlib
import matplotlib.colors as mc
import colorsys
from datetime import datetime

from algorithms import *

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

    x = nparray_to_3dvga_vector_array(pts_with_outliers)
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
    mask = bench_rot > 150
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
        print("Experiment:",exp)
        benchmark_values = benchmark(pts,exp['sigma'],algorithms,exp['rigid_trans'],exp['n_iters'],exp['n_outliers'],exp["max_dist_outlier"])

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

        dct['rigid_trans'] = U
        dct['sigma'] = get_value(sigma,i)
        dct['n_iters'] = get_value(niters,i)
        dct['n_outliers'] = get_value(noutliers,i)
        dct['max_dist_outlier'] = get_value(maxoutlier,i)
        lst[i] = dct
        
    return lst


filled_marker_style = dict(marker='o', linestyle=':', markersize=7,
                           color='darkgrey',
                           markerfacecolor='tab:blue',
                           markerfacecoloralt='lightsteelblue',
                           markeredgecolor='brown')


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def plot_experiments_2(bench,x_axis,title,xlabel_str,algorithms,marker_style,fig_name=None):
    ''' Plots experiments (worst and best in the same plot)'''
    color = cm.rainbow(np.linspace(0, 1, 2*len(algorithms)))
    n = len(algorithms)

    bench_worst,bench_best,bench_mean = bench

    fig,ax1 = plt.subplots(1,1)

    for i in range(n):
        marker_style['markerfacecolor'] = color[i+n]
        marker_style['color'] = color[i+n]
        marker_style['markeredgecolor'] = lighten_color(color[i+n],amount=1.3)
        ax1.plot(x_axis,bench_best.T[i],label=get_algorithm_name(algorithms[i]) + " (best)",fillstyle='full',**marker_style)
        marker_style['markerfacecolor'] = color[i]
        marker_style['color'] = color[i]
        marker_style['markeredgecolor'] = lighten_color(color[i],amount=1.3)
        ax1.plot(x_axis,bench_worst.T[i],label=get_algorithm_name(algorithms[i]) + " (worst)",fillstyle='full',**marker_style)

    ax1.set_xlabel(xlabel_str)
    ax1.set_title(title)
    ax1.legend()

    if fig_name is not None:
        plt.savefig("Plots/" + fig_name + ".pdf")



def plot_experiments(bench,x_axis,title,xlabel_str,algorithms,marker_style,fig_name=None):
    ''' Plots experiments (worst and best in the different plots)'''
    color = cm.rainbow(np.linspace(0, 1, len(algorithms)))
    n = len(algorithms)

    bench_worst,bench_best,bench_mean,bench_std = bench
    
    ratio = [822,583]
    scale = 0.007

    fig = plt.figure(figsize=(ratio[0]*scale, ratio[1]*scale))

    gs = fig.add_gridspec(2, hspace=0)
    ax1,ax2 = gs.subplots(sharex=True, sharey=False)
    fig.suptitle(title)

    for i in range(n):
        marker_style['markerfacecolor'] = color[i]
        marker_style['color'] = color[i]
        marker_style['markeredgecolor'] = lighten_color(color[i],amount=1.3)
        ax1.plot(x_axis,bench_worst.T[i],label=get_algorithm_name(algorithms[i]),fillstyle='full',**marker_style)

    ax1.set_xlabel(xlabel_str, fontsize=14)
    ax1.set_ylabel("Worst",fontsize=12)

    for i in range(n):
        marker_style['markerfacecolor'] = color[i]
        marker_style['color'] = color[i]
        marker_style['markeredgecolor'] = lighten_color(color[i],amount=1.3)
        ax2.plot(x_axis,bench_best.T[i],label=get_algorithm_name(algorithms[i]) + " (best)",fillstyle='full',**marker_style)
    ax2.set_xlabel(xlabel_str, fontsize=14)
    ax2.set_ylabel("Best",fontsize=12)

    if fig_name is None:
        ax1.legend()
    else:
        # Puts the legend outside of the plot (used for putting in the paper)
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.8),
            ncol=2, fancybox=True, shadow=True)
        # Save figure if file name is set
        plt.savefig("Plots/" + fig_name + ".pdf", bbox_inches="tight")

def plot_experiments_3(bench,x_axis,title,xlabel_str,algorithms,marker_style,capsize=10.0,fig_name=None,legend=True):
    ''' Plots experiments (worst and best in the different plots)'''
    color = cm.rainbow(np.linspace(0, 1, len(algorithms)))
    n = len(algorithms)

    bench_worst,bench_best,bench_mean,bench_std = bench
    
    ratio = [822,583]
    scale = 0.007

    fig = plt.figure(figsize=(ratio[0]*scale, ratio[1]*scale))
    # fig.suptitle(title)

    for i in range(n):
        marker_style['markerfacecolor'] = color[i]
        marker_style['color'] = color[i]
        marker_style['markeredgecolor'] = lighten_color(color[i],amount=1.3)
        plt.errorbar(x_axis,bench_mean.T[i],bench_std.T[i],label=get_algorithm_name(algorithms[i]),fillstyle='full',capsize=capsize,**marker_style)

    plt.xlabel(xlabel_str, fontsize=14)
    plt.ylabel(title + "  (Mean)",fontsize=12)

    if legend:
        if fig_name is not None: # Puts the legend outside of the plot (used for putting in the paper)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                ncol=2, fancybox=True, shadow=True)
            # Save figure if file name is set
            plt.savefig("Plots/" + fig_name + ".pdf", bbox_inches="tight")
        else:
            plt.legend()


if __name__ == '__main__':

    show_plot = False

    pcd = o3d.io.read_point_cloud(f"/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper_res2.ply")
    # pcd = o3d.io.read_point_cloud(f"/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper.ply")
    pts = np.asarray(pcd.points)
    # algorithms = [estimate_transformation_4,estimate_transformation_13]
    algorithms = []
    algorithms += [estimate_transformation_16,estimate_transformation_4,estimate_transformation_pasta]
    algorithms += [estimate_transformation_ICP]
    # algorithms += [estimate_transformation_15,estimate_transformation_9]
    
    # # Create dictionary for increasing number of outliers
    # rot_angle = 100
    # trans_scaling = 3
    # niters = 10
    # noutliers = np.arange(0,51,1)
    # experiments = get_outliers_experiments(noutliers,0,rot_angle,trans_scaling,niters,0.1)
    # x_axis = noutliers
    # xlabel_str = "number of outliers"

    # # Create dictionary for increasing the distance of an outlier
    # rot_angle = 100
    # trans_scaling = 3
    # niters = 10
    # max_dist_outlier = np.arange(0.0001,3,0.05)
    # experiments = get_max_dist_outliers_experiments(5,0,rot_angle,trans_scaling,niters,max_dist_outlier)
    # x_axis = max_dist_outlier
    # xlabel_str = "outlier max distance"

    '''Increasing the MAGNITUDE of the translation'''
    # niters = 10
    # rotangle = 100
    # sigmas = 0.01
    # tscaling = [0.001,1,2,3,4,5,6,7,8,9,10,15,25,50,100]
    # tscaling = np.arange(0.000,10.1,0.5)
    # tscaling = np.arange(0.000,1,0.05)
    # tscaling = [0.0001,0.001,0.01,0.1,1,5,10]
    # tscaling = [1,2,3,4,5,6]
    # tscaling = [0.0001,5,10,15,20,25,30,35]
    # n_exps = len(tscaling)
    # experiments =  get_experiments(n_exps,niters,sigmas,tscaling,rotangle,noutliers=0,maxoutlier=0)
    # experiments = get_trans_experiments(0,sigmas,rot_angle,trans_scaling,niters,0)
    # x_axis = trans_scaling
    # xlabel_str = r'Translation Magnitude $\|\mathbf{t}\|$'
    # fig_name_end = "sigma_" + str(sigmas).replace('.','_') + "_varpos"

    ''' Defining a random trajectory'''
    niters = 100
    ###### Rotation angle and translation increments for each experience
    rotangle = 1
    tscaling = 0.01
    n_exps = 20
    sigmas = 0.01
    experiments = get_experiments(n_exps,niters,sigmas,tscaling,rotangle,noutliers=0,maxoutlier=0,trajectory=True)
    x_axis = np.arange(n_exps)
    xlabel_str = "Iteration"
    fig_name_end = "sigma_" + str(sigmas).replace('.','_') + "_trajectory"

    '''Increasing the NOISE of the point clouds'''
    # niters = 100
    # rotangle = None # Set to none for random rotation angle
    # tscaling = 1
    # sigmas = np.arange(0.000,0.031,0.003)
    # sigmas = np.arange(0.000,0.21,0.02)
    # sigmas = np.arange(0.000,0.005,0.00025)
    # sigmas = np.arange(0.000,0.1,0.01)
    # n_exps = len(sigmas)
    # experiments = get_experiments(n_exps,niters,sigmas,tscaling,rotangle,noutliers=0,maxoutlier=0)
    # x_axis = sigmas
    # xlabel_str = r'Noise ($\sigma$)'
    # fig_name_end = "magpos_" + str(tscaling).replace('.','_') + "_varsigma"

    '''DUMMY experiments'''
    # niters = 1
    # rotangle = 10
    # tscaling = 0
    # sigmas = np.zeros(20)
    # x_axis = np.arange(len(sigmas))
    # n_exps = len(sigmas)
    # fig_name_end = "dummy"
    # xlabel_str = "Iterations"
    # experiments = get_experiments(n_exps,niters,sigmas,tscaling,rotangle,noutliers=0,maxoutlier=0,trajectory=True)

    

    bench_rot,bench_pos,bench_posangle = run_experiments(pts,experiments,algorithms)

    
    if show_plot:
        rot_file_name = None
        pos_file_name = None
        posangle_file_name = None
        # plt.style.use('dark_background')
        # plt.style.use('bmh')    
        # print(plt.style.available) # prints available styles
    else:
        now = datetime.now()
        dt_string = now.strftime('_%d_%m_%Y_%H_%M_%S')
        rot_file_name = "rot_angle_" + fig_name_end + dt_string
        pos_file_name = "pos_" + fig_name_end + dt_string
        posangle_file_name = "posangle_" + fig_name_end + dt_string
    
    
    angle_error_str = r"$\arccos(\mathbf{R}^{\dagger}*\widehat{\mathbf{R}})$ ($\degree$)"
    pos_error_str = r"$\|\mathbf{t} - \hat{\mathbf{t}}\|$"
    # posangle_error_str = r"$\arccos(\widehat{\mathbf{t}}\cdot\hat{\mathbf{t}})$ ($\degree$)"

    # Plot mean and standard deviation (removes outliers (180 degree angle error))
    capsize = 7.0
    plot_experiments_3(bench_rot,x_axis,angle_error_str,xlabel_str,algorithms,filled_marker_style,capsize,rot_file_name)
    plot_experiments_3(bench_pos,x_axis,pos_error_str,xlabel_str,algorithms,filled_marker_style,capsize,pos_file_name)
    # plot_experiments_3(bench_posangle,x_axis,posangle_error_str,xlabel_str,algorithms,filled_marker_style,capsize,posangle_file_name)

    if show_plot:
        # plot the worst and best
        plot_experiments(bench_rot,x_axis,angle_error_str,xlabel_str,algorithms,filled_marker_style,rot_file_name)
        plot_experiments(bench_pos,x_axis,pos_error_str,xlabel_str,algorithms,filled_marker_style,pos_file_name)
        # plot_experiments(bench_posangle,x_axis,posangle_error_str,xlabel_str,algorithms,filled_marker_style,posangle_file_name)
        plt.show()




    '''
    Benchmarks:
        Bench1: (Random Trajectory)
            niters = 100
            rotangle = 1
            tscaling = 0.01
            n_exps = 20
            sigmas = 0.01
            x_axis = np.arange(n_exps)
            xlabel_str = "Iteration"
            fig_name_end = "sigma_" + str(sigmas).replace('.','_') + "_trajectory"
            experiments = get_experiments(n_exps,niters,sigmas,tscaling,rotangle,noutliers=0,maxoutlier=0,trajectory=True)


    '''