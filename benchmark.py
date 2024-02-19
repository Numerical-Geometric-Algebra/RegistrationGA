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

def benchmark(pts,sigma,algorithms,rotangle,tscaling,niters,noutliers,maxoutlier):
    nalgorithms = len(algorithms)

    ang_error = np.zeros([nalgorithms,niters])
    pos_error = np.zeros([nalgorithms,niters])
    plane_error = np.zeros([nalgorithms,niters])
    trans_angle_error = np.zeros([nalgorithms,niters])
    
    for i in range(niters):
        T,R = gen_pseudordn_rigtr(rotangle,tscaling)
        t = -eo|T*2 # get the translation vector

        # Add random outliers
        outliers = maxoutlier*np.random.rand(noutliers,3)
        pts_with_outliers = np.r_[outliers,pts]
        npoints = pts.shape[0]
        
        x = nparray_to_3dvga_vector_array(pts)
        noise = rdn_gaussian_3dvga_vecarray(0,sigma,npoints)
        y = R*x*~R + t + noise
    
        x = nparray_to_3dvga_vector_array(pts_with_outliers)
        for j in range(nalgorithms):
            T_est,R_est,_,_ = algorithms[j](x,y,npoints)
            ang_error[j][i],pos_error[j][i],plane_error[j][i],trans_angle_error[j][i] = get_rigtr_error_metrics(R,R_est,T,T_est)

    return ang_error,pos_error,plane_error,trans_angle_error

def run_experiments(pts,experiments,algorithms):

    benchmark_worst_array = np.zeros([3,len(experiments),len(algorithms)])
    benchmark_best_array = np.zeros([3,len(experiments),len(algorithms)])
    benchmark_mean_array = np.zeros([3,len(experiments),len(algorithms)])

    i = 0
    for exp in experiments:
        print("Experiment:",exp)
        benchmark_values = benchmark(pts,exp['sigma'],algorithms,exp['rot_angle'],exp['trans_scaling'],exp['n_iters'],exp['n_outliers'],exp["max_dist_outlier"])

        for j in range(len(benchmark_worst_array)):
            benchmark_worst_array[j][i] =  benchmark_values[j].max(axis=1)
            benchmark_best_array[j][i] =  benchmark_values[j].min(axis=1)
            benchmark_mean_array[j][i] =  benchmark_values[j].sum(axis=1)/len(benchmark_values[0])
        i += 1

    bench_rot = [benchmark_worst_array[0],benchmark_best_array[0],benchmark_mean_array[0]]
    bench_pos = [benchmark_worst_array[1],benchmark_best_array[1],benchmark_mean_array[1]]
    bench_posangle = [benchmark_worst_array[2],benchmark_best_array[2],benchmark_mean_array[2]]
    

    return (bench_rot,bench_pos,bench_posangle)


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

def get_trans_experiments(n_outliers,sigma,rot_angle,trans_scaling,niters,max_dist_outlier):
    lst = [0]*len(trans_scaling)
    for i in range(len(trans_scaling)):
        dct = {}
        dct['sigma'] = sigma
        dct['rot_angle'] = rot_angle
        dct['trans_scaling'] = trans_scaling[i]
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


filled_marker_style = dict(marker='o', linestyle=':', markersize=8,
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

    bench_worst,bench_best,bench_mean = bench
    
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

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.8),
          ncol=2, fancybox=True, shadow=True)

    for i in range(n):
        marker_style['markerfacecolor'] = color[i]
        marker_style['color'] = color[i]
        marker_style['markeredgecolor'] = lighten_color(color[i],amount=1.3)
        ax2.plot(x_axis,bench_best.T[i],label=get_algorithm_name(algorithms[i]) + " (best)",fillstyle='full',**marker_style)
    ax2.set_xlabel(xlabel_str, fontsize=14)
    ax2.set_ylabel("Best",fontsize=12)

    # Save figure if file name is set
    if fig_name is not None:
        plt.savefig("Plots/" + fig_name + ".pdf", bbox_inches="tight")


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud(f"/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper_res2.ply")
    # pcd = o3d.io.read_point_cloud(f"/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper.ply")
    pts = np.asarray(pcd.points)

    algorithms = [estimate_transformation_13,estimate_transformation_4,estimate_transformation_pasta]
    algorithms += [estimate_transformation_15,estimate_transformation_9]
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

    # Create dictionary for increasing the magnitude of the translation
    # trans_scaling = [0.001,1,2,3,4,5,6,7,8,9,10,15,25,50,100]
    # trans_scaling = [1,2,3,4,5,6]
    # trans_scaling = [0.0001,5,10,15,20,25,30,35]
    # sigmas = 0.0
    # experiments = get_trans_experiments(0,sigmas,rot_angle,trans_scaling,niters,0)
    # x_axis = trans_scaling
    # xlabel_str = r'Translation Magnitude $\|\mathbf{t}\|$'
    # fig_name_end = "sigma_" + str(sigmas).replace('.','_') + "_varpos"


    # Create dictionary for increasing the noise of the point clouds
    niters = 10
    rot_angle = 100
    trans_scaling = 3
    sigmas = np.arange(0.000,0.005,0.00025)
    # sigmas = np.arange(0.000,0.051,0.002)
    experiments = get_noisy_experiments(0,sigmas,rot_angle,trans_scaling,niters,0.1)
    x_axis = sigmas
    xlabel_str = r'Noise ($\sigma$)'
    fig_name_end = "magpos_" + str(trans_scaling).replace('.','_') + "_varsigma"

    # Dummy experiments
    # niters = 1
    # rot_angle = 100
    # trans_scaling = 3
    # sigmas = np.zeros(3)
    # x_axis = np.arange(len(sigmas))
    # experiments = get_noisy_experiments(0,sigmas,rot_angle,trans_scaling,niters,0)
    # fig_name_end = "dummy"
    

    bench_rot,bench_pos,bench_posangle = run_experiments(pts,experiments,algorithms)

    now = datetime.now()
    dt_string = now.strftime('_%d_%m_%Y_%H_%M_%S')

    rot_file_name = "rot_angle_" + fig_name_end + dt_string
    pos_file_name = "pos_" + fig_name_end + dt_string
    posangle_file_name = "posangle_" + fig_name_end + dt_string


    # plt.style.use('dark_background')
    # plt.style.use('bmh')

    plot_experiments(bench_rot,x_axis,r"$\arccos(\mathbf{R}^{\dagger}*\mathbf{R}_{est})$ ($\degree$)",xlabel_str,algorithms,filled_marker_style,rot_file_name)
    plot_experiments(bench_pos,x_axis,r"$\|\mathbf{t} - \mathbf{t}_{est}\|$",xlabel_str,algorithms,filled_marker_style,pos_file_name)
    plot_experiments(bench_posangle,x_axis,r"$\arccos(\widehat{\mathbf{t}}\cdot\widehat{\mathbf{t}}_{est})$ ($\degree$)",xlabel_str,algorithms,filled_marker_style,posangle_file_name)
    # plt.show()




    '''
    Computing 
        Q_lst[0] /= (Q_lst[0]*Q_ref)(0)
        P_lst[0] /= (P_lst[0]*P_ref)(0)
    is worse than using only the sign 
        Q_lst[0] *= np.sign((Q_lst[0]*Q_ref)(0))
        P_lst[0] *= np.sign((P_lst[0]*P_ref)(0))

    Using an eigenvector instead of an eigenbivector is better!! Is there an eigenvector which gives a better estimate?? R: The first eigenvector is best!!
    '''