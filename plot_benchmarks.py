import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm
import matplotlib
import matplotlib.colors as mc
import colorsys
from datetime import datetime
import pickle
import numpy as np
from algorithms import get_algorithm_name


# Three consecutive figures. Ratios for plots:
ratio = [735,500]
scale = 0.007
legend = True
    
# Two consecutive figures. Ratios:
# ratio = [822,583]
# scale = 0.007
# legend = True

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

def legend_savefig(fig_name,ax,y_pos=1.3,legend_in=False,legend=True):
    if legend:
        if fig_name is None or legend_in:
            ax.legend()
        else:
            # Puts the legend outside of the plot (used for putting in the paper)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, y_pos),
                ncol=2, fancybox=True, shadow=True)
    plt.tight_layout()
    # Save figure if file name is set
    if fig_name is not None:
        plt.savefig("Plots/" + fig_name + ".pdf", bbox_inches="tight")
    

def plot_experiments(bench,x_axis,title,xlabel_str,algorithms,marker_style,fig_name=None):
    ''' Plots experiments (worst and best in the different plots)'''
    color = cm.rainbow(np.linspace(0, 1, len(algorithms)))
    n = len(algorithms)

    bench_worst,bench_best,bench_mean,bench_std = bench

    fig = plt.figure(figsize=(ratio[0]*scale, ratio[1]*scale))

    gs = fig.add_gridspec(2, hspace=0)
    ax1,ax2 = gs.subplots(sharex=True, sharey=False)
    fig.suptitle(title)

    for i in range(n):
        marker_style['markerfacecolor'] = color[i]
        marker_style['color'] = color[i]
        marker_style['markeredgecolor'] = lighten_color(color[i],amount=1.3)
        ax1.plot(x_axis,bench_worst.T[i],label=get_algorithm_name(algorithms[i]),fillstyle='full',**marker_style)

    ax1.set_xlabel(xlabel_str)
    ax1.set_ylabel("Worst")

    for i in range(n):
        marker_style['markerfacecolor'] = color[i]
        marker_style['color'] = color[i]
        marker_style['markeredgecolor'] = lighten_color(color[i],amount=1.3)
        ax2.plot(x_axis,bench_best.T[i],label=get_algorithm_name(algorithms[i]) + " (best)",fillstyle='full',**marker_style)
    ax2.set_xlabel(xlabel_str)
    ax2.set_ylabel("Best")

    legend_savefig(fig_name,plt)

def plot_experiments_3(bench,x_axis,title,xlabel_str,algorithms,marker_style,capsize=10.0,fig_name=None,legend=True):
    ''' Plots experiments (worst and best in the different plots)'''
    color = cm.rainbow(np.linspace(0, 1, len(algorithms)))
    n = len(algorithms)

    bench_worst,bench_best,bench_mean,bench_std = bench
    fig = plt.figure(figsize=(ratio[0]*scale, ratio[1]*scale))
    # fig.suptitle(title)

    for i in range(n):
        marker_style['markerfacecolor'] = color[i]
        marker_style['color'] = color[i]
        marker_style['markeredgecolor'] = lighten_color(color[i],amount=1.3)
        plt.errorbar(x_axis,bench_mean.T[i],bench_std.T[i],fillstyle='full',capsize=capsize,fmt='none',color=color[i])
        plt.plot(x_axis,bench_mean.T[i],label=get_algorithm_name(algorithms[i]),fillstyle='full',**marker_style)

    plt.xlabel(xlabel_str)
    plt.ylabel(title + "  (Mean)")

    
    legend_savefig(fig_name,plt,legend=legend,legend_in=True)

def plot_experiments_4(bench,x_axis,title,xlabel_str,algorithms,marker_style,capsize=10.0,fig_name=None,legend=True):
    bench_worst,bench_best,bench_mean,bench_std = bench
    color = cm.rainbow(np.linspace(0, 1, len(algorithms)))
    n = len(algorithms)

    fig = plt.figure(figsize=(ratio[0]*scale, ratio[1]*scale))
    for i in range(n):
        marker_style['markerfacecolor'] = color[i]
        marker_style['color'] = color[i]
        marker_style['markeredgecolor'] = lighten_color(color[i],amount=1.3)
        plt.plot(x_axis,bench_mean.T[i],label=get_algorithm_name(algorithms[i]),fillstyle='full',**marker_style)
        data = {
            'x': x_axis,
            'y1': [y - e for y, e in zip(bench_mean.T[i], bench_std.T[i])],
            'y2': [y + e for y, e in zip(bench_mean.T[i], bench_std.T[i])]
        }
        plt.fill_between(**data,color=color[i],alpha=.5)

    plt.xlabel(xlabel_str)
    plt.ylabel(title + "  (Mean)")

    legend_savefig(fig_name,plt,legend=legend)

def plot_experiments_5(bench_rot,bench_pos,x_axis,rot_worst_label,pos_worst_label,xlabel_str,algorithms,marker_style,capsize=10.0,fig_name=None,legend=True):
    bench_rot_worst,_,_,_ = bench_rot
    bench_pos_worst,_,_,_ = bench_pos

    color = cm.rainbow(np.linspace(0, 1, len(algorithms)))
    n = len(algorithms)

    fig = plt.figure(figsize=(ratio[0]*scale, ratio[1]*scale))
    gs = fig.add_gridspec(2, hspace=0)
    ax1,ax2 = gs.subplots(sharex=True, sharey=False)

    for i in range(n):
        marker_style['markerfacecolor'] = color[i]
        marker_style['color'] = color[i]
        marker_style['markeredgecolor'] = lighten_color(color[i],amount=1.3)
        ax1.plot(x_axis,bench_rot_worst.T[i],label=get_algorithm_name(algorithms[i]),fillstyle='full',**marker_style)

    ax1.set_xlabel(xlabel_str)
    ax1.set_ylabel(rot_worst_label)

    for i in range(n):
        marker_style['markerfacecolor'] = color[i]
        marker_style['color'] = color[i]
        marker_style['markeredgecolor'] = lighten_color(color[i],amount=1.3)
        ax2.plot(x_axis,bench_pos_worst.T[i],fillstyle='full',**marker_style)

    ax2.set_xlabel(xlabel_str)
    ax2.set_ylabel(pos_worst_label)

    legend_savefig(fig_name,ax1,legend_in=True,legend=legend)

def set_font_size():
    SMALL_SIZE = 14
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels

def check_if_in_list(a,list):
    for i in range(len(list)):
        if a == list[i]:
            return True
    return False

def plot_histogram(dct,nbins=100,rot_range=None,pos_range=None):
    

    sigma = dct["sigma"]
    bench_rot = dct["bench_rot"]
    bench_pos = dct["bench_pos"]
    algorithms = dct["algorithms"]
    fig_name_end = dct["fig_name_end"]
    dt_string = dct["dt_string"]

    fig_name = "histogram_" + fig_name_end + dt_string
    
    
    color = cm.rainbow(np.linspace(0, 1, len(algorithms)))
    label = []
    for i in range(len(algorithms)):
        label += [get_algorithm_name(algorithms[i])]


    set_font_size()

    fig,(ax1,ax2) = plt.subplots(2,figsize=(ratio[0]*scale, ratio[1]*scale))

    # Plot translation
    ax1.hist(bench_pos.T,nbins,histtype='bar',label=label,color=color,density=True,range=pos_range)
    ax1.set_xlabel("Translation Error (m)")
    ax1.legend()


    start, end = ax1.get_xlim()

    # num_ticks = 4
    # xticks = np.linspace(0,end,num_ticks)

    # stepsize = 0.001
    # xticks = np.arange(0, end, stepsize)
    # ax1.set_xticks(xticks)
    

    # Plot rotation
    ax2.hist(bench_rot.T,nbins,histtype='bar',label=label,color=color,density=True,range=rot_range)
    ax2.set_xlabel(r"Rotation Error ($\degree$)")
    # ax2.legend()



    plt.tight_layout()
    plt.savefig("Plots/" + fig_name + ".pdf",bbox_inches="tight")

    # plt.show()
    
def filter_dictionary_single_exp(dct,alg_names):
    bench_rot = dct["bench_rot"]
    bench_pos = dct["bench_pos"]
    algorithms = dct["algorithms"]
    
    bench_pos_list = []
    bench_rot_list = []
    algorithms_list = []

    for i in range(len(algorithms)):
        if(check_if_in_list(get_algorithm_name(algorithms[i]),alg_names)):
            bench_pos_list += [bench_pos[i]]
            bench_rot_list += [bench_rot[i]]
            algorithms_list += [algorithms[i]]

    dct["bench_rot"] = np.r_[bench_rot_list]
    dct["bench_pos"] = np.r_[bench_pos_list]
    dct["algorithms"] = np.r_[algorithms_list]

    return dct

def filter_dictionary(dct,alg_names):
    bench_rot = dct["bench_rot"]
    bench_pos = dct["bench_pos"]
    algorithms = dct["algorithms"]
    
    bench_pos_list = []
    bench_rot_list = []
    algorithms_list = []
    
    for i in range(len(bench_pos)): # iterates for each type of benchmark
        bench_pos_i = bench_pos[i].T
        bench_rot_i = bench_rot[i].T
        bench_pos_listj = []
        bench_rot_listj = []
        for j in range(len(algorithms)):
            if(check_if_in_list(get_algorithm_name(algorithms[j]),alg_names)):
                bench_pos_listj += [bench_pos_i[j]]
                bench_rot_listj += [bench_rot_i[j]]
        bench_pos_list += [np.r_[bench_pos_listj].T]
        bench_rot_list += [np.r_[bench_rot_listj].T]

    for j in range(len(algorithms)):
            if(check_if_in_list(get_algorithm_name(algorithms[j]),alg_names)):
                algorithms_list += [algorithms[j]]

    dct["bench_rot"] = bench_rot_list
    dct["bench_pos"] = bench_pos_list
    dct["algorithms"] = algorithms_list

    return dct
    
    

def plot_benchmarks(dct,show_plot=False):
    # with open(filename, "rb") as f:
    #     dct = pickle.load(f)

    # plt.style.use('dark_background')
    # plt.style.use('bmh')    
    # print(plt.style.available) # prints available styles

    bench_rot = dct["bench_rot"]
    bench_posangle = dct["bench_posangle"]
    bench_pos = dct["bench_pos"]
    x_axis = dct["x_axis"]
    algorithms = dct["algorithms"]
    fig_name_end = dct["fig_name_end"]
    xlabel_str = dct["xlabel_str"]
    dt_string = dct["dt_string"]
    
    # angle_error_str = r"$\arccos(\mathbf{R}^{\dagger}*\widehat{\mathbf{R}})$ ($\degree$)"
    # pos_error_str = r"$\|\mathbf{t} - \hat{\mathbf{t}}\|$"
    # posangle_error_str = r"$\arccos(\widehat{\mathbf{t}}\cdot\hat{\mathbf{t}})$ ($\degree$)"

    angle_error_str = r"Error ($\degree$)"
    pos_error_str = r"Error (m)"
    if show_plot:
        rot_file_name = None
        pos_file_name = None
        posangle_file_name = None
        worst_filename = None
    else:
        rot_file_name = "rot_angle_" + fig_name_end + dt_string
        pos_file_name = "pos_" + fig_name_end + dt_string
        posangle_file_name = "posangle_" + fig_name_end + dt_string
        worst_filename = "worst_" + fig_name_end + dt_string


    set_font_size()

    # Plot mean and standard deviation (removes outliers (180 degree angle error))
    capsize = 3.0
    plot_experiments_3(bench_rot,x_axis,angle_error_str,xlabel_str,algorithms,filled_marker_style,capsize,rot_file_name,legend=legend)
    plot_experiments_3(bench_pos,x_axis,pos_error_str,xlabel_str,algorithms,filled_marker_style,capsize,pos_file_name,legend=legend)

    # Plots the worst experiments in the same figure
    # plot_experiments_5(bench_rot,bench_pos,x_axis,angle_error_str,pos_error_str,xlabel_str,algorithms,filled_marker_style,capsize,worst_filename)

if __name__ == "__main__":
    alg_names = ["VGA CeofMass","CGA exactTRS"]
    filename = f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_magpos_1_varsigma_ArmadilloBack_0_01_03_2024_04_03_08.pickle"
    # filename = f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_magpos_1_varsigma_ArmadilloBack_0_29_02_2024_12_59_02.pickle"
    # filename = f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_magpos_1_varsigma_ArmadilloBack_0_29_02_2024_02_03_35.pickle"
    # filename = f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_magpos_1_varsigma_dragonMouth5_0_29_02_2024_07_01_11.pickle"
    with open(filename, "rb") as f:
        dct = pickle.load(f)
    dct = filter_dictionary(dct,alg_names)
    plot_benchmarks(dct)
    
    filename = f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_single_sigma_0_05ArmadilloBack_0_01_03_2024_04_20_19.pickle"
    # filename = f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_single_sigma_0_05ArmadilloBack_0_29_02_2024_13_00_45.pickle"
    # filename = f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_single_sigma_0_01ArmadilloBack_0_29_02_2024_02_28_25.pickle"
    # filename = f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_single_sigma_0_01dragonMouth5_0_29_02_2024_07_32_06.pickle"
    with open(filename, "rb") as f:
        dct = pickle.load(f)
    dct = filter_dictionary_single_exp(dct,alg_names)
    # plot_histogram(dct,nbins=30,rot_range=(0,0.6),pos_range=(0,0.002))
    plot_histogram(dct,nbins=100)

    plt.show()

