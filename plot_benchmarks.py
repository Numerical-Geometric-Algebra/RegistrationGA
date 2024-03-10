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
    plt.ylabel(title)

    
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

def plot_axis(ax,yaxis,yerror,xaxis,colors,capsize,alg_names,marker_style):
    ''' Plots the data into some ax '''
    n = len(alg_names)
    for i in range(n):
        marker_style['markerfacecolor'] = colors[i]
        marker_style['color'] = colors[i]
        marker_style['markeredgecolor'] = lighten_color(colors[i],amount=1.3)
        ax.errorbar(xaxis,yaxis.T[i],yerror.T[i],fillstyle='full',capsize=capsize,fmt='none',color=colors[i])
        ax.plot(xaxis,yaxis.T[i],label=alg_names[i],fillstyle='full',**marker_style)


    

    

def plot_experiments_multi_hists(dcts,titles,nbins=30,rot_range=None,pos_range=None):
    ratio = [2275,610]
    dpi = 96

    if pos_range is None:
        pos_range = []
        for i in range(len(dcts)):
            pos_range += [None]

    if rot_range is None:
        rot_range = []
        for i in range(len(dcts)):
            rot_range += [None]
    
    set_font_size(20,22,26)
    algorithms = dcts[0]["algorithms"]
    colors = [np.array([249, 99, 0])/255,np.array([0, 150, 249])/255]
    
    alg_names = []
    for i in range(len(algorithms)):
        alg_names += [get_algorithm_name(algorithms[i])]

    fig, axarr = plt.subplots(2,len(dcts),figsize=(ratio[0]/dpi,ratio[1]/dpi),dpi=dpi)

    for i in range(len(dcts)):
        axarr[0][i].hist(dcts[i]["bench_rot"].T,nbins,histtype='bar',label=alg_names,color=colors,range=rot_range[i],density=True)
        axarr[1][i].hist(dcts[i]["bench_pos"].T,nbins,histtype='bar',label=alg_names,color=colors,range=pos_range[i],density=True)
        axarr[0][i].set_title(titles[i],size=26)
        axarr[0][i].set_xlabel(r"RRE ($\degree$)")
        axarr[1][i].set_xlabel(r"RTE ($m$)")
        axarr[0][i].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        axarr[1][i].ticklabel_format(axis='both', style='sci', scilimits=(0,0))

    axarr[0][3].legend()
    axarr[0][0].set_ylabel(r"Prob. density")
    axarr[1][0].set_ylabel(r"Prob. density")

    plt.tight_layout()
    plt.subplots_adjust(top=0.93,bottom=0.135,left=0.038,right=0.987,hspace=0.482,wspace=0.11)
    plt.show()


def plot_experiments_multi_plots(dcts,titles):
    # plt.style.use('dark_background')
    ratio = [2275,610]
    dpi = 96
    print(dcts[0]["xlabel_str"])
    algorithms = dcts[0]["algorithms"]

    set_font_size(20,24,26)

    fig, axarr = plt.subplots(2,len(dcts),sharex=True,figsize=(ratio[0]/dpi,ratio[1]/dpi),dpi=dpi)
    alg_names = []
    for i in range(len(algorithms)):
        alg_names += [get_algorithm_name(algorithms[i])]
    colors = cm.rainbow(np.linspace(0, 1, len(algorithms)))
    # colors = [np.array([249, 99, 0])/255,np.array([0, 150, 249])/255]
    capsize = 3.0

    # end = dcts[i]["x_axis"].max()
    # stepsize = 0.005
    # xticks = np.arange(0, end+0.0001, stepsize)
    
    for i in range(len(dcts)):
        plot_axis(axarr[0][i],dcts[i]["bench_rot"][2],dcts[i]["bench_rot"][3],dcts[i]["x_axis"],colors,capsize,alg_names,filled_marker_style)
        plot_axis(axarr[1][i],dcts[i]["bench_pos"][2],dcts[i]["bench_pos"][3],dcts[i]["x_axis"],colors,capsize,alg_names,filled_marker_style)
        axarr[1][i].ticklabel_format(axis='y', style='sci', scilimits=(-3,-6))
        axarr[1][i].set_xlabel(dcts[i]["xlabel_str"])
        # axarr[1][i].set_xticks(xticks)
        axarr[0][i].set_title(titles[i],size=26)
        

    
    axarr[0][1].legend()

    axarr[0][0].set_ylabel(r"RRE ($\degree$)")
    axarr[1][0].set_ylabel(r"RTE (m)")

    
        
    # plt.grid()

    # axarr[0][1].ticklabel_format(axis='y', style='sci', scilimits=(-3,-6))

    plt.tight_layout()
    plt.subplots_adjust(top=0.937,bottom=0.12,left=0.044,right=0.995,hspace=0.16,wspace=0.093)
    # loc='upper center', bbox_to_anchor=(0.5, 1.4),ncol=1, fancybox=True, shadow=True

    plt.show()


def set_font_size(SMALL_SIZE = 14,MEDIUM_SIZE = 14,BIGGER_SIZE = 14):

    plt.rc('font', size=SMALL_SIZE)          # default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # axes title
    plt.rc('xtick', labelsize=SMALL_SIZE)    # xtick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # ytick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend 
    plt.rc('figure', titlesize=BIGGER_SIZE)  # figure title
    plt.rc('axes', labelsize=SMALL_SIZE)     # x and y labels

def check_if_in_list(a,list):
    for i in range(len(list)):
        if a == list[i]:
            return True
    return False

def plot_histogram(dct,nbins=100,rot_range=None,pos_range=None,save_plot=False):
    ''' Plots a single histogram '''

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

    # set_font_size()

    fig,(ax1,ax2) = plt.subplots(2,figsize=(ratio[0]*scale, ratio[1]*scale))
    fig.suptitle(r"$\sigma=$"+str(sigma))

    # Plot translation
    ax1.hist(bench_pos.T,nbins,histtype='bar',label=label,color=color,range=pos_range)
    ax1.set_xlabel("Translation Error (m)")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    
    # Plot rotation
    ax2.hist(bench_rot.T,nbins,histtype='bar',label=label,color=color,range=rot_range)
    ax2.set_xlabel(r"Rotation Error ($\degree$)")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    plt.tight_layout()
    if save_plot:
        plt.tight_layout()
        plt.savefig("Plots/" + fig_name + ".pdf",bbox_inches="tight")

    
def filter_dictionary_single_exp(dct,alg_names):
    ''' Used to filter out unwanted algorithms
        alg_names are the algorithms we want to show
    '''
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
    ''' Used to filter out unwanted algorithms
        alg_names are the names of the algorithms we want to show'''
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
    
def plot_multiple_benchmarks():
    '''Reads multiple benchmark files and plots things in the same figure'''
    
    # Chose multiple algorithms
    alg_names = ["ICP","CGA eigmvs","PASTA 3D"]

    # Chose multiple files    
    filenames = [f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_sigma_0_01_trajectory_bun_zipper_res2_05_03_2024_04_06_37.pickle",
                 f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_sigma_0_01_trajectory_ArmadilloBack_0_05_03_2024_09_30_05.pickle",
                 f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_sigma_0_01_trajectory_dragonMouth5_0_05_03_2024_16_11_04.pickle"]

    titles = ["Bunny","Armadillo","Dragon"]
    dcts = []
    for i in range(len(filenames)):
        with open(filenames[i], "rb") as f:
            dct = pickle.load(f)
        dct = filter_dictionary(dct,alg_names)
        dcts += [dct]

    plot_experiments_multi_plots(dcts,titles)


def plot_multiple_histograms():
    alg_names = ["VGA CeofMass","CGA eigmvs"]

    # Chose multiple files
    filenames = [f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_single_sigma_0_001_ArmadilloBack_0_05_03_2024_10_11_03.pickle",
                 f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_single_sigma_0_005_ArmadilloBack_0_05_03_2024_10_31_38.pickle",
                 f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_single_sigma_0_01_ArmadilloBack_0_05_03_2024_10_52_47.pickle",
                 f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_single_sigma_0_02_ArmadilloBack_0_05_03_2024_11_14_47.pickle"]

    titles = [r"$\sigma=0.001$",r"$\sigma=0.005$",r"$\sigma=0.01$",r"$\sigma=0.02$"]

    rot_range = [None,None,(0,0.8),(0,2)]
    pos_range = [None,None,(0,0.0015),(0,0.0035)]

    dcts = []
    for i in range(len(filenames)):
        with open(filenames[i], "rb") as f:
            dct = pickle.load(f)
        dct = filter_dictionary_single_exp(dct,alg_names)
        dcts += [dct]

    plot_experiments_multi_hists(dcts,titles,pos_range=pos_range,rot_range=rot_range)
    
def set_dark_background():
    plt.style.use('dark_background')

def show_plots():
    plt.show()

def plot_benchmarks(dct,show_plot=False):

    bench_rot = dct["bench_rot"]
    bench_posangle = dct["bench_posangle"]
    bench_pos = dct["bench_pos"]
    x_axis = dct["x_axis"]
    algorithms = dct["algorithms"]
    fig_name_end = dct["fig_name_end"]
    xlabel_str = dct["xlabel_str"]
    dt_string = dct["dt_string"]

    angle_error_str = r"RRE ($\degree$)"
    pos_error_str = r"RTE (m)"
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

if __name__ == "__main__":
    alg_names = ["CGA eigmvs","VGA CeofMass"]
    
    ''' Load a pickle file and plot the data'''
    filename = f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_sigma_0_01_trajectory_ArmadilloBack_0_05_03_2024_09_30_05.pickle"
    with open(filename, "rb") as f:
        dct = pickle.load(f)
    dct = filter_dictionary(dct,alg_names)
    plot_benchmarks(dct)
    
    ''' Load a pickle file and plot an histogram '''
    filename = f"/home/francisco/Code/RegistrationGA/Benchmarks/benchmark_single_sigma_0_005_ArmadilloBack_0_05_03_2024_10_31_38.pickle"
    with open(filename, "rb") as f:
        dct = pickle.load(f)
    dct = filter_dictionary_single_exp(dct,alg_names)
    plot_histogram(dct,nbins=30,rot_range=(0,0.6),pos_range=(0,0.002))

    plt.show()