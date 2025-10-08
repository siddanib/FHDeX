import sys                                                                      
import os                                                                       
import numpy as np                                                              
import torch
import math                                                                     
import h5py                                                                     
import yaml
from scipy.stats import skew                                                    
from scipy.stats import kurtosis as kurt
from scipy.special import erf
import matplotlib.pyplot as plt                                                 
#######################################################                         
                                                                                
def fhd_model_run ():
    device =  torch.device("cuda")
    dataset_name = "uniform"
    par_per_cell = 50
    dx = 0.01

    folder_list = [
        "/scratch/bsiddani/stochastic_ml/09_18_2025/multirun/2025-10-01/11-02-33",
        "/scratch/bsiddani/stochastic_ml/09_18_2025/multirun/2025-10-02/15-17-09",
        "/scratch/bsiddani/stochastic_ml/09_18_2025/multirun/2025-10-03/10-32-57",
        "/scratch/bsiddani/stochastic_ml/09_18_2025/multirun/2025-09-18/12-02-23"
                   ]
    cfl_list = [0.03/8, 0.03/4, 0.03/2, 0.03]
    # Blue is reserved for particle simulation
    color_list = ["r", "g", "m", "k"]

    n_steps_list = [8000, 4000, 2000, 1000]

    # List that holds auto correlation
    np_array_list = []
    # Load the entire data
    data_file_name = "structure_factor_different_temporal" 
    with h5py.File(data_file_name+".h5", mode="r") as f:
        for cfl in cfl_list:
            auto_corr_data = f[f"auto_corr_{cfl}"][:]
            np_array_list.append(auto_corr_data)

    save_fig = True                                                             
    show_fig = False                                                            
    fig, ax = plt.subplots(figsize=(10, 10))

    for list_id, cfl in enumerate(cfl_list):
        n_steps = n_steps_list[list_id]
        auto_corr = np_array_list[list_id]
        # Plot data on the axes
        dt = cfl*dx*dx
        time_array = np.linspace(0, n_steps, n_steps+1)*dt
        if cfl == 0.03/8:
            #ax.plot(time_array, auto_corr[0,:], color='blue', linestyle='-',                        
            #        marker='o', markersize=10, label="Particle")
            ax.plot(time_array, auto_corr[0,:], color='blue', linestyle='-',                        
                    markersize=10, label="Particle")

        #ax.plot(time_array, auto_corr[1,:], color='red', linestyle='-',
        #            marker='s', markersize=10, label=f"SPDE; cfl={cfl}")
        ax.plot(time_array, auto_corr[1,:], color=color_list[list_id],
                linestyle='-', label=f"SPDE; cfl={cfl}")

    # Add labels and title
    ax.set_xlabel(r'$k$'+" (time units)", fontsize=35)
    ax.set_title(r'$<N(t) N(t+k)> - <N(t)> <N(t+k)>$'+ "\n"+
       "ensembles : 20000"+
       "\n"+r"<N> = "+f"{par_per_cell}",
                 fontsize=35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(None, 1.0e-5)
    ## Add a legend
    ax.legend(fontsize=25)
    ax.grid(True)
    fig.tight_layout()
    if save_fig:
        fig.savefig('auto_correlation_different_times_loglog.jpg')
    if show_fig:
        plt.show()

    return None

if __name__ == "__main__":
    torch.set_default_device('cuda')
    fhd_model_run()
