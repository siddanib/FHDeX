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
    # Setting font size
    plt.rcParams.update({'font.size': 20})
    device =  torch.device("cuda")

    # The quantity being plotted depends on dx even for Particles & Theory
    # when plotted in dimensional time
    dimensional_time = False

    dataset_name = "uniform"
    par_per_cell = 1
    cfl = 0.03
    n_steps = 1000
    len_system = 1.0

    ncells_list = [800, 400, 200, 100
                   ]
    spde_type_list = ["Nonlin SPDE", "Nonlin SPDE", "Nonlin SPDE", "Nonlin SPDE",
                      ]
    # Black is for theory
    color_list = ["r", "g", "m", "b"]

    np_array_list = []
    # Load the entire data
    data_file_name = f"mean_change_different_dx_par_per_cell_{par_per_cell}" 
    with h5py.File(data_file_name+".h5", mode="r") as f:
        for ncells, spde_type in zip(ncells_list, spde_type_list):
            if "Nonlin" in spde_type:
                dd = f[f"mean_change_Nonlin_ncells_{ncells}"][:]
            else:
                dd = f[f"mean_change_Lin_ncells_{ncells}"][:]
            np_array_list.append(dd)

    save_fig = True                                                             
    show_fig = False                                                            
    fig, ax = plt.subplots(figsize=(12, 12)) 

    for list_id, ncells in enumerate(ncells_list):
        # Get data and spde_type
        dd = np_array_list[list_id]
        spde_tpe = spde_type_list[list_id]
        # Box size
        dx = len_system/ncells
        dt = cfl*dx*dx
        # Time array
        time_array = np.linspace(0, n_steps, n_steps+1)*dt
        # Note that D_0 in theorectical formulation is different
        diff_coeff = 0.5
        ### Scale the time
        scaled_time_array = time_array*((4.0*diff_coeff)/(dx*dx))
        theory_change = np.zeros((n_steps+1,))
        theory_change[1:] = erf(np.sqrt(1/scaled_time_array[1:]))
        theory_change[1:] += np.sqrt(scaled_time_array[1:]/np.pi)*(np.exp(-1/scaled_time_array[1:])-1)
        theory_change[1:] = theory_change[1:] # Note f()^{d} in Eq(2) where d = 1 for 1D system
        theory_change[1:] = 1.0 - theory_change[1:]
        theory_change[1:] *= 2*par_per_cell

        # Which time to use (dimensional or nondimensional)
        if dimensional_time:
            time_x_axis = time_array
        else:
            time_x_axis = scaled_time_array

        # Plot data on the axes
        if "Nonlin" in spde_type:
            ax.plot(time_x_axis, dd[1,:], color=color_list[list_id],
                linestyle='-', label=f"Nonlin SPDE; dx = {dx}")
            # Plotting particle and theory
            ax.plot(time_x_axis, dd[0,:], color=color_list[list_id], linestyle='-', 
                    marker='o', markersize=5, label=f"Particle; dx = {dx}")
            ax.plot(time_x_axis, theory_change, color="k", linestyle='-', 
                label=f"Theory; dx = {dx}")
        else:
            ax.plot(time_x_axis, dd[1,:], color=color_list[list_id],
                linestyle='--', label=f"Lin SPDE; dx = {dx}") 

    # Add labels and title
    if dimensional_time:
        ax.set_xlabel(r'$k$' + " (time units)", fontsize=35)
    else:
        ax.set_xlabel(r'$k \frac{4D_{0}}{dx^2}$' + " (nondimensional time)", fontsize=35)

    ax.set_title(r'$<(N(t)-N(t+k))^{2}>$'+ "\n"+
       "ensembles : 20000"+
       "\n"+r"<N> = "+f"{par_per_cell}",
                 fontsize=35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ## Add a legend
    ax.legend(fontsize=20)
    ax.grid(True)
    fig.tight_layout()
    if save_fig:
        if dimensional_time:
            fig.savefig(f'dx_mean_squared_change_dimensional_loglog_{par_per_cell}.jpg')
        else:
            fig.savefig(f'dx_mean_squared_change_nondimensional_loglog_{par_per_cell}.jpg')

    if show_fig:
        plt.show()

    return None

if __name__ == "__main__":
    torch.set_default_device('cuda')
    fhd_model_run()
