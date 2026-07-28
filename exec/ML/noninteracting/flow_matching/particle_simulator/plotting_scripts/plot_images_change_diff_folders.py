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
    cfl = 0.03
    dt = cfl*dx*dx
    n_steps = 1000
    time_array = np.linspace(0, n_steps, n_steps+1)*dt

    # First folder is non-linear SPDE and second is linear SPDE
    folder_list = [
        "/scratch/bsiddani/stochastic_ml/09_18_2025/multirun/2025-09-18/12-02-23",
        "/scratch/bsiddani/stochastic_ml/09_18_2025/multirun/2025-10-07/13-50-43"
                   ]
    spde_type_list = ["Nonlin SPDE", "Lin SPDE"]
    # Blue is reserved for particle simulation
    color_list = ["r", "g", "m", "k"]

    np_array_list = []

    save_fig = True                                                             
    show_fig = False                                                            
    fig, ax = plt.subplots(figsize=(10, 10))                                    

    for list_id, fldr in enumerate(folder_list):
        n_realizations = 0
        auto_corr = np.zeros((2,n_steps+1))
        for entry in os.listdir(fldr):
            full_path = os.path.join(fldr, entry)
            # Check whether it is a folder
            if not os.path.isdir(full_path):
                continue
            # Load the yaml file to see if it is the right one
            yaml_string = os.path.join(full_path, ".hydra/config.yaml")
            with open(yaml_string,"r") as yaml_file:
                data_yaml = yaml.safe_load(yaml_file)
            # Filter again based on dataset name, par_per_cell and cfl
            if dataset_name != data_yaml["dataset"]["name"]:
                print("Some folders don't meet dataset criterion")
                continue
            if np.abs(cfl-data_yaml["cfl"]) > 1.0e-5:
                print("Some folders don't meet cfl criterion")
                continue
            if par_per_cell != data_yaml["par_per_cell"]:
                print("Some folders don't meet par_per_cell criterion")
                continue

            data_file = os.path.join(full_path, dataset_name)
            # Ordering of data (Ensembles, Time Step, Cells)                            
            with h5py.File(data_file+".h5", mode="r") as f:                          
                n_ptcl_data = f["ground_truth_data"][:]
                if list_id == 0:
                    n_spde_data = f["gauss_data"][:]
                else:
                    n_spde_data = f["lin_gauss_data"][:]


            for i, n_version_data in enumerate([n_ptcl_data, n_spde_data]):
                data_tensor = torch.from_numpy(n_version_data).to(device)
                print(data_tensor.device)

                for j in range(1,n_steps+1):                                            
                    aa = data_tensor[:,:-j,:]                                        
                    bb = data_tensor[:,j:,:]
                    cc = torch.mean((aa-bb)**2)
                    auto_corr[i, j] += cc.cpu().numpy()

            n_realizations +=1 
        auto_corr /= n_realizations
        # Append this data to np_array_list
        np_array_list.append(auto_corr)

        # Plot data on the axes
        if list_id == 0:
            #ax.plot(time_array, auto_corr[0,:], color='blue', linestyle='-',                        
            #        marker='o', markersize=10, label="Particle")
            ax.plot(time_array, auto_corr[0,:], color='blue', linestyle='-',                        
                    marker='o', markersize=5, label="Particle")

        #ax.plot(time_array, auto_corr[1,:], color='red', linestyle='-',
        #            marker='s', markersize=10, label=f"SPDE; cfl={cfl}")
        ax.plot(time_array, auto_corr[1,:], color=color_list[list_id],
                linestyle='-', label=spde_type_list[list_id])
    
    # Note that D_0 in theorectical formulation is different
    list_id += 1
    diff_coeff = 0.5
    ### Scale the time
    scaled_time_array = time_array*((4.0*diff_coeff)/(dx*dx))
    theory_change = np.zeros((n_steps+1,))
    theory_change[1:] = erf(np.sqrt(1/scaled_time_array[1:]))
    theory_change[1:] += np.sqrt(scaled_time_array[1:]/np.pi)*(np.exp(-1/scaled_time_array[1:])-1)
    theory_change[1:] = theory_change[1:] # Note f()^{d} in Eq(2) where d = 1 for 1D system
    theory_change[1:] = 1.0 - theory_change[1:]
    theory_change[1:] *= 2*par_per_cell
    ax.plot(time_array, theory_change, color=color_list[list_id], linestyle='-',                        
            label="Theory")

    # Add labels and title
    ax.set_xlabel(r'$k$' + " (time units)", fontsize=35)
    ax.set_title(r'$<(N(t)-N(t+k))^{2}>$'+ "\n"+
       "ensembles : 20000"+
       "\n"+r"<N> = "+f"{np.mean(n_ptcl_data):.2e}",
                 fontsize=35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ## Add a legend
    ax.legend(fontsize=25)
    ax.grid(True)
    fig.tight_layout()
    if save_fig:
        fig.savefig(f'mean_squared_change_different_methods_loglog_{par_per_cell}.jpg')
    if show_fig:
        plt.show()

    return None

if __name__ == "__main__":
    torch.set_default_device('cuda')
    fhd_model_run()
