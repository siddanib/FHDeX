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

    np_array_list = []

    save_fig = True                                                             
    show_fig = False                                                            
    fig, ax = plt.subplots(figsize=(10, 10))                                    

    list_id = -1
    for fldr,cfl in zip(folder_list, cfl_list):
        list_id += 1
        n_realizations = 0
        n_steps = n_steps_list[list_id]
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
                n_spde_data = f["gauss_data"][:]
                dt          = f['dt'][()]
                dx          = f['dx'][()]

            for i, n_version_data in enumerate([n_ptcl_data, n_spde_data]):
                data_tensor = torch.from_numpy(n_version_data).to(device)
                print(data_tensor.device)
                i_ac = (torch.mean(data_tensor**2)
                        - torch.mean(data_tensor)**2)

                auto_corr[i, 0] += i_ac.cpu().numpy()

                for j in range(1,n_steps+1):                                            
                    aa = data_tensor[:,:-j,:]                                        
                    bb = data_tensor[:,j:,:]
                    cc = torch.mean(aa*bb)
                    cc -= torch.mean(aa)*torch.mean(bb)
                    auto_corr[i, j] += cc.cpu().numpy()

            n_realizations +=1 
        auto_corr /= n_realizations
        # Append this data to np_array_list
        np_array_list.append(auto_corr)

        # Plot data on the axes
        time_array = np.linspace(0, n_steps, n_steps+1)*dt
        if cfl == 0.03:
            #ax.plot(time_array, auto_corr[0,:], color='blue', linestyle='-',                        
            #        marker='o', markersize=10, label="Particle")
            ax.plot(time_array, auto_corr[0,:], color='blue', linestyle='-',                        
                    markersize=10, label="Particle")

        #ax.plot(time_array, auto_corr[1,:], color='red', linestyle='-',
        #            marker='s', markersize=10, label=f"SPDE; cfl={cfl}")
        ax.plot(time_array, auto_corr[1,:], color=color_list[list_id],
                linestyle='-', label=f"SPDE; cfl={cfl}")

    # Add labels and title
    ax.set_xlabel(r'$k$', fontsize=35)
    ax.set_title(r'$<N(t) N(t+k)> - <N(t)> <N(t+k)>$'+ "\n"+
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
        fig.savefig('auto_correlation_different_times_loglog.jpg')
    if show_fig:
        plt.show()

    # Save the np_array_list
    save_file_name = "structure_factor_different_temporal"
    with h5py.File(save_file_name+".h5", mode="w") as f:
        for dd,cfl in zip(np_array_list,cfl_list):
            f.create_dataset(f"auto_corr_{cfl}", data=dd, dtype = np.float32)

    return None

if __name__ == "__main__":
    torch.set_default_device('cuda')
    fhd_model_run()
