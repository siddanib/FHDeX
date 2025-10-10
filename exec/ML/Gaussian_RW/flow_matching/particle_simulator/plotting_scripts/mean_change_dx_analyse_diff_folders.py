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
    par_per_cell = 1
    cfl = 0.03
    n_steps = 1000

    # First folder is non-linear SPDE and second is linear SPDE
    folder_list = [
        "/scratch/bsiddani/stochastic_ml/09_18_2025/multirun/2025-10-08/14-20-54",
        "/scratch/bsiddani/stochastic_ml/09_18_2025/multirun/2025-10-08/14-20-54",
        "/scratch/bsiddani/stochastic_ml/09_18_2025/multirun/2025-10-08/14-20-54",
        "/scratch/bsiddani/stochastic_ml/09_18_2025/multirun/2025-10-08/14-20-54",
        "/scratch/bsiddani/stochastic_ml/09_18_2025/multirun/2025-10-08/16-14-53",
        "/scratch/bsiddani/stochastic_ml/09_18_2025/multirun/2025-10-08/16-14-53",
        "/scratch/bsiddani/stochastic_ml/09_18_2025/multirun/2025-10-08/16-14-53",
        "/scratch/bsiddani/stochastic_ml/09_18_2025/multirun/2025-10-08/16-14-53",
                   ]
    ncells_list = [800, 400, 200, 100,
                   800, 400, 200, 100,
                   ]
    spde_type_list = ["Nonlin SPDE", "Nonlin SPDE", "Nonlin SPDE", "Nonlin SPDE",
                      "Lin SPDE", "Lin SPDE", "Lin SPDE", "Lin SPDE",
                      ]

    np_array_list = []

    for list_id, fldr in enumerate(folder_list):
        ncells = ncells_list[list_id]
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
            if ncells != data_yaml["ncells"]:
                print("Some folders don't meet ncells criterion")
                continue

            data_file = os.path.join(full_path, dataset_name)
            # Ordering of data (Ensembles, Time Step, Cells)                            
            with h5py.File(data_file+".h5", mode="r") as f:                          
                dt          = f['dt'][()]
                dx          = f['dx'][()]
                n_ptcl_data = f["ground_truth_data"][:]

                if "Nonlin" in spde_type_list[list_id]:
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

    # Save the np_array_list
    save_file_name = f"mean_change_different_dx_par_per_cell_{par_per_cell}"
    with h5py.File(save_file_name+".h5", mode="w") as f:
        list_id = -1
        for dd,ncells in zip(np_array_list,ncells_list):
            list_id += 1
            if "Nonlin" in spde_type_list[list_id]:
                f.create_dataset(f"mean_change_Nonlin_ncells_{ncells}",
                                 data=dd, dtype = np.float32)
            else:
                f.create_dataset(f"mean_change_Lin_ncells_{ncells}",
                                 data=dd, dtype = np.float32) 
    return None

if __name__ == "__main__":
    torch.set_default_device('cuda')
    fhd_model_run()
