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
    device =  torch.device("cpu")
    full_path = "./multirun/2025-11-01/13-38-06/0"
    dataset_name = "ensembles_of_multi_steps"
    par_per_cell = 1
    dx = 0.01
    cfl = 0.03
    dt = cfl*dx*dx
    n_steps = 1000
    time_array = np.linspace(0, n_steps, n_steps+1)*dt

    # Blue is reserved for particle simulation
    np_array_list = []
    # This is for mean squared change
    mean_sq_corr = np.zeros((2,n_steps+1))
    # This is for structure factor
    auto_corr = np.zeros((2,n_steps+1))
    data_file = os.path.join(full_path, dataset_name)
    # Ordering of data (Ensembles, Time Step, Cells)                            
    with h5py.File(data_file+".h5", mode="r") as f:                          
        n_ptcl_data  = f["ground_truth_data"][:]
        n_model_data = f["model_data"][:]

    for i, n_version_data in enumerate([n_ptcl_data, n_model_data]):
        data_tensor = torch.from_numpy(n_version_data).to(device)
        print(data_tensor.device)

        for j in range(1,n_steps+1):                                            
            aa = data_tensor[:,:-j,:] 
            bb = data_tensor[:,j:,:]
            cc = torch.mean((aa-bb)**2)
            dd = torch.mean(aa*bb) - torch.mean(aa)*torch.mean(bb)
            mean_sq_corr[i, j] += cc.cpu().numpy()
            auto_corr[i, j] += dd.cpu().numpy()

    # Append this data to np_array_list
    np_array_list.append(mean_sq_corr)
    np_array_list.append(auto_corr)
 
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

    ## Final array that will be saved to text file
    final_array = np.zeros((n_steps+1, 6))
    final_array[:, 0] = time_array
    final_array[:, 1] = theory_change
    final_array[:, 2] = np_array_list[0][0,:]
    final_array[:, 3] = np_array_list[0][1,:]
    final_array[:, 4] = np_array_list[1][0,:]
    final_array[:, 5] = np_array_list[1][1,:]

    save_file_name = f"system_analysis_ML_avg_ppc_{par_per_cell}"
    np.savetxt(save_file_name, final_array,
               header="Time\tMS Theory\tMS Particle Simulation\tMS ML model\tAC Particle Simulation\t AC ML Model")

    return None

if __name__ == "__main__":
    fhd_model_run()
