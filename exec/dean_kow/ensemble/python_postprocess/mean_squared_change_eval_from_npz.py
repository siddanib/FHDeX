import os
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
import yt

device = torch.device("cuda")

# This script calculates mean squared change
### CAREFUL WITH DT
dt = 3.0e-6
dataset_name = "no_history"
par_per_cell =  10
parent_folder = "../particles_"+dataset_name+"_dt_3.0e-6"

fld_path = parent_folder
# Get plt files
onlyplotfiles = [fl for fl in os.listdir(fld_path) if "plt" in fl]

# Ensure to remove files that contain "old" string
onlyplotfiles = [fl for fl in onlyplotfiles if "old" not in fl]
onlyplotfiles.sort()
n_files = len(onlyplotfiles)

aa = np.load(dataset_name+"_total_data_file.npz")
total_data_dk = torch.from_numpy(aa['total_data_dk']).to(device)
total_data_ptcl = torch.from_numpy(aa['total_data_ptcl']).to(device)

frequency_array = np.zeros((n_files,))
mean_sq_array   = np.zeros((n_files,2))
time_array      = np.linspace(0,n_files-1, n_files)*dt

for i in range(n_files-1):
    ## Load the i^th file
    phi_i_dk_tnsr =  total_data_dk[i,:,:]
    phi_i_ptcl_tnsr = total_data_ptcl[i,:,:]

    for j in range(i+1, n_files):
        frequency_array[j-i] += 1
        ## Load the j^th file
        phi_j_dk_tnsr = total_data_dk[j,:,:]
        phi_j_ptcl_tnsr = total_data_ptcl[j,:,:]

        mean_sq_array[j-i, 0] += torch.mean((phi_j_ptcl_tnsr - phi_i_ptcl_tnsr)**2).cpu().numpy()
        mean_sq_array[j-i, 1] += torch.mean((phi_j_dk_tnsr - phi_i_dk_tnsr)**2).cpu().numpy()


# Scale these by frequency
mean_sq_array[1:, 0] /= frequency_array[1:]
mean_sq_array[1:, 1] /= frequency_array[1:]

final_array = np.zeros((n_files, 3))
final_array[:,0] = time_array[:]
final_array[:,1] = mean_sq_array[:,0]
final_array[:,2] = mean_sq_array[:,1]

save_file_name = dataset_name +f"_mean_squared_change_avg_ppc_{par_per_cell}"

np.savetxt(save_file_name, final_array,
               header="Time\tParticle Simulation\tSPDE\n")
