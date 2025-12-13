import sys
import os
import numpy as np
import torch
import math
import h5py
import yaml
from scipy.stats import skew
from scipy.stats import kurtosis as kurt
import matplotlib.pyplot as plt

torch.set_default_device('cuda')

dx = 0.01
dt = 0.03*dx*dx
directory = './multirun/2025-12-12/17-00-49/0/'

h5_files = [file for file in os.listdir(directory) if file.endswith('.h5')]
h5_files.sort()

ncells = 4
yaml_string = os.path.join(directory,".hydra/config.yaml")

with open(yaml_string,"r") as yaml_file:
    data_yaml = yaml.safe_load(yaml_file)
    n_left    = data_yaml["n_left"]
    n_right   = data_yaml["n_right"]
    n_samples = data_yaml["n_samples"]
    n_steps   = data_yaml["n_steps"]

total_density_data = np.zeros((n_samples, n_steps+1, ncells))
total_flux_data_ptcl    = np.zeros((n_samples, n_steps, 1))
total_flux_data_mdl     = np.zeros((n_samples, n_steps, 1))

for ii, fl_nm in enumerate(h5_files):
    data_file = os.path.join(directory,fl_nm)
    with h5py.File(data_file, mode="r") as f:
        dt  = f['dt'][()]
        total_density_data[ii,:,:]   = f["density_data"][:]
        total_flux_data_ptcl[ii,:,:] = f["particle_flux_data"][:]
        total_flux_data_mdl[ii,:,:]  = f["model_flux_data"][:]

total_density_tnsr   = torch.tensor(total_density_data)
total_flux_tnsr_ptcl = torch.tensor(total_flux_data_ptcl)
total_flux_tnsr_mdl  = torch.tensor(total_flux_data_mdl)
# Get theoretical flux
total_flux_theory = torch.zeros_like(total_flux_tnsr_ptcl)
# Left and right number of particles
left_density  = total_density_tnsr[:, :-1, 1]
right_density = total_density_tnsr[:, :-1, 2]

total_flux_theory[:,:,0] = (0.5*dt/(dx*dx))*(left_density - right_density)
total_flux_theory[:,:,0] += ((dt**0.5)/(dx))*(0.5*(torch.sqrt(torch.clamp(left_density,min=0.)) +
                                                   torch.sqrt(torch.clamp(right_density,min=0.))
                                                   ))*torch.randn_like(left_density)

cumsum_flux_tnsr_ptcl = torch.cumsum(total_flux_tnsr_ptcl,dim=1)

cumsum_flux_tnsr_mdl  = torch.cumsum(total_flux_tnsr_mdl,dim=1)

cumsum_flux_tnsr_theory = torch.cumsum(total_flux_theory,dim=1)

cumsum_flux_np_ptcl = cumsum_flux_tnsr_ptcl.cpu().numpy()
cumsum_flux_np_mdl = cumsum_flux_tnsr_mdl.cpu().numpy()
cumsum_flux_np_theory = cumsum_flux_tnsr_theory.cpu().numpy()

# Save a single npz file in that directory
npz_file = "total_data"

np.savez(os.path.join(directory,npz_file),
         density_data = total_density_data,

         particle_flux_data = total_flux_data_ptcl,
         particle_cumsum_flux_data = cumsum_flux_np_ptcl,

         model_flux_data = total_flux_data_mdl,
         model_cumsum_flux_data = cumsum_flux_np_mdl,

         theory_flux_data = total_flux_theory.cpu().numpy(),
         theory_cumsum_flux_data = cumsum_flux_np_theory
        )
