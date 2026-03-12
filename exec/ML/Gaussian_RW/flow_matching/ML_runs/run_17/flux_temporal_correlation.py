import sys
import os
import numpy as np
import torch
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})

torch.set_default_device('cuda')

parent_folder = "./outputs/2026-02-24/13-39-10/"

dir_list = [parent_folder,]

for directory in dir_list:
    npz_file = "total_data.npz"
    npz_file = os.path.join(directory,npz_file)
    aa = np.load(npz_file)

    ptcl_flux_data = aa["particle_flux_data"]
    mdl_flux_data  = aa["model_flux_data"]

ptcl_flux_data = torch.from_numpy(ptcl_flux_data)
mdl_flux_data = torch.from_numpy(mdl_flux_data)

n_step = ptcl_flux_data.size(1)

auto_corr = torch.zeros(n_step, 2)

auto_corr[0,:] = 1.

for i in range(1, n_step):
    #### Particle flux
    aa = ptcl_flux_data[:,:-i, :]
    bb = ptcl_flux_data[:,i:, :]
    aa_mn = torch.mean(aa)
    bb_mn = torch.mean(bb)
    auto_corr[i,0] = torch.mean((aa-aa_mn)*(bb-bb_mn))
    auto_corr[i,0] /= torch.std(aa)*torch.std(bb)

    #### Model flux
    aa = mdl_flux_data[:,:-i, :]
    bb = mdl_flux_data[:,i:, :]
    aa_mn = torch.mean(aa)
    bb_mn = torch.mean(bb)
    auto_corr[i,1] = torch.mean((aa-aa_mn)*(bb-bb_mn))
    auto_corr[i,1] /= torch.std(aa)*torch.std(bb)

print(auto_corr)
