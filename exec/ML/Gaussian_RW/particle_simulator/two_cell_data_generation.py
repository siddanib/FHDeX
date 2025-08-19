import sys
import os
import numpy as np
import torch
import h5py
from datetime import datetime
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
#######################################################
####### Local imports ################################
from random_walkers_pytorch import random_walk_v2
#######################################################

torch.set_default_device('cpu')

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def fhd_data_run (cfg):
    N_left  = cfg.n_left
    N_right = cfg.n_right
    dx = 1.0/100
    ncells = 2
    len_system=ncells*dx
    dt = 0.03*dx*dx
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    left_boundary  = ["periodic", 0]
    right_boundary = ["periodic", 0]
    
    # Generate the entire data first
    n_samples = cfg.n_samples
    total_flux_data = torch.zeros((n_samples,1))
    for itr in range(total_flux_data.size(0)):
        if N_left > 0:
            left_ptcls = torch.rand((N_left))*dx
        else:
            left_ptcls = torch.empty((0,))
        if N_right > 0:
            right_ptcls = torch.rand((N_right))*dx + dx
        else:
            right_ptcls = torch.empty((0,))

        initial_pos = torch.cat((left_ptcls,right_ptcls))
        _ , _ , flux = random_walk_v2(ncells, 1, dt, initial_pos,
                                      left_boundary, right_boundary,
                                      len_system = len_system)
        total_flux_data[itr,0] = flux[1]

    n_ptcl_data = np.zeros((n_samples,2))
    n_ptcl_data[:,0] = N_left
    n_ptcl_data[:,1] = N_right
    dataset_name = f"two_cells_{int(N_left)}_{int(N_right)}"
    with h5py.File(dataset_name+".h5", mode="w") as f:
        f.create_dataset("n_ptcl_data", data=n_ptcl_data, dtype = np.float32)
        f.create_dataset("flux_data"  , data=total_flux_data.cpu().numpy(), dtype = np.float32)
        f.create_dataset("ncells", data=ncells, dtype = 'i')
        f.create_dataset("dt", data=dt, dtype=float)
        f.create_dataset("len_system", data=len_system, dtype=float)

if __name__ == "__main__":
    fhd_data_run()
