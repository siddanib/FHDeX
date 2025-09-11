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

@hydra.main(version_base=None, config_path="./conf",
            config_name="config_two_cell")
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
    ptcls_at_cc = cfg.particles_at_cell_center
    if ptcls_at_cc == 0:
        print("All particles are initialized at cell centers")
        for itr in range(total_flux_data.size(0)):
            if N_left > 0:
                left_ptcls = torch.ones((N_left,))*0.5*dx
            else:
                left_ptcls = torch.empty((0,))
            if N_right > 0:
                right_ptcls = torch.ones((N_right))*0.5*dx + dx
            else:
                right_ptcls = torch.empty((0,))

            initial_pos = torch.cat((left_ptcls,right_ptcls))
            _ , _ , flux = random_walk_v2(ncells, 1, dt, initial_pos,
                                          left_boundary, right_boundary,
                                          len_system = len_system)
            total_flux_data[itr,0] = flux[1]
    elif ptcls_at_cc == 1:
        print("Particles are initialized uniformly in the cell.")
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
    elif ptcls_at_cc == 2:
        std_fctr = dx/6.21
        print("Particles are Normally distributed around cell center")
        for itr in range(total_flux_data.size(0)):
            if N_left > 0:
                left_ptcls = torch.randn((N_left))*std_fctr
                left_ptcls += 0.5*dx
                left_ptcls = torch.clamp(left_ptcls, min=0.,max=dx)
            else:
                left_ptcls = torch.empty((0,))
            if N_right > 0:
                right_ptcls = torch.randn((N_right))*std_fctr
                right_ptcls += 0.5*dx
                right_ptcls = torch.clamp(right_ptcls, min=0.,max=dx)
                right_ptcls += dx
            else:
                right_ptcls = torch.empty((0,))

            initial_pos = torch.cat((left_ptcls,right_ptcls))
            _ , _ , flux = random_walk_v2(ncells, 1, dt, initial_pos,
                                          left_boundary, right_boundary,
                                          len_system = len_system)
            total_flux_data[itr,0] = flux[1]
    else:
        print("Particles are Poisson uniformly initialized in the cell.")
        for itr in range(total_flux_data.size(0)):
            if N_left > 0:
                n_left_p = int(np.random.poisson(N_left,1))
                left_ptcls = torch.rand((n_left_p))*dx
            else:
                left_ptcls = torch.empty((0,))
            if N_right > 0:
                n_right_p = int(np.random.poisson(N_right,1))
                right_ptcls = torch.rand((n_right_p))*dx + dx
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

    dataset_name = os.path.join(HydraConfig.get().runtime.output_dir,
                                f"two_cells_{int(N_left)}_{int(N_right)}")

    with h5py.File(dataset_name+".h5", mode="w") as f:
        f.create_dataset("n_ptcl_data", data=n_ptcl_data, dtype = np.float32)
        f.create_dataset("flux_data"  , data=total_flux_data.cpu().numpy(),
                         dtype = np.float32)
        f.create_dataset("ncells", data=ncells, dtype = 'i')
        f.create_dataset("dt", data=dt, dtype=float)
        f.create_dataset("len_system", data=len_system, dtype=float)

if __name__ == "__main__":
    fhd_data_run()
