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
from random_walkers_pytorch import random_walk_just_evolve
from random_walkers_pytorch import get_particle_positions
from random_walkers_pytorch import get_density
#######################################################

torch.set_default_device('cpu')

@hydra.main(version_base=None, config_path="./conf",
            config_name="config_two_cell_temporal")
def fhd_data_run (cfg):
    # THIS CODE IS ONLY for RESERVOIR BOUNDARIES
    N_left  = int(cfg.n_left)
    N_right = int(cfg.n_right)
    dx = 1.0/100
    # The boundary cells are treated as reservoir
    ncells = 4
    len_system=ncells*dx
    dt = 0.03*dx*dx
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    left_boundary  = ["put", N_left]
    right_boundary = ["put", N_right]

    # Generate the entire data first
    n_samples = cfg.n_samples
    n_steps   = cfg.n_steps
    total_density_data = torch.zeros(n_samples,n_steps+1, ncells)
    total_flux_data = torch.zeros((n_samples,n_steps,1))
    for itr in range(total_flux_data.size(0)):
        ic_density = torch.zeros((ncells,))
        ic_density[0] = N_left
        ic_density[-1] = N_right
        ic_density[1] = int((N_right-N_left)/(ncells-1)) + N_left
        ic_density[2] = int(2.0*(N_right-N_left)/(ncells-1)) + N_left

        initial_pos = get_particle_positions(ic_density, dx)

        initial_pos = random_walk_just_evolve(ncells, 10000, dt,
                                          initial_pos.clone(),
                                          left_boundary, right_boundary,
                                          len_system = len_system)

        density = get_density(cell_centers, initial_pos.clone())
        total_density_data[itr, 0, :] = density[:]

        for i_step in range(n_steps):
            initial_pos, density, flux = random_walk_v2(ncells, 1, dt,
                                          initial_pos.clone(),
                                          left_boundary, right_boundary,
                                          len_system = len_system)

            total_density_data[itr, i_step+1, :] = density[:]
            total_flux_data[itr,i_step,0] = flux[2]

    # Note that you need to sum the fluxes across all previous steps
    torch_flux_data = torch.cumsum(total_flux_data,dim=-1)

    dataset_name = os.path.join(HydraConfig.get().runtime.output_dir,
                                f"two_cells_temporal_{int(N_left)}_{int(N_right)}")

    with h5py.File(dataset_name+".h5", mode="w") as f:
        f.create_dataset("density_data"  , data=total_density_data.cpu().numpy(),
                         dtype = np.float32)
        f.create_dataset("flux_data"  , data=total_flux_data.cpu().numpy(),
                         dtype = np.float32)
        f.create_dataset("ncells", data=ncells, dtype = 'i')
        f.create_dataset("N_left", data=N_left, dtype = 'i')
        f.create_dataset("N_right", data=N_right, dtype = 'i')
        f.create_dataset("dt", data=dt, dtype=float)
        f.create_dataset("len_system", data=len_system, dtype=float)

if __name__ == "__main__":
    fhd_data_run()
