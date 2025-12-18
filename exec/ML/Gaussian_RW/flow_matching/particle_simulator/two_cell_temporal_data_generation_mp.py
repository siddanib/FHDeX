import sys
import os
import numpy as np
import torch
import h5py
import time
from datetime import datetime
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
#####################################################
############# For multiprocessing ##################
import torch.multiprocessing as mp
from functools import partial
#####################################################
#######################################################
####### Local imports ################################
from random_walkers_pytorch import random_walk_v2
from random_walkers_pytorch import random_walk_just_evolve
from random_walkers_pytorch import get_particle_positions
from random_walkers_pytorch import get_density
#######################################################

torch.set_default_device('cpu')

def realization_process (itr, n_steps, N_left, N_right, ncells,
                         dx, dt, left_boundary, right_boundary,
                         len_system, dataset_name):

    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    ic_density = torch.zeros((ncells,))
    ic_density[0] = N_left
    ic_density[-1] = N_right
    ic_density[1] = int((N_right-N_left)/(ncells-1)) + N_left
    ic_density[2] = int(2.0*(N_right-N_left)/(ncells-1)) + N_left

    initial_pos = get_particle_positions(ic_density, dx)

    density = get_density(cell_centers, initial_pos.clone())
    itr_density_data = torch.zeros((n_steps+1, ncells))
    itr_density_data[0, :] = density[:]

    itr_flux_data = torch.zeros((n_steps, 1))

    for i_step in range(n_steps):
        initial_pos, density, flux = random_walk_v2(ncells, 1, dt,
                                      initial_pos.clone(),
                                      left_boundary, right_boundary,
                                      len_system = len_system)

        itr_density_data[i_step+1, :] = density[:]
        itr_flux_data[i_step,0] = flux[2]

    with h5py.File(dataset_name+f"_{itr}"+".h5", mode="w") as f:
        f.create_dataset("density_data",
                         data=itr_density_data.cpu().numpy(),
                         dtype = np.float32)
        f.create_dataset("particle_flux_data",
                         data=itr_flux_data.cpu().numpy(),
                         dtype = np.float32)
        f.create_dataset("ncells", data=ncells, dtype = 'i')
        f.create_dataset("N_left", data=N_left, dtype = 'i')
        f.create_dataset("N_right", data=N_right, dtype = 'i')
        f.create_dataset("dt", data=dt, dtype=float)
        f.create_dataset("len_system", data=len_system, dtype=float)

    return None

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
    left_boundary  = ["put", N_left]
    right_boundary = ["put", N_right]

    # Generate the entire data first
    n_samples = cfg.n_samples
    n_steps   = cfg.n_steps

    # Just leaving a few processors just in case
    num_processes = int(0.5*mp.cpu_count()) - 3

    dataset_name = os.path.join(HydraConfig.get().runtime.output_dir,
                                f"two_cells_temporal_{int(N_left)}_{int(N_right)}")

    with torch.multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(partial(realization_process,
                            n_steps = n_steps, N_left = N_left,
                            N_right = N_right, ncells = ncells,
                            dx = dx, dt = dt,
                            left_boundary = left_boundary,
                            right_boundary = right_boundary,
                            len_system = len_system,
                            dataset_name = dataset_name),
                           range(n_samples))

if __name__ == "__main__":
    start_time = time.time()
    fhd_data_run()
    end_time = time.time()
    print(f"Elapsed time : {(end_time - start_time):6f}")
