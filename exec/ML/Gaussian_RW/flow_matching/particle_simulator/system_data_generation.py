import os
import sys
import numpy as np
import torch
import h5py
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
#######################################################
####### Local imports ################################
from random_walkers_pytorch import get_uni_initial_pos, get_density
from random_walkers_pytorch import get_well_initial_pos
from random_walkers_pytorch import random_walk_v2, boundary_asserts
from random_walkers_pytorch import random_walk_just_evolve
#######################################################

torch.set_default_device('cpu')

@hydra.main(version_base=None, config_path="./conf",
            config_name="config_system")
def fhd_data_run (cfg):
    dataset_name = cfg.dataset_name

    assert dataset_name in ["uniform", "well", "reservoir"]

    print("Creating "+dataset_name+" dataset")

    dataset_dct = {"uniform"      : [20000, 1000],
                   "well"         : [20000, 1000],
                   "reservoir"    : [1, 1000000]}

    par_per_cell = cfg.par_per_cell
    ncells = cfg.ncells
    num_par = par_per_cell*ncells
    len_system=1.0
    dx = len_system/ncells
    dt = cfg.cfl*dx*dx # ensure <= 0.1*dx*dx
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)

    if dataset_name == "uniform":
        left_boundary  = ["periodic", 0]
        right_boundary = ["periodic", 0]
        initial_pos = get_uni_initial_pos(ncells,par_per_cell,
                                          len_system=len_system)
    elif dataset_name == "well":
        left_boundary  =["periodic", 0]
        right_boundary = ["periodic", 0]
        # x_1 and x_2 represent void region
        initial_pos = get_well_initial_pos(ncells, par_per_cell,
                                           x_1=0.25, x_2=0.75,
                                           len_system=len_system)
    elif dataset_name == "reservoir":
        left_boundary  = list(cfg.left_boundary)
        right_boundary = list(cfg.right_boundary)
        initial_pos = get_uni_initial_pos(ncells,par_per_cell,
                                          len_system=len_system)
    else:
        sys.exit(f"{dataset_name} dataset does not exist.")

    # Re-evaluate average particles per cell
    par_per_cell = torch.sum(get_density(cell_centers,initial_pos)).cpu().item()
    par_per_cell /= ncells

    n_initial_cond = cfg.n_ensembles
    n_temp_steps   = cfg.n_steps

    # Saving
    n_ptcl_data = np.zeros((n_initial_cond, n_temp_steps+1, ncells))

    if boundary_asserts(left_boundary, right_boundary):
        flux_data   = np.zeros((n_initial_cond, n_temp_steps  , ncells))
    else:
        flux_data   = np.zeros((n_initial_cond, n_temp_steps  , ncells+1))

    # Evolve the initial position to remove transients
    initial_pos = random_walk_just_evolve(ncells, cfg.n_thermalize_steps,
                                          dt, initial_pos.clone(),
                                          left_boundary, right_boundary,
                                          len_system = len_system)

    for i in range(n_initial_cond):
        # Get the first density
        n_ptcl_data[i,0,:] = get_density(cell_centers,initial_pos).cpu().numpy()
        iter_pos = initial_pos.clone()
        for j in range(1,n_temp_steps+1):
            # Evolve the initial condition by 1 step
            iter_pos, new_density, flux = random_walk_v2(ncells, 1, dt,
                                                        iter_pos.clone(),
                                                        left_boundary,
                                                        right_boundary,
                                                        len_system = len_system)
            # Position already updated;
            # Saved new density and flux over time step
            n_ptcl_data[i,j,:] = new_density.cpu().numpy()
            flux_data[i,j-1,:] = flux.cpu().numpy()

        if (i % 10):
            print(f"Completed {i} ensemble runs.")

    save_file_name = os.path.join(HydraConfig.get().runtime.output_dir,
                                  dataset_name)

    with h5py.File(save_file_name+".h5", mode="w") as f:
        f.create_dataset("n_ptcl_data", data=n_ptcl_data, dtype = np.float32)
        f.create_dataset("flux_data"  , data=flux_data  , dtype = np.float32)
        f.create_dataset("ppc", data=par_per_cell, dtype = float)
        f.create_dataset("ncells", data=ncells, dtype = 'i')
        f.create_dataset("dt", data=dt, dtype=float)
        f.create_dataset("len_system", data=len_system, dtype=float)

if __name__ == "__main__":
    fhd_data_run()
