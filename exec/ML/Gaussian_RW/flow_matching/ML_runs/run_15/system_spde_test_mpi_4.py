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
############# For MPI ##################
from mpi4py import MPI
#####################################################
#######################################################
####### Local imports ################################
from random_walkers_pytorch import random_walk_v2
from random_walkers_pytorch import random_walk_just_evolve
from random_walkers_pytorch import get_uni_initial_pos
from random_walkers_pytorch import get_well_initial_pos
from random_walkers_pytorch import get_nonzero_well_initial_pos
from random_walkers_pytorch import get_density
from spde_solver import get_dk_flux
#######################################################

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')

@torch.no_grad()
def realization_process (itr, cfg, output_dir):
    # Setting num_threads inside subprocess function seems
    # vital for proper scaling
    torch.set_num_threads(1)

    dx = 1.0/100
    ncells = cfg.ncells
    len_system=ncells*dx
    dt = 0.03*dx*dx
    left_boundary  = ["periodic", 0]
    right_boundary = ["periodic", 0]
    n_avg = cfg.n_avg
    ################################################################
    n_steps   = cfg.n_steps
    nmoves = cfg.nmoves # Number of steps of size dt
    ####################################################################
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    if cfg.ic_type == "uniform":
        initial_pos = get_uni_initial_pos(ncells, n_avg, len_system)
    elif cfg.ic_type == "well":
        initial_pos = get_well_initial_pos(ncells, n_avg, 0.25, 0.75,
                                           len_system)
    elif cfg.ic_type == "nonzero_well":
        initial_pos = get_nonzero_well_initial_pos(ncells, cfg.N_low,
                                                   cfg.N_high, 0.25, 0.75,
                                                   len_system)
    else:
        sys.exit("Unknown ic_type")
    ### Thermalize the system ####################################
    if cfg.n_thermal_steps > 0:
        initial_pos = random_walk_just_evolve(ncells, cfg.n_thermal_steps,
                                              dt, initial_pos.clone(),
                                              left_boundary, right_boundary,
                                              len_system = len_system)
    ##############################################################
    density = get_density(cell_centers, initial_pos.clone())

    ptcl_density_data = torch.zeros((n_steps+1, ncells))
    ptcl_density_data[0, :] = density[:]
    ptcl_flux_data = torch.zeros((n_steps, ncells))

    dk_density_data = torch.zeros_like(ptcl_density_data)
    dk_density_data[0, :] = density[:]
    dk_flux_data = torch.zeros_like(ptcl_flux_data)

    for i_step in range(n_steps):
        initial_pos, density, flux_ptcl = random_walk_v2(ncells, nmoves, dt,
                                            initial_pos.clone(),
                                            left_boundary, right_boundary,
                                            len_system = len_system)

        ptcl_density_data[i_step+1, :] = density[:]
        ptcl_flux_data[i_step,:] = flux_ptcl[:]
        ########################################################
        old_dens_dk = dk_density_data[i_step,:]
        new_dens_dk, flux_dk = get_dk_flux(old_dens_dk, ncells, dx, dt,
                                           left_boundary, right_boundary,
                                           nmoves)
        dk_flux_data[i_step,:] = flux_dk[:]
        dk_density_data[i_step+1,:] = new_dens_dk[:]

    dataset_name = os.path.join(output_dir,"system_temporal")

    with h5py.File(dataset_name+f"_{itr}"+".h5", mode="w") as f:
        f.create_dataset("ptcl_density_data"  , data=ptcl_density_data.cpu().numpy(),
                         dtype = np.float32)
        f.create_dataset("dk_density_data"  , data=dk_density_data.cpu().numpy(),
                         dtype = np.float32)
        f.create_dataset("ptcl_flux_data"  , data=ptcl_flux_data.cpu().numpy(),
                         dtype = np.float32)
        f.create_dataset("dk_flux_data"  , data=dk_flux_data.cpu().numpy(),
                         dtype = np.float32)
        f.create_dataset("ncells", data=ncells, dtype = 'i')
        f.create_dataset("N_avg", data=n_avg, dtype = 'i')
        f.create_dataset("dt", data=dt, dtype=float)
        f.create_dataset("len_system", data=len_system, dtype=float)

    return None

@hydra.main(version_base=None, config_path="./conf",
            config_name="config_system_test_mpi_3")
def fhd_data_run (cfg):
    # Get the global communicator
    app_comm = MPI.COMM_WORLD
    # Get the total number of processes
    app_size = app_comm.Get_size()
    # Get the unique rank of the current process
    app_rank = app_comm.Get_rank()

    if cfg.device == "cpu":
        torch.set_default_device('cpu')
        device = torch.device("cpu")
    else:
        device_id = app_rank %4
        torch.set_default_device(f"cuda:{device_id}")
        device = torch.device(f"cuda:{device_id}")

    n_samples = cfg.n_samples
    ### Get the output_dir location
    output_dir = None
    if app_rank == 0:
        output_dir = HydraConfig.get().runtime.output_dir
    output_dir = app_comm.bcast(output_dir,root=0)
    ###########################################################
    for ii in range(n_samples):
        itr = ii + app_rank*n_samples
        realization_process(itr, cfg, output_dir)

    app_comm.Barrier()

if __name__ == "__main__":
    fhd_data_run()
