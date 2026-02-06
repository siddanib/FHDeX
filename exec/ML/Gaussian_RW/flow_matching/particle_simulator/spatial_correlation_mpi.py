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
from hydra.core.hydra_config import HydraConfig
#####################################################
############# For MPI ##################
from mpi4py import MPI
#####################################################
#######################################################
####### Local imports ################################
from random_walkers_pytorch import random_walk_v2
from random_walkers_pytorch import random_walk_just_evolve
from random_walkers_pytorch import get_particle_positions
#######################################################

torch.set_default_device('cpu')

def realization_process (itr, cfg):
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
    nmoves = cfg.nmoves
    n_steps = cfg.n_steps
    n_thermal_steps = cfg.n_thermal_steps

    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)

    # Initializing with a Poisson distribution
    N_init_dist = torch.ones(ncells)*n_avg
    N_init_dist = torch.poisson(N_init_dist)
    initial_pos = get_particle_positions(N_init_dist, dx)

    if n_thermal_steps > 0:
        initial_pos = random_walk_just_evolve(ncells, n_thermal_steps,
                                      dt, initial_pos.clone(),
                                      left_boundary, right_boundary,
                                      len_system = len_system)

    mean_x = torch.zeros(1)
    mean_xy = torch.zeros(ncells)

    for i_step in range(n_steps):
        initial_pos, _, flux = random_walk_v2(ncells, nmoves, dt,
                                      initial_pos.clone(),
                                      left_boundary, right_boundary,
                                      len_system = len_system)
        mean_x += torch.mean(flux)
        mean_xy[0] += torch.mean(flux*flux)
        for id_y in range(1,ncells):
            # The imposed logic is only for periodic boundaries
            aa = flux.clone()
            bb_0, bb_1 = torch.split(flux.clone(),[id_y, ncells-id_y])
            # Concatenate the splits in opposite order
            bb = torch.cat((bb_1, bb_0))
            mean_xy[id_y] = torch.mean(aa*bb)

    # Need to divide by n_steps
    mean_x /= n_steps
    mean_xy /= n_steps

    return mean_x, mean_xy

@hydra.main(version_base=None, config_path="./conf",
            config_name="config_spatial_corr_mpi")
def fhd_data_run (cfg):
    # Get the global communicator
    app_comm = MPI.COMM_WORLD
    # Get the total number of processes
    app_size = app_comm.Get_size()
    # Get the unique rank of the current process
    app_rank = app_comm.Get_rank()
    n_samples = cfg.n_samples
    ncells = cfg.ncells
    n_avg = cfg.n_avg

    mean_x_np = np.zeros(1)
    mean_xy_np = np.zeros(ncells)

    for ii in range(n_samples):
        itr = ii + app_rank*n_samples
        mean_x_tnsr, mean_xy_tnsr = realization_process(itr, cfg)
        mean_x_np += mean_x_tnsr.numpy()
        mean_xy_np += mean_xy_tnsr.numpy()
    # Need to divide these by n_samples
    mean_x_np /= n_samples
    mean_xy_np /=  n_samples

    recvbuf_x = None
    recvbuf_xy = None
    if app_rank == 0:
        recvbuf_x = np.empty((app_size,1),dtype=mean_x_np.dtype)
        recvbuf_xy = np.empty((app_size,ncells),dtype=mean_x_np.dtype)

    app_comm.Gather(mean_x_np, recvbuf_x, root=0)
    app_comm.Gather(mean_xy_np, recvbuf_xy, root=0)

    if app_rank == 0:
        mean_x_root  = np.mean(recvbuf_x,axis=0)
        mean_xy_root = np.mean(recvbuf_xy,axis=0)

        corr_xy = mean_xy_root - mean_x_root*mean_x_root
        corr_xy /= (corr_xy[0])

        print(corr_xy)

        dataset_name = os.path.join(HydraConfig.get().runtime.output_dir,
                                    "spatial_correlation")

        with h5py.File(dataset_name+".h5", mode="w") as f:
            f.create_dataset("spatial_correlation_data", data=corr_xy,
                             dtype = np.float32)
            f.create_dataset("ncells", data=ncells, dtype = 'i')
            f.create_dataset("N_avg", data=n_avg, dtype = 'i')

if __name__ == "__main__":
    fhd_data_run()
