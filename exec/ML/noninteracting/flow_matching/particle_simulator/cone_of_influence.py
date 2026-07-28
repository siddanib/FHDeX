import sys
import os
import numpy as np
import math
import torch
import h5py
from datetime import datetime
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#######################################################
####### Local imports ################################
from random_walkers_pytorch import random_walk_v2
from random_walkers_pytorch import random_walk_just_evolve
from random_walkers_pytorch import get_particle_positions
from random_walkers_pytorch import get_well_initial_pos
from random_walkers_pytorch import get_uni_initial_pos
from random_walkers_pytorch import get_density
from random_walkers_pytorch import boundary_asserts
#######################################################

torch.set_default_device('cpu')

@torch.no_grad()
@hydra.main(version_base=None, config_path="./conf",
            config_name="config_cone")
def fhd_model_run (cfg):

    left_boundary  = ["periodic", 0]
    right_boundary = ["periodic", 0]

    periodic_boundary = boundary_asserts(left_boundary, right_boundary)

    n_total_steps = int(cfg.n_time_steps)

    ncells = 2*n_total_steps + 5 # some additional cells for buffer

    mid_cell_id = int(np.floor(ncells/2))

    dx = cfg.dx

    len_system = ncells*dx

    dt = cfg.cfl*dx*dx # ensure <= 0.03*dx*dx

    # gauss_data for PDE
    gauss_data     = torch.zeros((n_total_steps+1,ncells))

    dens_old = torch.zeros((ncells,))
    dens_new = torch.zeros_like(dens_old)
    left_dens = torch.zeros_like(dens_old)
    right_dens = torch.zeros_like(dens_old)

    ######### SPDE ###################################################
    # Put one particle in the middle of the domain
    dens_old[mid_cell_id] =  1.
    dens_old /= dx
    ###################################################################
    gauss_data[0, :] = dens_old[...]
    ###################################################################
    for i_t in range(1,n_total_steps+1):
        #### SPDE #########################
        right_dens[:-1] = dens_old[1:]
        ##### Boundary effects ###################
        if periodic_boundary:
            right_dens[-1] = dens_old[0]
        else:
            if right_boundary[1] > 0:
                right_dens[-1] = (np.random.poisson(
                                     float(right_boundary[1])))/dx
            else:
                right_dens[-1] = 0.
        #############################################
        left_dens[1:] = dens_old[:-1]
        ################ Periodic effects ##############
        if periodic_boundary:
            left_dens[0]  = dens_old[-1]
        else:
            if left_boundary[1] > 0:
                left_dens[0] = (np.random.poisson(
                                   float(left_boundary[1])))/dx
            else:
                left_dens[0] = 0.
        ##################################################
        flux_mean_p = (0.5/dx)*(right_dens-dens_old)
        flux_mean_m = (0.5/dx)*(dens_old-left_dens)

        dens_new = flux_mean_p - flux_mean_m
        dens_new /= dx
        dens_new *= dt
        dens_new += dens_old
        ################################################################
        gauss_data[i_t,:] = dens_new[...]
        ################################################################
        dens_old[...] = dens_new[...]
        ###############################################################

    gauss_data_np    = gauss_data.cpu().numpy()
    # Convert gauss data to Number of particles
    gauss_data_np *= dx

    cut_off = 0.01*0.5*cfg.cfl
    gauss_data_np[gauss_data_np < cut_off] = np.nan

    step_id = 30
    array_step = gauss_data_np[step_id,:]
    count = np.count_nonzero(~np.isnan(array_step))
    print(count)

    plt.rcParams.update({'font.size': 25})
    fig, ax = plt.subplots(figsize=(10, 10))
    # Display the image with a logarithmic color scale
    cax = ax.imshow(gauss_data_np, norm=LogNorm(), cmap='viridis')
    # Add a colorbar
    cbar = plt.colorbar(cax)
    # Optional: Customize the colorbar
    cbar.set_label('Log Scale Colorbar')
    ax.set_xlabel("Cell ID")
    ax.set_ylabel("Time step ID")
    ax.set_title(f"Cone of influence \n cutoff value = {cut_off:.3e}")
    plt.tight_layout()
    plt.savefig('cone_of_influence.jpeg', bbox_inches='tight')
    plt.show()

    #save_file_name = os.path.join(HydraConfig.get().runtime.output_dir,
    #                              "cone_data")

    #with h5py.File(save_file_name+".h5", mode="w") as f:
    #    f.create_dataset("gauss_data"  , data=gauss_data_np, dtype = np.float32)
    #    f.create_dataset("dt", data=dt, dtype=float)
    #    f.create_dataset("dx", data=dx, dtype=float)
    #    f.create_dataset("len_system", data=len_system, dtype=float)

if __name__ == "__main__":
    fhd_model_run()
