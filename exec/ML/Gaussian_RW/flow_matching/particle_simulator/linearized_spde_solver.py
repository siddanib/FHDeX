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
            config_name="config_linear_spde")
def fhd_model_run (cfg):
    dataset_name = cfg.dataset.name
    assert dataset_name in ["uniform", "well", "reservoir"]
    print("Creating "+dataset_name+" dataset")

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
        left_boundary  = list(cfg.dataset.left_boundary)
        right_boundary = list(cfg.dataset.right_boundary)
        initial_pos = get_uni_initial_pos(ncells,par_per_cell,
                                          len_system=len_system)
    else:
        sys.exit(f"{dataset_name} dataset does not exist.")

    # Re-evaluate average particles per cell
    par_per_cell = torch.sum(get_density(cell_centers,initial_pos)).cpu().item()
    par_per_cell /= ncells

    # Prefactor that is used for noise in linearized SPDE
    constant_prefactor = torch.tensor(par_per_cell/dx)

    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    periodic_boundary = boundary_asserts(left_boundary, right_boundary)

    n_total_steps = cfg.n_total_particle_steps
    n_samples   = cfg.n_samples

    # gauss_data for SPDE, grnd_trth_data for particle simulator 
    gauss_data     = torch.zeros((n_samples,n_total_steps+1,ncells))
    grnd_trth_data  = torch.zeros((n_samples,n_total_steps+1,ncells))

    # Same initial condition for all realizations in an ensemble
    same_ic = cfg.same_ic
    n_thermalize_steps = cfg.n_thermalize_steps

    if same_ic:
        # Evolve the initial position to remove transients
        initial_pos = random_walk_just_evolve(ncells, n_thermalize_steps,
                                              dt, initial_pos.clone(),
                                              left_boundary, right_boundary,
                                              len_system = len_system)

    dens_old = torch.zeros((ncells,))
    dens_new = torch.zeros_like(dens_old)
    left_dens = torch.zeros_like(dens_old)
    right_dens = torch.zeros_like(dens_old)

    for i_ens in range(n_samples):
        if (not same_ic):
            # Evolve the initial position to remove transients
            initial_pos = random_walk_just_evolve(ncells, n_thermalize_steps,
                                                  dt, initial_pos.clone(),
                                                  left_boundary, right_boundary,
                                                  len_system = len_system)
        ######### SPDE ###################################################
        initial_density_spde = get_density(cell_centers,
                                      initial_pos.clone()).float()
        initial_density_spde /= dx
        gauss_data[i_ens,0, :] = initial_density_spde[:]
        dens_old[...] = initial_density_spde[...]
        ###################################################################
        ####### Particle simulation #######################################
        initial_density_ps = get_density(cell_centers, initial_pos.clone())
        grnd_trth_data[i_ens,0, :] = initial_density_ps[:]
        iter_pos = initial_pos.clone()
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
            # Noise corresponding to faces
            ##################################################################
            ### Linearized version uses spatial mean as factor for noise here
            lin_value = constant_prefactor*torch.ones_like(dens_old)
            flux_fluc_p = torch.sqrt(torch.clamp(lin_value,min=0.))
            flux_fluc_p *= np.sqrt(1/(dt*dx))

            flux_fluc_m = torch.sqrt(torch.clamp(lin_value,min=0.))
            flux_fluc_m *= np.sqrt(1/(dt*dx))
            ##################################################################
            if periodic_boundary:
                noise = torch.randn_like(dens_old)
                noise_periodic = torch.cat([noise, noise[0:1]])
                flux_fluc_p *= noise_periodic[1:]
                flux_fluc_m *= noise_periodic[:-1]
            else:
                noise = torch.randn((ncells+1,))
                flux_fluc_p *= noise[1:]
                flux_fluc_m *= noise[:-1]

            dens_new = (flux_mean_p+flux_fluc_p) - (flux_mean_m+flux_fluc_m)
            dens_new /= dx
            dens_new *= dt
            dens_new += dens_old
            gauss_data[i_ens,i_t,:] = dens_new
            dens_old[...] = dens_new[...]
            ###############################################################
            ###### Particle Simulation ###################################
            iter_pos, new_density, _ = random_walk_v2(ncells, 1, dt,
                                           iter_pos.clone(),
                                           left_boundary, right_boundary,
                                           len_system = len_system)

            grnd_trth_data[i_ens,i_t,:] =  new_density[:]
            ###############################################################

        if (i_ens % 10 == 0):
            print(f"{i_ens+1} realizations completed.")

    grnd_trth_np   = grnd_trth_data.cpu().numpy()
    gauss_data_np    = gauss_data.cpu().numpy()
    # Convert gauss data to Number of particles
    gauss_data_np *= dx

    save_file_name = os.path.join(HydraConfig.get().runtime.output_dir,
                                  dataset_name)

    with h5py.File(save_file_name+".h5", mode="w") as f:
        f.create_dataset("ground_truth_data", data=grnd_trth_np, dtype = np.float32)
        f.create_dataset("lin_gauss_data"  , data=gauss_data_np, dtype = np.float32)
        f.create_dataset("dt", data=dt, dtype=float)
        f.create_dataset("dx", data=dx, dtype=float)
        f.create_dataset("len_system", data=len_system, dtype=float)

if __name__ == "__main__":
    fhd_model_run()
