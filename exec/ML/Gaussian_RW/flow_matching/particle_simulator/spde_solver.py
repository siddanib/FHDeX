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
from .random_walkers_pytorch import random_walk_v2
from .random_walkers_pytorch import get_particle_positions
from .random_walkers_pytorch import get_well_initial_pos
from .random_walkers_pytorch import get_density
#######################################################

torch.set_default_device('cuda')

@torch.no_grad()
@hydra.main(version_base=None, config_path="./conf",
            config_name="config_spde")
def fhd_model_run (cfg):
    dx = 1.0/100
    ncells = cfg.n_cells
    par_per_cell = cfg.par_per_cell
    len_system=ncells*dx
    dt = 0.03*dx*dx
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    left_boundary  = ["periodic", 0]
    right_boundary = ["periodic", 0]

    n_total_steps = cfg.n_total_particle_steps
    n_samples   = cfg.n_samples

    initial_pos = get_well_initial_pos(ncells, par_per_cell,
                                       x_1=0.25, x_2=0.75,
                                       len_system=len_system)

    initial_density = get_density(cell_centers,
                                  initial_pos.clone()).float()
    initial_density /= dx
    gauss_data     = torch.zeros((n_samples,n_total_steps+1,ncells))
    gauss_data[:,0, :] = initial_density.unsqueeze(0).repeat(n_samples,1)

    dens_old = initial_density.clone()
    dens_new = torch.zeros_like(dens_old)
    left_dens = torch.zeros_like(dens_old)
    right_dens = torch.zeros_like(dens_old)

    for i_ens in range(n_samples):
        dens_old[...] = initial_density[...]
        for i_t in range(1,n_total_steps+1):
            right_dens[:-1] = dens_old[1:]
            right_dens[-1] = dens_old[0]

            left_dens[1:] = dens_old[:-1]
            left_dens[0]  = dens_old[-1]

            flux_mean_p = (0.5/dx)*(right_dens-dens_old)
            flux_mean_m = (0.5/dx)*(dens_old-left_dens)

            # Noise corresponding to faces
            noise = torch.randn_like(dens_old)
            noise_periodic = torch.cat([noise, noise[0:1]])

            flux_fluc_p = 0.5*(torch.sqrt(torch.clamp(right_dens,min=0.))+
                               torch.sqrt(torch.clamp(dens_old,min=0.)))
            flux_fluc_p *= np.sqrt(1/(dt*dx))
            flux_fluc_p *= noise_periodic[1:]

            flux_fluc_m = 0.5*(torch.sqrt(torch.clamp(dens_old,min=0.))+
                               torch.sqrt(torch.clamp(left_dens,min=0.)))
            flux_fluc_m *= np.sqrt(1/(dt*dx))
            flux_fluc_m *= noise_periodic[:-1]

            dens_new = (flux_mean_p+flux_fluc_p) - (flux_mean_m+flux_fluc_m)
            dens_new /= dx
            dens_new *= dt
            dens_new += dens_old
            gauss_data[i_ens,i_t,:] = dens_new
            dens_old[...] = dens_new[...]

    # Particle-based solver
    initial_density = get_density(cell_centers, initial_pos.clone())
    grnd_trth_data  = torch.zeros((n_samples,n_total_steps+1,ncells))
    grnd_trth_data[:,0, :] = initial_density.unsqueeze(0).repeat(n_samples,1)
    for iter_val in range(n_samples):
        iter_pos = initial_pos.clone()
        for i_t in range(n_total_steps):
            iter_pos, new_density, _ = random_walk_v2(ncells, 1, dt,
                                           iter_pos.clone(),
                                           left_boundary, right_boundary,
                                           len_system = len_system)
            grnd_trth_data[iter_val,i_t+1,:] =  new_density[...]

    grnd_trth_np   = grnd_trth_data.cpu().numpy()
    gauss_data_np    = gauss_data.cpu().numpy()
    # Convert gauss data to Number of particles
    gauss_data_np *= dx

    dataset_name = os.path.join(HydraConfig.get().runtime.output_dir,
                                "ensembles_of_multi_steps")

    with h5py.File(dataset_name+".h5", mode="w") as f:
        f.create_dataset("ground_truth_data", data=grnd_trth_np, dtype = np.float32)
        f.create_dataset("gauss_data"  , data=gauss_data_np, dtype = np.float32)
        f.create_dataset("dt", data=dt, dtype=float)
        f.create_dataset("dx", data=dx, dtype=float)
        f.create_dataset("len_system", data=len_system, dtype=float)

if __name__ == "__main__":
    fhd_model_run()
