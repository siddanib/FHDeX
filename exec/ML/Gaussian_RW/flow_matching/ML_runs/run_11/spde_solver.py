import sys
import os
import numpy as np
import math
import torch
#######################################################
####### Local imports ################################
from random_walkers_pytorch import update_density
from random_walkers_pytorch import boundary_asserts
#######################################################

torch.set_default_device('cpu')

def get_dk_flux (N_cell_tnsr, ncells, dx, dt,
            left_boundary, right_boundary, nmoves):
    len_system=dx*ncells
    cell_centers_dx = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    periodic_boundary = boundary_asserts(left_boundary, right_boundary)

    n_total_steps = nmoves
    # gauss_data for SPDE 
    gauss_data_flux = torch.zeros((ncells,))

    dens_old = torch.zeros((ncells,))
    dens_new = torch.zeros_like(dens_old)
    left_dens = torch.zeros_like(dens_old)
    right_dens = torch.zeros_like(dens_old)

    ######### SPDE ###################################################
    initial_density_spde = N_cell_tnsr.clone() 
    initial_density_spde /= dx
    ###################################################################
    dens_old[...] = initial_density_spde[...]
    ###################################################################
    for i_t in range(nmoves):
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
        flux_fluc_p = 0.5*(torch.sqrt(torch.clamp(right_dens,min=0.))+
                           torch.sqrt(torch.clamp(dens_old,min=0.)))
        flux_fluc_p *= np.sqrt(1/(dt*dx))

        flux_fluc_m = 0.5*(torch.sqrt(torch.clamp(dens_old,min=0.))+
                           torch.sqrt(torch.clamp(left_dens,min=0.)))
        flux_fluc_m *= np.sqrt(1/(dt*dx))

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
        ################################################################
        dens_old[...] = dens_new[...]
        g_dat_flux = -(flux_mean_m+flux_fluc_m)
        g_dat_flux /= dx
        g_dat_flux *= dt
        gauss_data_flux += g_dat_flux
        ###############################################################

    # Final Number of Particles
    dens_new *= dx
    # Net particles that have crossed from "Left to Right"
    gauss_data_flux *= dx

    return gauss_data_flux, dens_new

def check_dk_flux (N_cell_tnsr, ncells, dx, dt,
                left_boundary, right_boundary, nmoves):

    gauss_data_flux, dens_new = get_dk_flux (N_cell_tnsr, ncells, dx, dt,
                                    left_boundary, right_boundary, nmoves)

    # Cross verification
    dens_new_check = update_density(N_cell_tnsr, gauss_data_flux,
                                    left_boundary, right_boundary)

    print(dens_new, dens_new_check)

    assert torch.allclose(dens_new, dens_new_check)

if __name__ == "__main__":
    dx = 1.0/100
    ncells = 10
    dt = 0.03*dx*dx
    left_boundary  = ["periodic", 0]
    right_boundary = ["periodic", 0]
    nmoves = 45 # Number of steps of size dt
    N_cell_tnsr = torch.randint(1, 20, (ncells,)).float()

    check_dk_flux(N_cell_tnsr, ncells, dx, dt,
            left_boundary, right_boundary, nmoves)
