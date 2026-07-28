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

#### Only implementing potential of the form
#### V(x) = (x-alpha)^2 (x-beta)^2
def add_external_potential_flux(face_pos, dens_left, dens_right,
                                alpha, beta, gamma,
                                len_system, periodic_boundaries):
    dist_1 = face_pos - alpha
    dist_2 = face_pos - beta
    if periodic_boundaries:
        dist_1 = torch.where(dist_1 > 0.5*len_system,
                               dist_1-len_system, dist_1)
        dist_1 = torch.where(dist_1 < -0.5*len_system,
                               dist_1 + len_system, dist_1)

        dist_2 = torch.where(dist_2 > 0.5*len_system,
                               dist_2 - len_system, dist_2)
        dist_2 = torch.where(dist_2 < -0.5*len_system,
                               dist_2 + len_system, dist_2)

    flux_pot = 2.*dist_1*dist_2*dist_2
    flux_pot += 2.*dist_2*dist_1*dist_1
    flux_pot *= (dens_left+dens_right)/(2.*gamma)

    return flux_pot

def get_dk_flux (N_cell_tnsr, ncells, dx, dt,
                 left_boundary, right_boundary, nmoves,
                 add_potential=False, alpha=0.3, beta=0.7,
                 gamma=5.0e-4):
    len_system=dx*ncells
    assert len_system > alpha
    assert len_system > beta
    face_centers = torch.linspace(0.,len_system,ncells+1)
    periodic_boundary = boundary_asserts(left_boundary, right_boundary)

    n_total_steps = nmoves
    # gauss_data for SPDE
    nfaces = ncells + 1 ## Number of unique faces
    if periodic_boundary:
        nfaces -= 1
    gauss_data_flux = torch.zeros((nfaces,))
    g_dat_flux = torch.zeros_like(gauss_data_flux)

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

        flux_p = flux_mean_p+flux_fluc_p
        flux_m = flux_mean_m+flux_fluc_m
        #### External potential related ###########
        if add_potential:
            ##### Left faces ##############
            face_m = face_centers[:-1]
            flux_m += add_external_potential_flux(face_m, left_dens,
                                                  dens_old, alpha, beta,
                                                  gamma, len_system,
                                                  periodic_boundary)
            ###### Right faces ############
            face_p = face_centers[1:]
            flux_p += add_external_potential_flux(face_p, dens_old,
                                                  right_dens, alpha, beta,
                                                  gamma, len_system,
                                                  periodic_boundary)
        ###########################################
        dens_new = flux_p - flux_m
        dens_new /= dx
        dens_new *= dt
        dens_new += dens_old
        ################################################################
        dens_old[...] = dens_new[...]
        if periodic_boundary:
            g_dat_flux = -flux_m
        else:
            g_dat_flux[:-1] = -flux_m
            g_dat_flux[-1] = -flux_p[-1]
        g_dat_flux /= dx
        g_dat_flux *= dt
        gauss_data_flux += g_dat_flux
        ###############################################################
    # Final Number of Particles
    dens_new *= dx
    # Net particles that have crossed from "Left to Right"
    gauss_data_flux *= dx

    return dens_new, gauss_data_flux

def check_dk_flux (N_cell_tnsr, ncells, dx, dt,
                left_boundary, right_boundary, nmoves,
                   add_potential=False,alpha=0.3, beta=0.7,
                   gamma=5.0e-4):

    dens_new, gauss_data_flux = get_dk_flux (N_cell_tnsr, ncells, dx, dt,
                                    left_boundary, right_boundary, nmoves,
                                    add_potential=add_potential, alpha=alpha,
                                    beta=beta, gamma=gamma)

    # Cross verification
    dens_new_check = update_density(N_cell_tnsr, gauss_data_flux,
                                    left_boundary, right_boundary)

    print(dens_new, dens_new_check)

    assert torch.allclose(dens_new, dens_new_check)

if __name__ == "__main__":
    #torch.set_default_dtype(torch.float64)
    dx = 1.0/100
    ncells = 100
    dt = 0.03*dx*dx
    #left_boundary  = ["periodic", 0]
    #right_boundary = ["periodic", 0]
    left_boundary  = ["put", 0]
    right_boundary = ["put", 0]
    nmoves = 1 # Number of steps of size dt
    N_cell_tnsr = torch.randint(1, 20, (ncells,)).float()

    #check_dk_flux(N_cell_tnsr, ncells, dx, dt,
    #        left_boundary, right_boundary, nmoves,
    #        add_potential=False)

    check_dk_flux(N_cell_tnsr, ncells, dx, dt,
            left_boundary, right_boundary, nmoves,
            add_potential=True)
