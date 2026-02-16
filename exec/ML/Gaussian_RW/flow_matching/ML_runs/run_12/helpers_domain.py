import sys
import os
import numpy as np
import math
import torch
import numpy as np
####### Local imports ################################
from random_walkers_pytorch import get_uni_initial_pos
from random_walkers_pytorch import get_density
from random_walkers_pytorch import random_walk_v2
from random_walkers_pytorch import random_walk_just_evolve
from random_walkers_pytorch import get_particle_positions
from spde_solver import get_dk_flux
#######################################################
"""
This function creates (input, output) pairs for faces
"""
@torch.no_grad()
def get_particle_data(batch_size, ppc_min, ppc_max,
                      ncells, dx, dt, left_boundary,
                      right_boundary, nmoves):
    len_system = ncells*dx
    cell_centers = torch.linspace(0.5*dx,(ncells-0.5)*dx,ncells)
    # total data
    input_batch =  torch.zeros((batch_size, 1, ncells))
    output_batch = torch.zeros_like(input_batch)
    x_0 =  torch.zeros_like(input_batch)
    for ii in range(batch_size):
        # Create the mini-batch
        par_per_cell = np.random.randint(ppc_min,ppc_max)
        # Create a uniform system and thermalize for few steps
        # Sample from a Poisson distribution
        N_init_dist = torch.ones(batch_size)*par_per_cell
        N_init_dist = torch.poisson(N_init_dist)
        init_pos = get_particle_positions(N_init_dist, dx)
        init_pos = random_walk_just_evolve(ncells, 5, dt,
                                           init_pos.clone(), left_boundary,
                                           right_boundary, len_system)
        N_cell_batch = get_density(cell_centers, init_pos)
        input_batch[ii,0,:] = N_cell_batch[...]
        # Do the final step
        _, _, flux = random_walk_v2(ncells, nmoves, dt, init_pos,
                                    left_boundary, right_boundary,
                                    len_system = len_system)
        output_batch[ii,0,:] = flux[...]
        ### Get the corresponding DK flux
        flux_dk, _ = get_dk_flux(N_cell_batch, ncells, dx, dt,
                                 left_boundary, right_boundary, nmoves) 
        x_0[ii, 0, :] = flux_dk[:]

    return input_batch, output_batch, x_0


if __name__ == "__main__":
    N_min = 0
    N_max = 50
    dx = 1.0/100
    ncells = 10
    len_system=ncells*dx
    dt = 0.03*dx*dx
    left_boundary  = ["periodic", 0]
    right_boundary = ["periodic", 0]
    batch_size=5
    nmoves = 34 

    input_batch, output_batch, x_0 = get_particle_data(batch_size, 1, 51,
                                                  ncells, dx, dt, left_boundary,
                                                  right_boundary, nmoves)
    x_1 = output_batch
    print(x_1)
    # Scale the output instead of input
    ##############################################################
    N_left_t  = torch.zeros_like(input_batch)
    N_left_t[:,:,1:] = input_batch[:,:,:-1]
    N_left_t[:,:,0] = input_batch[:,:,-1]
    N_right_t = torch.zeros_like(input_batch)
    N_right_t[...] = input_batch
    # Shift the mean based on (N_left-N_right)
    x_1 -= (0.5*nmoves*dt/(dx*dx))*(N_left_t-N_right_t)
    # Change the standard deviation
    std_scale = 0.5*(torch.sqrt(torch.clamp(N_left_t, min=0.))
                     + torch.sqrt(torch.clamp(N_right_t,min=0.)))
    std_scale = torch.clamp(std_scale,min=0.5)
    std_scale *= np.sqrt(nmoves*dt)/dx
    x_1 /= std_scale
    print(x_1)


