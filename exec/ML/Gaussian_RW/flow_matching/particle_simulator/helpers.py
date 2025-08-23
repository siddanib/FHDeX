import sys
import os
import numpy as np
import math
import torch
import h5py
####### Local imports ################################
from random_walkers_pytorch import random_walk_v2
from random_walkers_pytorch import get_particle_positions
#######################################################

"""
This function creates (input, output) pairs ONLY for interior faces
"""
@torch.no_grad()
def get_particle_data(N_cell_tnsr, n_hist_steps,
                      dx, dt, left_boundary,
                      right_boundary):
    ncells = N_cell_tnsr.size(0)
    len_system = ncells*dx
    # Ensuring that atleast 1 particle exists in the system
    if torch.sum(N_cell_tnsr) == 0:
        N_cell_tnsr[0] = 1

    initial_pos = get_particle_positions(N_cell_tnsr,dx)
    # total data
    density_data =  torch.zeros((n_hist_steps+1,ncells))
    density_data[0,:] = N_cell_tnsr
    for jj in range(1,n_hist_steps+1):
        initial_pos, density, _ = random_walk_v2(ncells, 1, dt, initial_pos,
                                                 left_boundary, right_boundary,
                                                 len_system = len_system)
        density_data[jj,:] = density

    # Do the final step
    _, _, flux = random_walk_v2(ncells, 1, dt, initial_pos,
                                left_boundary, right_boundary,
                                len_system = len_system)

    n_batch_size = ncells - 1
    input_batch  = torch.zeros((n_batch_size, n_hist_steps+1, 2))
    output_batch = torch.zeros((n_batch_size, 1))

    for jj in range(n_hist_steps+1):
        input_batch[:,jj,0] = density_data[jj,:-1]
        input_batch[:,jj,1] = density_data[jj,1:]

    input_batch = torch.reshape(input_batch,(n_batch_size,-1))
    output_batch[:,0] = flux[1:]

    return input_batch,output_batch
