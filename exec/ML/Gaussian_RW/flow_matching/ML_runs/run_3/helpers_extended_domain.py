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
This function creates (input, output) pairs for faces
"""
@torch.no_grad()
def get_particle_data(N_cell_tnsr, n_hist_steps,
                      dx, dt, left_boundary,
                      right_boundary, half_window, nmoves):
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
        initial_pos, density, _ = random_walk_v2(ncells, nmoves, dt, initial_pos,
                                                 left_boundary, right_boundary,
                                                 len_system = len_system)
        density_data[jj,:] = density

    # Do the final step
    _, _, flux = random_walk_v2(ncells, nmoves, dt, initial_pos,
                                left_boundary, right_boundary,
                                len_system = len_system)

    n_batch_size = ncells
    # Apply circular padding
    pad_mdl = torch.nn.CircularPad1d(half_window)
    input_batch = pad_mdl(density_data)
    input_batch = input_batch.unfold(-1,2*half_window,1)
    # Need to narrow dim=1 as duplicate exists at the end
    input_batch = torch.narrow(input_batch,1,0,n_batch_size)
    # Need to swap dims 0 and 1
    input_batch = torch.swapdims(input_batch,0,1)
    # Flatten the temporal direction
    input_batch = torch.reshape(input_batch,(n_batch_size,-1))

    output_batch = torch.zeros((n_batch_size, 1))
    output_batch[:,0] = flux[...]

    return input_batch,output_batch

if __name__ == "__main__":
    N_min = 0
    N_max = 50
    dx = 1.0/100
    ncells = 10
    len_system=ncells*dx
    dt = 0.03*dx*dx
    left_boundary  = ["periodic", 0]
    right_boundary = ["periodic", 0]
    half_window = 5
    batch_size=ncells

    N_cell_batch = torch.randint(low=N_min, high=N_max+1,
                                 size=(batch_size,),dtype=torch.float32)

    input_batch, output_batch = get_particle_data(N_cell_batch, 1,
                                    dx, dt, left_boundary,
                                    right_boundary, half_window, 34)
    print(input_batch)
    #print(output_batch)

    N_left_t  = torch.narrow(input_batch,-1,-half_window-1,1)
    N_right_t = torch.narrow(input_batch,-1,-half_window,1)
    print(N_left_t)
    print(N_right_t)
