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
    # total flux data
    flux_data = torch.zeros(n_hist_steps, ncells)
    for jj in range(1,n_hist_steps+1):
        initial_pos, density, flux = random_walk_v2(ncells, nmoves, dt, initial_pos,
                                                 left_boundary, right_boundary,
                                                 len_system = len_system)
        density_data[jj,:] = density
        flux_data[jj-1,:] = flux

    # Do the final step
    _, _, flux = random_walk_v2(ncells, nmoves, dt, initial_pos,
                                left_boundary, right_boundary,
                                len_system = len_system)

    n_batch_size = ncells
    # Apply circular padding for density data
    pad_mdl = torch.nn.CircularPad1d(half_window)
    input_batch_N = pad_mdl(density_data)
    input_batch_N = input_batch_N.unfold(-1,2*half_window,1)
    # Need to narrow dim=1 as duplicate exists at the end
    input_batch_N = torch.narrow(input_batch_N,1,0,n_batch_size)
    # Need to swap dims 0 and 1
    input_batch_N = torch.swapdims(input_batch_N,0,1)

    # Unsqueeze last dim for flux_data and swap 0 and -1
    input_batch_F = flux_data.unsqueeze(0)
    input_batch_F = torch.swapdims(input_batch_F, 0, -1)

    output_batch = torch.zeros((n_batch_size, 1, 1))
    output_batch[:,0, 0] = flux[...]

    return input_batch_N, input_batch_F, output_batch

"""
This function converts density and flux data into model inputs.
ASSUMES PERIODIC BOUNDARIES.
Assumes density_data has a shape (n_time_steps, ncells)
Assumes flux_data has a shape    (n_time_steps-1, ncells)
"""
@torch.no_grad()
def convert_system_data_to_model_inputs (density_data, flux_data, half_window):
    n_batch_size = density_data.size(-1)
    # Apply circular padding for density data
    pad_mdl = torch.nn.CircularPad1d(half_window)
    input_batch_N = pad_mdl(density_data)
    input_batch_N = input_batch_N.unfold(-1,2*half_window,1)
    # Need to narrow dim=1 as duplicate exists at the end
    input_batch_N = torch.narrow(input_batch_N,1,0,n_batch_size)
    # Need to swap dims 0 and 1
    input_batch_N = torch.swapdims(input_batch_N,0,1)

    # Unsqueeze last dim for flux_data and swap 0 and -1
    input_batch_F = flux_data.unsqueeze(0)
    input_batch_F = torch.swapdims(input_batch_F, 0, -1)

    return input_batch_N, input_batch_F

"""
This function converts ML model's output to system data
ONLY designed for periodic boundaries for now.
THIS ONLY PROVIDES INCREMENT FOR EACH CELL
Expected model_ouputs shape:  (n_cells, 1)
"""
@torch.no_grad()
def convert_model_outputs_to_system_data (model_outputs):
    flux_left = torch.zeros_like(model_outputs)
    flux_right = torch.zeros_like(flux_left)

    flux_left[...] = model_outputs[...]
    flux_right[:-1, 0] = model_outputs[1:,0]
    flux_right[-1,0] = model_outputs[0,0]

    return (flux_left - flux_right).squeeze(-1)

if __name__ == "__main__":
    N_min = 0
    N_max = 50
    dx = 1.0/100
    ncells = 10
    len_system=ncells*dx
    dt = 0.03*dx*dx
    left_boundary  = ["periodic", 0]
    right_boundary = ["periodic", 0]
    half_window = 1
    n_hist_len = 2
    nmoves = 1
    batch_size=ncells

    N_cell_batch = torch.randint(low=N_min, high=N_max+1,
                                 size=(batch_size,),dtype=torch.float32)

    ##### get_particle_data function testing ##########################
    #input_batch_N, input_batch_F, output_batch = get_particle_data(N_cell_batch,
    #                                                n_hist_len, dx, dt,
    #                                               left_boundary, right_boundary,
    #                                                half_window, nmoves)
    ##print(input_batch_N)
    ##print(input_batch_F)
    #print(output_batch.shape)

    #N_left_t  = torch.narrow(input_batch,-1,-half_window-1,1)
    #N_right_t = torch.narrow(input_batch,-1,-half_window,1)
    #print(N_left_t)
    #print(N_right_t)
    ###################################################################
    ###### convert_system_data_to_model_inputs test ##################
    #initial_pos = get_particle_positions(N_cell_batch,dx)
    #density_data =  torch.zeros((n_hist_len+1,ncells))
    #density_data[0,:] = N_cell_batch
    ## total flux data
    #flux_data = torch.zeros(n_hist_len, ncells)
    #for jj in range(1,n_hist_len+1):
    #    initial_pos, density, flux = random_walk_v2(ncells, nmoves, dt, initial_pos,
    #                                             left_boundary, right_boundary,
    #                                             len_system = len_system)
    #    density_data[jj,:] = density
    #    flux_data[jj-1,:] = flux
    #input_batch_N, input_batch_F = convert_system_data_to_model_inputs(density_data,
    #                                                  flux_data, half_window)
    #print(input_batch_N)
    #print(input_batch_F)
    ####################################################################
    ####### convert_model_outputs_to_system_data test ###################
    initial_pos = get_particle_positions(N_cell_batch,dx)

    _, new_density, flux = random_walk_v2(ncells, 40, dt, initial_pos,
                                      left_boundary, right_boundary,
                                      len_system = len_system)

    increment_density = convert_model_outputs_to_system_data(
                                      flux.clone().unsqueeze(-1))

    print(new_density)
    print(N_cell_batch+increment_density)
    #####################################################################
