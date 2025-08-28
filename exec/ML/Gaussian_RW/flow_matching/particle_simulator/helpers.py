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

"""
This function also return the position of particles at step t.
"""
@torch.no_grad()
def get_particle_data_and_initial_pos(N_cell_tnsr, n_hist_steps,
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

    return input_batch,output_batch, initial_pos


"""
This function converts system data to ML model's input
ONLY designed for periodic boundaries for now.
Expects N_cell_tnsr shape: (B, n_hist+1, ncells)
ncells leads to ONLY ncells faces for periodic
"""
@torch.no_grad()
def convert_system_data_to_model_inputs (N_cell_tnsr,
                                         left_boundary, right_boundary):
    assert (left_boundary[0] == "periodic" and
            right_boundary[0] == "periodic")

    assert N_cell_tnsr.dim() == 3

    ncells = N_cell_tnsr.size(-1)
    n_hist = N_cell_tnsr.size(-2) - 1

    model_inputs = torch.zeros((N_cell_tnsr.size(0),
                                ncells,n_hist+1,2))
    for jj in range(n_hist+1):
        # Left cell info at each time step
        model_inputs[:,1:,jj,0] = N_cell_tnsr[:,jj,:-1]
        # The left cell for the first face is the right most cell
        model_inputs[:,0,jj,0] = N_cell_tnsr[:,jj,-1]
        # Right cell info at each time step
        model_inputs[:,:,jj,1] = N_cell_tnsr[:,jj,:]

    # Combine the last two dimensions
    orig_shape = model_inputs.shape
    model_inputs = torch.reshape(model_inputs,
                                 (*orig_shape[:-2],-1))
    return model_inputs

"""
This function converts ML model's output to system data
ONLY designed for periodic boundaries for now.
THIS ONLY PROVIDES INCREMENT FOR EACH CELL
Expected model_ouputs shape:  (B, n_cells, 1)
"""
@torch.no_grad()
def convert_model_outputs_to_system_data (model_outputs,
                                         left_boundary, right_boundary):
    assert (left_boundary[0] == "periodic" and
            right_boundary[0] == "periodic")

    assert model_outputs.dim() == 3

    ncells = model_outputs.size(-2)

    flux_left = torch.zeros_like(model_outputs)
    flux_right = torch.zeros_like(flux_left)

    flux_left[...]  = model_outputs[...]
    flux_right[:,:-1,0] = model_outputs[:,1:, 0]
    # Last element in flux_right has to be treated correctly
    flux_right[:,-1,0] = model_outputs[:,0,0]

    return flux_left - flux_right
