"""
The intention of this script is to see how the stats of flux across a face
look when multiple small "dt" steps are taken.
How does this flux scale when compared to N_L and N_R?
"""
import numpy as np
import torch
############### Local imports #######################
from random_walkers_pytorch import random_walk_v2
from random_walkers_pytorch import get_uni_initial_pos
from random_walkers_pytorch import get_particle_positions
#####################################################

n_datasets = 50
n_realizations = 5000

ncells = 100
nmoves = 50 # This is based on flux autocorrelation
len_system=1.0
dx = len_system/ncells
dt = 0.03*dx*dx
left_boundary =  ["periodic",0]
right_boundary = ["periodic",0]
cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
face_centers = torch.linspace(0,len_system,ncells+1)

# Generate random initial particle positions 
N_min = 0
N_max = 50
half_window = 5
face_id = 50

flux_ensemble = torch.zeros(n_realizations, 1)

left_array  = np.zeros(n_datasets)
right_array = np.zeros(n_datasets)
mean_array  = np.zeros((n_datasets, 2))
std_array   = np.zeros((n_datasets, 2))

for jj in range(n_datasets):
    N_left  = np.random.randint(N_min, N_max+1, 1).item()
    N_right = np.random.randint(N_min, N_max+1, 1).item()

    if (N_left == 0) and (N_left == N_right):
        N_left = 1

    left_array[jj] = N_left
    right_array[jj] = N_right

    for ii in range(n_realizations):    
        N_cell_system = torch.randint(low=N_min, high=N_max+1,
                                     size=(ncells,),dtype=torch.float32)
        # Fix left and right cell values
        N_cell_system[face_id-1] = N_left
        N_cell_system[face_id] = N_right
        
        initial_pos = get_particle_positions(N_cell_system, dx)
    
        _, _, flux = random_walk_v2(ncells,nmoves,dt,
                                    initial_pos,
                                    left_boundary,
                                    right_boundary,
                                    len_system=len_system)
    
        flux_ensemble[ii,0] = flux[face_id]
    
    # Mean from particle simulation
    mean_array[jj, 0] = torch.mean(flux_ensemble).item()
    # Mean from using DK
    dk_mean = (0.5*nmoves*dt/(dx*dx))*(N_left-N_right)
    mean_array[jj, 1] = dk_mean
    
    # Std from particle simulation
    std_array[jj, 0] = torch.std(flux_ensemble).item()
    # Std from DK
    dk_std = 0.5*(np.sqrt(N_left) + np.sqrt(N_right))
    dk_std *= np.sqrt(nmoves*dt)/dx
    std_array[jj, 1] = dk_std

np.savez(f"scaling_analysis_nmoves_{nmoves}",
         left_array = left_array, right_array=right_array,
         mean_array = mean_array, std_array = std_array,
         dt=dt, n_realizations = n_realizations)
