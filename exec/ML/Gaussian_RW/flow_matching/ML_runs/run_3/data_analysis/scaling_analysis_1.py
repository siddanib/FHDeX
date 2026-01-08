"""
The intention of this script is to see how the stats of flux across a face
look when multiple small "dt" steps are taken.
How does this flux scale when compared to N_L and N_R
"""
import numpy as np
import torch
############### Local imports #######################
from random_walkers_pytorch import random_walk_v2
from random_walkers_pytorch import get_uni_initial_pos
from random_walkers_pytorch import get_particle_positions
#####################################################

n_realizations = 1000

ncells = 100
nmoves = 34 # This is based on flux autocorrelation
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

N_left  = 49
N_right = 0
face_id = 50
flux_ensemble = torch.zeros(n_realizations, 1)

# Trying different scalings
flux_ens_dk_1 = torch.zeros(n_realizations, 1)
flux_ens_dk_2 = torch.zeros(n_realizations, 1)

half_window = 5

for ii in range(n_realizations):    
    N_cell_system = torch.randint(low=N_min, high=N_max+1,
                                 size=(ncells,),dtype=torch.float32)
    # Fix left and right cell values
    N_cell_system[face_id-1] = N_left
    N_cell_system[face_id] = N_right
    
    initial_pos = get_particle_positions(N_cell_system, dx)

    # Collect window on left and right
    left_window = torch.narrow(N_cell_system, 0, face_id-half_window,
                               half_window)
    right_window = torch.narrow(N_cell_system, 0, face_id, half_window)
    _, _, flux = random_walk_v2(ncells,nmoves,dt,
                                initial_pos,
                                left_boundary,
                                right_boundary,
                                len_system=len_system)

    flux_ensemble[ii,0] = flux[face_id]

    # For the first one use average
    left_avg = torch.mean(left_window)
    right_avg = torch.mean(right_window)
    flux_ens_dk_1[ii,0] = (0.5*nmoves*dt/(dx*dx))*(left_avg-right_avg)
    fluct_1 = torch.randn_like(left_avg)
    fluct_1 *= 0.5*(torch.sqrt(torch.clamp(left_avg, min = 0.))
                                 +
                    torch.sqrt(torch.clamp(right_avg, min = 0.))
                    )
    fluct_1 *= ((nmoves*dt)**0.5)/dx
    flux_ens_dk_1 += fluct_1

    # For the second one use sum
    left_sum = torch.sum(left_window)
    right_sum = torch.sum(right_window)
    big_dx = half_window*dx
    flux_ens_dk_2[ii,0] = (0.5*nmoves*dt/(big_dx*big_dx))*(left_avg-right_avg)
    fluct_2 = torch.randn_like(left_sum)
    fluct_2 *= 0.5*(torch.sqrt(torch.clamp(left_sum, min = 0.))
                                 +
                    torch.sqrt(torch.clamp(right_sum, min = 0.))
                    )
    fluct_2 *= ((nmoves*dt)**0.5)/big_dx
    flux_ens_dk_2 += fluct_2

# Mean from particle simulation
print(torch.mean(flux_ensemble))
print(torch.mean(flux_ens_dk_1))
print(torch.mean(flux_ens_dk_2))
# Mean from using DK
dk_mean = (0.5*nmoves*dt/(dx*dx))*(N_left-N_right)
print(dk_mean)

# Std from particle simulation
print(torch.std(flux_ensemble))
print(torch.std(flux_ens_dk_1))
print(torch.std(flux_ens_dk_2))
# Std from DK
dk_std = 0.5*(np.sqrt(N_left) + np.sqrt(N_right))
dk_std *= np.sqrt(nmoves*dt)/dx
print(dk_std)
