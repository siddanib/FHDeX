import sys
import os
import numpy as np
import math
import torch
import h5py
from scipy.stats import skew, kurtosis
from datetime import datetime
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
#######################################################
####### Local imports ################################
from random_walkers_pytorch import random_walk_v2
from random_walkers_pytorch import random_walk_just_evolve
from random_walkers_pytorch import get_particle_positions
from random_walkers_pytorch import get_uni_initial_pos
from random_walkers_pytorch import get_density
from model import Hierarchical_Model
from helpers import convert_system_data_to_model_inputs
from helpers import convert_model_outputs_to_system_data
#######################################################


@torch.no_grad()
@hydra.main(version_base=None, config_path="./conf",
            config_name="config_test")
def fhd_model_run (cfg):
    device_id = int(HydraConfig.get().job.num)%4
    torch.set_default_device('cuda')
    device = torch.device("cuda")
    torch.cuda.set_device(f'cuda:{device_id}')

    dx = 1.0/100
    ncells = cfg.ncells
    par_per_cell = cfg.par_per_cell
    len_system=ncells*dx
    dt = 0.03*dx*dx
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    left_boundary  = ["periodic", 0]
    right_boundary = ["periodic", 0]

    n_total_steps = cfg.n_total_particle_steps


    n_layers = cfg.model.n_layers
    layer_width = cfg.model.layer_width
    residual_con = cfg.model.residual_con
    history_length = int(cfg.model.history_length)

    act_func     = instantiate(cfg.model.act_func)
    flow = Hierarchical_Model(history_length, 2, 1, n_layers,
                         layer_width,
                         act_func = act_func,
                         residual_con=residual_con)
    # Load the trained ML model
    chpt_fl = torch.load(cfg.model.file_name, weights_only=False)
    flow.load_state_dict(chpt_fl['model_state_dict'])
    # Max level to leverage
    max_level = int(cfg.max_level)
    flow.max_level = max_level
    flow.train(False)
    # Turn off gradients for the parameters
    flow.train_levels([])
    # Put the model on device
    flow.to(device)
    print(next(flow.parameters()).device)
    # Using torch.compile
    flow.compile()
    print("Model compilation done")

    N_scale = cfg.n_scale
    n_sampling_steps = cfg.n_sampling_steps
    noise_std_fctr = 0.5/3
    # Create StudentT distribution
    student_t = torch.distributions.StudentT(cfg.df, loc=0., scale=1.0)
    ####################################################
    # Let us sample using the model
    batch_size = cfg.batch_size
    n_samples = cfg.n_samples
    n_batches = int(np.ceil(n_samples/batch_size))
    grnd_trth_data = torch.zeros((n_samples,n_total_steps+1,ncells))
    mdl_data       = torch.zeros_like(grnd_trth_data)

    for iter_initial in range(n_samples):
        initial_pos = get_uni_initial_pos(ncells, par_per_cell,
                                      len_system=len_system)

        initial_pos = random_walk_just_evolve(ncells, 10000, dt,
                        initial_pos.clone(), left_boundary,
                        right_boundary, len_system=len_system)

        initial_density = get_density(cell_centers, initial_pos.clone())
        mdl_data[iter_initial,0, :] = initial_density[...]
        grnd_trth_data[iter_initial,0,:] = initial_density[...]

    ############# Separate out ML data from Particle data #####################
    ############ ML data ##########################################
    with torch.no_grad():
        for iter_val in range(n_batches):
            if iter_val % 10 == 0:
                print(f"{iter_val} batches completed")
            b_start = iter_val*batch_size
            b_end   = b_start + batch_size
            mdl_data_batch = mdl_data[b_start:b_end,:,:]

            for i_t in range(n_total_steps):
                if (max_level > 0) and (i_t > 0):
                    old_density = torch.narrow(mdl_data_batch,1,i_t-1,2)
                    flow.max_level = max_level
                else:
                    old_density = torch.narrow(mdl_data_batch,1,i_t,1)
                    flow.max_level = 0

                input_batch = convert_system_data_to_model_inputs(
                                            old_density.clone(),
                                            left_boundary, right_boundary)
                N_left_t  = torch.narrow(input_batch,-1,-2,1)
                N_right_t = torch.narrow(input_batch,-1,-1,1)
                ##############################################################
                x_0 = student_t.sample(N_left_t.size())
                output_batch = flow.sample(x_0,input_batch/N_scale,
                                           n_steps=n_sampling_steps)
                ##############################################################
                # Change the standard deviation
                std_scale = torch.sqrt(torch.clamp(N_left_t, min=0.0)
                                     +torch.clamp(N_right_t,min=0.))
                std_scale = torch.clamp(std_scale,min=1.0)
                output_batch *= 0.2537*std_scale
                # Shift the mean based on (N_left-N_right)
                output_batch += 0.069*(N_left_t-N_right_t)
                output_batch = output_batch.detach()
                # Clamp the data
                output_batch = torch.clamp(output_batch,
                                           min=-N_right_t, max=N_left_t)
                new_delta_density = convert_model_outputs_to_system_data(
                                                   output_batch,
                                                   left_boundary,
                                                   right_boundary)
                # new_delta_density has shape (B, ncells, 1)
                new_delta_density = new_delta_density.permute(0,2,1)
                new_density = (torch.narrow(old_density,1,-1,1)
                               + new_delta_density)
                ##########################################################################
                mdl_data_batch[:,i_t+1:i_t+2,:] = new_density[...]
                if new_density.isnan().any():
                    print("Coming here")
                    break
    ####################################################################################
    #### Collect particle data #########################
    for iter_val in range(n_samples):
        N_cell_tnsr = torch.zeros((ncells,))
        N_cell_tnsr[...] = grnd_trth_data[iter_val,0,:]
        iter_pos = get_particle_positions(N_cell_tnsr, dx)
        for i_t in range(n_total_steps):
            iter_pos, new_density, _ = random_walk_v2(ncells, 1, dt,
                                           iter_pos.clone(),
                                           left_boundary, right_boundary,
                                           len_system = len_system)
            grnd_trth_data[iter_val,i_t+1,:] =  new_density[...]

    grnd_trth_np   = grnd_trth_data.cpu().numpy()
    mdl_data_np    = mdl_data.cpu().numpy()

    dataset_name = os.path.join(HydraConfig.get().runtime.output_dir,
                                "ensembles_of_multi_steps")

    with h5py.File(dataset_name+".h5", mode="w") as f:
        f.create_dataset("ground_truth_data", data=grnd_trth_np, dtype = np.float32)
        f.create_dataset("model_data"  , data=mdl_data_np, dtype = np.float32)
        f.create_dataset("dt", data=dt, dtype=float)
        f.create_dataset("dx", data=dx, dtype=float)
        f.create_dataset("len_system", data=len_system, dtype=float)

if __name__ == "__main__":
    fhd_model_run()
