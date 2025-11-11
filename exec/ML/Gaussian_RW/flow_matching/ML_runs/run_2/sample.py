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
from helpers import get_particle_data
from model import Hierarchical_Model 
#######################################################

torch.set_default_device('cpu')

@torch.no_grad()
@hydra.main(version_base=None, config_path="./conf",
            config_name="config_sample")
def fhd_model_run (cfg):
    N_scale = cfg.n_scale
    dx = 1.0/100
    ncells = 2
    dt = 0.03*dx*dx
    left_boundary  = ["periodic", 0]
    right_boundary = ["periodic", 0]

    n_sampling_steps = cfg.n_sampling_steps
    batch_size = cfg.batch_size

    noise_std_fctr = 0.5/3
    n_layers = cfg.model.n_layers
    layer_width = cfg.model.layer_width
    residual_con = cfg.model.residual_con
    history_length = int(cfg.model.history_length)
    # This variable is just to use for get_particle_data
    hist_for_get_ptcl_data = int(cfg.hist_for_get_ptcl_data)
    # This variable is to decide whether noise should be added to input or not
    add_noise_to_input = cfg.add_noise_to_input 

    act_func     = instantiate(cfg.model.act_func)
    flow = Hierarchical_Model(history_length, 2, 1, n_layers, layer_width,
                         act_func = act_func,
                         residual_con=residual_con)

    # Create StudentT distribution
    student_t = torch.distributions.StudentT(cfg.df, loc=0., scale=1.0)
    # Load the trained ML model
    chpt_fl = torch.load(cfg.model.file_name)
    flow.load_state_dict(chpt_fl['model_state_dict'])
    # Max level to leverage
    flow.max_level = int(cfg.max_level)
    flow.train(False)

    ### Sampling is slightly different #####
    ### The Particle simulation data is generated at the final step
    #### The previous time step data for ML model is created based on
    #### N_{L} + N_{R} = constant at every time step
    ###################################################
    # Let us sample using this model
    n_samples = cfg.n_samples
    grnd_trth_data = torch.zeros((n_samples,))
    mdl_data       = torch.zeros_like(grnd_trth_data)
    N_left  = int(cfg.n_left)
    N_right = int(cfg.n_right)
    N_min   = min(cfg.n_range)
    N_max   = max(cfg.n_range)
    ############# Separate out ML data from Particle data #####################
    ############ ML data ##########################################
    with torch.no_grad():
        for iter_val in range(n_samples):
            # Randomly generate a particle distribution
            N_cell_batch = torch.randint(low=N_min, high=N_max+1,
                                         size=(batch_size+1,),dtype=torch.float32)
            # Replace two adjacent cells with (N_left, N_right) values
            idx = int(batch_size*0.5)
            N_cell_batch[idx]   = N_left
            N_cell_batch[idx+1] = N_right
            input_batch, output_batch = get_particle_data(N_cell_batch,
                                                hist_for_get_ptcl_data,
                                                dx, dt, left_boundary,
                                                right_boundary)

            grnd_trth_data[iter_val] = output_batch[idx,0].detach()
            ##############################################################
            if add_noise_to_input:
                # Add noise to the input_batch
                input_batch += torch.randn_like(input_batch)/3.0

            N_left_t  = torch.narrow(input_batch,-1,-2,1)
            N_right_t = torch.narrow(input_batch,-1,-1,1)
            x_0 = student_t.sample((1,1))
            ##################################################################
            output_batch = flow.sample(x_0,input_batch/N_scale,n_steps=n_sampling_steps)
            ###########################################################################
            # Change the standard deviation
            std_scale = torch.sqrt(torch.clamp(N_left_t, min=0.0)
                                 +torch.clamp(N_right_t,min=0.))
            std_scale = torch.clamp(std_scale,min=1.0)
            output_batch *= 0.2537*std_scale
            # Shift the mean based on (N_left-N_right)
            output_batch += 0.069*(N_left_t-N_right_t)
            ##########################################################################
            mdl_data[iter_val] = output_batch[idx,0].detach()
    ####################################################################################
    grnd_trth_np = grnd_trth_data.cpu().numpy()
    mdl_data_np  = mdl_data.cpu().numpy()

    dataset_name = os.path.join(HydraConfig.get().runtime.output_dir,
                                f"samples_two_cells_{int(N_left)}_{int(N_right)}")

    with h5py.File(dataset_name+".h5", mode="w") as f:
        f.create_dataset("ground_truth_data", data=grnd_trth_np, dtype = np.float32)
        f.create_dataset("model_data"  , data=mdl_data_np, dtype = np.float32)
        f.create_dataset("N_left", data=int(N_left), dtype = 'i')
        f.create_dataset("N_right", data=int(N_right), dtype = 'i')
        f.create_dataset("dt", data=dt, dtype=float)

    data_test = np.zeros(4)
    data_test[0] = np.mean(grnd_trth_np) - np.mean(mdl_data_np)
    data_test[1] = np.std(grnd_trth_np)  - np.std(mdl_data_np)
    data_test[2] = skew(grnd_trth_np,axis=None) - skew(mdl_data_np,axis=None)
    data_test[3] = (kurtosis(grnd_trth_np,axis=None,fisher=False) -
                        kurtosis(mdl_data_np,axis=None, fisher=False))
    print(np.mean(grnd_trth_np), np.std(grnd_trth_np),
          skew(grnd_trth_np,axis=None),
          kurtosis(grnd_trth_np,axis=None,fisher=False))

    print(np.mean(mdl_data_np), np.std(mdl_data_np),
          skew(mdl_data_np,axis=None),
          kurtosis(mdl_data_np,axis=None,fisher=False))

    print(np.abs(data_test))
    return np.sum(np.abs(data_test))

if __name__ == "__main__":
    fhd_model_run()
