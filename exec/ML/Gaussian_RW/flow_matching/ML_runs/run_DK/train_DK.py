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
from random_walkers_pytorch import get_particle_positions
from model import Hierarchical_Model
from helpers import get_particle_data
#######################################################

def save_model (len_system, noise_std_fctr, n_layers, layer_width,
                residual_con, batch_size, cfg, learning_rate,
                history_length, flow, optimizer, n_sampling_steps):
    # Get current date and time
    now = datetime.now()
    # Format as string (e.g., "2024-06-13_15-27-45")
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    torch.save({
                'len_system'          : len_system,
                'noise_stc_fctr'      : noise_std_fctr,
                'n_layers'            : n_layers,
                'layer_width'         : layer_width,
                'residual_con'        : residual_con,
                'batch_size'          : batch_size,
                'act_func'            : cfg.model.act_func,
                'learning_rate'       : learning_rate,
                'history_length'      : history_length,
                'model_state_dict'    : flow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'n_sampling_steps'    : n_sampling_steps,
                },
                os.path.join(HydraConfig.get().runtime.output_dir,
                             "flow_deeponet_"+timestamp_str+".tar")
                )

torch.set_default_device('cuda')

@hydra.main(version_base=None, config_path="./conf", config_name="config_DK")
def fhd_model_run (cfg):
    torch.set_default_device('cuda')
    N_min  = min(cfg.n_range)
    N_max  = max(cfg.n_range)
    N_scale = cfg.n_scale
    dx = 1.0/100
    ncells = cfg.batch_size - 1
    len_system=ncells*dx
    dt = 0.03*dx*dx
    left_boundary  = ["periodic", 0]
    right_boundary = ["periodic", 0]

    n_sampling_steps = cfg.n_sampling_steps

    noise_std_fctr = 0.5/3
    n_layers = cfg.model.n_layers
    layer_width = cfg.model.layer_width
    residual_con = cfg.model.residual_con
    history_length = int(cfg.model.history_length)
    act_func     = instantiate(cfg.model.act_func)
    flow = Hierarchical_Model(history_length, 2, 1, n_layers, layer_width,
                         act_func = act_func,
                         residual_con=residual_con)

    batch_size = cfg.batch_size
    learning_rate = 1.0e-4
    optimizer = torch.optim.AdamW(flow.parameters(), learning_rate)
    loss_fn = torch.nn.MSELoss()
    # Create StudentT distribution
    student_t = torch.distributions.StudentT(cfg.df, loc=0., scale=1.0)

    # Max level to leverage
    flow.max_level = 0
    # Which levels must be trained
    flow.train_levels([0,])

    epoch_start = 0
    # Look if there a previous version to start
    if 'file_name' in cfg['model']:
        if cfg.model.file_name != "":
            chpt_fl = torch.load(cfg.model.file_name,weights_only=False)
            flow.load_state_dict(chpt_fl['model_state_dict'])
            epoch_start = cfg.model.epoch_start

    if cfg.model.train:
        n_iter_per_epoch = min(cfg.max_iter,cfg.n_iter_per_epoch)
        max_epoch = int(cfg.max_iter/n_iter_per_epoch)
        chunk_size = max_epoch//(history_length+1)
        for epch in range(epoch_start, max_epoch):
            # Decide which levels to train
            lev_train = epch//chunk_size
            lev_train = min(lev_train, history_length)
            flow.max_level = lev_train
            flow.train_levels([lev_train,])
            print(lev_train)

            mean_loss = 0
            n_mn_vals = 0
            for _ in range(int(n_iter_per_epoch)+1):
                n_mn_vals += 1
                # Create the mini-batch
                N_cell_batch = torch.randint(low=N_min, high=N_max+1,
                                             size=(batch_size+1,),dtype=torch.float32)

                ###### NOTE THAT YOU ARE ONLY TAKING INPUT from particle simulation

                input_batch, _ = get_particle_data(N_cell_batch, history_length,
                                                   dx, dt, left_boundary,
                                                   right_boundary)
                #### Use N_L and N_R to construct DK fluxes ##################
                N_left_DK  = torch.narrow(input_batch,-1,-2,1)
                N_right_DK = torch.narrow(input_batch,-1,-1,1)

                output_batch = (0.5*dt/(dx*dx))*(N_left_DK - N_right_DK)
                dk_fluct = torch.randn_like(N_left_DK)
                dk_fluct *= 0.5*(torch.sqrt(torch.clamp(N_left_DK, min = 0.))
                                 +
                                 torch.sqrt(torch.clamp(N_right_DK, min = 0.))
                                 )
                dk_fluct *= (dt**0.5)/dx
                output_batch += dk_fluct
                ##############################################################
                # Add noise to the input_batch
                input_batch += torch.randn_like(input_batch)/3.0
                x_1  = output_batch
                # Add some noise to the flux
                x_1 += torch.randn_like(x_1)/3.0
                # Scale the output instead of input
                ##############################################################
                N_left_t  = torch.narrow(input_batch,-1,-2,1)
                N_right_t = torch.narrow(input_batch,-1,-1,1)
                # Shift the mean based on (N_left-N_right)
                x_1 -= (0.5*dt/(dx*dx))*(N_left_t-N_right_t)
                # Change the standard deviation
                std_scale = 0.5*(torch.sqrt(torch.clamp(N_left_t, min=0.))
                                 +
                                 torch.sqrt(torch.clamp(N_right_t,min=0.)))
                std_scale = torch.clamp(std_scale,min=1.0)
                x_1 /= ((dt**0.5)/dx)*std_scale
                ##############################################################
                x_0 = student_t.sample(x_1.size())
                t    = torch.rand_like(x_1)
                x_t  = torch.cos(0.5*math.pi*t)*x_0 + torch.sin(0.5*math.pi*t)*x_1
                dx_t = torch.cos(0.5*math.pi*t)*x_1 - torch.sin(0.5*math.pi*t)*x_0
                dx_t *= (0.5*math.pi)
                optimizer.zero_grad()
                loss_batch = loss_fn(flow(x_t, input_batch/N_scale, t), dx_t)
                loss_batch.backward()
                optimizer.step()
                mean_loss += loss_batch.item()
            mean_loss /= n_mn_vals
            print(f"Epoch: {epch}, loss: {mean_loss}")
            if epch % 10 == 0:
                save_model(len_system, noise_std_fctr, n_layers, layer_width,
                           residual_con, batch_size, cfg, learning_rate,
                           history_length, flow, optimizer, n_sampling_steps)

        # Final model
        save_model(len_system, noise_std_fctr, n_layers, layer_width,
                   residual_con, batch_size, cfg, learning_rate,
                   history_length, flow, optimizer, n_sampling_steps)
    else:
        chpt_fl = torch.load(cfg.model.file_name)
        flow.load_state_dict(chpt_fl['model_state_dict'])
    return None

if __name__ == "__main__":
    fhd_model_run()
