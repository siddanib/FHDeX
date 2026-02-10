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
from random_walkers_pytorch import get_uni_initial_pos
from random_walkers_pytorch import get_density
from random_walkers_pytorch import random_walk_just_evolve
from model import Flow_DeepONet
from helpers_domain import get_particle_data
#######################################################

def save_model (len_system, noise_std_fctr, n_layers, layer_width,
                residual_con, batch_size, cfg, learning_rate,
                flow, optimizer):
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
                'model_state_dict'    : flow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                },
                os.path.join(HydraConfig.get().runtime.output_dir,
                             "flow_deeponet_"+timestamp_str+".tar")
                )

torch.set_default_device('cuda')

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def fhd_model_run (cfg):
    torch.set_default_device('cuda')
    torch.autograd.set_detect_anomaly(True)
    N_min  = min(cfg.n_range)
    N_max  = max(cfg.n_range)
    N_scale = cfg.n_scale
    dx = 1.0/100
    ncells = cfg.ncells
    len_system=ncells*dx
    dt = 0.03*dx*dx
    left_boundary  = ["periodic", 0]
    right_boundary = ["periodic", 0]
    nmoves = cfg.nmoves # Number of steps of size dt
    cell_centers = torch.linspace(0.5*dx,(ncells-0.5)*dx,ncells)

    noise_std_fctr = 0
    kernel_size = cfg.model.kernel_size
    first_kernel_size = cfg.model.first_kernel_size
    normalize = cfg.model.normalize
    n_layers = cfg.model.n_layers
    layer_width = cfg.model.layer_width
    residual_con = cfg.model.residual_con
    act_func     = instantiate(cfg.model.act_func)
    flow = Flow_DeepONet(1, kernel_size, n_layers,
                         layer_width, act_func = act_func,
                         residual_con=residual_con,
                         normalize=normalize,
                         first_kernel_size=first_kernel_size)

    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate
    optimizer = torch.optim.AdamW(flow.parameters(), learning_rate)
    loss_fn = torch.nn.MSELoss()

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

        for epch in range(epoch_start, max_epoch):
            mean_loss = 0
            n_mn_vals = 0
            for _ in range(int(n_iter_per_epoch)+1):
                n_mn_vals += 1
                # Create the mini-batch
                input_batch, output_batch = get_particle_data(batch_size, N_min+1, N_max+1,
                                                              ncells, dx, dt, left_boundary,
                                                              right_boundary, nmoves)
                x_1  = output_batch
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
                ##############################################################
                x_0 = torch.randn_like(x_1)
                t   = torch.rand(batch_size, 1, 1)
                sigma_min = 1.0e-4 
                x_t  = (1.-(1.-sigma_min))*t*x_0 + t*x_1
                dx_t = x_1 - (1.-sigma_min)*x_0
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
                           flow, optimizer)

        # Final model
        save_model(len_system, noise_std_fctr, n_layers, layer_width,
                   residual_con, batch_size, cfg, learning_rate,
                   flow, optimizer)
    else:
        chpt_fl = torch.load(cfg.model.file_name)
        flow.load_state_dict(chpt_fl['model_state_dict'])
    return None

if __name__ == "__main__":
    fhd_model_run()
