import sys
import os
import numpy as np
import torch
import h5py
import time
from datetime import datetime
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
#####################################################
############# For MPI ##################
from mpi4py import MPI
#####################################################
#######################################################
####### Local imports ################################
from random_walkers_pytorch import random_walk_v2
from random_walkers_pytorch import random_walk_just_evolve
from random_walkers_pytorch import get_uni_initial_pos
from random_walkers_pytorch import get_well_initial_pos
from random_walkers_pytorch import get_density
from model import Flow_DeepONet
#######################################################

torch.set_default_device('cpu')

def realization_process (itr, cfg, flow):
    # Setting num_threads inside subprocess function seems
    # vital for proper scaling
    torch.set_num_threads(1)

    dx = 1.0/100
    ncells = cfg.ncells
    len_system=ncells*dx
    dt = 0.03*dx*dx
    left_boundary  = ["periodic", 0]
    right_boundary = ["periodic", 0]
    n_avg = cfg.n_avg
    ####### ML Model related ################################
    n_steps   = cfg.n_steps
    nmoves = cfg.nmoves # Number of steps of size dt
    N_scale = cfg.n_scale
    n_sampling_steps = cfg.n_sampling_steps
    ####################################################################
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    if cfg.ic_type == "uniform":
        initial_pos = get_uni_initial_pos(ncells, n_avg, len_system)
    elif cfg.ic_type == "well":
        initial_pos = get_well_initial_pos(ncells, n_avg, 0.25, 0.75,
                                           len_system)
    else:
        sys.exit("Unknown ic_type")
    ### Thermalize the system ####################################
    if cfg.n_thermal_steps > 0:
        initial_pos = random_walk_just_evolve(ncells, cfg.n_thermal_steps,
                                              dt, initial_pos.clone(),
                                              left_boundary, right_boundary,
                                              len_system = len_system)
    ##############################################################
    density = get_density(cell_centers, initial_pos.clone())

    ptcl_density_data = torch.zeros((n_steps+1, ncells))
    ptcl_density_data[0, :] = density[:]

    mdl_density_data = torch.zeros_like(ptcl_density_data)
    mdl_density_data[0, :] = density[:]

    for i_step in range(n_steps):
        initial_pos, density, _ = random_walk_v2(ncells, nmoves, dt,
                                      initial_pos.clone(),
                                      left_boundary, right_boundary,
                                      len_system = len_system)

        ptcl_density_data[i_step+1, :] = density[:]
        ###### ML model prediction ################################
        old_density = torch.narrow(mdl_density_data, 0, i_step, 1).clone()
        # Remove negative numbers that may have occured
        old_density = torch.clamp(old_density,min=0.)
        ## These density states can be reals so convert them to integers
        od_floor = torch.floor(old_density)
        od_ceil = torch.ceil(old_density)
        prob_tensr = torch.rand_like(old_density)
        old_density_int = torch.where(prob_tensr < od_ceil-old_density,
                                      od_floor, od_ceil)
        #####################################################################
        input_batch = old_density_int.unsqueeze(0)
        N_left_t  = torch.zeros_like(input_batch)
        N_left_t[:,:,1:] = input_batch[:,:,:-1]
        N_left_t[:,:,0] = input_batch[:,:,-1]
        N_right_t = torch.zeros_like(input_batch)
        N_right_t[...] = input_batch

        x_0 = torch.randn_like(input_batch)
        output_batch = flow.sample(x_0,input_batch/N_scale,
                                   n_steps=n_sampling_steps)
        # Change the standard deviation based on (N_left, N_right)
        std_scale = 0.5*(torch.sqrt(torch.clamp(N_left_t, min=0.))
                         + torch.sqrt(torch.clamp(N_right_t,min=0.)))
        std_scale = torch.clamp(std_scale,min=0.5)
        std_scale *= np.sqrt(nmoves*dt)/dx
        output_batch *= std_scale
        # Shift the mean based on (N_left-N_right)
        output_batch += (0.5*nmoves*dt/(dx*dx))*(N_left_t-N_right_t)
        output_batch = output_batch.detach()
        ################################################################
        #### Get new ML density from ML flux ##############
        flux_left = torch.zeros(ncells)
        flux_left[:] = output_batch[0,0,:]
        flux_right = torch.zeros(ncells)
        flux_right[:-1] = output_batch[0,0,1:]
        flux_right[-1] = output_batch[0,0,0]
        new_density = flux_left - flux_right
        mdl_density_data[i_step+1,:] = new_density + mdl_density_data[i_step,:]
        ###########################################################

    print(mdl_density_data[-2,:])
    print(mdl_density_data[-1,:])

    dataset_name = os.path.join(HydraConfig.get().runtime.output_dir,
                                "system_temporal")

    with h5py.File(dataset_name+f"_{itr}"+".h5", mode="w") as f:
        f.create_dataset("ptcl_density_data"  , data=ptcl_density_data.cpu().numpy(),
                         dtype = np.float32)
        f.create_dataset("mdl_density_data"  , data=mdl_density_data.cpu().numpy(),
                         dtype = np.float32)
        f.create_dataset("ncells", data=ncells, dtype = 'i')
        f.create_dataset("N_avg", data=n_avg, dtype = 'i')
        f.create_dataset("dt", data=dt, dtype=float)
        f.create_dataset("len_system", data=len_system, dtype=float)

    return None

@hydra.main(version_base=None, config_path="./conf",
            config_name="config_system_test_mpi")
def fhd_data_run (cfg):
    # Get the global communicator
    app_comm = MPI.COMM_WORLD
    # Get the total number of processes
    app_size = app_comm.Get_size()
    # Get the unique rank of the current process
    app_rank = app_comm.Get_rank()

    n_samples = cfg.n_samples

    ####### ML Model related ################################
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
    # Load the trained ML model
    chpt_fl = torch.load(cfg.model.file_name, weights_only=False,
                         map_location=torch.device('cpu'))
    flow.load_state_dict(chpt_fl['model_state_dict'])
    flow.train(False)
    # Turn off gradients for the parameters
    for param in flow.parameters():
        param.requires_grad = False
    flow.compile()
    ###########################################################
    for ii in range(n_samples):
        itr = ii + app_rank*n_samples
        realization_process(itr, cfg, flow)

    app_comm.Barrier()

if __name__ == "__main__":
    fhd_data_run()
