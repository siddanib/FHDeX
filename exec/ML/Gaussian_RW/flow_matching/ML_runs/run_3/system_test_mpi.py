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
from model import Hierarchical_Model
from helpers_extended_domain import convert_system_data_to_model_inputs
from helpers_extended_domain import convert_model_outputs_to_system_data
#######################################################

torch.set_default_device('cpu')

def realization_process (itr, cfg):
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
    half_window = cfg.model.half_window
    n_layers = cfg.model.n_layers
    layer_width = cfg.model.layer_width
    residual_con = cfg.model.residual_con
    history_length = int(cfg.model.history_length)
    act_func     = instantiate(cfg.model.act_func)
    flow = Hierarchical_Model(history_length, half_window, 1, n_layers,
                              layer_width,
                              act_func = act_func,
                              residual_con=residual_con)
    # Load the trained ML model
    chpt_fl = torch.load(cfg.model.file_name, weights_only=False,
                         map_location=torch.device('cpu'))
    flow.load_state_dict(chpt_fl['model_state_dict'])
    # Max level to leverage
    max_level = int(cfg.max_level)
    flow.max_level = max_level
    flow.train(False)
    # Turn off gradients for the parameters
    flow.train_levels([])
    flow.compile()
    ################################################################
    n_steps   = cfg.n_steps
    nmoves = cfg.nmoves # Number of steps of size dt
    N_scale = cfg.n_scale
    n_sampling_steps = cfg.n_sampling_steps
    # Create StudentT distribution
    student_t = torch.distributions.StudentT(cfg.df, loc=0., scale=1.0)
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
        a_level = int(min(i_step, max_level))
        flow.max_level = a_level
        old_density = torch.narrow(mdl_density_data, 0, i_step-a_level,
                                   a_level+1).clone()
        # Remove negative numbers that may have occured
        old_density = torch.clamp(old_density,min=0.)
        input_batch = convert_system_data_to_model_inputs(old_density,
                                                          half_window)
        N_left_t  = torch.narrow(input_batch,-1,-half_window-1,1)
        N_right_t = torch.narrow(input_batch,-1,-half_window,1)
        ## Also get total particles in left and right half windows
        left_window = torch.narrow(input_batch,-1,0,half_window).clone()
        right_window = torch.narrow(input_batch,-1,-half_window, half_window).clone()
        x_0 = student_t.sample(N_left_t.size())
        output_batch = flow.sample(x_0,input_batch/N_scale,
                                   n_steps=n_sampling_steps)
        # Change the standard deviation based on (N_left, N_right)
        std_scale = 0.5*(torch.sqrt(torch.clamp(N_left_t, min=0.))
                         + torch.sqrt(torch.clamp(N_right_t,min=0.)))
        std_scale = torch.clamp(std_scale,min=1.0)
        std_scale *= np.sqrt(nmoves*dt)/dx
        output_batch *= 0.93*std_scale
        # Shift the mean based on (N_left-N_right)
        output_batch += 0.61*(0.5*nmoves*dt/(dx*dx))*(N_left_t-N_right_t)
        ### THE EXTRA 0.61 AND 0.93 FACTORS ARE IN 01_06_2026 SLIDES
        output_batch = output_batch.detach()
        ##### Clamp the output based on left and right windows
        left_window = torch.sum(left_window,dim=-1,keepdim=True)
        right_window = torch.sum(right_window,dim=-1,keepdim=True)
        output_batch = torch.clamp(output_batch,min=-right_window,
                                   max=left_window)
        ################################################################
        #### Get new ML density from ML flux ##############
        ### Note this function only return increment #######
        new_density = convert_model_outputs_to_system_data(output_batch)
        mdl_density_data[i_step+1,:] = new_density + mdl_density_data[i_step,:]
        ###########################################################

    dataset_name = os.path.join(HydraConfig.get().runtime.output_dir,
                                "system_temporal")
    print(mdl_density_data)
    print(torch.sum(mdl_density_data,dim=-1))
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

    for ii in range(n_samples):
        itr = ii + app_rank*n_samples
        realization_process (itr, cfg)

    app_comm.Barrier()

if __name__ == "__main__":
    fhd_data_run()
