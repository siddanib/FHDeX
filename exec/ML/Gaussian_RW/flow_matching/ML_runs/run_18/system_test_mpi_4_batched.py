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
from random_walkers_pytorch import get_nonzero_well_initial_pos
from random_walkers_pytorch import get_particle_positions_from_list
from random_walkers_pytorch import apply_boundary_effects
from random_walkers_pytorch import get_density
from random_walkers_pytorch import boundary_asserts
from model_reflect import Flow_Transformer
from helpers_extended_domain import convert_batched_system_data_to_model_inputs
from helpers_extended_domain import convert_model_outputs_to_batched_system_data
from helpers_extended_domain import add_extrnl_pot_net_ptcls_crsng
from helpers_extended_domain import apply_N_boundary_effects
#######################################################

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')

@torch.no_grad()
def realization_process (itr, cfg, flow, output_dir):
    # Setting num_threads inside subprocess function seems
    # vital for proper scaling
    torch.set_num_threads(1)
    #### This is the (dx, dt) combination for which the model
    #### has been trained #####
    dx = 1.0/100
    dt = 0.03*dx*dx
    left_boundary  = list(cfg.bc_type.left_boundary)
    right_boundary = list(cfg.bc_type.right_boundary)
    ### Do not use "ignore"  bc_type
    assert "ignore" not in left_boundary
    assert "ignore" not in right_boundary
    periodic_boundary = boundary_asserts(left_boundary,
                                         right_boundary)
    ### The particle-solver and ML model treat the first and last cells
    ### as GHOST cells if the boundaries are NON-PERIODIC
    ncells = cfg.ncells
    if not periodic_boundary:
        ncells += 2
    len_system=ncells*dx
    nfaces = ncells +1 
    if periodic_boundary:
        nfaces -= 1 # Number of unique faces
    #############################################################
    n_avg = cfg.n_avg
    ##### External potential related #############################
    add_pot = cfg.external_pot.add_potential
    pot_alpha = cfg.external_pot.alpha
    pot_beta = cfg.external_pot.beta
    pot_gamma = cfg.external_pot.gamma
    if not periodic_boundary:
        pot_alpha += dx
        pot_beta += dx
    ####### ML Model related ################################
    half_window = cfg.model.half_window
    if not periodic_boundary:
        assert half_window == 1
    ################################################################
    n_steps   = cfg.n_steps
    nmoves = cfg.nmoves # Number of steps of size dt
    N_scale = cfg.n_scale
    n_sampling_steps = cfg.n_sampling_steps
    ## history_length
    hist_len = cfg.model.history_length
    #### batch_size
    batch_size = cfg.batch_size
    # Create StudentT distribution
    student_t = torch.distributions.StudentT(cfg.df, loc=0., scale=1.0)
    ####################################################################
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    ####################################################################
    ptcl_density_data = torch.zeros((batch_size, n_steps+1, ncells))
    ptcl_flux_data = torch.zeros((batch_size, n_steps, nfaces))

    mdl_density_data = torch.zeros_like(ptcl_density_data)
    mdl_flux_data = torch.zeros_like(ptcl_flux_data)

    for i_batch in range(batch_size):
        if cfg.ic_type == "uniform":
            initial_pos = get_uni_initial_pos(ncells, n_avg, len_system)
        elif cfg.ic_type == "well":
            initial_pos = get_well_initial_pos(ncells, n_avg, 0.25, 0.75,
                                               len_system)
        elif cfg.ic_type == "nonzero_well":
            initial_pos = get_nonzero_well_initial_pos(ncells, cfg.N_low,
                                                       cfg.N_high, 0.25, 0.75,
                                                       len_system)
        elif cfg.ic_type == "custom_pos_list":
            x_list = list(cfg.custom_pos_list.x_list)
            N_list = list(cfg.custom_pos_list.N_list)
            assert len(x_list)+1 == len(N_list)
            ### Note: There needs to be correction to supplied list
            ### when the boundaries are non-periodic
            if not periodic_boundary:
                x_list = [x_val+dx for x_val in x_list]
            initial_pos = get_particle_positions_from_list(ncells, dx,
                                                           x_list, N_list) 
        else:
            sys.exit("Unknown ic_type")
        #### Apply boundary effects to perform a cleanup of
        #### particle positions
        initial_pos = apply_boundary_effects(initial_pos.clone(), len_system,
                                             ncells, left_boundary,
                                             right_boundary)
        ### Thermalize the system ####################################
        if cfg.n_thermal_steps > 0:
            initial_pos = random_walk_just_evolve(ncells, cfg.n_thermal_steps,
                                                  dt, initial_pos.clone(),
                                                  left_boundary, right_boundary,
                                                  len_system = len_system,
                                                  add_potential= add_pot,
                                                  alpha=pot_alpha, beta=pot_beta,
                                                  gamma=pot_gamma)
        ##############################################################
        density = get_density(cell_centers, initial_pos.clone())
        ptcl_density_data[i_batch, 0, :] = density[:]
        mdl_density_data[i_batch, 0, :] = density[:]
        for i_step in range(n_steps):
            initial_pos, density, flux_ptcl = random_walk_v2(ncells, nmoves, dt,
                                                initial_pos.clone(),
                                                left_boundary, right_boundary,
                                                len_system = len_system,
                                                add_potential= add_pot,
                                                alpha=pot_alpha, beta=pot_beta,
                                                gamma=pot_gamma)

            ptcl_density_data[i_batch, i_step+1, :] = density[:]
            ptcl_flux_data[i_batch, i_step,:] = flux_ptcl[:]
    ####### ML model based time stepping #############################
    for i_step in range(n_steps):
        ###### ML model prediction ################################
        #### Need a local hist_len variable as there is no flux history
        #### at the beginning ######
        l_h_len = min(i_step, hist_len)
        old_density = torch.narrow(mdl_density_data, 1, i_step-l_h_len,
                                   l_h_len+1).clone()
        # Remove negative numbers that may have occured
        old_density = torch.clamp(old_density,min=0.)
        # Get the N_left_reals and N_right_reals
        old_density_r = torch.narrow(old_density,1,-1, 1).clone()
        limit_batch_N, _  = convert_batched_system_data_to_model_inputs(
                                            old_density_r,
                                            torch.empty((batch_size, 0, nfaces)),
                                            half_window,
                                            periodic_boundary)
        N_left_r = torch.narrow(limit_batch_N, -2, -1, 1)
        N_left_r = torch.narrow(N_left_r, -1,-half_window-1,1)

        N_right_r = torch.narrow(limit_batch_N, -2, -1, 1)
        N_right_r = torch.narrow(N_right_r, -1,-half_window, 1)
        ## These density states can be reals so convert them to integers
        od_floor = torch.floor(old_density)
        od_ceil = torch.ceil(old_density)
        prob_tensr = torch.rand_like(old_density)
        old_density_int = torch.where(prob_tensr < od_ceil-old_density,
                                      od_floor, od_ceil)
        # Similar for Flux
        if i_step == 0:
            #### This is dummy when i_step = 0
            old_flux_int = torch.empty((batch_size, 0, 1))
        else:
            old_flux = torch.narrow(mdl_flux_data, 1, i_step-l_h_len,
                                    l_h_len).clone()
            ## These fluxes can be reals so convert them to integers
            od_floor = torch.floor(old_flux)
            od_ceil = torch.ceil(old_flux)
            prob_tensr = torch.rand_like(old_flux)
            old_flux_int = torch.where(prob_tensr < od_ceil-old_flux,
                                       od_floor, od_ceil)
        #####################################################################
        input_batch_N, input_batch_F = convert_batched_system_data_to_model_inputs(
                                                          old_density_int,
                                                          old_flux_int,
                                                          half_window,
                                                          periodic_boundary)
        # Scale the output instead of input
        N_left_t = torch.narrow(input_batch_N, -2, -1, 1)
        N_left_t = torch.narrow(N_left_t, -1,-half_window-1,1)

        N_right_t = torch.narrow(input_batch_N, -2, -1, 1)
        N_right_t = torch.narrow(N_right_t, -1,-half_window, 1)

        x_0 = student_t.sample(N_left_t.size())
        output_batch = flow.sample(x_0,input_batch_N/N_scale,
                                   input_batch_F/N_scale,
                                   n_steps=n_sampling_steps)
        # Change the standard deviation based on (N_left, N_right)
        std_scale = torch.sqrt(torch.clamp(N_left_t, min=0.0)
                               +torch.clamp(N_right_t,min=0.))
        std_scale = torch.clamp(std_scale,min=0.5)
        output_batch *= 0.2537*std_scale
        # Shift the mean based on (N_left-N_right)
        output_batch += 0.069*(N_left_t-N_right_t)
        ### Clamp based on N_left and N_right
        output_batch = torch.clamp(output_batch, min=-N_right_r,
                                   max=N_left_r)
        ########################################################
        output_batch = output_batch.detach()
        # Reshaped output_batch
        re_output_batch = torch.reshape(output_batch,
                                        (batch_size, -1, 1, 1))
        ### The number of faces for which ML model is queried is different
        ### for periodic and non-periodic boundaries
        ### For periodic; ML is called on "ncells" faces
        ### For non-periodic; ML is called on "ncells-1" faces
        if periodic_boundary:
            mdl_flux_data[:, i_step,:] = re_output_batch[:,:, 0, 0]
        else:
            mdl_flux_data[:, i_step,1:-1] = re_output_batch[:,:, 0, 0]
        ################################################################
        #### Get new ML density from ML flux ##############
        ### Note this function only returns increment #######
        new_density = convert_model_outputs_to_batched_system_data(
                                        output_batch.clone().squeeze(-1),
                                        batch_size, periodic_boundary)
        ##### External potential related ########################
        old_N = mdl_density_data[:, i_step,:]
        if add_pot:
            pot_dens_incr, pot_flux = add_extrnl_pot_net_ptcls_crsng(old_N.clone(),
                                        dt, pot_alpha, pot_beta, pot_gamma,
                                        len_system, periodic_boundary)
            mdl_flux_data[:, i_step,:] += pot_flux
            new_density += pot_dens_incr
        new_N = new_density + old_N
        #### Apply non-periodic boundary effects
        mdl_density_data[:, i_step+1,:] = apply_N_boundary_effects(new_N,
                                                left_boundary,
                                                right_boundary)
        ###########################################################
    ##### Need to remove the ghost cells before saving the data
    if not periodic_boundary:
        ncells -= 2
        len_system = ncells*dx
        ptcl_density_data = torch.narrow(ptcl_density_data, -1, 1,
                                         ncells)
        mdl_density_data = torch.narrow(mdl_density_data, -1, 1,
                                         ncells)
        ptcl_flux_data = torch.narrow(ptcl_flux_data, -1, 1,
                                      ncells+1)
        mdl_flux_data = torch.narrow(mdl_flux_data, -1, 1,
                                      ncells+1)

    dataset_name = os.path.join(output_dir,"system_temporal")

    for i_batch in range(batch_size):
        itr_sv = itr+i_batch
        itr_ptcl_density_data = ptcl_density_data[i_batch,:,:]
        itr_mdl_density_data = mdl_density_data[i_batch,:,:]
        itr_ptcl_flux_data = ptcl_flux_data[i_batch,:,:]
        itr_mdl_flux_data = mdl_flux_data[i_batch, :,:]
        with h5py.File(dataset_name+f"_{itr_sv}"+".h5", mode="w") as f:
            f.create_dataset("ptcl_density_data"  ,
                             data=itr_ptcl_density_data.cpu().numpy(),
                             dtype = np.float32)
            f.create_dataset("mdl_density_data",
                             data=itr_mdl_density_data.cpu().numpy(),
                             dtype = np.float32)
            f.create_dataset("ptcl_flux_data",
                             data=itr_ptcl_flux_data.cpu().numpy(),
                             dtype = np.float32)
            f.create_dataset("mdl_flux_data",
                             data=itr_mdl_flux_data.cpu().numpy(),
                             dtype = np.float32)
            f.create_dataset("ncells", data=ncells, dtype = 'i')
            f.create_dataset("N_avg", data=n_avg, dtype = 'i')
            f.create_dataset("dt", data=dt, dtype=float)
            f.create_dataset("len_system", data=len_system, dtype=float)

    return None

@hydra.main(version_base=None, config_path="./conf",
            config_name="config_batched_system_test_mpi_3")
def fhd_data_run (cfg):
    # Get the global communicator
    app_comm = MPI.COMM_WORLD
    # Get the total number of processes
    app_size = app_comm.Get_size()
    # Get the unique rank of the current process
    app_rank = app_comm.Get_rank()

    if cfg.device == "cpu":
        torch.set_default_device('cpu')
        device = torch.device("cpu")
    else:
        device_id = app_rank %4
        torch.set_default_device(f"cuda:{device_id}")
        device = torch.device(f"cuda:{device_id}")

    n_samples = cfg.n_samples
    batch_size = cfg.batch_size
    ### Get the output_dir location
    output_dir = None
    if app_rank == 0:
        output_dir = HydraConfig.get().runtime.output_dir
    output_dir = app_comm.bcast(output_dir,root=0)
    ####### ML Model related ################################
    half_window = cfg.model.half_window
    n_layers = cfg.model.n_layers
    d_model = cfg.model.d_model
    history_length = int(cfg.model.history_length)
    act_func     = instantiate(cfg.model.act_func)
    flow = Flow_Transformer(input_N_dim=2*half_window, input_F_dim=1,
                            d_model = d_model, nhead=cfg.model.n_head,
                            num_encoder_layers=n_layers,
                            num_decoder_layers=n_layers,
                            dropout = cfg.model.dropout,
                            max_len=50, d_embed=d_model,
                            n_layers=n_layers,act_func=act_func,
                            residual_con=cfg.model.residual_con)
    # Load the trained ML model
    chpt_fl = torch.load(cfg.model.file_name, weights_only=False,
                         map_location=device)
    flow.load_state_dict(chpt_fl['model_state_dict'])
    flow.train(False)
    # Turn off gradients for the parameters
    for param in flow.parameters():
        param.requires_grad = False
    # Torch compile
    flow.compile()
    ###########################################################
    for ii in range(n_samples):
        itr = (ii + app_rank*n_samples)*batch_size
        realization_process(itr, cfg, flow, output_dir)

    app_comm.Barrier()

if __name__ == "__main__":
    fhd_data_run()
