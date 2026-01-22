"""
 This code is to check the weights of the N_L and N_R branch of the model
"""
import sys
import os
import numpy as np
import torch
import h5py
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
#####################################################
#######################################################
####### Local imports ################################
from random_walkers_pytorch import random_walk_v2
from random_walkers_pytorch import random_walk_just_evolve
from random_walkers_pytorch import get_uni_initial_pos
from random_walkers_pytorch import get_density
from model import Hierarchical_Model
from helpers_extended_domain import convert_system_data_to_model_inputs
from helpers_extended_domain import convert_model_outputs_to_system_data
#######################################################

torch.set_default_device('cpu')

@hydra.main(version_base=None, config_path="./conf",
            config_name="config_weight_check")
def fhd_data_run (cfg):
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

    wgts_intrst = flow.hierarchy_levels[0].module_list_2[0].module_list[0].weight
    #print(torch.var(torch.abs(wgts_intrst[:,0])))
    #print(torch.var(torch.abs(wgts_intrst[:,-half_window])))

    print(torch.mean(wgts_intrst[:,0]))
    print(torch.mean(wgts_intrst[:,-1]))
    print(torch.mean(wgts_intrst[:,1]))
    print(torch.mean(wgts_intrst[:,-2]))
    print(torch.mean(wgts_intrst[:,2]))
    print(torch.mean(wgts_intrst[:,-3]))
    print(torch.mean(wgts_intrst[:,-half_window-1]))
    print(torch.mean(wgts_intrst[:,-half_window]))

    print(torch.var(wgts_intrst[:,0]))
    print(torch.var(wgts_intrst[:,-1]))
    print(torch.var(wgts_intrst[:,1]))
    print(torch.var(wgts_intrst[:,-2]))
    print(torch.var(wgts_intrst[:,2]))
    print(torch.var(wgts_intrst[:,-3]))
    print(torch.var(wgts_intrst[:,-half_window-1]))
    print(torch.var(wgts_intrst[:,-half_window]))

if __name__ == "__main__":
    fhd_data_run()
