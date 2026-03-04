import sys
import os
import numpy as np
import h5py
import yaml

#parent_folder = "./outputs/2026-03-03/12-22-06/"
parent_folder = "./outputs/2026-03-03/13-38-04/"

yaml_string = os.path.join(parent_folder,".hydra/config.yaml")

with open(yaml_string,"r") as yaml_file:
    data_yaml = yaml.safe_load(yaml_file)
    n_steps = data_yaml["n_steps"]
    ncells  = data_yaml["ncells"]

dir_list = [parent_folder,]

for directory in dir_list:
    h5_files = [file for file in os.listdir(directory) if file.endswith('.h5')]
    h5_files.sort()

    total_samples = len(h5_files)

    particle_density_data = np.zeros((total_samples, n_steps+1, ncells))
    model_density_data    = np.zeros((total_samples, n_steps+1, ncells))
    particle_flux_data = np.zeros((total_samples, n_steps, ncells))
    model_flux_data    = np.zeros((total_samples, n_steps, ncells))

    for ii, fl_nm in enumerate(h5_files):
        data_file = os.path.join(directory,fl_nm)
        with h5py.File(data_file, mode="r") as f:
            particle_density_data[ii,:,:] = f["ptcl_density_data"][:]
            model_density_data[ii,:,:] = f["mdl_density_data"][:]
            particle_flux_data[ii,:,:] = f["ptcl_flux_data"][:]
            model_flux_data[ii,:,:] = f["mdl_flux_data"][:]

    # Save a single npz file in that directory
    npz_file = "total_data"

    np.savez(os.path.join(directory,npz_file),
             particle_density_data = particle_density_data,
             model_density_data = model_density_data,
             particle_flux_data = particle_flux_data,
             model_flux_data = model_flux_data
            )
