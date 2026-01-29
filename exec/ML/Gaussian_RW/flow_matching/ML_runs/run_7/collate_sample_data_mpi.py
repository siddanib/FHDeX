import sys
import os
import numpy as np
import h5py
import yaml

parent_folder = "./outputs/2026-01-27/09-59-14/"

yaml_string = os.path.join(parent_folder,".hydra/config.yaml")

with open(yaml_string,"r") as yaml_file:
    data_yaml = yaml.safe_load(yaml_file)
    n_samples = data_yaml["n_samples"]

dir_list = []

for itm in os.listdir(parent_folder):
    # Construct full path
    full_path = os.path.join(parent_folder, itm)
    # Check if it's a directory
    if os.path.isdir(full_path):
        if ".hydra" in full_path:
            continue
        dir_list.append(full_path)

for directory in dir_list:
    h5_files = [file for file in os.listdir(directory) if file.endswith('.h5')]
    h5_files.sort()

    num_processes = len(h5_files)
    total_samples = num_processes*n_samples

    particle_flux_data = np.zeros((total_samples, 1))
    model_flux_data    = np.zeros((total_samples, 1))

    for ii, fl_nm in enumerate(h5_files):
        data_file = os.path.join(directory,fl_nm)
        with h5py.File(data_file, mode="r") as f:
            particle_flux_data[(ii*n_samples):(ii+1)*n_samples,0] = f["ground_truth_data"][:]
            model_flux_data[(ii*n_samples):(ii+1)*n_samples,0] = f["model_data"][:]
            n_left = f['N_left'][()]
            n_right = f['N_right'][()]


    # Save a single npz file in that directory
    npz_file = "total_data"

    np.savez(os.path.join(directory,npz_file),
             n_left = n_left,
             n_right = n_right,
             particle_flux_data = particle_flux_data,
             model_flux_data = model_flux_data
            )
