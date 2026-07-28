import sys
import os
import numpy as np
import torch
import math
import h5py
import yaml
from scipy.stats import skew
from scipy.stats import kurtosis as kurt
import matplotlib.pyplot as plt

#directory = './multirun/2025-12-10/12-37-30/3'
directory = './multirun/2025-12-11/08-54-22/0'

h5_files = [file for file in os.listdir(directory) if file.endswith('.h5')]

h5_files.sort()

ncells = 4
yaml_string = os.path.join(directory,".hydra/config.yaml")

with open(yaml_string,"r") as yaml_file:
    data_yaml = yaml.safe_load(yaml_file)
    n_left    = data_yaml["n_left"]
    n_right   = data_yaml["n_right"]
    n_samples = data_yaml["n_samples"]
    n_steps   = data_yaml["n_steps"]

total_density_data = np.zeros((n_samples, n_steps+1, ncells))
total_flux_data    = np.zeros((n_samples, n_steps, 1))

for ii, fl_nm in enumerate(h5_files):
    data_file = os.path.join(directory,fl_nm)
    with h5py.File(data_file, mode="r") as f:
        dt  = f['dt'][()]
        total_density_data[ii,:,:] = f["density_data"][:]
        total_flux_data[ii,:,:] = f["flux_data"][:]

# Save a single npz file in that directory
npz_file = "total_data"

np.savez(os.path.join(directory,npz_file),
         density_data = total_density_data,
         flux_data    = total_flux_data)
