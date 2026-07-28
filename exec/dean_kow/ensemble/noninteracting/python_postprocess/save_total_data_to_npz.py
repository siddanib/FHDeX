import os
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
import yt

# This script calculates mean squared change
dataset_name = "no_history"
par_per_cell =  10
parent_folder = "../particles_"+dataset_name+"_dt_3.0e-6/"

fld_path = parent_folder
# Get plt files
onlyplotfiles = [fl for fl in os.listdir(fld_path) if "plt" in fl]

# Ensure to remove files that contain "old" string
onlyplotfiles = [fl for fl in onlyplotfiles if "old" not in fl]
onlyplotfiles.sort()
n_files = len(onlyplotfiles)

total_data_dk = np.zeros((n_files, 100, 2048))
total_data_ptcl = np.zeros((n_files, 100, 2048))

for i in range(n_files):
    ## Load the i^th file
    plotfile_nm = os.path.join(fld_path, onlyplotfiles[i])
    ds = yt.load(plotfile_nm)
    #print(plotfile_nm)
    time_i = float(ds.current_time)
    #print(ds.field_list)
    dx = (ds.domain_right_edge[0]-ds.domain_left_edge[0])/ds.domain_dimensions[0]
    dy = (ds.domain_right_edge[1]-ds.domain_left_edge[1])/ds.domain_dimensions[1]

    all_data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)

    phi_i_dk   = all_data['boxlib', 'phi0'].to_ndarray()[:,:]
    phi_i_ptcl = all_data['boxlib', 'phi1'].to_ndarray()[:,:]
    phi_i_dk *= dx*dy
    phi_i_ptcl *= dx*dy

    total_data_dk[i, :, :] = phi_i_dk[...,0]
    total_data_ptcl[i, :, :] = phi_i_ptcl[...,0]
    print(i, time_i)

np.savez(dataset_name+"_total_data_file",total_data_dk=total_data_dk,
         total_data_ptcl = total_data_ptcl)
