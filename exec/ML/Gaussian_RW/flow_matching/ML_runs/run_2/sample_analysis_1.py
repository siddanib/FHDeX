##### THis script is for history_length == 1
import sys
import os
import numpy as np
import math
import torch
import h5py
import yaml
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from datetime import datetime
#######################################################
plt.rcParams.update({'font.size': 25}) 
torch.set_default_device('cpu')

@torch.no_grad()
def fhd_model_run ():
    n_left = 25
    n_right = 25 
    history_length = 1
    #parent_folder_list = [ "multirun/2025-10-30/11-32-44/",
    #                       "multirun/2025-10-30/11-32-44/",
    #                     ]

    parent_folder_list = [ "multirun/2025-11-01/11-32-36/",
                           "multirun/2025-11-01/11-32-36/",
                         ]
    max_level_list = [ 0, 1]

    fig, ax = plt.subplots(figsize=(12, 12))

    for i_fldr, parent_folder in enumerate(parent_folder_list):
        max_level  =  max_level_list[i_fldr]
        print(f"n_left = {n_left}, n_right = {n_right}, max_level = {max_level}")
        # Go through the parent folder to get the correct directory
        for entry in os.listdir(parent_folder):
            full_path = os.path.join(parent_folder, entry)
            # Check whether it is a folder
            if not os.path.isdir(full_path):
                continue
            # Load the yaml file to see if it is the right one
            yaml_string = os.path.join(full_path, ".hydra/config.yaml")
            with open(yaml_string,"r") as yaml_file:
                data_yaml = yaml.safe_load(yaml_file)

            if n_left !=  data_yaml["n_left"]:
                continue

            if n_right !=  data_yaml["n_right"]:
                continue

            if max_level !=  data_yaml["max_level"]:
                continue

            if history_length !=  data_yaml["model"]["history_length"]:
                continue

            # Load the h5 file
            dataset_name = os.path.join(full_path,
                                f"samples_two_cells_{int(n_left)}_{int(n_right)}")

            with h5py.File(dataset_name+".h5", mode="r") as f:
                grnd_trth_np = f["ground_truth_data"][:]
                mdl_data_np  = f["model_data"][:]

            gt_min = np.floor(np.min(grnd_trth_np)) - 0.5
            gt_max = np.ceil(np.max(grnd_trth_np)) + 0.5
            gt_pdf, _ = np.histogram(grnd_trth_np,
                         bins= np.linspace(gt_min, gt_max, int(gt_max-gt_min+1)),
                                  density=True)

            if i_fldr == 0:
                plt.plot(np.linspace(gt_min+0.5, gt_max-0.5, int(gt_max-gt_min)),
                         gt_pdf, marker="*", label="Particle Simulation")

            mdl_min = np.floor(np.min(mdl_data_np)) - 0.5
            mdl_max = np.ceil(np.max(mdl_data_np)) + 0.5
            mdl_pdf, _ = np.histogram(mdl_data_np,
                         bins= np.linspace(mdl_min, mdl_max, int(mdl_max-mdl_min+1)),
                                  density=True)

            if max_level == 0:
                plt.plot(np.linspace(mdl_min+0.5, mdl_max-0.5, int(mdl_max-mdl_min)),
                         mdl_pdf, marker="o", label="ML-No history")
            else:
                plt.plot(np.linspace(mdl_min+0.5, mdl_max-0.5, int(mdl_max-mdl_min)),
                         mdl_pdf, marker="s", label=f"ML-{max_level} history")

            data_test = np.zeros(4)
            data_test[0] = np.mean(grnd_trth_np) - np.mean(mdl_data_np)
            data_test[1] = np.std(grnd_trth_np)  - np.std(mdl_data_np)
            data_test[2] = skew(grnd_trth_np,axis=None) - skew(mdl_data_np,axis=None)
            data_test[3] = (kurtosis(grnd_trth_np,axis=None,fisher=False) -
                                kurtosis(mdl_data_np,axis=None, fisher=False))

            print(np.mean(grnd_trth_np), np.std(grnd_trth_np),
                  skew(grnd_trth_np,axis=None),
                  kurtosis(grnd_trth_np,axis=None,fisher=False))

            print(np.mean(mdl_data_np), np.std(mdl_data_np),
                  skew(mdl_data_np,axis=None),
                  kurtosis(mdl_data_np,axis=None,fisher=False))

            print(np.abs(data_test))

    ax.set_title(r'$(N_{L}^{(t-1)}, N_{R}^{(t-1)} )$'+ f" = ({n_left}, {n_right})"+ "\n"+
                 f"realizations = {grnd_trth_np.size}",
                 fontsize=35)
    ax.legend(fontsize=20)
    fig.tight_layout()
    fig.savefig(f'pdf_n_left_{n_left}_n_right_{n_right}_no_noise.jpg')
    plt.show()

if __name__ == "__main__":
    fhd_model_run()
