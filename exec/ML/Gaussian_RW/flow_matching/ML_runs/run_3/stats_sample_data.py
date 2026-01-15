import sys
import os
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 25})

#parent_folder = "./multirun/2026-01-08/17-59-17/"
#parent_folder = "./multirun/2026-01-12/11-54-41/"
#parent_folder = "./multirun/2026-01-12/17-50-08/"
#parent_folder = "./multirun/2026-01-13/14-11-08/"
#parent_folder = "./multirun/2026-01-14/12-25-39/"
parent_folder = "./outputs/2026-01-15/13-04-50/"

dir_list = []

for itm in os.listdir(parent_folder):
    # Construct full path
    full_path = os.path.join(parent_folder, itm)
    # Check if it's a directory
    if os.path.isdir(full_path):
        if ".hydra" in full_path:
            continue
        dir_list.append(full_path)

data_array = np.zeros((len(dir_list), 10))

for ii, directory in enumerate(dir_list):
    npz_file = "total_data.npz"
    npz_file = os.path.join(directory,npz_file)
    aa = np.load(npz_file)
    n_left = aa['n_left']
    n_right = aa['n_right']
    grnd_trth_np = aa['particle_flux_data']
    mdl_data_np  = aa['model_flux_data']

    # What if I round the data I obtain from the model
    #mdl_data_np = np.round(mdl_data_np)

    data_array[ii, 0] = n_left
    data_array[ii, 1] = n_right
    # Means 
    data_array[ii, 2] = np.mean(grnd_trth_np)
    data_array[ii, 3] = np.mean(mdl_data_np)
    # Std
    data_array[ii, 4] = np.std(grnd_trth_np)
    data_array[ii, 5] = np.std(mdl_data_np)
    # Skewness
    data_array[ii, 6] = skew(grnd_trth_np, axis=None)
    data_array[ii, 7] = skew(mdl_data_np, axis=None)
    # Kurtosis
    data_array[ii, 8] = kurtosis(grnd_trth_np, axis=None, fisher=False)
    data_array[ii, 9] = kurtosis(mdl_data_np, axis=None, fisher=False)

    #data_test = np.zeros(4)
    #data_test[0] = np.mean(grnd_trth_np) - np.mean(mdl_data_np)
    #data_test[1] = np.std(grnd_trth_np)  - np.std(mdl_data_np)
    #data_test[2] = skew(grnd_trth_np,axis=None) - skew(mdl_data_np,axis=None)
    #data_test[3] = (kurtosis(grnd_trth_np,axis=None,fisher=False) -
    #                    kurtosis(mdl_data_np,axis=None, fisher=False))

    #print(f"N_Left = {n_left}, N_right = {n_right}")
    #print("Particle stats: Mean, Std, Skewness, Kurtosis")
    #print(np.mean(grnd_trth_np), np.std(grnd_trth_np),
    #      skew(grnd_trth_np,axis=None),
    #      kurtosis(grnd_trth_np,axis=None,fisher=False))

    #print("Model stats: Mean, Std, Skewness, Kurtosis")
    #print(np.mean(mdl_data_np), np.std(mdl_data_np),
    #      skew(mdl_data_np,axis=None),
    #      kurtosis(mdl_data_np,axis=None,fisher=False))

    #print(np.abs(data_test))

n_start = 8
title_string = "Kurtosis"

plot_data = data_array[:,n_start:n_start+2]
print(np.mean(np.abs(plot_data[:,0]-plot_data[:,1])))


fig, ax  = plt.subplots(1,1,figsize=(10,10))
scatter = ax.scatter(plot_data[:,0], plot_data[:,1])
min_val = np.min(plot_data)
max_val = np.max(plot_data)
xy_line = np.linspace(min_val,max_val)
ax.plot(xy_line,xy_line,color="red",label="x=y")

ax.set_title(title_string)
ax.set_xlabel('Particle Simulation')
ax.set_ylabel('ML')
#ax.set_xlabel(r'$N_{L}$')
#ax.set_ylabel(r'$N_{R}$')
ax.grid(True)
plt.tight_layout()
#plt.savefig(title_string+".jpeg")
plt.show()
