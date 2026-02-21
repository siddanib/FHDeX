import sys
import os
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})

#parent_folder = "./outputs/2026-02-20/17-54-45/"
parent_folder = "./outputs/2026-02-20/18-09-46/"

dir_list = [parent_folder,]

#### Which step to look at
n_step = 1

for directory in dir_list:
    npz_file = "total_data.npz"
    npz_file = os.path.join(directory,npz_file)
    aa = np.load(npz_file)

    ptcl_dens_data = aa["particle_density_data"]
    mdl_dens_data  = aa["model_density_data"]

    # Clamp the ML data as it produces negative values
    mdl_dens_data = np.maximum(mdl_dens_data,0.)

    ptcl_dens_mean = np.mean(ptcl_dens_data, axis=0)
    mdl_dens_mean  = np.mean(mdl_dens_data, axis=0)

    ptcl_dens_var = np.var(ptcl_dens_data, axis=0)
    mdl_dens_var  = np.var(mdl_dens_data, axis=0)

    ptcl_dens_skew = skew(ptcl_dens_data, axis=0)
    mdl_dens_skew  = skew(mdl_dens_data, axis=0)

    ptcl_dens_kurt = kurtosis(ptcl_dens_data, axis=0, fisher=False)
    mdl_dens_kurt  = kurtosis(mdl_dens_data, axis=0, fisher = False)


fig, ax  = plt.subplots(2,2,figsize=(12,12))

ax[0,0].plot(ptcl_dens_mean[n_step,:],color="red",label="Particle")
ax[0,0].plot(mdl_dens_mean[n_step,:],color="blue",label="ML")
ax[0,0].set_title(f"Time Step = {n_step}")
ax[0,0].set_xlabel('Cell ID')
ax[0,0].set_ylabel('Mean')
ax[0,0].grid(True)
ax[0,0].legend()

ax[0,1].plot(ptcl_dens_var[n_step,:],color="red",label="Particle")
ax[0,1].plot(mdl_dens_var[n_step,:],color="blue",label="ML")
ax[0,1].set_title(f"Time Step = {n_step}")
ax[0,1].set_xlabel('Cell ID')
ax[0,1].set_ylabel('Variance')
ax[0,1].grid(True)
ax[0,1].legend()

ax[1,0].plot(ptcl_dens_skew[n_step,:],color="red",label="Particle")
ax[1,0].plot(mdl_dens_skew[n_step,:],color="blue",label="ML")
ax[1,0].set_title(f"Time Step = {n_step}")
ax[1,0].set_xlabel('Cell ID')
ax[1,0].set_ylabel('Skewness')
ax[1,0].grid(True)
ax[1,0].legend()

ax[1,1].plot(ptcl_dens_kurt[n_step,:],color="red",label="Particle")
ax[1,1].plot(mdl_dens_kurt[n_step,:],color="blue",label="ML")
ax[1,1].set_title(f"Time Step = {n_step}")
ax[1,1].set_xlabel('Cell ID')
ax[1,1].set_ylabel('Kurtosis')
ax[1,1].grid(True)
ax[1,1].legend()

plt.tight_layout()
plt.savefig(f"n_step_{n_step}"+".jpeg")
plt.show()
