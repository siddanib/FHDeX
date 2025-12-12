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

plt.rcParams['font.size'] = 25
torch.set_default_device('cuda')

dt = 3.0e-6
dx = 0.01

directory = './multirun/2025-12-10/12-06-38/0/'
#directory = './multirun/2025-12-10/12-37-30/3'
#directory = './multirun/2025-12-11/08-54-22/0'

npz_file = "total_data.npz"

data_npz = np.load(os.path.join(directory,npz_file))

total_density_np = data_npz['density_data']
total_flux_np    = data_npz['flux_data']

left_reservoir = np.mean(total_density_np[:,:,0])
right_reservoir = np.mean(total_density_np[:,:,-1])

print(f"Left reservoir: {left_reservoir}")
print(f"Right reservoir: {right_reservoir}")

total_density_tnsr = torch.tensor(total_density_np)
total_flux_tnsr    = torch.tensor(total_flux_np)

# Get theoretical flux
total_theory_flux = torch.zeros_like(total_flux_tnsr)
# Left number of particles
left_density  = total_density_tnsr[:, :-1, 1]
right_density = total_density_tnsr[:, :-1, 2]

total_theory_flux[:,:,0] = (0.5*dt/(dx*dx))*(left_density - right_density)
total_theory_flux[:,:,0] += ((dt**0.5)/(dx))*(0.5*(torch.sqrt(torch.clamp(left_density,min=0.)) +
                                                   torch.sqrt(torch.clamp(right_density,min=0.))
                                                   ))*torch.randn_like(left_density)
cumsum_flux_tnsr = torch.cumsum(total_flux_tnsr,dim=1)

cumsum_theory_tnsr = torch.cumsum(total_theory_flux,dim=1)

cumsum_flux_np = cumsum_flux_tnsr.cpu().numpy()

cumsum_mn = torch.mean(cumsum_flux_tnsr, dim=0)
cumsum_var = torch.var(cumsum_flux_tnsr, dim=0)

cumsum_mn_np = cumsum_mn.cpu().numpy()
cumsum_var_np = cumsum_var.cpu().numpy()

cumsum_skew_np = skew(cumsum_flux_np,axis=0)
cumsum_kurt_np = kurt(cumsum_flux_np,axis=0, fisher=False)

cumsum_theory_mn = torch.mean(cumsum_theory_tnsr, dim=0)
cumsum_theory_var = torch.var(cumsum_theory_tnsr, dim=0)

cumsum_theory_mn_np = cumsum_theory_mn.cpu().numpy()
cumsum_theory_var_np = cumsum_theory_var.cpu().numpy()

n_steps = cumsum_mn_np.size

time_array = np.linspace(0,n_steps-1,num=n_steps)*dt

# Note that this is difference
fig, ax = plt.subplots(figsize=(10, 10))
#ax.loglog(time_array, cumsum_mn_np - cumsum_theory_mn_np)
ax.semilogx(time_array, (cumsum_mn_np - cumsum_theory_mn_np)/cumsum_mn_np)
ax.grid(True)
ax.set_xlabel("Time")
ax.set_ylabel(r"${Mean(\sum_{k=0}^{t} F^{k})}_{P} - {Mean(\sum_{k=0}^{t} F^{k})}_{T}$")
ax.set_title(f"Left reservoir = {left_reservoir:.2e}\n"+
             f"Right reservoir = {right_reservoir:.2e}"
            )
plt.tight_layout()
plt.savefig(f"diff_mean_L_{int(left_reservoir)}_R_{int(right_reservoir)}.jpg")
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
#ax.loglog(time_array, cumsum_mn_np, label="Particle Simulation")
#ax.loglog(time_array, cumsum_theory_mn_np, label="Analytical")
ax.semilogx(time_array, cumsum_mn_np, label="Particle Simulation")
ax.semilogx(time_array, cumsum_theory_mn_np, label="Analytical")
ax.grid(True)
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel(r"Mean$(\sum_{k=0}^{t} F^{k})$")
ax.set_title(f"Left reservoir = {left_reservoir:.2e}\n"+
             f"Right reservoir = {right_reservoir:.2e}"
            )
plt.tight_layout()
plt.savefig(f"mean_L_{int(left_reservoir)}_R_{int(right_reservoir)}.jpg")
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
#ax.loglog(time_array, cumsum_var_np, label="Particle Simulation" )
#ax.loglog(time_array, cumsum_theory_var_np, label="Analytical")
ax.semilogx(time_array, cumsum_var_np, label="Particle Simulation" )
ax.semilogx(time_array, cumsum_theory_var_np, label="Analytical")
ax.grid(True)
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel(r"Var$(\sum_{k=0}^{t} F^{k})$")
ax.set_title(f"Left reservoir = {left_reservoir:.2e}\n"+
             f"Right reservoir = {right_reservoir:.2e}"
            )
plt.tight_layout()
plt.savefig(f"var_L_{int(left_reservoir)}_R_{int(right_reservoir)}.jpg")
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
ax.semilogx(time_array, cumsum_skew_np, label="Particle Simulation")
ax.grid(True)
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel(r"Skewness$(\sum_{k=0}^{t} F^{k})$")
ax.set_title(f"Left reservoir = {left_reservoir:.2e}\n"+
             f"Right reservoir = {right_reservoir:.2e}"
            )
plt.tight_layout()
plt.savefig(f"skew_L_{int(left_reservoir)}_R_{int(right_reservoir)}.jpg")
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
ax.semilogx(time_array, cumsum_kurt_np, label="Particle Simulation")
ax.grid(True)
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel(r"Kurtosis$(\sum_{k=0}^{t} F^{k})$")
ax.set_title(f"Left reservoir = {left_reservoir:.2e}\n"+
             f"Right reservoir = {right_reservoir:.2e}"
            )
plt.tight_layout()
plt.savefig(f"kurt_L_{int(left_reservoir)}_R_{int(right_reservoir)}.jpg")
plt.show()
