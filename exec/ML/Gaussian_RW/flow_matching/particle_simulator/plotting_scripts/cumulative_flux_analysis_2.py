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

#directory = './multirun/2025-12-10/12-06-38/0/'
#directory = './multirun/2025-12-10/12-37-30/3'
directory = './multirun/2025-12-11/08-54-22/0'
npz_file = "total_data.npz"

data_npz = np.load(os.path.join(directory,npz_file))

total_density_np = data_npz['density_data']
total_flux_np    = data_npz['flux_data']

n_steps = np.size(total_flux_np, axis=1)

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

# This is cummulative flux
cumsum_flux_tnsr = torch.cumsum(total_flux_tnsr,dim=1)
cumsum_theory_tnsr = torch.cumsum(total_theory_flux,dim=1)

time_block_mn_ptcl = np.zeros((n_steps-1,))
time_block_mn_theory = np.zeros((n_steps-1,))

for ii in range(1,n_steps):
    aa = cumsum_flux_tnsr[:,:-ii,0]
    bb = cumsum_flux_tnsr[:,ii:,0]
    cc = bb-aa
    time_block_mn_ptcl[ii-1] = torch.mean(cc).cpu().numpy()

    aa = cumsum_theory_tnsr[:,:-ii,0]
    bb = cumsum_theory_tnsr[:,ii:,0]
    cc = bb-aa
    time_block_mn_theory[ii-1] = torch.mean(cc).cpu().numpy()

time_diff = np.linspace(1, n_steps-1, num=n_steps-1)

fig, ax = plt.subplots(figsize=(10, 10))
#ax.loglog(time_diff, time_block_mn_ptcl-time_block_mn_theory)
ax.semilogx(time_diff, time_block_mn_ptcl-time_block_mn_theory)
ax.grid(True)
ax.set_xlabel(r"$b-a$")
ax.set_ylabel(r"${Mean(\sum_{k=a}^{b} F^{k})}_{P} - {Mean(\sum_{k=a}^{b} F^{k})}_{T}$")
ax.set_title(f"Left reservoir = {left_reservoir:.2e}\n"+
             f"Right reservoir = {right_reservoir:.2e}"
            )
plt.tight_layout()
plt.savefig(f"diff_mean_L_{int(left_reservoir)}_R_{int(right_reservoir)}_time_block.jpg")
plt.show()
