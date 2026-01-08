"""
The intention of this script is to see how the stats of flux across a face
look when multiple small "dt" steps are taken.
How does this flux scale when compared to N_L and N_R?
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 25})

nmoves = 50
aa = np.load(f"scaling_analysis_nmoves_{nmoves}.npz")

left_array = aa['left_array']
right_array = aa['right_array']

mean_array = aa['mean_array']
std_array = aa['std_array']

print(mean_array[:,0]/mean_array[:,1])
print(std_array[:,0]/std_array[:,1])
print(left_array - right_array)
print(left_array, right_array)

mean_ratio = mean_array[:,0]/mean_array[:,1]
mean_ratio[np.isinf(mean_ratio)] = np.nan
std_ratio = std_array[:,0]/std_array[:,1]
std_ratio[np.isinf(std_ratio)] = np.nan

print(np.nanmean(mean_ratio))
print(np.nanmean(std_ratio))

#fig, ax  = plt.subplots(1,1,figsize=(10,10))
#scatter = ax.scatter(mean_array[:,1], mean_array[:,0])
#min_val = np.min(mean_array)
#max_val = np.max(mean_array)
#xy_line = np.linspace(min_val,max_val)
#ax.plot(xy_line,xy_line,color="red",label="x=y")
#
#ax.set_title(f'nmoves = {nmoves}, Mean')
#ax.set_xlabel('DK')
#ax.set_ylabel('Particle Simulation')
#ax.grid(True)
#plt.show()

fig, ax  = plt.subplots(1,1,figsize=(10,10))
scatter = ax.scatter(std_array[:,1], std_array[:,0])
min_val = np.min(std_array)
max_val = np.max(std_array)
xy_line = np.linspace(min_val,max_val)
ax.plot(xy_line,xy_line,color="red",label="x=y")

ax.set_title(f'nmoves = {nmoves}, Std')
ax.set_xlabel('DK')
ax.set_ylabel('Particle Simulation')
ax.grid(True)
plt.show()
