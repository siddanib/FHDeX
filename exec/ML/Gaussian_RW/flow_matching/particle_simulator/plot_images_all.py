import sys                                                                      
import os                                                                       
import numpy as np                                                              
import torch
import math                                                                     
import h5py                                                                     
from scipy.stats import skew                                                    
from scipy.stats import kurtosis as kurt                                        
import matplotlib.pyplot as plt                                                 
#######################################################                         
                                                                                
def fhd_model_run ():
    device =  torch.device("cuda")
    ## ppc_id = {0,1,2}
    ## correspond to {1, 10, 50} average particles per cell
    ## dataset_id = {0, 1}
    ## correspond to {periodic, reservoir} boundaries
    ppc_list = [1, 10, 50]
    dataset_id = 0
    n_realizations = 50

    ppc_id = 0
    fld_start_id = ppc_id*(2*n_realizations) + dataset_id*n_realizations
    #####################################################################
    # Just reading the first file for some information
    subfolder = "./"+str(fld_start_id)+"/"
    if dataset_id == 0:
        dataset_name = os.path.join(subfolder,"uniform")
    elif dataset_id == 1:
        dataset_name = os.path.join(subfolder,"reservoir")
    # Ordering of data (Ensembles, Time Step, Cells)                            
    with h5py.File(dataset_name+".h5", mode="r") as f:                          
        n_ptcl_data = f["ground_truth_data"][:]                                 
        dt          = f['dt'][()]                                               
        dx          = f['dx'][()]
    n_steps = np.size(n_ptcl_data,axis=1) - 1                                   
    cfl = dt/(dx*dx)

    ppc_id_list = [0,1,2]

    auto_corr = np.zeros((len(ppc_id_list), 2,n_steps+1))
        ###################################################################
        #### Reading all files now
    for ppc_id in ppc_id_list:
        fld_start_id = ppc_id*(2*n_realizations) + dataset_id*n_realizations
        for fld_id in range(fld_start_id, fld_start_id+n_realizations):
            subfolder = "./"+str(fld_id)+"/"
            if dataset_id == 0:
                dataset_name = os.path.join(subfolder,"uniform")
            elif dataset_id == 1:
                dataset_name = os.path.join(subfolder,"reservoir")
            # Ordering of data (Ensembles, Time Step, Cells)                            
            with h5py.File(dataset_name+".h5", mode="r") as f:                          
                n_ptcl_data = f["ground_truth_data"][:]                                 
                n_spde_data = f["gauss_data"][:]                                        
                dt          = f['dt'][()]                                               
                dx          = f['dx'][()]
            for i, n_version_data in enumerate([n_ptcl_data, n_spde_data]):
                data_tensor = torch.from_numpy(n_version_data).to(device)
                i_ac = (torch.mean(data_tensor**2)
                        - torch.mean(data_tensor)**2)

                auto_corr[ppc_id, i, 0] += i_ac.cpu().numpy()

                for j in range(1,n_steps+1):                                            
                    aa = data_tensor[:,:-j,:]                                        
                    bb = data_tensor[:,j:,:]
                    cc = torch.mean(aa*bb)
                    cc -= torch.mean(aa)*torch.mean(bb)
                    auto_corr[ppc_id, i, j] += cc.cpu().numpy()

    auto_corr /= n_realizations
    # Normalize each by the zeroth element
    auto_corr[:,:,:] /= auto_corr[:,:,0:1]

    save_fig = True                                                             
    show_fig = False                                                            
    fig, ax = plt.subplots(figsize=(10, 10))                                    
    # Plot data on the axes
    color_list = ["red", "green", "blue"]
    for ppc_id, ppc in enumerate(ppc_list):
        ax.plot(auto_corr[ppc_id, 0,:], color=color_list[ppc_id], linestyle='-',                        
                #marker='o', markersize=10,
                label=f"Particle - {ppc}")
        ax.plot(auto_corr[ppc_id, 1,:], color=color_list[ppc_id], linestyle='--',
                #marker='s', markersize=10,
                label=f"SPDE - {ppc}")
    # Add labels and title
    ax.set_xlabel(r'$k$', fontsize=35)
    ax.set_title(r'$<\delta N(t) \delta N(t+k)>/<\delta N(t)^{2}>$'+ "\n"+
       f"cfl = {cfl:.2e}, ensembles : {n_realizations*np.size(n_version_data,axis=0)}",
                 fontsize=35)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ## Add a legend
    ax.legend(fontsize=25)
    ax.grid(True)
    #ax.set_xlim(0, 250)
    fig.tight_layout()
    if save_fig:
        fig.savefig('auto_correlation_all_log_log.jpg')
    if show_fig:
        plt.show()

    print(auto_corr[0,0,:]-auto_corr[1, 0, :])
    print(auto_corr[1,0,:]-auto_corr[2, 0, :])
    print(auto_corr[2,0,:]-auto_corr[0, 0, :])

    return None

if __name__ == "__main__":
    torch.set_default_device('cuda')
    fhd_model_run()
