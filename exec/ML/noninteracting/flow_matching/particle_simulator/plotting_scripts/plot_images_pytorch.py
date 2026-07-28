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
    ppc_id = 2
    ## dataset_id = {0, 1}
    ## correspond to {periodic, reservoir} boundaries
    dataset_id = 1
    n_realizations = 50

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
    auto_corr = np.zeros((2,n_steps+1))
    ###################################################################
    #### Reading all files now
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
            print(data_tensor.device)
            i_ac = (torch.mean(data_tensor**2)
                    - torch.mean(data_tensor)**2)

            auto_corr[i, 0] += i_ac.cpu().numpy()

            for j in range(1,n_steps+1):                                            
                aa = data_tensor[:,:-j,:]                                        
                bb = data_tensor[:,j:,:]
                cc = torch.mean(aa*bb)
                cc -= torch.mean(aa)*torch.mean(bb)
                auto_corr[i, j] += cc.cpu().numpy()

    auto_corr /= n_realizations
    save_fig = True                                                             
    show_fig = False                                                            
    fig, ax = plt.subplots(figsize=(10, 10))                                    
    # Plot data on the axes                                                     
    ax.plot(auto_corr[0,:], color='blue', linestyle='-',                        
            marker='o', markersize=10, label="Particle")
    ax.plot(auto_corr[1,:], color='red', linestyle='-',
            marker='s', markersize=10, label="SPDE")
    # Add labels and title
    ax.set_xlabel(r'$k$', fontsize=35)
    ax.set_title(r'$<N(t) N(t+k)> - <N(t)> <N(t+k)>$'+ "\n"+
       f"cfl = {cfl:.2e}, ensembles : {n_realizations*np.size(n_version_data,axis=0)}"+
       "\n"+r"<N> = "+f"{np.mean(n_ptcl_data):.2e}",
                 fontsize=35)
    #ax.set_xscale("log")
    ## Add a legend
    ax.legend(fontsize=25)
    ax.grid(True)
    fig.tight_layout()
    if save_fig:
        fig.savefig('auto_correlation_2.jpg')
    if show_fig:
        plt.show()

    print(auto_corr[:,0])

    return None

if __name__ == "__main__":
    torch.set_default_device('cuda')
    fhd_model_run()
