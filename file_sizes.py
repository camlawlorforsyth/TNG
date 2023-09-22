
import os
from os.path import exists, getsize
import numpy as np

import h5py

from core import determine_mass_bin_indices, get_mpb_radii_and_centers

with h5py.File('TNG50-1/TNG50-1_99_sample_SFHs(t).hdf5', 'r') as hf :
    times = hf['times'][:]
    SFMS = hf['SFMS'][:] # 6337 galaxies
    subIDs = hf['SubhaloID'][:]
    masses = hf['SubhaloMassStars'][:]

with h5py.File('TNG50-1/TNG50-1_99_quenched_SFHs(t).hdf5', 'r') as hf :
    q_subIDs = hf['SubhaloID'][:]
    q_masses = hf['SubhaloMassStars'][:]
    onset_indices = hf['onset_indices'][:].astype(int)
    # tonsets = hf['onset_times'][:]
    term_indices = hf['termination_indices'][:].astype(int)
    # tterms = hf['termination_times'][:]

massive = (q_masses > 10.17)

IDs = q_subIDs[massive]
onsets = onset_indices[massive]
terms = term_indices[massive]
logMs = q_masses[massive]

sort = np.argsort(logMs)
IDs = IDs[sort]
onsets = onsets[sort]
terms = terms[sort]
logMs = logMs[sort]

'''
number, los, his = [], [], []
for ID, onset, term, logM in zip(IDs, onsets, terms, logMs) :
    mass_bin = determine_mass_bin_indices(masses[SFMS], logM, hw=0.1, minNum=50)
    los.append(logM - np.min(masses[SFMS][mass_bin]))
    his.append(np.max(masses[SFMS][mass_bin]) - logM)
    number.append(len(masses[SFMS][mass_bin]))
import plotting as plt
plt.plot_scatter_err(logMs, number, los, his, xmin=10, xmax=12, ymin=45, ymax=200,
                     xlabel=r'$\log(M_{*}/{\rm M}_{\odot}$',
                     ylabel='# control galaxies within 0.1 dex, min 50 required')
'''

def check_minimum_data_volume(delete=False) :
    
    cutouts, missing, size = [], [], 0
    for ID, onset, term, logM in zip(IDs, onsets, terms, logMs) :
        
        # get basic MPB info about the quenched galaxy
        snapNums, mpb_IDs, _, _ = get_mpb_radii_and_centers('TNG50-1', 99, ID)
        snaps = snapNums[onset:term+1] # we want information from the onset
        # until the termination, but including the termination
        
        # loop over all the snapshots from the onset until the termination
        for snap in snaps :
            cutout = 'TNG50-1/mpb_cutouts_099/cutout_{}_{}.hdf5'.format(
                snap, mpb_IDs[snap])
            if cutout not in cutouts :
                cutouts.append(cutout)
                # size += getsize(cutout) # getsize returns size in bytes
        
        # loop over the control galaxies
        mass_bin = determine_mass_bin_indices(masses[SFMS], logM, hw=0.1,
                                              minNum=50)
        control = subIDs[SFMS][mass_bin]
        
        for control_ID in control :
            _, control_mpb_IDs, _, _ = get_mpb_radii_and_centers(
                'TNG50-1', 99, control_ID)
            
            for snap in snaps :
                cutout = 'TNG50-1/mpb_cutouts_099/cutout_{}_{}.hdf5'.format(
                    snap, control_mpb_IDs[snap])
                if exists(cutout) :
                    if cutout not in cutouts :
                        cutouts.append(cutout)
                        # size += getsize(cutout) # getsize returns size in bytes
                else :
                    missing.append(cutout)
        
        # string = ('subID {:6}, logM = {:.2f}, '.format(ID, logM) +
        #           'delta_t = {:.1f} Gyr, '.format(times[term]-times[onset]) +
        #           'total size = {:.2f} TB, '.format(size/1099511627776) +
        #           'missing {} files'.format(len(np.unique(missing))))
        # print(string)
    
    return

def delete_unnecessary_files(unnecessary_files) :
    
    for file in unnecessary_files :
        os.remove(file)
    
    return

# all_files = np.load('TNG50-1/cutouts_file_list_all.npy')
# necessary_files = np.load('TNG50-1/cutouts_file_list_necessary.npy')
# size = 0
# for file in necessary_files :
#     size += getsize(file)
# print('{:.2f} TB'.format(size/1099511627776))
