
from os.path import exists
import numpy as np

import h5py

from core import (add_dataset, bsPath, determine_mass_bin_indices, find_nearest,
                  get, mpbCutoutPath)

def determine_mpb_cutouts_to_download(simName='TNG50-1', snapNum=99,
                                      hw=0.1, minNum=50, save=True) :
    
    # define the input directory and file, and output file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    outfile = inDir + '/{}_{}_mpb_cutouts_to_download.hdf5'.format(simName, snapNum)
    
    # get the relevant information for the quenched sample
    with h5py.File(infile, 'r') as hf :
        times = hf['times'][:]
        subIDs = hf['SubhaloID'][:]
        logM = hf['logM'][:]
        SFMS = hf['SFMS'][:].astype(bool) # (boolean) SFMS at each snapshot
        quenched = hf['quenched'][:]
        ionsets = hf['onset_indices'][:].astype(int)
        tonsets = hf['onset_times'][:]
        iterms = hf['termination_indices'][:].astype(int)
        tterms = hf['termination_times'][:]
    
    with h5py.File('TNG50-1/all_subIDs.hdf5', 'r') as hf :
        all_subIDs = hf['subIDs'][:].astype(int)
    
    # define the snapshots in the simulation
    snapshots = np.arange(100)
    
    # let's only use quenched galaxies with 8 <= logM <= 11
    last_logM = logM[:, -1]
    mask = quenched & (last_logM >= 8) & (last_logM < 11)
    q_subIDs = subIDs[mask]
    q_logM = logM[mask]
    q_ionsets = ionsets[mask]
    q_tonsets = tonsets[mask]
    q_iterms = iterms[mask]
    q_tterms = tterms[mask]
    
    # these galaxies have a termination index before an onset index -> update
    # onset index to only before termination? this is simple enough
    bad_mask = (q_iterms - q_ionsets < 0)
    # print(q_subIDs[bad_mask])
    good = ~bad_mask
    q_subIDs = q_subIDs[good]
    q_logM = q_logM[good]
    q_ionsets = q_ionsets[good]
    q_tonsets = q_tonsets[good]
    q_iterms = q_iterms[good]
    q_tterms = q_tterms[good]
    
    # loop through the quenched galaxies in the sample
    snap_list, subIDs_to_download = [], []
    for q_subID, q_logM, q_ionset, q_tonset, q_iterm, q_tterm in zip(q_subIDs,
        q_logM, q_ionsets, q_tonsets, q_iterms, q_tterms):
        
        # work through the snapshots from onset until termination
        # we don't need to worry about when the galaxy is above logM = 10^4.45
        # as the onset indices are all greater than `init` from quenched.py
        
        # use alternative times based on the final snapshot being 75% of the
        # quenching mechanism duration
        index_times = np.array([q_tonset,
                                q_tonset + 0.375*(q_tterm - q_tonset),
                                q_tonset + 0.75*(q_tterm - q_tonset)])
        indices = find_nearest(times, index_times)
        
        for snap, mass in zip(snapshots[indices], # snapshots[q_ionset:q_iterm+1]
                              q_logM[indices]) :  # q_logM[q_ionset:q_iterm+1]
            
            # get values at the snapshot
            logM_at_snap = logM[:, snap]
            SFMS_at_snap = SFMS[:, snap]
            all_subIDs_at_snap = all_subIDs[:, snap]
            
            # create a mask for the SFMS galaxy masses at that snapshot
            SFMS_at_snap_masses_mask = np.where(SFMS_at_snap > 0,
                                                logM_at_snap, False)
            
            # find galaxies in a similar mass range as the galaxy, but that
            # are on the SFMS at that snapshot
            mass_bin = determine_mass_bin_indices(SFMS_at_snap_masses_mask,
                mass, hw=hw, minNum=minNum)
            
            # find the subIDs for the comparison control sample at each snapshot
            control_subIDs_at_snap = all_subIDs_at_snap[mass_bin]
            for ID in control_subIDs_at_snap :
                snap_list.append(snap)
                subIDs_to_download.append(ID)
    
    concatenated = []
    for snap, ID in zip(snap_list, subIDs_to_download) :
        val = '{}_{}'.format(snap, ID)
        concatenated.append(val)
    
    snaps, IDs = [], []
    for val in np.unique(concatenated) :
        snap, ID = val.split('_')
        snaps.append(int(snap))
        IDs.append(int(ID))
    
    if save :
        with h5py.File(outfile, 'w') as hf :
            add_dataset(hf, np.array(snaps), 'list_of_snaps')
            add_dataset(hf, np.array(IDs), 'list_of_subIDs')
    
    return snaps, IDs

def download_mpb_cutouts(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and file, and output directory for the mpb cutouts
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_mpb_cutouts_to_download.hdf5'.format(simName, snapNum)
    outDir = mpbCutoutPath(simName, snapNum)
    
    # get the mpb cutouts to download
    with h5py.File(infile, 'r') as hf :
        list_of_snaps = hf['list_of_snaps'][:]
        list_of_subIDs = hf['list_of_subIDs'][:]
    
    # define the parameters that are requested for each particle in the cutout
    params = {'stars':'Coordinates,GFM_InitialMass,GFM_StellarFormationTime'}
    
    # loop over all the required mpb cutouts
    for snap, subID in zip(list_of_snaps, list_of_subIDs) :
        
        # define the URL for the galaxy at the redshift of interest
        url = 'https://www.tng-project.org/api/{}/snapshots/{}/subhalos/{}'.format(
            simName, snap, subID)
        
        # save the cutout file into the output directory if it doesn't exist
        filename = 'cutout_{}_{}.hdf5'.format(snap, subID)
        if not exists(outDir + filename) :
            get(url + '/cutout.hdf5', directory=outDir, params=params,
                filename=filename)
    
    return

def estimate_remaining_time(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and file, and output directory for the mpb cutouts
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_mpb_cutouts_to_download.hdf5'.format(simName, snapNum)
    outDir = mpbCutoutPath(simName, snapNum)
    
    # get the mpb cutouts to download
    with h5py.File(infile, 'r') as hf :
        list_of_snaps = hf['list_of_snaps'][:]
        list_of_IDs = hf['list_of_subIDs'][:]
    
    # check if the files exist
    to_download, downloaded = len(list_of_snaps), 0
    for snap, subID in zip(list_of_snaps, list_of_IDs) :
        if exists(outDir + 'cutout_{}_{}.hdf5'.format(snap, subID)) :
            downloaded +=1
    
    # determine the remaining number of files to download, and use an estimate
    # for the download rate
    remaining = to_download - downloaded
    rate = 2000 # ~2000 files per hour
    
    print('{} files remaining, at ~2000 files/hr -> {:.2f} hr remaining'.format(
        remaining, remaining/rate))
    
    return
