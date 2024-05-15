
from os.path import exists
import numpy as np

import h5py
import imageio.v3 as iio

from core import add_dataset, bsPath
import plotting as plt

def compute_SFMS_percentile_limits(simName='TNG50-1', snapNum=99) :
    
    # define the input directory, input file, and the output helper file
    inDir = bsPath(simName)
    sample_file = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    helper_file = inDir + '/{}_{}_SFMS_helper.hdf5'.format(simName, snapNum)
    
    # get the stellar masses and SFHs for the entire sample
    with h5py.File(sample_file, 'r') as hf :
        snaps = hf['snapshots'][:]
        logM = hf['logM'][:]
        SFHs = hf['SFH'][:]
        exclude = hf['exclude'][:] # 103 galaxies should be excluded, see sfhs.py
        # subIDfinals = hf['SubhaloID'][:]
    
    # create the helper file
    if not exists(helper_file) :
        hf = h5py.File(helper_file, 'w')
        hf.close()
    
    # loop over every snapshot
    for snap in snaps :
        
        # get the stellar masses, SFRs in correct units, and the quiescent mask
        masses, SFRs, q_mask = get_logM_and_SFRs_at_snap(logM, SFHs, snap, exclude)
        
        # save a basic version of the plot
        # outfile = '{}/figures/SFMS(t)_without-bands/SFMS_{}.png'.format(simName, snap)
        # plt.plot_scatter(masses, SFRs, 'k', '', 'o',
        #     xlabel=r'$\log(M_{*}/{\rm M}_{\odot})$',
        #     ylabel=r'$\log({\rm SFR}/{\rm M}_{\odot}~{\rm yr}^{-1}$)',
        #     xmin=6.8, xmax=12.8, ymin=-6.4, ymax=3, outfile=outfile, save=True)
        
        # define the edges for the mass bins, using only galaxies with both
        # a good mass and a good SFR
        temp = masses[~(np.isnan(masses) | np.isnan(SFRs))]
        temp = temp[temp > 4.45]
        start = (np.min(temp)//0.2)*0.2
        end = (np.max(temp)//0.2)*0.2 + 0.3 # add 0.3 instead of 0.2 to
        edges = np.arange(start, end, 0.2)  # account for floating point errors
        
        # find galaxies that have active SF
        SF_logM, SF_SFRs = masses[~q_mask], SFRs[~q_mask]
        
        # now find the 16th and 84th percentiles for the SFR of galaxies in
        # each mass bin
        centers, los, his = [], [], [] # centers is only used for plotting
        for first, second in zip(edges, edges[1:]) :
            centers.append(np.mean([first, second]))
            
            bin_mask = (SF_logM >= first) & (SF_logM < second)
            
            # we require at least one galaxy to take the percentiles
            if np.sum(bin_mask) > 1 :
                lo, hi = np.nanpercentile(SF_SFRs[bin_mask], [16, 84])
            else :
                lo, hi = np.nan, np.nan
            
            # append those values
            los.append(lo)
            his.append(hi)
        
        # save the 16th and 84th percentiles for future use
        if exists(helper_file) :
            with h5py.File(helper_file, 'a') as hf :
                add_dataset(hf, edges, 'Snapshot_{}/edges'.format(snap))
                add_dataset(hf, np.array(los), 'Snapshot_{}/los'.format(snap))
                add_dataset(hf, np.array(his), 'Snapshot_{}/his'.format(snap))
        
        # plot the SFMS with those percentiles
        outfile = '{}/figures/SFMS(t)_before-checking/SFMS_{}.png'.format(simName, snap)
        plt.plot_scatter_with_bands(masses, SFRs, centers, los, his,
            xlabel=r'$\log(M_{*}/{\rm M}_{\odot})$',
            ylabel=r'$\log({\rm SFR}/{\rm M}_{\odot}~{\rm yr}^{-1}$)',
            xmin=6.8, xmax=12.8, ymin=-6.4, ymax=3, outfile=outfile, save=True)
    
    return

def get_logM_and_SFRs_at_snap(logM, SFHs, snap, exclude) :
    
    # get the stellar masses and SFHs at the given snapshot
    masses, SFRs = logM[:, snap], SFHs[:, snap]
    
    # mask out NaN values
    masses = np.where(~exclude, masses, np.nan)
    SFRs = np.where(~exclude, SFRs, np.nan)
    
    # define a mask for the quiescent population
    q_mask = (SFRs == 0)
    
    # update values in the array to prepare for taking the logarithm
    SFRs[q_mask] = 1e-5
    SFRs = np.log10(SFRs)
    
    # perturb the quiescent values
    SFRs[q_mask] = -5 - np.random.rand(np.sum(q_mask))
    
    return masses, SFRs, q_mask

def determine_SFMS(simName='TNG50-1', snapNum=99) :
    
    # define the input directory, input file, and the output helper file
    inDir = bsPath(simName)
    sample_file = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    helper_file = inDir + '/{}_{}_SFMS_helper.hdf5'.format(simName, snapNum)
    
    # get the stellar masses and SFHs for the entire sample
    with h5py.File(sample_file, 'r') as hf :
        snaps = hf['snapshots'][:]
        logM = hf['logM'][:]
        SFHs = hf['SFH'][:]
        exclude = hf['exclude'][:] # 103 galaxies should be excluded, see sfhs.py
    
    # the edges, 16th, and 84th percentiles will be loaded from the helper file
    helper = h5py.File(helper_file, 'r')
    
    # loop over every snapshot
    for snap in snaps :
        
        # get the stellar masses and SFRs in correct units
        masses, SFRs, _ = get_logM_and_SFRs_at_snap(logM, SFHs, snap, exclude)
        
        # get the edges, 16th, and 84th percentile limits
        edges = helper['Snapshot_{}/edges'.format(snap)][:]
        los = helper['Snapshot_{}/los'.format(snap)][:]
        his = helper['Snapshot_{}/his'.format(snap)][:]
        
        # check if each galaxy is on the SFMS at the given snapshot
        SFMS_at_snap, below_SFMS_at_snap = [], []
        for mass, SFR in zip(masses, SFRs) :
            
            # if the mass and SFR are intact find the mass bin where the galaxy
            if (np.isfinite(mass) & np.isfinite(SFR)) : # is located
                idx = np.where(mass < edges)[0][0] - 1
                
                # determine if the galaxy is on the SFMS
                SFMS_quality = (SFR >= los[idx]) & (SFR <= his[idx])
                SFMS_at_snap.append(SFMS_quality)
                below_SFMS_at_snap.append(SFR < los[idx])
            else : # if the mass isn't intact then the galaxy isn't on the SFMS
                SFMS_at_snap.append(False)
                below_SFMS_at_snap.append(False)
        
        # place the values at the given snapshot into the array for all snapshots
        with h5py.File(sample_file, 'a') as hf :
            hf['SFMS'][:, snap] = np.array(SFMS_at_snap).astype(bool)
            hf['below_SFMS'][:, snap] = np.array(below_SFMS_at_snap).astype(bool)
    
    helper.close()
    
    return

def save_SFMS_evolution_gif(simName='TNG50-1', snapNum=99) :
    
    # define the input directory, input file, and the output helper file
    inDir = bsPath(simName)
    sample_file = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    helper_file = inDir + '/{}_{}_SFMS_helper.hdf5'.format(simName, snapNum)
    
    # get the stellar masses and SFHs for the entire sample
    with h5py.File(sample_file, 'r') as hf :
        snaps = hf['snapshots'][:]
        logM = hf['logM'][:]
        SFHs = hf['SFH'][:]
        SFMS = hf['SFMS'][:]
        exclude = hf['exclude'][:] # 103 galaxies should be excluded, see sfhs.py
    
    # the edges, 16th, and 84th percentiles will be loaded from the helper file
    helper = h5py.File(helper_file, 'r')
    
    # loop over every snapshot
    for snap in snaps :
        
        # get the stellar masses, SFRs in correct units, and the quiescent mask
        masses, SFRs, q_mask = get_logM_and_SFRs_at_snap(logM, SFHs, snap, exclude)
        
        # get the centers, 16th and 84th percentiles
        centers = helper['Snapshot_{}/edges'.format(snap)][:-1] + 0.1
        los = helper['Snapshot_{}/los'.format(snap)][:]
        his = helper['Snapshot_{}/his'.format(snap)][:]
        
        # define a mask for the SFMS, and create three populations
        SFMS_mask = SFMS[:, snap].astype(bool)
        SFMS_masses, SFMS_SFRs = masses[SFMS_mask], SFRs[SFMS_mask]
        
        q_masses, q_SFRs = masses[q_mask], SFRs[q_mask]
        
        other_mask = (~SFMS_mask) & (~q_mask)
        other_masses, other_SFRs = masses[other_mask], SFRs[other_mask]
        
        # plot the SFMS with those percentiles
        outfile = 'TNG50-1/figures/SFMS(t)/SFMS_{}.png'.format(snap)
        plt.plot_scatter_multi_with_bands(SFMS_masses, SFMS_SFRs,
            q_masses, q_SFRs, other_masses, other_SFRs, centers, los, his,
            xlabel=r'$\log(M_{*}/{\rm M}_{\odot})$',
            ylabel=r'$\log({\rm SFR}/{\rm M}_{\odot}~{\rm yr}^{-1}$)',
            xmin=6.4, xmax=12.8, ymin=-6.4, ymax=3,
            outfile=outfile, save=False)
    
    helper.close()
    
    frames = [iio.imread(f'TNG50-1/figures/SFMS(t)/SFMS_{i}.png') for i in range(100)]
    iio.imwrite('TNG50-1/figures/SFMS(t).gif', np.stack(frames, axis=0))
    
    return
