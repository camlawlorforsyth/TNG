
from os.path import exists
import pickle
import numpy as np

import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from core import add_dataset, bsPath, get_SFH_limits
import plotting as plt

def determine_quenched_systems(simName, snapNum, mass_bin_edges, kernel=2) :
    
    # define the input directory and the input file for the SFHs
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    
    # open the dictionary containing the limits for the SFMS SFHs
    limits_file = inDir + '/{}_{}_SFMS_SFH_limits(t).pkl'.format(simName, snapNum)
    with open(limits_file, 'rb') as file :
        limits = pickle.load(file)
    
    # open the SFHs for the sample of candidate primary and satellite systems
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        masses = hf['SubhaloMassStars'][:]
        SFHs = hf['SFH'][:]
        times = hf['times'][:]
    
    # add empty quenched mask, onset and termination times into the HDF5
    # file to populate later
    if exists(infile) :
        with h5py.File(infile, 'a') as hf :
            if 'quenched_mask' not in hf.keys() :
                add_dataset(hf, np.full(len(subIDs), False), 'quenched_mask')
            if 'onset_times' not in hf.keys() :
                add_dataset(hf, np.full(len(subIDs), np.nan), 'onset_times')
            if 'termination_times' not in hf.keys() :
                add_dataset(hf, np.full(len(subIDs), np.nan), 'termination_times')
    
    # loop through the galaxies in the sample
    for i, (subID, mass, SFH) in enumerate(zip(subIDs, masses, SFHs)) :
        
        # smooth the SFH of the specific galaxy
        smoothed = gaussian_filter1d(SFH, kernel=kernel)
        smoothed[smoothed < 0] = 0
        
        # get the corresponding lower and upper two sigma limits for that mass
        lo_SFH, hi_SFH = get_SFH_limits(limits, np.array(mass_bin_edges), mass)
        
        # set the indices where the limits don't have to be strictly observed,
        # essentially a small tolerance given the smoothing and low SFRs at
        # early times
        start_lim, end_lim = 3, -3 # originally was -5
        
        # now select galaxies that are within the SFR limits then quench
        lo_diff = smoothed - lo_SFH
        hi_diff = smoothed - hi_SFH
        termination_index = np.argmax(lo_diff[start_lim:] < 0) + start_lim
        # or use np.where(lo_diff[start_lim:] < 0)[0][0] + start_lim
        
        # these are the criteria for the galaxy to be quenched
        if (np.all(lo_diff[end_lim:] <= 0) and
            np.all(hi_diff[start_lim:] <= 0) and
            np.all(lo_diff[termination_index:end_lim] <= 0)) :
            
            # find the peaks of the smoothed curve
            maxima, props = find_peaks(smoothed, height=0.4*np.max(smoothed))
            
            # use the final peak as the onset of quenching
            onset_index = maxima[-1]
            
            # if the galaxy is quenched, change the value in the mask, and
            # update the onset and termination times
            with h5py.File(infile, 'a') as hf :
                hf['quenched_mask'][i] = True
                hf['onset_times'][i] = times[onset_index]
                hf['termination_times'][i] = times[termination_index]
    
    return

def get_quenched_systems_info(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and the input file for the SFHs
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
    # open the dictionary containing the limits for the SFMS SFHs
    limits_file = inDir + '/{}_{}_SFMS_SFH_limits(t).pkl'.format(simName, snapNum)
    with open(limits_file, 'rb') as file :
        limits = pickle.load(file)
    
    # open the SFHs for the quenched systems
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        masses = hf['SubhaloMassStars'][:]
        R_es = hf['SubhaloHalfmassRadStars'][:]
        SFRs = hf['SubhaloSFRinRad'][:]
        SFHs = hf['SFH'][:]
        times = hf['times'][:]
        onset_times = hf['onset_times'][:]
        termination_times = hf['termination_times'][:]
    
    return (subIDs, masses, R_es, SFRs, SFHs, times, onset_times,
            termination_times, limits)

def plot_quenched_systems_in_massBin(mass_bin_edges, simName='TNG50-1',
                                     snapNum=99, kernel=2) :
    
    # retrieve the relevant information about the quenched systems
    _, masses, _, _, SFHs, times, _, _, _ = get_quenched_systems_info(
        simName, snapNum)
    
    # iterate over all the mass bins
    for i, (lo, hi) in enumerate(zip(mass_bin_edges[:-1], mass_bin_edges[1:])) :
        
        smoothed_SFHs = []
        # loop through the galaxies in the quenched sample in the mass
        # range/bin of interest
        for SFH in SFHs[(masses >= lo) & (masses < hi)] :
            # smooth the SFH of the specific galaxy
            smoothed = gaussian_filter1d(SFH, kernel)
            smoothed[smoothed < 0] = 0
            smoothed_SFHs.append(smoothed)
        
        # TODO - also plot the median of the quenched SFHs in the mass bin
        
        # plot and save the SFMS in each mass bin
        outfile = 'output/quenched_SFHs_in_massBin(t)/massBin_{}.png'.format(i)
        plt.plot_simple_many(times, smoothed_SFHs,
                             xlabel=r'$t$ (Gyr)',
                             ylabel=r'SFR ($M_{\odot}$ yr$^{-1}$)',
                             xmin=-0.1, xmax=13.8, save=True, outfile=outfile)
    
    return

def plot_termination_times(simName, snapNum) :
    
    # retrieve the relevant information about the quenched systems
    (_, masses, _, _, _, _, _,
     termination_times, _) = get_quenched_systems_info(simName, snapNum)
    
    # plot the quenching time as a function of stellar mass
    plt.plot_scatter(masses, termination_times, 'k', 'data', 'o',
                     xlabel=r'$\log(M_{*}/M_{\odot})$',
                     ylabel=r'$t_{\rm termination}$ (Gyr)')
    
    return

def reduce_SFH_to_quenched_systems(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and the input file for the SFHs
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    outfile = inDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
    # read the input file and populate the information
    with h5py.File(infile, 'r') as hf :
        
        # the arrays that will be copied over as existing
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        
        # the arrays that will be masked to the quenched systems
        subIDs = hf['SubhaloID'][:]
        masses = hf['logM'][:]
        SFHs = hf['SFH'][:]
        SFMS = hf['SFMS'][:]
        
        quenched_mask = hf['quenched_mask'][:]
        onset_indices = hf['onset_indices'][:]
        onset_times = hf['onset_times'][:]
        termination_indices = hf['termination_indices'][:]
        termination_times = hf['termination_times'][:]
    
    # mask and copy that information to the new file
    with h5py.File(outfile, 'w') as hf :
        # the arrays that are not masked
        add_dataset(hf, redshifts, 'redshifts')
        add_dataset(hf, times, 'times')
        
        # the arrays masked to only the quenched systems
        add_dataset(hf, subIDs[quenched_mask], 'SubhaloID')
        add_dataset(hf, masses[quenched_mask], 'SubhaloMassStars')
        add_dataset(hf, SFHs[quenched_mask], 'SFH')
        add_dataset(hf, SFMS[quenched_mask], 'SFMS')
        
        add_dataset(hf, quenched_mask[quenched_mask], 'quenched_mask')
        add_dataset(hf, onset_indices[quenched_mask], 'onset_indices')
        add_dataset(hf, onset_times[quenched_mask], 'onset_times')
        add_dataset(hf, termination_indices[quenched_mask], 'termination_indices')
        add_dataset(hf, termination_times[quenched_mask], 'termination_times')
    
    return
