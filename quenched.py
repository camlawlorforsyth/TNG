
import pickle
import numpy as np

import h5py
from scipy.signal import find_peaks, savgol_filter

from core import add_dataset, bsPath
import plotting as plt

def determine_quenched_systems(simName, snapNum, mass_bin_edges,
                               window_length, polyorder) :
    
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
    
    # add empty quenched mask, onset and quench lookbacktimes into the HDF5
    # file to populate later
    with h5py.File(infile, 'a') as hf :
        add_dataset(hf, np.full(len(subIDs), False), 'quenched_mask')
        add_dataset(hf, np.full(len(subIDs), np.nan), 'onset_times')
        add_dataset(hf, np.full(len(subIDs), np.nan), 'quench_times')
    
    # loop through the galaxies in the sample
    for i, (subID, mass, SFH) in enumerate(zip(subIDs, masses, SFHs)) :
        
        # smooth the SFH of the specific galaxy
        smoothed = savgol_filter(SFH, window_length, polyorder)
        smoothed[smoothed < 0] = 0
        
        # get the corresponding lower and upper two sigma limits for that mass
        lo_SFH, hi_SFH = get_SFH_limits(limits, np.array(mass_bin_edges), mass)
        
        # now select galaxies that are within the SFR limits then quench
        lo_diff = smoothed - lo_SFH
        hi_diff = smoothed - hi_SFH
        quench_index = np.argmax(lo_diff > 0) - 1 # TODO - update after time conversion
        # equivalent to np.where(lo_diff > 0)[0][0] - 1
        
        # set the indices where the limits don't have to be strictly observed,
        # essentially a small tolerance given the smoothing and low SFRs at
        # early times
        start_lim, end_lim = 3, -3 # originally was -5
        
        # TODO - determine onset index, and onset time
        maxima, props = find_peaks(smoothed, height=0.4*np.max(smoothed))
        onset_index = maxima[-1]
        
        # these are the criteria for the galaxy to be quenched
        # TODO - update after time conversion
        if (np.all(lo_diff[:start_lim] <= 0) and np.all(hi_diff[:end_lim] <= 0) and
            np.all(lo_diff[quench_index+1:end_lim] >= 0)) :
            
            # if the galaxy is quenched, change the value in the mask, and
            # update the onset and quench lookbacktimes
            with h5py.File(infile, 'a') as hf :
                hf['quenched_mask'][i] = True
                hf['onset_times'][i] = times[onset_index]
                hf['quench_times'][i] = times[quench_index]
    
    return

def get_SFH_limits(limits_dic, edges, mass) :
    
    # find the index of the corresponding mass bin
    idx = np.where((mass >= edges[:-1]) & (mass <= edges[1:]))[0][0]
    subdic = limits_dic['mass_bin_{}'.format(idx)]
    
    return subdic['lo_SFH'], subdic['hi_SFH'] # return those limits

def get_quenched_systems_info(simName, snapNum) :
    
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
        quench_times = hf['quench_times'][:]
    
    return (subIDs, masses, R_es, SFRs, SFHs, times, onset_times, quench_times,
            limits)

def plot_quenched_systems(simName, snapNum, mass_bin_edges, window_length,
                          polyorder) : 
    
    # retrieve the relevant information about the quenched systems
    (subIDs, masses, _, _, SFHs, times, onset_times,
     quench_times, limits) = get_quenched_systems_info(simName, snapNum)
    
    # loop through the galaxies in the quenched sample
    for (subID, mass, SFH, onset_time,
         quench_time) in zip(subIDs, masses, SFHs, onset_times, quench_times) :
        
        # smooth the SFH of the specific galaxy
        smoothed = savgol_filter(SFH, window_length, polyorder)
        smoothed[smoothed < 0] = 0
        
        # get the corresponding lower and upper two sigma limits for that mass
        lo_SFH, hi_SFH = get_SFH_limits(limits, np.array(mass_bin_edges), mass)
        
        # now plot the curves
        outfile = 'output/quenched_SFHs/quenched_SFH_subID_{}.png'.format(subID)
        plt.plot_simple_multi_with_times([times, times,
                                          times, times],
                                         [SFH, smoothed, lo_SFH, hi_SFH],
                                         ['data', 'smoothed', 'lo, hi', ''],
                                         ['grey', 'k', 'lightgrey', 'lightgrey'],
                                         ['', '', '', ''],
                                         ['--', '-', '-.', '-.'],
                                         [0.5, 1, 1, 1],
                                         np.nan, onset_time, quench_time,
                                         xlabel=r'$t_{\rm lookback}$ (Gyr)',
                                         ylabel=r'SFR ($M_{\odot}$ yr$^{-1}$)',
                                         xmin=-0.1, xmax=13.8,
                                         scale='linear', save=True, outfile=outfile)
    
    return

def plot_quenched_systems_in_massBin(simName, snapNum, mass_bin_edges,
                                     window_length, polyorder) :
    
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
            smoothed = savgol_filter(SFH, window_length, polyorder)
            smoothed[smoothed < 0] = 0
            smoothed_SFHs.append(smoothed)
        
        # TODO - also plot the median of the quenched SFHs in the mass bin
        
        # plot and save the SFMS in each mass bin
        outfile = 'output/quenched_SFHs_in_massBin/massBin_{}.png'.format(i)
        plt.plot_simple_many(times, smoothed_SFHs,
                             xlabel=r'$t_{\rm lookback}$ (Gyr)',
                             ylabel=r'SFR ($M_{\odot}$ yr$^{-1}$)',
                             xmin=-0.1, xmax=13.8, save=True, outfile=outfile)
    
    return

def plot_quenching_times(simName, snapNum) :
    
    # retrieve the relevant information about the quenched systems
    (_, masses, _, _, _, _, _,
     quench_times, _) = get_quenched_systems_info(simName, snapNum)
    
    # plot the quenching time as a function of stellar mass
    plt.plot_scatter(masses, quench_times, 'k', 'data', 'o',
                     xlabel=r'$\log(M_{*}/M_{\odot})$',
                     ylabel=r'$t_{\rm lookback, quench}$ (Gyr)')
    
    return

def reduce_SFH_to_quenched_systems(simName, snapNum) :
    
    # define the input directory and the input file for the SFHs
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    outfile = inDir + '/{}_{}_quenched_SFHs(t)s.hdf5'.format(simName, snapNum)
    
    # read the input file and populate the information
    with h5py.File(infile, 'r') as hf :
        # the arrays that will be copied over as existing
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        edges = hf['edges'][:]
        time_in_bins = hf['time_in_bins'][:]
        
        # the arrays that will be masked to the quenched systems
        subIDs = hf['SubhaloID'][:]
        masses = hf['SubhaloMassStars'][:]
        SFRs = hf['SubhaloSFRinRad'][:]
        R_es = hf['SubhaloHalfmassRadStars'][:]
        SFHs = hf['SFH'][:]
        onset_times = hf['onset_times'][:]
        quench_times = hf['quench_times'][:]
        quenched_mask = hf['quenched_mask'][:]
    
    # mask and copy that information to the new file
    with h5py.File(outfile, 'w') as hf :
        # the arrays that are not masked
        add_dataset(hf, redshifts, 'redshifts')
        add_dataset(hf, times, 'times')
        add_dataset(hf, edges, 'edges')
        add_dataset(hf, time_in_bins, 'time_in_bins')
        
        # the arrays masked to only the quenched systems
        add_dataset(hf, subIDs[quenched_mask], 'SubhaloID')
        add_dataset(hf, masses[quenched_mask], 'SubhaloMassStars')
        add_dataset(hf, SFRs[quenched_mask], 'SubhaloSFRinRad')
        add_dataset(hf, R_es[quenched_mask], 'SubhaloHalfmassRadStars')
        add_dataset(hf, SFHs[quenched_mask], 'SFH')
        add_dataset(hf, onset_times[quenched_mask], 'onset_times')
        add_dataset(hf, quench_times[quenched_mask], 'quench_times')
    
    return
