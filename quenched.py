
from os.path import exists
import pickle
import numpy as np

import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter

from core import add_dataset, bsPath, get_SFH_limits, determine_mass_bin_indices
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
        smoothed = savgol_filter(SFH, window_length, polyorder)
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

def determine_quenched_systems_relative(simName, snapNum) :
    
    # define the input directory and the input file for the SFHs
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    
    # open the SFHs for the sample of candidate primary and satellite systems
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        masses = hf['SubhaloMassStars'][:]
        SFHs = hf['SFH'][:]
        # times = hf['times'][:]
    
    # select the SFHs corresponding to the SFMS at z = 0, and their masses
    SFMS_at_z0_mask = (SFHs[:, -1] > 0.001)
    SFMS_SFHs = SFHs[SFMS_at_z0_mask]
    SFMS_masses = masses[SFMS_at_z0_mask]
    
    minimumNum = 50
    hw = 0.10
    
    new_quenched_subIDs = []
    new_quenched_mass = []
    # loop through the galaxies in the sample
    for i, (subID, mass, SFH) in enumerate(zip(subIDs, masses, SFHs)) :
        
        # smooth the SFH of the specific galaxy
        smoothed = gaussian_filter1d(SFH, 2)
        smoothed[smoothed < 0] = 0
        
        # find galaxies in a similar mass range as the galaxy
        mass_bin = determine_mass_bin_indices(SFMS_masses, mass, halfwidth=hw,
                                              minNum=minimumNum)
        # !!! - with halfwidth=0.05, minNum=30,  we find 190 systems
        # !!! - with halfwidth=0.05, minNum=50,  we find 191 systems
        # !!! - with halfwidth=0.05, minNum=100, we find 192 systems
        
        # !!! - with halfwidth=0.10, minNum=30,  we find 220 systems
        # !!! - with halfwidth=0.10, minNum=50,  we find 220 systems
        # !!! - with halfwidth=0.10, minNum=100, we find 218 systems
        
        # !!! - with halfwidth=0.15, minNum=30,  we find 268 systems
        # !!! - with halfwidth=0.15, minNum=50,  we find 268 systems
        # !!! - with halfwidth=0.15, minNum=100, we find 268 systems
        
        # use the SFH values for those comparison galaxies to determine percentiles
        comparison_SFHs = SFMS_SFHs[mass_bin]
        lo_SFH, hi_SFH = np.nanpercentile(comparison_SFHs, [2.5, 97.5], axis=0)
        lo_SFH = gaussian_filter1d(lo_SFH, 2)
        hi_SFH = gaussian_filter1d(hi_SFH, 2)
        
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
            
            new_quenched_subIDs.append(subID)
            new_quenched_mass.append(mass)
            
            '''
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
            '''
    
    from astropy.table import Table
    t = Table()
    t['q_subIDs'] = new_quenched_subIDs
    t['mass'] = new_quenched_mass
    outfile = 'relative_quenched_subIDs_minNum-{}_halfwidth-{}.fits'.format(
        minimumNum, hw)
    t.write(outfile)
    
    return

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
        termination_times = hf['termination_times'][:]
    
    return (subIDs, masses, R_es, SFRs, SFHs, times, onset_times,
            termination_times, limits)

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

def reduce_SFH_to_quenched_systems(simName, snapNum) :
    
    # define the input directory and the input file for the SFHs
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    outfile = inDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
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
        termination_times = hf['termination_times'][:]
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
        add_dataset(hf, termination_times[quenched_mask], 'termination_times')
    
    return
