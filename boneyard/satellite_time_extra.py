
from os.path import exists
import numpy as np

import h5py
from scipy.ndimage import gaussian_filter1d

from core import add_dataset, bsPath, determine_mass_bin_indices, get
import plotting as plt

def add_primary_flags(simName, snapNum) :
    
    # define the output directory and the output file
    outDir = bsPath(simName)
    outfile = outDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
    # check if the outfile exists and has good primary flags
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            if 'primary_flags' in hf.keys() :
                if np.all(~np.isnan(hf['primary_flags'])) :
                    print('File already exists with all non-NaN primary flags')
    
    # add empty primary flag info into the HDF5 file to populate later
    if exists(outfile) :
        with h5py.File(outfile, 'a') as hf :
            if 'primary_flags' not in hf.keys() :
                add_dataset(hf, np.full(hf['SFH'].shape, np.nan), 'primary_flags')
    
    # if the outfile exists, read the relevant information
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            subIDs = hf['SubhaloID'][:]
            x_flags = hf['primary_flags'][:]
    
    # now iterate over every subID in subIDs and get the flag for that subID
    for i, subID in enumerate(subIDs) :
        
        # if the primary flags don't exist for the galaxy, populate the flags
        if np.all(np.isnan(x_flags[i, :])) :
            # determine the primary flags for the galaxy
            url = 'http://www.tng-project.org/api/{}/snapshots/{}/subhalos/{}'.format(
                simName, snapNum, subID)
            flags = determine_primary_flags(url)
            
            # append those values into the outfile
            with h5py.File(outfile, 'a') as hf :
                
                # the galaxy may not have existed from the first snapshot, so
                # we have to limit what we replace of the empty array
                hf['primary_flags'][i, 100-len(flags):] = flags
    
    return

def check_for_missing_flags() :
    
    files = ['TNG50-1/TNG50-1_99_primary_flags(t)_cluster.hdf5',
             'TNG50-1/TNG50-1_99_primary_flags(t)_hmGroup1.hdf5',
             'TNG50-1/TNG50-1_99_primary_flags(t)_hmGroup2.hdf5',
             'TNG50-1/TNG50-1_99_primary_flags(t)_lmGroup1.hdf5',
             'TNG50-1/TNG50-1_99_primary_flags(t)_lmGroup2.hdf5',
             'TNG50-1/TNG50-1_99_primary_flags(t)_field1.hdf5',
             'TNG50-1/TNG50-1_99_primary_flags(t)_field2.hdf5',
             'TNG50-1/TNG50-1_99_primary_flags(t)_field3.hdf5']

    missing = 0
    for file in files :
        with h5py.File(file, 'r') as hf :
            subIDs = hf['subIDfinals'][:]
            flags = hf['flags'][:]
        
        for subID, row in zip(subIDs, flags) :
            if np.all(np.isnan(row)) :
                # print(subID)
                missing += 1

    print(missing)
    
    return

def check_if_cluster_flags_are_stable() :

    with h5py.File('cluster_primary_flags.hdf5', 'r') as hf :
        subIDs = hf['cluster_subIDs'][:]
        flags = hf['primary_flags'][:]
    
    # primary to satellite (as we work forward in time)
    ps = [1, 0]
    
    one_transition = 0
    two_transitions = 0
    more = 0
    for subID, flag in zip(subIDs, flags) :
        
        length = len(flag) - len(ps) + 1
        indices = [x for x in range(length) if list(flag)[x:x+len(ps)] == ps]
        indices = np.array(indices) + 1
        
        if len(indices) == 1 : # galaxies that have only one transition
            one_transition += 1
            # print(subID)
            # print(flag)
            # print()
        
        if len(indices) == 2 : # galaxies with two transitions
            two_transitions += 1
            # print(subID)
            # print(flag)
            # print()
        
        if len(indices) >= 3 :
            more += 1
            print(subID)
            print(flag)
            print()
    
    print('{} subIDs located in the 10^14.2 Mstar cluster halo'.format(len(subIDs)))
    print('{} systems with one transition'.format(one_transition))
    print('{} systems with two transitions'.format(two_transitions))
    print('{} systems with more than one transition'.format(more))
    
    return

def combine_flags_for_different_environments() :

    infile = 'TNG50-1/TNG50-1_99_sample(t).hdf5'
    outfile = 'TNG50-1/TNG50-1_99_primary_flags(t).hdf5'
    
    with h5py.File(infile, 'r') as hf :
        snaps = hf['snapshots'][:]
        subIDs = hf['subIDs'][:].astype(int)
        subIDfinals = hf['SubhaloID'][:]
        quenched = hf['quenched'][:]
        cluster = hf['cluster'][:]
        hm_group = hf['hm_group'][:]
        lm_group = hf['lm_group'][:]
        field = hf['field'][:]
    
    flags = np.full((8260, 100), np.nan)
    
    with h5py.File('TNG50-1/TNG50-1_99_primary_flags(t)_cluster.hdf5', 'r') as hf :
        cluster_subIDs = hf['subIDfinals'][:]
        cluster_flags = hf['flags'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_primary_flags(t)_hmGroup1.hdf5','r') as hf :
        hm1_subIDs = hf['subIDfinals'][:]
        hm1_flags = hf['flags'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_primary_flags(t)_hmGroup2.hdf5','r') as hf :
        hm2_subIDs = hf['subIDfinals'][:]
        hm2_flags = hf['flags'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_primary_flags(t)_lmGroup1.hdf5','r') as hf :
        lm1_subIDs = hf['subIDfinals'][:]
        lm1_flags = hf['flags'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_primary_flags(t)_lmGroup2.hdf5','r') as hf :
        lm2_subIDs = hf['subIDfinals'][:]
        lm2_flags = hf['flags'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_primary_flags(t)_field1.hdf5','r') as hf :
        field1_subIDs = hf['subIDfinals'][:]
        field1_flags = hf['flags'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_primary_flags(t)_field2.hdf5','r') as hf :
        field2_subIDs = hf['subIDfinals'][:]
        field2_flags = hf['flags'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_primary_flags(t)_field3.hdf5','r') as hf :
        field3_subIDs = hf['subIDfinals'][:]
        field3_flags = hf['flags'][:]
    
    for i, subID in enumerate(subIDfinals) :
        if subID in cluster_subIDs :
            loc = np.where(cluster_subIDs == subID)[0][0]
            flag = cluster_flags[loc]
        
        elif subID in hm1_subIDs :
            loc = np.where(hm1_subIDs == subID)[0][0]
            flag = hm1_flags[loc]
        
        elif subID in hm2_subIDs :
            loc = np.where(hm2_subIDs == subID)[0][0]
            flag = hm2_flags[loc]
        
        elif subID in lm1_subIDs :
            loc = np.where(lm1_subIDs == subID)[0][0]
            flag = lm1_flags[loc]
        
        elif subID in lm2_subIDs :
            loc = np.where(lm2_subIDs == subID)[0][0]
            flag = lm2_flags[loc]
        
        elif subID in field1_subIDs :
            loc = np.where(field1_subIDs == subID)[0][0]
            flag = field1_flags[loc]
        
        elif subID in field2_subIDs :
            loc = np.where(field2_subIDs == subID)[0][0]
            flag = field2_flags[loc]
        
        elif subID in field3_subIDs :
            loc = np.where(field3_subIDs == subID)[0][0]
            flag = field3_flags[loc]
        
        else :
            flag = np.full(100, np.nan)
        
        flags[i, :] = flag
    
    if not exists(outfile) :
        with h5py.File(outfile, 'w') as hf :
            add_dataset(hf, subIDfinals, 'subIDfinals')
            add_dataset(hf, flags, 'flags')
    
    return

def compare_satellite_times(flag, indices, times, max_age) :
    
    consistency_measures = []
    for index in indices :
        
        # index = 0 gives nan because flag[:index] returns an empty list
        if index == 0 :
            before = 0.0
        else :
            before = times[index]*np.std(flag[:index+1])
        
        after = (max_age - times[index+1])*np.std(flag[index+1:])
        
        consistency_measures.append(before + after)
    
    return indices[np.argmin(consistency_measures)]

def determine_primary_flags(url) :
    
    sub = get(url)
    
    flags = []
    flags.append(sub['primary_flag']) # get the first flag at z = 0
    
    # now work back through the main progenitor branch tree
    while sub['prog_sfid'] != -1 :
        # request the full subhalo details of the progenitor by following the sublink URL
        sub = get(sub['related']['sublink_progenitor'])
        
        # now get the flag for the subprogenitor, working back through the tree
        flags.append(sub['primary_flag'])
    
    return np.flip(flags) # return the array in time increasing order

def determine_satellite_time(simName, snapNum) :
    
    # define the output directory and the output file
    outDir = bsPath(simName)
    outfile = outDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
    # get the relevant information to determine the satellite transition time
    # for the quenched sample
    with h5py.File(outfile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        times = hf['times'][:]
        flags = hf['primary_flags'][:]
    
    # add empty satellite times and a satellite flag into the HDF5 file to
    # populate later
    with h5py.File(outfile, 'a') as hf :
        if 'satellite_times' not in hf.keys() :
            add_dataset(hf, np.full(len(subIDs), np.nan), 'satellite_times')
        if 'satellite' not in hf.keys() :
            add_dataset(hf, np.full(len(subIDs), np.nan), 'satellite')
    
    # primary to satellite (as we work forward in time)
    ps = [1, 0]
    
    # now loop through all the primary flags, trying to find the definitive
    # time when the galaxy transitioned from a primary to a satellite
    for i, flag in enumerate(flags) :
        
        length = len(flag) - len(ps) + 1
        indices = [x for x in range(length) if list(flag)[x:x+len(ps)] == ps]
        # indices returns the location of the start of the subarray, so we
        # need to increase by 1 to find the first index where a galaxy is
        # a satellite, given that the transitions are primary-to-satellite
        indices = np.array(indices) + 1
        
        # now use the length of the returned list of indices to determine
        # three main populations of galaxies: primaries, satellites, and
        # ambiguous cases
        if len(indices) == 0 : # galaxies that have always been primaries
            index, time, satellite_flag = 99, times[99], 0
            # primary galaxies have tsat_lookback = 0, so tsat = 13.8 Gyr
        
        elif len(indices) == 1 : # galaxies that have only one transition
            # special cases
            if i in [130, 149, 169, 172, 197, # early, single snapshot when satellite
                     142] : # early, multiple consecutive snapshots when satellite
                index, time, satellite_flag = 99, times[99], 0
            elif i in [192, 193, 198, 204, 207, 214, 216, 219] : # few late primary flags
                index, time, satellite_flag = np.nan, np.nan, 2
            
            # for non-special cases, use the found index for the satellite time
            else :
                index, time, satellite_flag = indices[0], times[indices[0]], 1
        
        elif len(indices) == 2 :
            # special cases
            if i in [136, 152, 157, 164, 177, 180, 182, 201, ] : # random single
            # snapshots when satellite
                index, time, satellite_flag = 99, times[99], 0
            
            elif i in [  1,   4,  9,  19,  24,  47,  51,  53,  54,  55,
                        61,  70, 84, 100, 114, 115, 122, 127, 135, 140,
                       156, 174, # early, single snapshot when satellite
                        20,  30, 31, 43, 65, 73, 74, 75, 99, 145, 
                       153, # early, few consecutive snapshots when satellite
                       202, 203, 217] : # late conseutive snapshots when primary
            # with subsequent transition later
                index, time, satellite_flag = indices[-1], times[indices[-1]], 1
            
            elif i in [50, 52, 57, 66, 129, 134, # late, single snapshot when satellite
                       101, # late, few consecutive snapshots when satellite
                       ] :
            # with previous transition earlier
                index, time, satellite_flag = indices[0], times[indices[0]], 1
            
            # for non-special cases, the transition is ambiguous
            else :
                index, time, satellite_flag = np.nan, np.nan, 2
        
        else : # galaxies with multiple transitions are generally ambiguous
            index, time, satellite_flag = np.nan, np.nan, 2
        
        # update the satellite times and the satellite flag
        with h5py.File(outfile, 'a') as hf :
            hf['satellite_times'][i] = time
            hf['satellite'][i] = satellite_flag
    
    return

def plot_primary_flags_in_massBin(simName, snapNum, mass_bin_edges) :
    
    # define the output directory and the output file
    outDir = bsPath(simName)
    outfile = outDir + '/{}_{}_quenched_SFHs.hdf5'.format(simName, snapNum)
    
    with h5py.File(outfile, 'r') as hf :
        primary_flags = hf['primary_flag'][:]
        times = hf['times'][:]
        masses = hf['SubhaloMassStars'][:]
    
    # iterate over all the mass bins
    for i, (lo, hi) in enumerate(zip(mass_bin_edges[:-1], mass_bin_edges[1:])) :
        
        # loop through the galaxies in the quenched sample in the mass
        # range/bin of interest
        flags = []
        mass_mask = (masses >= lo) & (masses < hi)
        for mass, flag in zip(masses[mass_mask], primary_flags[mass_mask]) :
            flags.append(mass + flag)
        
        # plot and save the primary flags in each mass bin
        outfile = 'output/primary_flags_in_massBin/massBin_{}.png'.format(i)
        plt.plot_simple_many(times, flags,
                             xlabel=r'$t$ (Gyr)', ylabel='Primary Flag',
                             xmin=-0.1, xmax=13.8, save=True, outfile=outfile)
    
    return

def plot_quenched_systems(simName, snapNum, hw=0.1, minNum=50, kernel=2) :
    
    # define the input directory and the input file for the SFHs
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    sample_file = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    
    # retrieve the relevant information about the quenched systems
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        masses = hf['SubhaloMassStars'][:]
        SFHs = hf['SFH'][:]
        times = hf['times'][:]
        satellite_times = hf['satellite_times'][:]
        onset_times = hf['onset_times'][:]
        termination_times = hf['termination_times'][:]
    
    # open the SFHs for the sample of all galaxies, to find those for the SFMS
    with h5py.File(sample_file, 'r') as hf :
        all_subIDs = hf['SubhaloID'][:]
        all_masses = hf['SubhaloMassStars'][:]
        all_SFHs = hf['SFH'][:]
        SFMS = hf['SFMS'][:] # 6337 star forming main sequence (at z = 0) galaxies
        quenched_mask = hf['quenched_mask'][:]
    
    nearly_quenched = (~SFMS) & (~quenched_mask)
    other_subIDs = all_subIDs[nearly_quenched]
    other_masses = all_masses[nearly_quenched]
    other_SFHs = all_SFHs[nearly_quenched]
    
    for subID, mass, SFH in zip(other_subIDs, other_masses, other_SFHs) :
        
        # smooth the SFH of the specific galaxy
        smoothed = gaussian_filter1d(SFH, kernel)
        smoothed[smoothed < 0] = 0 # the SFR cannot be negative
        
        # find galaxies in a similar mass range as the galaxy, but that are on
        # the SFMS at z = 0
        mass_bin = determine_mass_bin_indices(all_masses[SFMS], mass, hw=hw,
                                              minNum=minNum)
        
        # use the SFH values for those comparison galaxies to determine percentiles
        comparison_SFHs = all_SFHs[SFMS][mass_bin]
        lo_SFH, hi_SFH = np.nanpercentile(comparison_SFHs, [2.5, 97.5], axis=0)
        lo_SFH = gaussian_filter1d(lo_SFH, kernel)
        hi_SFH = gaussian_filter1d(hi_SFH, kernel)
        
        # now plot the curves
        outfile = 'TNG50-1/nearly_quenched_SFHs(t)/quenched_SFH_subID_{}.png'.format(subID)
        plt.plot_simple_multi_with_times([times, times, times, times],
                                         [SFH, smoothed, lo_SFH, hi_SFH],
                                         ['data', 'smoothed', 'lo, hi', ''],
                                         ['grey', 'k', 'lightgrey', 'lightgrey'],
                                         ['', '', '', ''],
                                         ['--', '-', '-.', '-.'],
                                         [0.5, 1, 1, 1],
                                         np.nan, np.nan, np.nan,
                                         xlabel=r'$t$ (Gyr)',
                                         ylabel=r'SFR ($M_{\odot}$ yr$^{-1}$)',
                                         xmin=-0.1, xmax=13.8, scale='linear',
                                         save=True, outfile=outfile, loc=0)
    
    # loop through the galaxies in the quenched sample
    for subID, mass, SFH, tsat, tonset, tterm in zip(subIDs, masses, SFHs,
        satellite_times, onset_times, termination_times) :
        
        # smooth the SFH of the specific galaxy
        smoothed = gaussian_filter1d(SFH, kernel)
        smoothed[smoothed < 0] = 0 # the SFR cannot be negative
        
        # find galaxies in a similar mass range as the galaxy, but that are on
        # the SFMS at z = 0
        mass_bin = determine_mass_bin_indices(all_masses[SFMS], mass, hw=hw,
                                              minNum=minNum)
        
        # use the SFH values for those comparison galaxies to determine percentiles
        comparison_SFHs = all_SFHs[SFMS][mass_bin]
        lo_SFH, hi_SFH = np.nanpercentile(comparison_SFHs, [2.5, 97.5], axis=0)
        lo_SFH = gaussian_filter1d(lo_SFH, kernel)
        hi_SFH = gaussian_filter1d(hi_SFH, kernel)
        
        # now plot the curves
        outDir = 'TNG50-1/quenched_SFHs(t)/'
        outfile = outDir + 'quenched_SFH_subID_{}.png'.format(subID)
        plt.plot_simple_multi_with_times([times, times, times, times],
                                         [SFH, smoothed, lo_SFH, hi_SFH],
                                         ['data', 'smoothed', 'lo, hi', ''],
                                         ['grey', 'k', 'lightgrey', 'lightgrey'],
                                         ['', '', '', ''],
                                         ['--', '-', '-.', '-.'],
                                         [0.5, 1, 1, 1],
                                         tsat, tonset, tterm,
                                         xlabel=r'$t$ (Gyr)',
                                         ylabel=r'SFR ($M_{\odot}$ yr$^{-1}$)',
                                         xmin=-0.1, xmax=13.8, scale='linear',
                                         save=True, outfile=outfile, loc=0)
        
        # now plot the curves without the upper and lower limits
        outDir = 'output/quenched_SFHs_without_lohi(t)/'
        outfile = outDir + 'quenched_SFH_subID_{}.png'.format(subID)
        plt.plot_simple_multi_with_times([times, times], [SFH, smoothed],
                                         ['data', 'smoothed'], ['grey', 'k'],
                                         ['', ''], ['--', '-'], [0.5, 1],
                                         tsat, tonset, tterm,
                                         xlabel=r'$t$ (Gyr)',
                                         ylabel=r'SFR ($M_{\odot}$ yr$^{-1}$)',
                                         xmin=0, xmax=14, scale='linear',
                                         save=True, outfile=outfile, loc=0)
    
    return

def plot_satellite_times(simName, snapNum) :
    
    # define the output directory and the output file
    outDir = bsPath(simName)
    outfile = outDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
    # get the masses, satellite and termination times for the quenched sample
    with h5py.File(outfile, 'r') as hf :
        masses = hf['SubhaloMassStars'][:]
        tterm = hf['termination_times'][:]
        tsatellite = hf['satellite_times'][:]
        confidence = hf['primary_confidence'][:]
    
    # convert the confidences into colours for plotting
    colours = np.full(len(masses), '')
    colours[confidence == 5] = 'r'
    colours[confidence == 4] = 'k'
    colours[confidence == 3] = 'm'
    colours[confidence == 2] = 'b'
    colours[confidence == 1] = 'w'
    
    # plot those satellite times versus the quenching times
    plt.plot_scatter(tterm, tsatellite, list(colours), 'data', 'o',
                     xlabel=r'$t_{\rm termination}$ (Gyr)',
                     ylabel=r'$t_{\rm satellite}$ (Gyr)')
    
    # plot the time difference as a function of stellar mass
    ylabel = r'$\Delta t = t_{\rm termination} - t_{\rm satellite}$ (Gyr)'
    plt.plot_scatter(masses, tterm - tsatellite, list(colours),
                     'data', 'o', xlabel=r'$\log(M_{*}/M_{\odot})$',
                     ylabel=ylabel)
    
    return
