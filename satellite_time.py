
from os.path import exists
import numpy as np

from astropy.cosmology import Planck15 as cosmo
import h5py

from core import add_dataset, bsPath, get
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
    
    # now iterate over every subID in subIDs and get the SFH for that subID
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
                hf['primary_flags'][i, :len(flags)] = flags
    
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
    flags.append(sub['primary_flags']) # get the first flag at z = 0
    
    # now work back through the main progenitor branch tree
    while sub['prog_sfid'] != -1 :
        # request the full subhalo details of the progenitor by following the sublink URL
        sub = get(sub['related']['sublink_progenitor'])
        
        # now get the flag for the subprogenitor, working back through the tree
        flags.append(sub['primary_flags'])
    
    return np.flip(flags) # return the array in time increasing order

def determine_satellite_time(simName, snapNum, plot=False, save=False) :
    
    # by definition, the primary galaxies have tsat_lookback = 0, so tsat = 13.8 Gyr
    max_age = cosmo.age(0.0).value
    
    # define the output directory and the output file
    outDir = bsPath(simName)
    outfile = outDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)    
    
    # get the relevant information to determine the satellite transition time
    # for the quenched sample
    with h5py.File(outfile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        times = hf['times'][:]
        flags = hf['primary_flags'][:]
    
    # add empty satellite times and primary confidence measures into the HDF5
    # file to populate later
    with h5py.File(outfile, 'a') as hf :
        if 'satellite_times' not in hf.keys() :
            add_dataset(hf, np.full(len(subIDs), np.nan), 'satellite_times')
        if 'primary_confidence' not in hf.keys() :
            add_dataset(hf, np.full(len(subIDs), np.nan), 'primary_confidence')
    
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
            time, confidence = max_age, 5
        
        elif len(indices) == 1 : # galaxies that have only one transition
            # special cases
            if i in [164, 183, 185, 221, 222, 241, # early, single snapshot when satellite
                     206, 216, # early, multiple consecutive snapshots when satellite
                     254, 258, 262, 264, 266] : # primary to satellite to primary
                time, confidence = max_age, 5
            elif i in [244, 253] : # less likely primaries - few late primary flags
                time, confidence = times[indices[0]], 3
            
            # for non-special cases, use the found index for the satellite time
            else :
                time, confidence = times[indices[0]], 2
        
        else : # galaxies with multiple transitions
            
            # special cases
            if i in [175, 189, 193, 199, 202, 203, 207, 209, 211, 219,
                     220, 223, 232, 235, 236, 239, 240, 243, 245, 249,
                     250, 257, 259, 260, 261, 263, 265, 267] : # very likely primaries
                time, confidence = max_age, 5
            elif i in [242, 256] : # possible primaries
                time, confidence = times[indices[-1]], 4
            elif i in [246, 251, 252, 255, 268] : # unlikely primaries
                time, confidence = times[indices[-1]], 3
            
            # for non-special cases, use the most recent transition for simplicity
            else :
                time, confidence = times[indices[-1]], 1
        
        # update the satellite times and confidences
        with h5py.File(outfile, 'a') as hf :
            hf['satellite_times'][i] = time
            hf['primary_confidence'][i] = confidence # rated out of 5
            # where primaries are 5, possible primaries are 4,
            # less likely/unlikely primaries are 3, satellites are 2,
            # and ambiguous cases are 1
    
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
