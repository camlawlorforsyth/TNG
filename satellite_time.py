
from os.path import exists
import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
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

def compare_satellite_times(flag, indices, lookbacktimes, max_age) :
    
    consistency_measures = []
    for index in indices :
        
        # index = 0 gives nan because flag[:index] returns an empty list
        if index == 0 :
            before = 0.0
        else :
            before = lookbacktimes[index]*np.std(flag[:index+1])
        
        after = (max_age - lookbacktimes[index+1])*np.std(flag[index+1:])
        
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
    
    max_age = cosmo.age(0.0).value
    
    # define the output directory and the output file
    outDir = bsPath(simName)
    outfile = outDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)    
    
    with h5py.File(outfile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        masses = hf['SubhaloMassStars'][:]
        times = hf['times'][:]
        tquench = hf['quench_times'][:]
        flags = hf['primary_flags'][:]
    
    # primary to satellite (as we work forward in time)
    ps = [1, 0]
    
    # now loop through all the primary flags, trying to find the definitive
    # time when the galaxy transitioned from a primary to a satellite
    tsatellite, colours = [], []
    for i, flag in enumerate(flags) :
        
        length = len(flag) - len(ps) + 1
        indices = [x for x in range(length) if list(flag)[x:x+len(ps)] == ps]
        # indices returns the location of the start of the subarray, so we
        # need to increase by 1 to find the first index where a galaxy is
        # a satellite
        indices = np.array(indices) + 1
        
        # now use the length of the returned list of indices to determine
        # three main populations of galaxies: primaries, satellites, and
        # ambiguous cases
        if len(indices) == 0 : # galaxies that have always been primaries
            time, colour = max_age, 'red'
        
        elif len(indices) == 1 : # galaxies that have only one transition
            
            # special cases
            if i in [164, 183, 185, 221, 222, 241, # early, single snapshot when satellite
                     206, 216, # early, multiple consecutive snapshots when satellite
                     254, 258, 262, 264, 266] : # primary to satellite to primary
                time, colour = max_age, 'red'
            elif i in [244, 253] : # less likely primaries - few late primary flags
                time, colour = times[indices[0]], 'gold'
            
            # for non-special cases, use the found index for the satellite time
            else :
                time, colour = times[indices[0]], 'b'
        
        else : # galaxies with multiple transitions
            
            # special cases
            if i in [175, 189, 193, 199, 202, 203, 207, 209, 211, 219,
                     220, 223, 232, 235, 236, 239, 240, 243, 245, 249,
                     250, 257, 259, 260, 261, 263, 265, 267] : # very likely primaries
                time, colour = max_age, 'red'
            elif i in [242, 256] : # possible primaries
                time, colour = times[indices[-1]], 'darkorange'
            elif i in [246, 251, 252, 255, 268] : # unlikely primaries
                time, colour = times[indices[-1]], 'gold'
            
            # for non-special cases, use the most recent transition for simplicity
            else :
                time, colour = times[indices[-1]], 'w'
        
        tsatellite.append(time)
        colours.append(colour)
    
    if plot :
        # plot those satellite times versus the quenching times
        plt.plot_scatter(tquench, tsatellite, colours, 'data', 'o',
                         xlabel=r'$t_{\rm quench}$ (Gyr)',
                         ylabel=r'$t_{\rm satellite}$ (Gyr)')
        
        # plot the time difference as a function of stellar mass
        ylabel = r'$\Delta t = t_{\rm quench} - t_{\rm sat}$ (Gyr)'
        plt.plot_scatter(masses, tquench - tsatellite, colours,
                         'data', 'o', xlabel=r'$\log(M_{*}/M_{\odot})$',
                         ylabel=ylabel)
    
    if save : # create a table for cirtical information and save to file
        table = Table([subIDs, masses, tquench, tsatellite],
                       names=('subID', 'mass', 't_quench', 't_sat'))
        table.write('output/tquench_vs_tsat.fits')
    
    return

def plot_primary_flags_in_massBin(simName, snapNum, mass_bin_edges) :
    
    # define the output directory and the output file
    outDir = bsPath(simName)
    outfile = outDir + '/{}_{}_quenched_SFHs.hdf5'.format(simName, snapNum)
    
    with h5py.File(outfile, 'r') as hf :
        primary_flags = hf['primary_flag'][:]
        lookbacktimes = hf['lookbacktimes'][:]
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
        plt.plot_simple_many(lookbacktimes, flags,
                             xlabel=r'$t_{\rm lookback}$ (Gyr)',
                             ylabel='Primary Flag',
                             xmin=-0.1, xmax=13.8, save=True, outfile=outfile)
    
    return

# TODO - run after the quenched systems have been determined
# with h5py.File('TNG50-1/output/TNG50-1_99_flags(t).hdf5', 'r') as hf :
#     flags = hf['primary_flag'][:]

# with h5py.File('TNG50-1/output/TNG50-1_99_quenched_SFHs.hdf5', 'a') as hf :
#     add_dataset(hf, flags, 'primary_flags')
