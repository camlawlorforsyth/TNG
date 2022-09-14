
from os.path import exists
import numpy as np

from astropy.cosmology import Planck15 as cosmo
import h5py

from core import add_dataset, bsPath, get
import plotting as plt

def add_primary_flags(simName, snapNum, redshift) :
    
    # define the output directory and the output file
    outDir = bsPath(simName)
    outfile = outDir + '/{}_{}_quenched_SFHs.hdf5'.format(simName, snapNum)
    
    # check if the outfile exists and has good primary flags
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            if 'primary_flag' in hf.keys() :
                if np.all(~np.isnan(hf['primary_flag'])) :
                    print('File already exists with all non-NaN primary flags')
    
    # add empty primary flag info into the HDF5 file to populate later
    with h5py.File(outfile, 'a') as hf :
        add_dataset(hf, np.full(hf['SFH'].shape, np.nan), 'primary_flag')
    
    # if the outfile exists, read the relevant information
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            subIDs = hf['SubhaloID'][:]
            x_flags = hf['primary_flag'][:]
    
    # now iterate over every subID in subIDs and get the SFH for that subID
    for i, subID in enumerate(subIDs) :
        
        # if the primary flags don't exist for the galaxy, populate the flags
        if np.all(np.isnan(x_flags[i, :])) :
            # determine the primary flags for the galaxy
            url = 'http://www.tng-project.org/api/{}/snapshots/z={}/subhalos/{}'.format(
                simName, redshift, subID)
            flags = determine_primary_flags(url)
            
            # append those values into the outfile
            with h5py.File(outfile, 'a') as hf :
                
                # the galaxy may not have existed from the first snapshot, so
                # we have to limit what we replace of the empty array
                hf['primary_flag'][i, :len(flags)] = flags
    
    return

def compare_satellite_times(flag, indices, sp, lookbacktimes, max_age) :
    
    vals = []
    for index in indices :
        
        forward = lookbacktimes[index]*np.std(flag[:index])
        backward = (max_age - lookbacktimes[index])*np.std(flag[index:])
        
        vals.append(forward + backward)
    
    return indices[np.argmin(vals)]

def determine_primary_flags(url) :
    
    sub = get(url)
    
    flags = []
    while sub['prog_sfid'] != -1 :
        # request the full subhalo details of the progenitor by following the sublink URL
        sub = get(sub['related']['sublink_progenitor'])
        
        flags.append(sub['primary_flag'])
    
    return flags

def determine_satellite_time(simName, snapNum) :
    
    max_age = cosmo.lookback_time(np.inf).value
    
    # define the output directory and the output file
    outDir = bsPath(simName)
    outfile = outDir + '/{}_{}_quenched_SFHs.hdf5'.format(simName, snapNum)
    
    with h5py.File(outfile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        masses = hf['SubhaloMassStars'][:]
        primary_flags = hf['primary_flag'][:]
        lookbacktimes = hf['lookbacktimes'][:]
        quench_index = hf['quench_index'][:]
    
    # good, missing, one_extra, two_extra = 0, 0, 0, 0
    sp = [0, 0, 0, 1, 1, 1] # satelitte to primary (as we work back in time)
    offset = int(len(sp)/2)
    
    satellite_times = []
    for i, flag in enumerate(primary_flags) :
        
        # mask out the NaN values
        flag = flag[~np.isnan(flag)]
        
        length = len(flag) - len(sp) + 1
        index = [x for x in range(length) if list(flag)[x:x+len(sp)] == sp]
        
        if len(index) == 0 :
            # repeat the same procedure as before, but with a shorter sublist
            ssp = [0, 1]
            offset = int(len(ssp)/2)
            length = len(flag) - len(ssp) + 1
            new = [x for x in range(length) if list(flag)[x:x+len(ssp)] == ssp]
            
            # find the galaxies that have always been primaries
            if len(new) == 0 :
                if np.all(flag == 1) :
                    satellite_times.append(0.0)
            
            elif len(new) == 1 :
                # find the galaxies that have always been primaries
                if np.all(np.delete(flag, new[0]) == 1) :
                    satellite_times.append(0.0)
                else :
                    if np.all(flag[:50] == 1) :
                        satellite_times.append(0.0)
                    else :
                        satellite_times.append(lookbacktimes[new[0]])
            
            elif len(new) > 1 :
                val = compare_satellite_times(flag, new, ssp, lookbacktimes, max_age)
                satellite_times.append(lookbacktimes[val])
        
        elif len(index) == 1 :
            satellite_times.append(lookbacktimes[index[0]])
        
        elif len(index) > 1 :
            index = np.array(index) + offset
            index = compare_satellite_times(flag, index, sp, lookbacktimes, max_age)
            satellite_times.append(lookbacktimes[index])
    
    # determine the quenching lookbacktimes
    tquench_lookbacktimes = lookbacktimes[quench_index.astype(int)]
    
    # plot those satellite times versus the quenching times
    plt.plot_scatter(tquench_lookbacktimes, satellite_times, 'k', 'data', 'o',
                     xlabel=r'$t_{\rm lookback, quench}$ (Gyr)',
                     ylabel=r'$t_{\rm lookback, satellite}$ (Gyr)')
    
    # plot the time difference as a function of stellar mass
    ylabel = r'$\Delta t = t_{\rm lookback, quench} - t_{\rm lookback, sat}$ (Gyr)'
    plt.plot_scatter(masses, tquench_lookbacktimes - satellite_times,
                     'k', 'data', 'o', xlabel=r'$\log(M_{*}/M_{\odot})$',
                     ylabel=ylabel)
    
    from astropy.table import Table
    table = Table([subIDs, masses, tquench_lookbacktimes, satellite_times],
                  names=('subID', 'mass', 't_quench', 't_sat'))
    # table.write('output.fits')
    table.pprint(max_lines=-1)
    
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
