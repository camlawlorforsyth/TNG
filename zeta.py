
from os.path import exists
import numpy as np

import astropy.units as u
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from core import (add_dataset, bsPath, determine_mass_bin_indices,
                  get_particles, get_sf_particles)
import plotting as plt

def compute_sf_rms(rs) :
    
    if len(rs) == 0 :
        rms = 0.0
    else :
        rms = np.sqrt(np.mean(np.square(rs)))
    
    return rms

def determine_zeta(simName, snapNum, time, snap, mpbsubID, center,
                   delta_t=100*u.Myr) :
    
    # get all particles
    ages, masses, rs = get_particles(simName, snapNum, snap, mpbsubID,
                                     center)
    
    # only proceed if the ages, masses, and distances are intact
    if (ages is not None) and (masses is not None) and (rs is not None) :
        
        # find the stellar half mass radius for all particles
        stellar_halfmass_radius = compute_halfmass_radius(masses, rs)
        
        # get the SF particles
        _, sf_masses, sf_rs = get_sf_particles(ages, masses, rs, time,
                                               delta_t=delta_t)
        
        # find the stellar half mass radius for SF particles
        sf_halfmass_radius = compute_halfmass_radius(sf_masses, sf_rs)
        
        # now compute the ratio of the half mass radius of the SF
        # particles to the half mass radius of all particles
        zeta = sf_halfmass_radius/stellar_halfmass_radius
    else :
        zeta = np.nan
    
    return zeta

def determine_zeta_fxn(simName, snapNum, times, subID, delta_t=100*u.Myr) :
    
    # get the mpb snapshot numbers, subIDs, stellar halfmassradii, and
    # galaxy centers
    snapNums, mpb_subIDs, _, centers = get_mpb_radii_and_centers(
        simName, snapNum, subID)
    
    # limit the time axis to valid snapshots
    ts = times[len(times)-len(snapNums):]
    
    # now get the star particle ages, masses, and distances at each
    # snapshot/time
    zetas = []
    for time, snap, mpbsubID, center in zip(ts, snapNums, mpb_subIDs, centers) :
        
        zeta = determine_zeta(simName, snapNum, time, snap, mpbsubID, center,
                              delta_t=delta_t)
        zetas.append(zeta)
    
    return ts, zetas

def save_zeta_for_sample(simName, snapNum) :

    # define the input directory and file for the sample, and the output file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    outfile = inDir + '/{}_{}_highMassGals_zeta(t).hdf5'.format(
        simName, snapNum)
    
    # get basic information for the sample of primary and satellite systems
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        times = hf['times'][:]
        masses = hf['SubhaloMassStars'][:]
    
    # limit the sample to the highest mass objects
    subIDs = subIDs[(masses > 10.07) & (masses <= 12.75)]
    
    # add empty zeta(t) into the HDF5 file to populate later
    if not exists(outfile) :
        with h5py.File(outfile, 'w') as hf :
            if 'subID' not in hf.keys() :
                add_dataset(hf, subIDs, 'SubhaloID')
            if 'zeta(t)' not in hf.keys() :
                add_dataset(hf, np.full((len(subIDs), len(times)), np.nan),
                            'zeta(t)')
    
    # if the outfile exists, read the relevant information
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            subIDs = hf['SubhaloID'][:]
            x_zeta = hf['zeta(t)'][:]
    
    # now iterate over every subID in subIDs and get zeta(t)
    for i, subID in enumerate(subIDs) :
        
        # if zeta(t) doesn't exist for the galaxy, populate the values
        if np.all(np.isnan(x_zeta[i, :])) :
            
            _, zeta = determine_zeta_fxn(simName, snapNum, times, subID,
                                         delta_t=100*u.Myr)
            start_index = len(x_zeta[i, :]) - len(zeta)
            
            # append the determined values for zeta(t) into the outfile
            with h5py.File(outfile, 'a') as hf :
                hf['zeta(t)'][i, start_index:] = zeta
        
        print('{} - {} done'.format(i, subID))
    
    return

def zeta_for_sample(simName, snapNum, delta_t=100*u.Myr, plot=False, save=False) :
    
    # define the output directory
    outDir = 'output/zeta(t)/'
    
    '''
    # define the input directory and file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
    # get the masses, satellite and termination times for the quenched sample
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        tsats = hf['satellite_times'][:]
        tonsets = hf['onset_times'][:]
        tterms = hf['termination_times'][:]
    '''
    
    # get attributes for the test data
    from core import get_test_data
    _, times, test_subIDs, tsats, tonsets, tterms = get_test_data()
    
    # get relevant information for the larger sample of 8260 galaxies
    infile = bsPath(simName) + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    with h5py.File(infile, 'r') as hf :
        # sample_subIDs = hf['SubhaloID'][:]
        sample_masses = hf['SubhaloMassStars'][:]
    mask = (sample_masses > 10.07) & (sample_masses <= 12.75)
    # sample_subIDs = sample_subIDs[mask]
    sample_masses = sample_masses[mask]
    
    # get information about high mass galaxies and their zeta parameter
    infile = bsPath(simName) + '/{}_{}_highMassGals_zeta(t).hdf5'.format(
        simName, snapNum)
    with h5py.File(infile, 'r') as hf :
        highMass_subIDs = hf['SubhaloID'][:]
        highMass_zetas = hf['zeta(t)'][:]
    
    # loop over all the galaxies in the quenched sample
    for subID, tsat, tonset, tterm in zip(test_subIDs, tsats, tonsets, tterms) :
        
        # ts, zetas = determine_zeta_fxn(simName, snapNum, times, subID,
        #                                delta_t=delta_t)
        
        # find the corresponding index for the galaxy
        loc = np.where(highMass_subIDs == subID)[0][0]
        
        # find galaxies in a similar mass range as the galaxy
        mass = sample_masses[loc]
        mass_bin = determine_mass_bin_indices(sample_masses, mass, halfwidth=0.05)
        
        # use the zeta values for those comparison galaxies to determine percentiles
        comparison_zetas = highMass_zetas[mass_bin]
        lo_zeta, hi_zeta = np.nanpercentile(comparison_zetas, [16, 84], axis=0)
        lo_zeta = gaussian_filter1d(lo_zeta, 2)
        hi_zeta = gaussian_filter1d(hi_zeta, 2)
        
        # get zeta for the galaxy
        zetas = highMass_zetas[loc]
        
        if plot :
            # smooth the function for plotting purposes
            smoothed = gaussian_filter1d(zetas, 2)
            
            outfile = outDir + 'zeta_subID_{}.png'.format(subID)
            ylabel = r'$\zeta = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}}$'
            plt.plot_simple_multi_with_times(
                [times, times, times, times],
                [zetas, smoothed, lo_zeta, hi_zeta],
                ['data', 'smoothed', 'lo, hi', ''],
                ['grey', 'k', 'lightgrey', 'lightgrey'],
                ['', '', '', ''],
                ['--', '-', '-.', '-.'],
                [0.5, 1, 1, 1],
                tsat, tonset, tterm, xlabel=r'$t$ (Gyr)', ylabel=ylabel,
                xmin=0, xmax=14, ymin=0, ymax=6,
                scale='linear', save=save, outfile=outfile)
    
    return
