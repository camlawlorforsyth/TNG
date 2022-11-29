
from os.path import exists
import numpy as np

import astropy.units as u
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from core import (add_dataset, bsPath, get_mpb_radii_and_centers, get_particles,
                  get_sf_particles)
import plotting as plt

def compute_halfmass_radius(masses, rs) :
    
    if (len(masses) == 0) and (len(rs) == 0) :
        halfmass_radius = np.nan
    else :
        sort_order = np.argsort(rs)
        masses = masses[sort_order]
        rs = rs[sort_order]
        halfmass_radius = np.interp(0.5*np.sum(masses), np.cumsum(masses), rs)
    
    return halfmass_radius

def compute_sf_rms(rs) :
    
    if len(rs) == 0 :
        rms = 0.0
    else :
        rms = np.sqrt(np.mean(np.square(rs)))
    
    return rms

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
            zetas.append(zeta)
        else :
            zetas.append(np.nan)
    
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
            # append the determined values for zeta(t) into the outfile
            with h5py.File(outfile, 'a') as hf :
                _, hf['zeta(t)'][i, :] = determine_zeta_fxn(simName, snapNum,
                    times, subID, delta_t=100*u.Myr)
    
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
    
    from core import get_test_data
    _, times, subIDs, tsats, tonsets, tterms = get_test_data()
    
    # loop over all the galaxies in the quenched sample
    for subID, tsat, tonset, tterm in zip(subIDs, tsats, tonsets, tterms) :
        
        ts, zetas = determine_zeta_fxn(simName, snapNum, times, subID,
                                       delta_t=delta_t)
        
        if plot :
            # smooth the function for plotting purposes
            smoothed = gaussian_filter1d(zetas, 2)
            
            outfile = outDir + 'zeta_subID_{}.png'.format(subID)
            ylabel = r'$\zeta = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}}$'
            plt.plot_simple_multi_with_times([ts, ts], [zetas, smoothed],
                ['data', 'smoothed'], ['grey', 'k'], ['', ''], ['--', '-'], [0.5, 1],
                tsat, tonset, tterm, xlabel=r'$t$ (Gyr)', ylabel=ylabel,
                xmin=0, xmax=14, ymin=0, ymax=6,
                scale='linear', save=save, outfile=outfile)
    
    return
