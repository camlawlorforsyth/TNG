
from os.path import exists
import numpy as np

from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from core import (add_dataset, bsPath, determine_mass_bin_indices,
                  get_mpb_radii_and_centers, get_particles, get_sf_particles)
import plotting as plt

def compute_xi(redshift, masses, rs) :
    
    if (len(masses) == 0) and (len(rs) == 0) :
        xi = np.nan
    else :
        
        # convert the distances to physical kpc
        rs = rs/(1 + redshift)/cosmo.h
        
        # define masks for both regions
        kpc_mask = (rs <= 1)
        
        # if there are star particles that are within that radius range
        if (np.sum(kpc_mask) > 0) :
            
            # determine the total mass formed within the inner region
            total_mass_formed_within_kpc = np.sum(masses[kpc_mask])
            total_mass_formed = np.sum(masses)
            
            xi = total_mass_formed_within_kpc/total_mass_formed
        else :
            xi = np.nan
    
    return xi

def determine_xi_fxn(simName, snapNum, redshifts, times, subID, 
                     delta_t=100*u.Myr) :
    
    # get the mpb snapshot numbers, subIDs, stellar halfmassradii, and
    # galaxy centers
    snapNums, mpb_subIDs, _, centers = get_mpb_radii_and_centers(
        simName, snapNum, subID)
    
    # limit the time axis to valid snapshots
    ts = times[len(times)-len(snapNums):]
    
    # now get the star particle ages, masses, and distances at each
    # snapshot/time
    xis = []
    for redshift, time, snap, mpbsubID, center in zip(redshifts, ts,
        snapNums, mpb_subIDs, centers) :
                        
        # get all particles
        ages, masses, rs = get_particles(simName, snapNum, snap, mpbsubID,
                                         center)
        
        # only proceed if the ages, masses, and distances are intact
        if (ages is not None) and (masses is not None) and (rs is not None) :
            
            # get the SF particles
            _, masses, rs = get_sf_particles(ages, masses, rs, time,
                                             delta_t=delta_t)
            
            # now compute the ratio of the SFR density within 1 kpc
            # relative to the total SFR
            xi = compute_xi(redshift, masses, rs)
            xis.append(xi)
        else :
            xis.append(np.nan)
    
    return ts, xis

def save_xi_for_sample(simName, snapNum) :

    # define the input directory and file for the sample, and the output file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    outfile = inDir + '/{}_{}_highMassGals_xi(t).hdf5'.format(
        simName, snapNum)
    
    # get basic information for the sample of primary and satellite systems
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        masses = hf['SubhaloMassStars'][:]
    
    # limit the sample to the highest mass objects
    subIDs = subIDs[(masses > 10.07) & (masses <= 12.75)]
    
    # add empty xi(t) into the HDF5 file to populate later
    if not exists(outfile) :
        with h5py.File(outfile, 'w') as hf :
            if 'subID' not in hf.keys() :
                add_dataset(hf, subIDs, 'SubhaloID')
            if 'xi(t)' not in hf.keys() :
                add_dataset(hf, np.full((len(subIDs), len(times)), np.nan),
                            'xi(t)')
    
    # if the outfile exists, read the relevant information
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            subIDs = hf['SubhaloID'][:]
            x_xi = hf['xi(t)'][:]
    
    # now iterate over every subID in subIDs and get xi(t)
    for i, subID in enumerate(subIDs) :
        
        # if xi(t) doesn't exist for the galaxy, populate the values
        if np.all(np.isnan(x_xi[i, :])) :
            
            _, xi = determine_xi_fxn(simName, snapNum, redshifts, times, subID,
                                     delta_t=100*u.Myr)
            start_index = len(x_xi[i, :]) - len(xi)
            
            # append the determined values for xi(t) into the outfile
            with h5py.File(outfile, 'a') as hf :
                hf['xi(t)'][i, start_index:] = xi
        
        print('{} - {} done'.format(i, subID))
    
    return

def xi_for_sample(simName, snapNum, delta_t=100*u.Myr, plot=False, save=False) :
    
    # define the output directory
    outDir = 'output/xi(t)/'
    
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
    redshifts, times, subIDs, tsats, tonsets, tterms = get_test_data()
    
    # get relevant information for the larger sample of 8260 galaxies
    infile = bsPath(simName) + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    with h5py.File(infile, 'r') as hf :
        # sample_subIDs = hf['SubhaloID'][:]
        sample_masses = hf['SubhaloMassStars'][:]
    mask = (sample_masses > 10.07) & (sample_masses <= 12.75)
    # sample_subIDs = sample_subIDs[mask]
    sample_masses = sample_masses[mask]
    
    # get information about high mass galaxies and their xi parameter
    infile = bsPath(simName) + '/{}_{}_highMassGals_xi(t).hdf5'.format(
        simName, snapNum)
    with h5py.File(infile, 'r') as hf :
        highMass_subIDs = hf['SubhaloID'][:]
        highMass_xis = hf['xi(t)'][:]
    
    # loop over all the galaxies in the quenched sample
    for subID, tsat, tonset, tterm in zip(subIDs, tsats, tonsets, tterms) :
        
        # ts, xis = determine_xi_fxn(simName, snapNum, subID, redshifts, times,
        #                            delta_t=delta_t)
        
        # find the corresponding index for the galaxy
        loc = np.where(highMass_subIDs == subID)[0][0]
        
        # find galaxies in a similar mass range as the galaxy
        mass = sample_masses[loc]
        mass_bin = determine_mass_bin_indices(sample_masses, mass, halfwidth=0.05)
        
        # use the xi values for those comparison galaxies to determine percentiles
        comparison_xis = highMass_xis[mass_bin]
        lo_xi, hi_xi = np.nanpercentile(comparison_xis, [16, 84], axis=0)
        lo_xi = gaussian_filter1d(lo_xi, 2)
        hi_xi = gaussian_filter1d(hi_xi, 2)
        
        # get xi for the galaxy
        xis = highMass_xis[loc]
        
        if plot :            
            # smooth the function for plotting purposes
            smoothed = gaussian_filter1d(xis, 2)
            
            outfile = outDir + 'xi_subID_{}.png'.format(subID)
            ylabel = r'$\xi = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$'
            plt.plot_simple_multi_with_times(
                [times, times, times, times],
                [xis, smoothed, lo_xi, hi_xi],
                ['data', 'smoothed', 'lo, hi', ''],
                ['grey', 'k', 'lightgrey', 'lightgrey'],
                ['', '', '', ''],
                ['--', '-', '-.', '-.'],
                [0.5, 1, 1, 1],
                tsat, tonset, tterm, xlabel=r'$t$ (Gyr)', ylabel=ylabel,
                xmin=0, xmax=14, ymin=0, ymax=1,
                scale='linear', save=save, outfile=outfile)
    
    return

def scatter() :
    
    simName = 'TNG50-1'
    snapNum = 99
    
    # get information about high mass galaxies and their xi parameter
    infile = bsPath(simName) + '/{}_{}_highMassGals_xi(t).hdf5'.format(
        simName, snapNum)
    with h5py.File(infile, 'r') as hf :
        highMass_subIDs = hf['SubhaloID'][:]
        highMass_xis = hf['xi(t)'][:]
    
    # get information about high mass galaxies and their zeta parameter
    infile = bsPath(simName) + '/{}_{}_highMassGals_zeta(t).hdf5'.format(
        simName, snapNum)
    with h5py.File(infile, 'r') as hf :
        highMass_subIDs = hf['SubhaloID'][:]
        highMass_zetas = hf['zeta(t)'][:]
    
    return