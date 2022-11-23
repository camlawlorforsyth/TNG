
import numpy as np

from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from core import (bsPath, get_mpb_radii_and_centers, get_particles,
                  get_sf_particles)
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

def determine_xi(simName, snapNum, delta_t=100*u.Myr, plot=False, save=False) :
    
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
    
    from core import get_test_data
    redshifts, times, subIDs, tsats, tonsets, tterms = get_test_data()
    
    # loop over all the galaxies in the quenched sample
    for subID, tsat, tonset, tterm in zip(subIDs, tsats, tonsets, tterms) :
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
        
        if plot :            
            # smooth the function for plotting purposes
            smoothed = gaussian_filter1d(xis, 2)
            
            outfile = outDir + 'xi_subID_{}.png'.format(subID)
            ylabel = r'$\xi = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$'
            plt.plot_simple_multi_with_times([ts, ts], [xis, smoothed],
                ['data', 'smoothed'], ['grey', 'k'], ['', ''], ['--', '-'], [0.5, 1],
                tsat, tonset, tterm, xlabel=r'$t$ (Gyr)', ylabel=ylabel,
                xmin=0, xmax=14, ymin=0, ymax=1,
                scale='linear', save=save, outfile=outfile)
    
    return
