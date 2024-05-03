
import numpy as np

import astropy.units as u
import h5py
from scipy.ndimage import gaussian_filter1d

from core import bsPath, get_mpb_radii_and_centers, get_particles, get_sf_particles
import plotting as plt

def determine_SFH_along_mpb(simName, snapNum, times, subID, delta_t=100*u.Myr) :
    
    # get the mpb snapshot numbers, subIDs, stellar halfmassradii, and
    # galaxy centers
    snapNums, mpb_subIDs, radii, centers = get_mpb_radii_and_centers(
        simName, snapNum, subID)
    
    # limit the time axis to valid snapshots
    ts = times[len(times)-len(snapNums):]
    
    # now get the star particle ages, masses, and distances at each
    # snapshot/time
    SFH = []
    for time, snap, mpbsubID, center, Re in zip(ts, snapNums, mpb_subIDs,
                                                centers, radii) :
                        
        # get all particles
        ages, masses, rs = get_particles(simName, snapNum, snap, mpbsubID,
                                         center)
        
        # only proceed if the ages, masses, and distances are intact
        if (ages is not None) and (masses is not None) and (rs is not None) :
            
            # get the SF particles
            _, masses, rs = get_sf_particles(ages, masses, rs, time,
                                             delta_t=delta_t)
            
            # now compute the SFR for particles within 2Re
            if len(masses) == 0 :
                SFR = 0
            else :
                SFR = np.sum(masses[rs <= 2*Re])/delta_t*u.solMass
                SFR = (SFR.to(u.solMass/u.yr)).value
            SFH.append(SFR)
        else :
            SFH.append(0)
    
    return SFH

def determine_SFH_from_mpb(simName, snapNum, subID) :
    
    # define the input directory and the input file for the MPB SFHs
    inDir = bsPath(simName)
    infile = inDir + '/mpbs_099/sublink_mpb_{}.hdf5'.format(subID)
    
    with h5py.File(infile, 'r') as hf :
        mpb_SFRs = np.flip(hf['SubhaloSFRinRad'][:])
    
    return mpb_SFRs

def plot_quenched_systems_with_mpb(simName, snapNum) :
    
    # define the input directory and the input file for the SFHs
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
    # retrieve the relevant information about the quenched systems
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        masses = hf['SubhaloMassStars'][:]
        SFHs = hf['SFH'][:]
        times = hf['times'][:]
        tsats = hf['satellite_times'][:]
        tonsets = hf['onset_times'][:]
        tterms = hf['termination_times'][:]
    
    test_IDs = [
        # satellites
        14 #, 41, 63878, 167398, 184946, 220605, 324126,
        # primaries
        # 545003, 547545, 548151, 556699, 564498,
        # 592021, 604066, 606223, 607654, 623367
        ]
    locs = []
    for ID in test_IDs :
        locs.append(np.where(subIDs == ID)[0][0])
    mask = np.full(len(subIDs), False)
    for loc in locs :
        mask[loc] = True
    subIDs = subIDs[mask]
    masses = masses[mask]
    SFHs = SFHs[mask]
    tsats = tsats[mask]
    tonsets = tonsets[mask]
    tterms = tterms[mask]        
    
    # loop through the galaxies in the quenched sample
    for subID, mass, SFH, tsat, tonset, tterm in zip(subIDs, masses, SFHs,
        tsats, tonsets, tterms) :
        
        # using the orignally-computed SFH of the galaxy, smooth the SFH
        smoothed = gaussian_filter1d(SFH, 2)
        smoothed[smoothed < 0] = 0 # the SFR cannot be negative
        
        # get the SFR averaged over the past 100 Myr along the MPB, and pad the
        # front of the SFH if it doesn't go back to the earliest snapshot
        avg_SFH = np.full(100, np.nan)
        avg_SFRs = determine_SFH_along_mpb(simName, snapNum, times, subID, delta_t=100*u.Myr)
        avg_SFH[100-len(avg_SFRs):] = avg_SFRs
        avg_sm = gaussian_filter1d(avg_SFH, 2)
        
        # get the time-averaged SFRs from the TNG catalog, and pad the front
        catalog_SFH = np.full(100, np.nan)
        catalog_SFRs = determine_SFH_from_catalog(simName, snapNum, subID, times)
        catalog_SFH[100-len(catalog_SFRs):] = catalog_SFRs
        catalog_sm = gaussian_filter1d(catalog_SFH, 2)
        
        # get the instantaneous SFR along the MPB, and pad the front
        mpb_SFH = np.full(100, np.nan)
        mpb_SFRs = determine_SFH_from_mpb(simName, snapNum, subID)
        mpb_SFH[100-len(mpb_SFRs):] = mpb_SFRs
        mpb_sm = gaussian_filter1d(mpb_SFH, 2)
        
        print(avg_sm) # blue
        print()
        print(catalog_sm) # green
        
        '''
        # now plot the curves without the upper and lower limits
        outfile = 'output/quenched_SFHs(t)_comparison/quenched_SFH_subID_{}.png'.format(subID)
        plt.plot_simple_multi_with_times(
            [times, times, times, times, times, times, times, times],
            [SFH, smoothed, mpb_SFH, mpb_sm, avg_SFH, avg_sm, catalog_SFH, catalog_sm],
            ['star particles', 'smoothed (sm.)', r'MPB$_{\rm inst.}$',
             r'MPB$_{\rm inst.}$ sm.', r'MPB$_{\rm 100~Myr}$',
             r'MPB$_{\rm 100~Myr}$ sm.', r'catalog$_{\rm 100 Myr}$',
             r'catalog$_{\rm 100 Myr}$ sm.'],
            ['grey', 'k', 'orangered', 'r', 'dodgerblue', 'b', 'lime', 'g'],
            ['', '', '', '', '', '', '', ''],
            [':', '-', ':', '-', ':', '--', ':', '--'],
            [0.2, 1, 0.2, 1, 0.3, 1, 0.2, 1],
            tsat, tonset, tterm,
            xlabel=r'$t$ (Gyr)', ylabel=r'SFR ($M_{\odot}$ yr$^{-1}$)',
            xmin=0, xmax=14, scale='linear', save=False, outfile=outfile, loc=0)
        '''
    
    return


