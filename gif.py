
import numpy as np

import astropy.units as u

from core import (get_mpb_radii_and_centers, get_particle_positions,
                  get_sf_particle_positions)
import plotting as plt

def find_nearest(times, index_times) :
    
    indices = []
    for time in index_times :
        index = (np.abs(times - time)).argmin()
        indices.append(index)
    
    return indices

def quenching_mechanism(simName, snapNum, delta_t=100*u.Myr) :
    
    # define the output directory
    outDir = 'output/evolution(t)/'
    
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
        
        index_times = np.array([tonset, tonset + 0.5*(tterm - tonset), tterm])
        indices = find_nearest(times, index_times)
        
        # now get the star particle ages, masses, and distances at each
        # snapshot/time
        for time, snap, mpbsubID, center in zip(ts, snapNums, mpb_subIDs,
                                                centers) :
            
            if snap in indices :
                # get all particles
                ages, masses, dx, dy, dz = get_particle_positions(simName,
                    snapNum, snap, mpbsubID, center)
                
                # get the SF particles
                _, sf_masses, sf_dx, sf_dy, sf_dz = get_sf_particle_positions(
                    ages, masses, dx, dy, dz, time, delta_t=delta_t)
                
                # plot the results
                outfile = outDir + 'evolution_subID_{}_snap_{}.png'.format(subID, snap)
                plt.plot_scatter_multi(masses, dx, dy, dz,
                                       sf_masses, sf_dx, sf_dy, sf_dz,
                                       xlabel=r'$\Delta x$ (ckpc/h)',
                                       ylabel=r'$\Delta y$ (ckpc/h)',
                                       zlabel=r'$\Delta z$ (ckpc/h)',
                                       save=False, outfile=outfile)
    
    return
