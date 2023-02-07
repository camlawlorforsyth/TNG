
import numpy as np

from astropy.table import Table
import astropy.units as u
import scipy.stats as st

from core import (get_mpb_radii_and_centers, get_particle_positions,
                  get_sf_particle_positions)
import plotting as plt

def find_nearest(times, index_times) :
    
    indices = []
    for time in index_times :
        index = (np.abs(times - time)).argmin()
        indices.append(index)
    
    return indices

def quenching_mechanism(simName, snapNum, delta_t=100*u.Myr, fast=True, save=False) :
    
    # define the output directory
    outDir = 'output/evolution(t)/'
    
    from core import get_test_data
    redshifts, times, subIDs, tsats, tonsets, tterms = get_test_data()
    
    # loop over all the galaxies in the quenched sample
    for subID, tsat, tonset, tterm in zip(subIDs, tsats, tonsets, tterms) :
        # get the mpb snapshot numbers, subIDs, stellar halfmassradii, and
        # galaxy centers
        snapNums, mpb_subIDs, radii, centers = get_mpb_radii_and_centers(
            simName, snapNum, subID)
        
        # limit the time axis to valid snapshots
        ts = times[len(times)-len(snapNums):]
        
        index_times = np.array([tonset, tonset + 0.5*(tterm - tonset), tterm])
        indices = find_nearest(times, index_times)
        
        # now get the star particle ages, masses, and distances at each
        # snapshot/time
        for time, snap, mpbsubID, center, Re in zip(ts, snapNums, mpb_subIDs,
                                                    centers, radii) :
            
            if snap in indices :
                # get all particles
                ages, masses, dx, dy, dz = get_particle_positions(simName,
                    snapNum, snap, mpbsubID, center)
                
                # get the SF particles
                _, sf_masses, sf_dx, sf_dy, sf_dz = get_sf_particle_positions(
                    ages, masses, dx, dy, dz, time, delta_t=delta_t)
                
                # plot the results
                outfile = outDir + 'evolution_subID_{}_snap_{}.png'.format(subID, snap)
                # plt.plot_scatter_multi(dx, dy, dz,
                #                        sf_dx, sf_dy, sf_dz,
                #                        xlabel=r'$\Delta x$ (ckpc/h)',
                #                        ylabel=r'$\Delta y$ (ckpc/h)',
                #                        zlabel=r'$\Delta z$ (ckpc/h)',
                #                        save=False, outfile=outfile)
                
                # from astropy.table import Table
                # table = Table([rand, dx, dz], names=('rand', 'dx', 'dz'))
                # table = table[table['rand'] < 0.0024]
                # df = table.to_pandas()
                # edges = np.linspace(-5*Re, 5*Re, 41)
                # plt.plot_scatter_CASTOR(dx, dz, sf_dx, sf_dz, Re, df=df,
                #                         bins=[edges, edges],
                #                         xlabel=r'$\Delta x$ (ckpc/h)',
                #                         ylabel=r'$\Delta z$ (ckpc/h)',
                #                         xmin=-5*Re, xmax=5*Re,
                #                         ymin=-5*Re, ymax=5*Re,
                #                         save=False, outfile=outfile)
                
                # define an array of random numbers to select ~1000 particles
                np.random.seed(0)
                rand = np.random.random(len(dx))
                
                table = Table([rand, dx/Re, dz/Re, masses/1e8],
                              names=('rand', 'dx', 'dz', 'masses'))
                if fast :
                    table = table[table['rand'] < 0.0024]
                df = table.to_pandas()
                # df = None
                
                if snap == indices[-1] :
                    legend = True
                    if df is not None :
                        figwidth = 8.7
                else :
                    figwidth = 7
                    legend = False
                
                title = (r'$z = {:.2f}$'.format(redshifts[snap]) + ', ' +
                         r'$\Delta t_{\rm since~onset} = $' +
                         '{:.1f} Gyr'.format(time - tonset))
                
                edges = np.linspace(-5, 5, 41)
                plt.plot_scatter_CASTOR(dx/Re, dz/Re, sf_dx/Re, sf_dz/Re,
                                        sf_masses/1e8, 1, df=df,
                                        radii=[2, 4], legend=legend,
                                        bins=[edges, edges], title=title,
                                        xlabel=r'$\Delta x$ ($R_{\rm e}$)',
                                        ylabel=r'$\Delta z$ ($R_{\rm e}$)',
                                        xmin=-5, xmax=5, ymin=-5, ymax=5,
                                        figsizewidth=figwidth,
                                        save=save, outfile=outfile)
    
    return
