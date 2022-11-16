
import numpy as np

from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from core import (bsPath, get_mpb_radii_and_centers, get_particles,
                  get_sf_particles)
import plotting as plt

def compute_nabla_psi(redshift, masses, rs, radius, snap, delta_t=100*u.Myr) :
    
    if (len(rs) == 0) and (len(masses) == 0) :
        gradient = np.nan
    else :
        
        # some particles have distances of 0, so we need to increase their
        # distances by a small amount, equal to the distance of the next most
        # inner star particle, for taking the logarithm
        if len(rs) > 1 :
            rs[rs == 0.0] = np.sort(rs)[1]
        else :
            rs[rs == 0.0] = 0.001
        
        # scale the radius by the effective radius
        rs = np.log10(rs/radius)
        
        # define the shell edges, based on the region that's the most physically
        # interesting
        edges = np.linspace(-1, 0.5, 11)
        
        # now determine the middle points in those shells, and the total
        # SFR within that shell/volume
        centers, psis = [], []
        for first, second in zip(edges, edges[1:]) :
            
            # determine the middle points
            center = np.mean([first, second])
            centers.append(center)
            
            # limit the star particles to the given shell
            mask = (rs > first) & (rs <= second)
            
            # if there are star particles that are within that radius range
            if np.sum(mask) > 0 :
                
                # determine the total mass formed within that bin
                masses_in_bin = masses[mask]
                total_mass = np.sum(masses_in_bin)*u.solMass
                
                # determine the total volume in the shell, in physical units
                outer_r = np.power(10, second*radius)/(1 + redshift)/cosmo.h
                inner_r = np.power(10, first*radius)/(1 + redshift)/cosmo.h
                volume = 4/3*np.pi*(np.power(outer_r, 3) -
                                    np.power(inner_r, 3))*(u.kpc**3)
                
                # determine the SFR within that bin, and also the SFR density
                SFR = (total_mass/delta_t).to(u.solMass/u.yr)
                psi = SFR/volume
                psis.append(psi.value) # ensure that the
                # SFR densities are unitless for future masking, but note
                # that they have units of solMass/yr/kpc^3
            else :
                psis.append(np.nan)
        
        # mask out the nan values
        centers = np.array(centers)[~np.isnan(psis)]
        psis = np.log10(psis)[~np.isnan(psis)]
        
        # if we have sufficient data, then compute the gradient
        if (len(centers) > 1) and (len(psis) > 1) :
            gradient, intercept = np.polyfit(centers, psis, 1)
        else :
            gradient = np.nan
    
    return gradient

def determine_psi(simName, snapNum, delta_t=100*u.Myr, plot=False, save=False) :
    
    # define the output directory
    outDir = 'output/psi(t)/'
    
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
        snapNums, mpb_subIDs, radii, centers = get_mpb_radii_and_centers(
            simName, snapNum, subID)
        
        # limit the time axis to valid snapshots
        ts = times[len(times)-len(snapNums):]
        
        # now get the star particle ages, masses, and distances at each
        # snapshot/time
        gradients = []
        for redshift, time, snap, mpbsubID, center, Re in zip(redshifts,
            ts, snapNums, mpb_subIDs, centers, radii) :
                            
            # get all particles
            ages, masses, rs = get_particles(simName, snapNum, snap, mpbsubID,
                                             center)
            
            # only proceed if the ages, masses, and distances are intact
            if (ages is not None) and (masses is not None) and (rs is not None):
                
                # get the SF particles
                _, masses, rs = get_sf_particles(ages, masses, rs, time,
                                                 delta_t=delta_t)
                
                # now compute the SFR density gradient at each snapshot/time
                gradient = compute_nabla_psi(redshift, masses, rs, Re, snap,
                                             delta_t=delta_t)
                gradients.append(gradient)
            else :
                gradients.append(np.nan)
        
        if plot :
            # smooth the function for plotting purposes
            smoothed = gaussian_filter1d(gradients, 2)
            
            # determine the y-axis limits based on minima +/- 1 Gyr around
            # the quenching episode
            window = np.where((ts >= tonset - 1) & (ts <= tterm + 1))
            lo, hi = np.min(smoothed[window]), np.max(smoothed[window])
            
            outfile = outDir + 'SFR_density_gradient_subID_{}.png'.format(subID)
            ylabel = r'$\nabla \left[ \log (\psi/M_{\odot}~{\rm yr}^{-1}~{\rm kpc}^{-3})/\log (r/R_{\rm e}) \right]$'
            plt.plot_simple_multi_with_times([ts, ts], [gradients, smoothed],
                ['data', 'smoothed'], ['grey', 'k'], ['', ''], ['--', '-'], [0.5, 1],
                tsat, tonset, tterm, xlabel=r'$t$ (Gyr)', ylabel=ylabel,
                xmin=0, xmax=14,
                scale='linear', save=save, outfile=outfile)
    
    return
