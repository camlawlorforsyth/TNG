
from os.path import exists
import numpy as np

from astropy.cosmology import Planck15 as cosmo, z_at_value
import astropy.units as u
import h5py
from scipy.signal import savgol_filter

from catalogs import convert_distance_units
from core import bsPath, cutoutPath, get, mpbPath
import plotting as plt

def get_mpb_halfmassradii(simName, snapNum, subID) :
    
    # define the mpb directory and file
    mpbDir = mpbPath(simName, snapNum)
    mpbfile = mpbDir + 'sublink_mpb_{}.hdf5'.format(subID)
    
    # get the mpb subIDs, stellar halfmassradii, and galaxy centers
    with h5py.File(mpbfile, 'r') as hf :
        mpb_subIDs = hf['SubfindID'][:]
        halfmassradii = hf['SubhaloHalfmassRadType'][:, 4]
        centers = hf['SubhaloPos'][:]
    
    # given that the mpb files are arranged such that the most recent redshift
    # (z = 0) is at the beginning of arrays, we'll flip the arrays to conform
    # to an increasing-time convention
    mpb_subIDs = np.flip(mpb_subIDs)
    radii = np.flip(convert_distance_units(halfmassradii))
    centers = np.flip(centers, axis=0)
    
    return mpb_subIDs, radii, centers

def download_mpb_cutouts(simName, snapNum, subID) :
    
    # define the parameters that are requested for each particle in the cutout
    params = {'stars':'Coordinates,GFM_InitialMass,GFM_StellarFormationTime'}
    
    url = 'https://www.tng-project.org/api/{}/snapshots/{}/subhalos/{}'.format(
        simName, snapNum, subID)
    
    # retrieve information about the galaxy at the redshift of interest
    sub = get(url)
    
    # save the cutout file into the output directory
    get(sub['meta']['url'] + '/cutout.hdf5', directory='test/', params=params)
    
    while sub['prog_sfid'] != -1 :
        # request the full subhalo details of the progenitor by following the sublink URL
        sub = get(sub['related']['sublink_progenitor'])
        
        # check if the cutout file exists
        if not exists('test/cutout_{}.hdf5'.format(sub['id'])) :
            # save the cutout file into the output directory
            get(sub['meta']['url'] + '/cutout.hdf5', directory='test/',
                params=params)
    
    return 

def compute_sf_rms(simName, snapNum, subID, center, time, delta_t=100*u.Myr) :
    
    # define the cutout directory and file
    cutoutDir = cutoutPath(simName, snapNum)
    cutout_file = cutoutDir + 'cutout_{}.hdf5'.format(subID)
    
    # cosmo.age(redshift) is slow for very large arrays, so we'll work in units
    # of scalefactor and convert delta_t. t_minus_delta_t is in units of redshift
    t_minus_delta_t = z_at_value(cosmo.age, time*u.Gyr - delta_t, zmax=np.inf)
    limit = 1/(1 + t_minus_delta_t) # in units of scalefactor
    
    try :
        with h5py.File(cutout_file) as hf :
            dx = hf['PartType4']['Coordinates'][:, 0] - center[0]
            dy = hf['PartType4']['Coordinates'][:, 1] - center[1]
            dz = hf['PartType4']['Coordinates'][:, 2] - center[2]
            
            # formation ages are in units of scalefactor
            formation_ages = hf['PartType4']['GFM_StellarFormationTime'][:]
        
        mask = formation_ages >= limit
        
        rs = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))[mask]
        
        if len(rs) == 0 :
            rms = 0.0
        else :
            rms = np.sqrt(np.mean(np.square(rs)))
        return rms
    
    except KeyError :
        return 0.0

def determine_sf_rms_function(simName, snapNum, window_length, polyorder,
                              plot=False) :
    
    # define the output directory and the output file
    outDir = bsPath(simName)
    outfile = outDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
    # get the masses, satellite and termination times for the quenched sample
    with h5py.File(outfile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        times = hf['times'][:]
        tsats = hf['satellite_times'][:]
        tonsets = hf['onset_times'][:]
        tterms = hf['termination_times'][:]
    
    # loop over all the galaxies in the quenched sample
    for subID, tsat, tonset, tterm in zip(subIDs, tsats, tonsets, tterms) :
        # get the mpb subIDs, stellar halfmassradii, and galaxy centers
        mpb_subIDs, radii, centers = get_mpb_halfmassradii(simName, snapNum, subID)
        
        # ensure that the quotient will be real
        radii[radii == 0.0] = np.inf
        
        # now compute the stellar rms values at each snapshot/time
        rmses = []
        for time, subID, center in zip(times, mpb_subIDs, centers) :
            rms = compute_sf_rms(simName, snapNum, subID, center, time)
            rmses.append(rms)
        rmses = np.array(rmses)
        
        # now compute the quantity zeta, which is the quotient of the RMS of the
        # radial distances of the star forming particles to the stellar half mass radii
        zeta = rmses/radii
        
        # smooth the function for plotting purposes
        smoothed = savgol_filter(zeta, window_length, polyorder)
        smoothed[smoothed < 0] = 0
        
        if plot :
            # now plot the results, but limit the time axis to valid snapshots
            ts = times[len(times)-len(zeta):]
            
            ylabel=r'$\zeta =$ RMS$_{\rm SF}/$Stellar Half Mass Radius$_{\rm SF}$'
            plt.plot_simple_multi_with_times([ts, ts], [zeta, smoothed],
                ['data', 'smoothed'], ['grey', 'k'], ['', ''], ['--', '-'], [0.5, 1],
                tsat, tonset, tterm, xlabel=r'$t$ (Gyr)', ylabel=ylabel,
                xmin=-0.1, xmax=13.8, scale='linear')
    
    return
