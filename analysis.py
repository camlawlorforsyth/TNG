
from os.path import exists
import numpy as np

from astropy.cosmology import Planck15 as cosmo, z_at_value
from astropy.table import Table
import astropy.units as u
import h5py
from scipy.signal import savgol_filter

from catalogs import convert_distance_units
from core import cutoutPath, get, mpbPath
import plotting as plt

import warnings
warnings.filterwarnings('ignore')

def get_mpb_halfmassradii(simName, snapNum, subID) :
    
    infile = mpbPath(simName, snapNum) + 'sublink_mpb_{}.hdf5'.format(subID)
    with h5py.File(infile, 'r') as hf :
        mpb_subIDs = hf['SubfindID'][:]
        halfmassradii = hf['SubhaloHalfmassRadType'][:, 4]
        centers = hf['SubhaloPos'][:]
    
    return mpb_subIDs, convert_distance_units(halfmassradii), centers

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
    
    # cosmo.age(redshift) is slow for very large arrays, so we'll work in units
    # of scalefactor and convert delta_t
    t_minus_delta_t = z_at_value(cosmo.age, time - delta_t, zmax=np.inf) # in units of redshift
    limit = 1/(1 + t_minus_delta_t)
    
    cutout_file = cutoutPath(simName, snapNum) + 'cutout_{}.hdf5'.format(subID)
    
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

def determine_sf_rms_function(simName, snapNum, subID, plot=False) :
    
    window_length, polyorder = 15, 3
    
    # table = Table.read('output/tquench_vs_tsat.fits')
    # loc = np.where(table['subID'] == subID)[0][0]
    # tquench, tsat = table['t_quench'][loc], table['t_sat'][loc]
    
    table = Table.read('output/snapshot_redshifts.fits')
    redshifts = table['Redshift'].value
    ts = cosmo.age(redshifts)
    
    # given that the MPB files are arranged such that the most recent redshift
    # (z = 0) is at the beginning of arrays, we'll work in that convention
    # and then flip the arrays at the end when plotting
    mpb_subIDs, radii, centers = get_mpb_halfmassradii(simName, snapNum, subID)
    
    radii[radii == 0.0] = np.inf # ensure that the quotient will be real
    
    rms = []
    for time, subID, center in zip(np.flip(ts), mpb_subIDs, centers) :
        rms.append(compute_sf_rms(simName, snapNum, subID, center, time))
    
    # now compute the quantity zeta, which is the quotient of the RMS of the
    # radial distances of the star forming particles to the stellar half mass radii
    zeta = np.flip(rms)/np.flip(radii)
    
    smoothed = savgol_filter(zeta, window_length, polyorder)
    smoothed[smoothed < 0] = 0
    
    tsat, tonset, tquench = 5.4266043438955816, 5.4266043438955816, 7.127169581183778
    
    if plot :
        # now plot the results, but limit the time axis to valid snapshots
        
        ts = ts[len(ts)-len(zeta):]
        
        plt.plot_simple_multi_with_times([ts, ts], [zeta, smoothed],
            ['data', 'smoothed'], ['grey', 'k'], ['', ''], ['--', '-'], [0.75, 1],
            tsat, tonset, tquench, xlabel=r'$t$ (Gyr)',
            ylabel=r'$\zeta =$ RMS$_{\rm SF}/$Stellar Half Mass Radius$_{\rm SF}$',
            xmin=-0.1, xmax=13.8, scale='linear')
        
        # plt.plot_simple_with_vertical_lines(
        #     ts[len(ts)-len(zeta):], zeta, tsat, tonset, tquench, xlabel=r'$t$ (Gyr)',
        #     ylabel=r'$\zeta =$ RMS$_{\rm SF}/$Half Mass Radius$_{\rm SF}$')
    
    return
