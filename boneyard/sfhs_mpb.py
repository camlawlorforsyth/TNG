
import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
import h5py

import plotting as plt

def history_from_mpb(mpbfilename) :
    
    # read the main progenitor branch merger tree and load relevant properties
    with h5py.File(mpbfilename, 'r') as mpb :
        SFR = mpb['SubhaloSFRinRad'][:] # most recent to earlier time
        snaps = mpb['SnapNum'][:]
    
    # get the redshifts for all the snapshots
    table = Table.read('output/snapshot_redshifts.fits')
    
    # sort the redshifts according to the snapshot order from the MPB file
    # could also simply use redshifts = np.flip(table['Redshift'])
    redshifts = []
    for snap in snaps :
        redshifts.append(table['Redshift'][np.where(table['SnapNum'] == snap)][0])
    
    lookbacktimes = cosmo.lookback_time(redshifts).value
    
    # some galaxies aren't in very early snapshots
    lookbacktimes = lookbacktimes[:len(SFR)]
    
    return lookbacktimes, SFR

def save_comparisons() :
    
    # open the table of subhalos in the sample that we want SFHs for
    subhalos = Table.read(
        'TNG50-1/output/groups_099/subhalos_catalog_TNG50-1_99_sample.fits')
    
    for subID in subhalos['SubhaloID'] :
        # save the SFHs using the two different methods
        times_mpb, SFR_mpb = history_from_mpb(
            'TNG50-1/output/mpbs_099/sublink_mpb_{}.hdf5'.format(subID))
        
        times_cut, SFR_cut = history_from_cutout(
            'TNG50-1/output/cutouts_099/cutout_{}_masked.npz'.format(subID))
        
        # plot and compare the results
        plt.plot_simple_multi([times_mpb, times_cut], [SFR_mpb, SFR_cut],
                              ['MPB', 'star ages'], ['k', 'r'], ['', ''],
                              ['-', '-'], alphas=[1, 1], scale='linear', loc=0,
                              xlabel=r'$t_{\rm lookback}$ (Gyr)',
                              ylabel=r'SFR ($M_{\odot}$ yr$^{-1}$)',
                              xmin=-0.1, xmax=13.8, save=False,
                              outfile='output/SFH_subID_{}.png'.format(subID))
        
    '''
    for subID in [97] : # [52, 58, 87, 97, 101]
        # subIDs 10, 25, 40 are good examples of them matching up
        # 30, 50 are good examples of a mismatch
    
        snapNum_mpb, SFR_mpb = history_from_mpb(
            'TNG50-1/output/mpbs_099/sublink_mpb_{}.hdf5'.format(subID))
        plt.plot_simple_dumb(snapNum_mpb, SFR_mpb,
                             xlabel='Snapshot Number', #r'$t_{\rm lookback}$ (Gyr)',
                             ylabel=r'SFR ($M_{\odot}$ yr$^{-1}$)',
                             xmin=-1, xmax=100)
    '''
    return
