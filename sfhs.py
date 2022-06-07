
from os.path import exists
import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
import h5py

from catalogs import convert_mass_units
from core import add_dataset, cutoutPath, gcPath, get, mpbPath
import plotting as plt

def download_all_cutouts(simName='TNG50-1', snapNum=99, redshift=0.0) :
    
    cutoutDir = cutoutPath(simName, snapNum)
    groupsDir = gcPath(simName, snapNum)
    
    # define the parameters that are requested for each particle in the cutout
    params = {'stars':
              'Coordinates,GFM_InitialMass,GFM_Metallicity,GFM_StellarFormationTime'}
    
    # open the table of subhalos in the sample that we want SFHs for
    subhalos = Table.read(
        groupsDir + 'subhalos_catalog_{}_{}_sample.fits'.format(simName, snapNum))
    
    subIDs = subhalos['SubhaloID']
    halfMassRadii = subhalos['SubhaloHalfmassRadStars']
    # badsubIDs = [] # IDs that aren't available - not on server
    
    for subID, R_e in zip(subIDs, halfMassRadii) :
        filename = cutoutDir + 'cutout_{}.hdf5'.format(subID)
        numpyfilename = cutoutDir + 'cutout_{}_masked.npz'.format(subID)
        subID_URL = 'http://www.tng-project.org/api/{}/snapshots/z={}/subhalos/{}'.format(
            simName, redshift, subID)
        
        # check if the cutout file exists
        if not exists(filename) :
            # retrieve information about the galaxy at the redshift of interest
            sub = get(subID_URL)
            
            # save the cutout file into the output directory
            get(sub['meta']['url'] + '/cutout.hdf5', directory=cutoutDir,
                params=params)
        
        # check if the cutout file exists and if the numpy file exists
        if exists(filename) and not exists(numpyfilename) :
            # retrieve information about the galaxy at the redshift of interest
            sub = get(subID_URL)
            
            # resave masked data into a numpy file for faster loading, and
            # to take up less disk space
            resave(filename, numpyfilename, sub, radius=2*R_e)
    
    return

def download_all_mpbs(simName='TNG50-1', snapNum=99, redshift=0.0) :
    
    mpbDir = mpbPath(simName, snapNum)
    groupsDir = gcPath(simName, snapNum)
    
    # open the table of subhalos in the sample that we want SFHs for
    subhalos = Table.read(
        groupsDir + 'subhalos_catalog_{}_{}_sample.fits'.format(simName, snapNum))
    
    subIDs = subhalos['SubhaloID']
    badsubIDs = [64192] # IDs that aren't available - not on server
    
    for subID in [subID for subID in subIDs if subID not in badsubIDs] :
        filename = mpbDir + 'sublink_mpb_{}.hdf5'.format(subID)
        subID_URL = 'http://www.tng-project.org/api/{}/snapshots/z={}/subhalos/{}'.format(
            simName, redshift, subID)
        
        # check if the main progenitor branch file exists
        if not exists(filename) :
            # retrieve information about the galaxy at the redshift of interest
            sub = get(subID_URL)
            
            # save the main progenitor branch file into the output directory
            get(sub['trees']['sublink_mpb'], directory=mpbDir)
    
    return

def history_from_cutout(numpyfilename, redshift=0.0) :
    
    # get the redshifts for all the snapshots
    table = Table.read('output/snapshot_redshifts.fits')
    redshifts = np.flip(table['Redshift'])
    
    # include an initial and final redshift
    redshifts = np.concatenate(([0.0], redshifts, [np.inf]))
    
    # calculate the lookbacktimes edges
    lookbacktimes = cosmo.lookback_time(redshifts).value
    lookbacktime_edges = lookbacktimes[:-1] + np.diff(lookbacktimes)/2
    
    '''
    # number of bins in lookback time
    Ntimes = 100
    
    # define the SFH bin boundaries in Gyr
    lookbacktime_edges = np.logspace(-4, np.log10(cosmo.age(redshift).value),
                                     num=Ntimes+1)
    lookbacktimes = lookbacktime_edges[:-1] + np.diff(lookbacktime_edges)/2
    '''
    
    # load information from saved numpy file
    numpyfile = np.load(numpyfilename)
    formation_ages = numpyfile['formation_ages']
    # metallicities = numpyfile['metallicities']
    initial_masses = numpyfile['initial_masses']
    
    # convert the formation_ages (in units of scalefactor) to redshifts
    formation_redshifts = 1.0/formation_ages - 1
    
    # determine the formation ages in terms of lookback times
    formation_lookbacktimes = (cosmo.lookback_time(formation_redshifts).value -
                               cosmo.lookback_time(redshift).value)
    
    # histogram the data to determine SFH(t) and Z(t)
    SFR, _ = np.histogram(formation_lookbacktimes, bins=lookbacktime_edges,
                          weights=np.power(10, initial_masses))
    # zz, _ = np.histogram(formation_lookbacktimes, bins=lookbacktime_edges,
    #                      weights=np.power(10, initial_masses)*metallicities)
    
    # mask the metallicity history based on if the SFH is valid
    # zh, mask = np.zeros(Ntimes), sfh > 0
    # zh[mask] = zz[mask]/sfh[mask]
    
    # plot the results
    '''
    lengths = np.diff(lookbacktime_edges)*1e9
    plt.plot_simple_dumb(lookbacktimes, SFR/lengths,
                         xlabel=r'$t_{\rm lookback}$ (Gyr)',
                         ylabel=r'SFR ($M_{\odot}$ yr$^{-1}$)',
                         xmin=-0.1, xmax=13.8)
    '''
    time_in_bins = np.diff(lookbacktime_edges)*1e9 # in yr
    
    return lookbacktimes[1:-1], SFR/time_in_bins

def history_from_mpb(mpbfilename) :
    
    # read the main progenitor branch merger tree and load relevant properties
    with h5py.File(mpbfilename, 'r') as mpb :
        SFR = mpb['SubhaloSFRinRad'][:] # most recent to earlier time
    
    # get the redshifts for all the snapshots
    table = Table.read('output/snapshot_redshifts.fits')
    redshifts = np.flip(table['Redshift'])
    lookbacktimes = cosmo.lookback_time(redshifts).value
    
    # some galaxies aren't in very early snapshots
    lookbacktimes = lookbacktimes[:len(SFR)]
    
    return lookbacktimes, SFR

def resave(filename, numpyfilename, sub, radius=None) :
    
    with h5py.File(filename, 'r') as hf :
        # get the formation ages (in units of scalefactor), metallicities, and
        # initial masses of all the star particles
        formation_ages = hf['PartType4']['GFM_StellarFormationTime'][:]
        metallicities = hf['PartType4']['GFM_Metallicity'][:]
        initial_masses = hf['PartType4']['GFM_InitialMass'][:]
        
        # if the radius is provided, only use star particles within that radius
        if radius :
            dx = hf['PartType4']['Coordinates'][:, 0] - sub['pos_x']
            dy = hf['PartType4']['Coordinates'][:, 1] - sub['pos_y']
            dz = hf['PartType4']['Coordinates'][:, 2] - sub['pos_z']
            rr = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
            
            # convert ckpc/h to physical kpc, and note that scale_factor=1 at z=0
            rr = rr/cosmo.h
            
            # mask out wind particles and constrain to radius
            mask = np.where((formation_ages > 0) & (rr < radius))
        else :
            mask = np.where(formation_ages > 0) # mask out wind particles
    
    #  mask out the wind particles and/or regions and save to numpy file
    np.savez(numpyfilename,
             formation_ages=formation_ages[mask],
             metallicities=metallicities[mask],
             initial_masses=convert_mass_units(initial_masses[mask]))
    
    return

def save_comparisons() :
    
    # open the table of subhalos in the sample that we want SFHs for
    subhalos = Table.read(
        'TNG50-1/output/groups_099/subhalos_catalog_TNG50-1_99_sample.fits')
    
    for subID in [97] : #subhalos['SubhaloID'] :
        # subIDs 10, 25, 40 are good examples of them matching up
        # 30, 50 are good examples of a mismatch
        
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
    
    return

def all_histories(simName='TNG50-1', snapNum=99, redshift=0.0) :
    
    # mpbfilename = mpbDir + 'sublink_mpb_{}.hdf5'.format(subID)
    
    groupsDir = gcPath(simName, snapNum)
    
    outfile = groupsDir + 'SFHs_sample.hdf5'
    
    # check if the outfile exists and has good SFHs
    # if exists(outfile) :
    #     with h5py.File(outfile, 'r') as hf :
    #         if 'SFH' in hf.keys() :
    #             if np.all(~np.isnan(hf['SFH'])) :
    #                 print('File already exists with all non-NaN SFHs')
    
    # open the table of subhalos in the sample that we want SFHs for
    subhalos = Table.read(
        groupsDir + 'subhalos_catalog_{}_{}_sample.fits'.format(simName, snapNum))
    
    # number of galaxies; number of bins in lookback time
    Ngals = len(subhalos)
    
    # check if the outfile exists, and if not, populate key information into it
    if not exists(outfile) :
        with h5py.File(outfile, 'w') as hf :
            # add basic information from the table into the HDF5 file
            add_dataset(hf, subhalos['SubhaloID'], 'SubhaloID')
            add_dataset(hf, subhalos['SubhaloFlag'], 'SubhaloFlag')
            add_dataset(hf, subhalos['SubhaloMassStars'], 'SubhaloMassStars')
            add_dataset(hf, subhalos['SubhaloSFRinRad'], 'SubhaloSFRinRad')
            add_dataset(hf, subhalos['SubhaloHalfmassRadStars'],
                        'SubhaloHalfmassRadStars')
            
            # add information about the lookback time bin centers and edges
            add_dataset(hf, np.array([redshift]), 'redshift',
                        dtype=type(redshift))
            # add_dataset(hf, lookbacktimes, 'lookbacktimes')
            # add_dataset(hf, lookbacktime_edges, 'lookbacktime_edges')
            
            # add empty SFH information into the HDF5 file to populate later
            add_dataset(hf, np.full(Ngals, np.nan), 'primary_flag')
            # add_dataset(hf, np.full((Ngals, Ntimes), np.nan), 'SFH')
            # add_dataset(hf, np.full((Ngals, Ntimes), np.nan), 'Zhistory')
    
    # if the outfile exists, read the relevant information
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            subIDs = hf['SubhaloID'][:]
            R_e = hf['SubhaloHalfmassRadStars'][:]
            x_sfhs = hf['SFH'][:]
            # x_zhs = hf['Zhistory'][:]
    
    # now iterate over every subID in subIDs and get the SFH for that subID
    for i, subID in enumerate(subIDs) :
        
        # if the SFHs don't exist for the galaxy, populate the SFHs
        if np.all(np.isnan(x_sfhs[i, :])) :
            
            # retrieve information about the galaxy at the redshift of interest
            sub = get(
                'http://www.tng-project.org/api/{}/snapshots/z={}/subhalos/{}'.format(
                    simName, redshift, subID))
            
            # determine the SFH and metallicity history for the galaxy
            SFH = history_from_cutout(simName, snapNum, sub, subID,
                                      lookbacktime_edges, Ntimes,
                                      radius=2*R_e[i], overwrite=False,
                                      redshift=redshift)
            
            # append those values, and a flag if the galaxy is the primary galaxy,
            # into the outfile
            with h5py.File(outfile, 'a') as hf :
                hf['SFH'][i, :] = SFH
                # hf['Zhistory'][i, :] = Zhistory
                hf['primary_flag'][i] = sub['primary_flag']
    
    return
