
from os.path import exists
import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
import h5py
import requests

from core import (add_dataset, bsPath, get, get_mpb_masses,
                  get_mpb_radii_and_centers, mpbPath)

def determine_all_histories_from_catalog(simName='TNG50-1', snapNum=99,
                                         delta_t=100) :
    
    # define the output directory and the output file
    outDir = bsPath(simName)
    outfile = outDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    
    # check if the outfile exists and has good SFHs
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            if 'SFH' in hf.keys() :
                if np.all(~np.isnan(hf['SFH'])) :
                    print('File already exists with all non-NaN SFHs')
    
    # open the table of subhalos in the sample that we want SFHs for
    subhalos = Table.read(outDir + '/{}_{}_sample.fits'.format(simName, snapNum))
    
    # get the redshifts and times (ages of the universe) for all the snapshots
    table = Table.read('TNG50-1/snapshot_redshifts.fits')
    redshifts = table['Redshift'].value
    times = cosmo.age(redshifts).value
    
    # check if the outfile exists, and if not, populate key information into it
    if not exists(outfile) :
        with h5py.File(outfile, 'w') as hf :
            # add information about the snapshot redshifts and times, and subIDs
            add_dataset(hf, redshifts, 'redshifts')
            add_dataset(hf, times, 'times')
            add_dataset(hf, subhalos['SubhaloID'], 'SubhaloID')
            
            # add empty SFH, masses, and SFMS info
            add_dataset(hf, np.full((8260, 100), np.nan), 'SFH')
            add_dataset(hf, np.full((8260, 100), np.nan), 'logM')
            add_dataset(hf, np.full((8260, 100), np.nan), 'SFMS')
    
    # if the outfile exists, read the relevant information
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            subIDs = hf['SubhaloID'][:]
            x_sfhs = hf['SFH'][:]
            x_logMs = hf['logM'][:]
    
    # now iterate over every subID in subIDs and get the SFH and logM along
    # the MPB for that subID
    for i, subID in enumerate(subIDs) :
        
        # if the SFH values don't exist, determine the SFH and populate
        if np.all(np.isnan(x_sfhs[i, :])) :
            with h5py.File(outfile, 'a') as hf : # use the catalog values
                hf['SFH'][i, :] = history_from_catalog(subID,
                    simName=simName, snapNum=snapNum, delta_t=delta_t)
        
        # if the logM values don't exist, determine the values and populate
        if np.all(np.isnan(x_logMs[i, :])) :
            with h5py.File(outfile, 'a') as hf :
                hf['logM'][i, :] = get_mpb_masses(subID)
    
    return

def download_all_mpbs(simName='TNG50-1', snapNum=99) :
    
    mpbDir = mpbPath(simName, snapNum)
    outDir = bsPath(simName)
    
    # open the table of subhalos in the sample that we want SFHs for
    subhalos = Table.read(outDir + '/{}_{}_sample.fits'.format(simName, snapNum))
    
    for subID in subhalos['SubhaloID'] :
        outfile = mpbDir + 'sublink_mpb_{}.hdf5'.format(subID)
        url = 'http://www.tng-project.org/api/{}/snapshots/{}/subhalos/{}'.format(
            simName, snapNum, subID)
        
        # check if the main progenitor branch file exists
        if not exists(outfile) :
            # retrieve information about the galaxy at the redshift of interest
            sub = get(url)
            
            try :
                # save the main progenitor branch file into the output directory
                get(sub['trees']['sublink_mpb'], directory=mpbDir)
            
            # if the subID mpb file doesn't exist on the server, save critical
            # information to a similarly formatted hdf5 file
            except requests.exceptions.HTTPError :
                with h5py.File(outfile, 'w') as hf :
                    # we need the center and the stellar halfmassradius to
                    # compute the SFH
                    centers = np.array([[sub['pos_x'], sub['pos_y'], sub['pos_z']]])
                    halfmassradii = np.array([[sub['halfmassrad_gas'],
                                               sub['halfmassrad_dm'], 0.0, 0.0,
                                               sub['halfmassrad_stars'],
                                               sub['halfmassrad_bhs']]])
                    
                    # add that information into the outfile
                    add_dataset(hf, centers, 'SubhaloPos')
                    add_dataset(hf, halfmassradii, 'SubhaloHalfmassRadType')
    
    return

def history_from_catalog(subID, simName='TNG50-1', snapNum=99, delta_t=100) :
    
    catalog_file = 'TNG50-1/Donnari_Pillepich_star_formation_rates.hdf5'
    
    # get the mpb snapshot numbers and subIDs
    snapNums, mpb_subIDs, _, _ = get_mpb_radii_and_centers(simName, snapNum, subID)
    
    SFH = []
    for snap, mpb_subID in zip(snapNums, mpb_subIDs) :
        if snap <= 1 :
            SFH.append(np.nan)
        
        # the catalog only exists for snapshots above snapshot 1
        if snap >= 2 :
            
            # get the relevant SFRs for all subIDs in the snapshot
            with h5py.File(catalog_file, 'r') as hf :
                subIDs_in_snap = hf['Snapshot_{}/SubfindID'.format(snap)][:]
                SFRs_in_snap = hf['Snapshot_{}/SFR_MsunPerYrs_in_InRad_{}Myrs'.format(
                    snap, delta_t)][:]
            
            # check to see if the subID of interest has a value in the catalog
            exists = np.where(subIDs_in_snap == mpb_subID)[0]
            
            # if the subID has a value, append that value to the SFH
            if len(exists) > 0 :
                loc = exists[0]
                SFR = SFRs_in_snap[loc]
                SFH.append(SFR)
            
            # if the subID isn't in the catalog, append a NaN
            else :
                SFH.append(np.nan)
    
    return SFH
