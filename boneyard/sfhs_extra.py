
from os.path import exists
import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
import h5py

from core import add_dataset, bsPath, get, mpbPath

def determine_all_histories(simName, snapNum) :
    
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
    
    # get the times for all the snapshots
    table = Table.read('output/snapshot_redshifts.fits')
    
    # we additionally need a first redshift and a final redshift in order to take
    # the difference to find the bin edges
    redshifts = np.concatenate(([np.inf], table['Redshift'].value, [0.0]))
    
    # define the times, edges (in units of scalefactor), and the time in bins
    times = cosmo.age(redshifts).value
    scalefactors = 1/(1 + redshifts)
    edges = scalefactors[:-1] + np.diff(scalefactors)/2
    time_in_bins = np.diff(cosmo.age(1/edges - 1)).value*1e9 # in yr
    
    # check if the outfile exists, and if not, populate key information into it
    if not exists(outfile) :
        with h5py.File(outfile, 'w') as hf :
            # add information about the snapshot redshifts
            add_dataset(hf, redshifts[1:-1], 'redshifts')
            
            # add information about the time bin centers and edges, and time
            # in bins
            add_dataset(hf, times[1:-1], 'times')
            add_dataset(hf, edges, 'edges') # in units of scalefactor
            add_dataset(hf, time_in_bins, 'time_in_bins')
            
            # add basic information from the table into the HDF5 file
            add_dataset(hf, subhalos['SubhaloID'], 'SubhaloID')
            add_dataset(hf, subhalos['SubhaloMassStars'], 'SubhaloMassStars')
            add_dataset(hf, subhalos['SubhaloSFRinRad'], 'SubhaloSFRinRad')
            add_dataset(hf, subhalos['SubhaloHalfmassRadStars'],
                        'SubhaloHalfmassRadStars')
            
            # add empty SFH information into the HDF5 file to populate later,
            # based on the number of galaxies and number of time points
            add_dataset(hf, np.full((len(subhalos), len(times[1:-1])), np.nan),
                        'SFH')
    
    # if the outfile exists, read the relevant information
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            subIDs = hf['SubhaloID'][:]
            x_sfhs = hf['SFH'][:]
    
    # now iterate over every subID in subIDs and get the SFH for that subID
    for i, subID in enumerate(subIDs) :
        
        # if the SFHs don't exist for the galaxy, populate the SFHs
        if np.all(np.isnan(x_sfhs[i, :])) :
            # determine the SFH for the galaxy, using edges in units of
            # scalefactor as cosmo.age is slow for very large arrays
            mass_formed_in_time_bin = history_from_cutout(simName, snapNum,
                                                          subID, edges)
            SFH = mass_formed_in_time_bin/time_in_bins
            
            # append those values into the outfile
            with h5py.File(outfile, 'a') as hf :
                hf['SFH'][i, :] = SFH
    
    return

def history_from_catalog(snapNums, mpb_subIDs, simName='TNG50-1', snapNum=99,
                         delta_t=100) :
    
    # define the input directory and file
    inDir = bsPath(simName)
    catalog_file = inDir + '/Donnari_Pillepich_star_formation_rates.hdf5'
    
    SFH = [np.nan, np.nan]
    for snap, mpb_subID in zip(snapNums, mpb_subIDs) :
        
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
    
    return np.array(SFH)

def history_from_cutout(simName, snapNum, subID, edges) :
    
    # define the cutout and mpb directories and the corresponding files
    cutoutDir = cutoutPath(simName, snapNum)
    mpbDir = mpbPath(simName, snapNum)
    
    cutout_file = cutoutDir + 'cutout_{}.hdf5'.format(subID)
    mpb_file = mpbDir + 'sublink_mpb_{}.hdf5'.format(subID)
    
    with h5py.File(mpb_file, 'r') as hf :
        # get the galaxy center coordinates and stellar halfmassradius
        center = hf['SubhaloPos'][0]
        radius = 2*hf['SubhaloHalfmassRadType'][:, 4][0] # in ckpc/h
    
    with h5py.File(cutout_file, 'r') as hf :
        # get the distances, formation ages (in units of scalefactor), and
        # initial masses of all the star particles
        dx = hf['PartType4']['Coordinates'][:, 0] - center[0]
        dy = hf['PartType4']['Coordinates'][:, 1] - center[1]
        dz = hf['PartType4']['Coordinates'][:, 2] - center[2]
        rs = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
        
        formation_ages = hf['PartType4']['GFM_StellarFormationTime'][:]
        initial_masses = hf['PartType4']['GFM_InitialMass'][:]    
    
    # if the radius is provided, only use star particles within that radius
    if radius :
        # mask out wind particles and constrain to radius
        mask = np.where((formation_ages > 0) & (rs <= radius))
    else :
        mask = np.where(formation_ages > 0) # mask out wind particles
    
    # mask out the wind particles and/or regions, and convert masses into
    # solar masses
    ages = formation_ages[mask]
    masses = initial_masses[mask]*1e10/cosmo.h
    
    # histogram the data to determine the total stellar content formed in each
    # time bin
    mass_formed_in_time_bin, _ = np.histogram(ages, bins=edges, weights=masses)
    
    return mass_formed_in_time_bin
