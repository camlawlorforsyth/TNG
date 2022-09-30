
from os.path import exists
import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
import h5py

from catalogs import convert_mass_units
from core import add_dataset, bsPath, cutoutPath, get

def determine_all_histories(simName, snapNum) :
    
    # define the output directory and the output file
    outDir = bsPath(simName)
    outfile = outDir + '/{}_{}_sample_SFHs.hdf5'.format(simName, snapNum)
    
    # check if the outfile exists and has good SFHs
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            if 'SFH' in hf.keys() :
                if np.all(~np.isnan(hf['SFH'])) :
                    print('File already exists with all non-NaN SFHs')
    
    # open the table of subhalos in the sample that we want SFHs for
    subhalos = Table.read(outDir + '/{}_{}_sample.fits'.format(simName, snapNum))
    
    # get the redshifts for all the snapshots
    table = Table.read('output/snapshot_redshifts.fits')
    redshifts = np.flip(table['Redshift'])
    
    # include an initial and final redshift
    redshifts = np.concatenate(([0.0], redshifts, [np.inf]))
    
    # define the lookbacktimes and their corresponding edges
    lookbacktimes = cosmo.lookback_time(redshifts).value
    lookbacktime_edges = lookbacktimes[:-1] + np.diff(lookbacktimes)/2
    
    # number of galaxies; number of bins in lookback time (which should be 100)
    Ngals, Ntimes = len(subhalos), len(lookbacktimes[1:-1])
    
    # check if the outfile exists, and if not, populate key information into it
    if not exists(outfile) :
        with h5py.File(outfile, 'w') as hf :
            # add basic information from the table into the HDF5 file
            add_dataset(hf, subhalos['SubhaloID'], 'SubhaloID')
            add_dataset(hf, subhalos['SubhaloMassStars'], 'SubhaloMassStars')
            add_dataset(hf, subhalos['SubhaloSFRinRad'], 'SubhaloSFRinRad')
            add_dataset(hf, subhalos['SubhaloHalfmassRadStars'],
                        'SubhaloHalfmassRadStars')
            
            # add information about the redshifts and most recent redshift
            add_dataset(hf, np.array([0.0]), 'last_redshift', dtype=float)
            add_dataset(hf, redshifts[1:-1], 'redshifts')
            
            # add information about the lookback time bin centers and edges
            add_dataset(hf, lookbacktimes[1:-1], 'lookbacktimes')
            add_dataset(hf, lookbacktime_edges, 'lookbacktime_edges')
            
            # add empty SFH information into the HDF5 file to populate later
            add_dataset(hf, np.full((Ngals, Ntimes), np.nan), 'SFH')
    
    # if the outfile exists, read the relevant information
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            subIDs = hf['SubhaloID'][:]
            x_sfhs = hf['SFH'][:]
    
    # now iterate over every subID in subIDs and get the SFH for that subID
    for i, subID in enumerate(subIDs) :
        
        # if the SFHs don't exist for the galaxy, populate the SFHs
        if np.all(np.isnan(x_sfhs[i, :])) :
            # determine the SFH for the galaxy
            SFH = history_from_cutout(lookbacktime_edges,
                outDir + '/cutouts_0{}/cutout_{}_masked.npz'.format(snapNum, subID),
                last_redshift=0.0)
            
            # append those values into the outfile
            with h5py.File(outfile, 'a') as hf :
                hf['SFH'][i, :] = SFH
    
    return

def download_all_cutouts(simName, snapNum) :
    
    cutoutDir = cutoutPath(simName, snapNum)
    outDir = bsPath(simName)
    
    # define the parameters that are requested for each particle in the cutout
    params = {'stars':
              'Coordinates,GFM_InitialMass,GFM_Metallicity,GFM_StellarFormationTime'}
    
    # open the table of subhalos in the sample that we want SFHs for
    subhalos = Table.read(outDir + '/{}_{}_sample.fits'.format(simName, snapNum))
    
    subIDs = subhalos['SubhaloID']
    halfMassRadii = subhalos['SubhaloHalfmassRadStars']
    
    for subID, R_e in zip(subIDs, halfMassRadii) :
        filename = cutoutDir + 'cutout_{}.hdf5'.format(subID)
        numpyfilename = cutoutDir + 'cutout_{}_masked.npz'.format(subID)
        subID_URL = 'http://www.tng-project.org/api/{}/snapshots/{}/subhalos/{}'.format(
            simName, snapNum, subID)
        
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
            download_all_cutouts_as_npz(filename, numpyfilename, sub, radius=2*R_e)
    
    return

def download_all_cutouts_as_npz(filename, numpyfilename, sub, radius=None) :
    
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

def history_from_cutout(lookbacktime_edges, numpyfilename, last_redshift=0.0) :
    
    # load information from saved numpy file
    numpyfile = np.load(numpyfilename)
    formation_ages = numpyfile['formation_ages']
    # metallicities = numpyfile['metallicities']
    initial_masses = numpyfile['initial_masses']
    
    # convert the formation_ages (in units of scalefactor) to redshifts
    formation_redshifts = 1.0/formation_ages - 1
    
    # determine the formation ages in terms of lookback times
    formation_lookbacktimes = (cosmo.lookback_time(formation_redshifts).value -
                               cosmo.lookback_time(last_redshift).value)
    
    # histogram the data to determine SFH(t) and Z(t), and note that the "SFR"
    # is really the total stellar content formed in each time bin
    SFR, _ = np.histogram(formation_lookbacktimes, bins=lookbacktime_edges,
                          weights=np.power(10, initial_masses))
    # zz, _ = np.histogram(formation_lookbacktimes, bins=lookbacktime_edges,
    #                      weights=np.power(10, initial_masses)*metallicities)
    
    # mask the metallicity history based on if the SFH is valid
    # zh, mask = np.zeros(Ntimes), sfh > 0
    # zh[mask] = zz[mask]/sfh[mask]
    
    time_in_bins = np.diff(lookbacktime_edges)*1e9 # in yr
    
    return SFR/time_in_bins
