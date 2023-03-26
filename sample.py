
from os.path import exists
import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table, join
import h5py
import requests

from core import add_dataset, bsPath, get, get_mpb_values,  mpbPath

def build_final_sample(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and file, and the output file
    inDir = bsPath(simName)
    infile = 'F:/{}/{}_{}_subhalos_catalog.fits'.format(
        simName, simName, snapNum)
    env_file = inDir + '/{}_{}_env.fits'.format(simName, snapNum)
    flags_file = inDir + '/{}_{}_primary-satellite-flags.fits'.format(
        simName, snapNum)
    outfile = inDir + '/{}_{}_sample.fits'.format(simName, snapNum)
    
    # read the constituent tables
    subhalos = Table.read(infile)
    subhalo_flags = Table.read(flags_file)
    subhalo_envs = Table.read(env_file)
    
    # join the subhalos and flags tables
    table = join(subhalos, subhalo_flags, keys=['SubhaloID'])
    
    # mask the resulting table based on the 'SubhaloFlag' for "real" galaxies
    table = table[table['SubhaloFlag'] == True]
    
    # join that table with the environment table
    final = join(table, subhalo_envs, keys='SubhaloID')
    
    # save the table
    final.write(outfile)
    
    return

def create_mask(length, groupIDs, halo_logM, haloMassMin, haloMassMax) :
    
    mask = np.zeros(length, dtype=bool)
    mass_range = (halo_logM >= haloMassMin) & (halo_logM < haloMassMax)
    
    indices = groupIDs[mass_range]
    mask[indices] = True
    
    return mask, indices

def determine_environment(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and files, and the output file
    inDir = bsPath(simName)
    halos_infile = 'F:/{}/{}_{}_halos_catalog.fits'.format(
        simName, simName, snapNum)
    subhalos_infile = 'F:/{}/{}_{}_subhalos_catalog.fits'.format(
        simName, simName, snapNum)
    outfile = inDir + '/{}_{}_env.fits'.format(simName, snapNum)
    
    # read catalogs
    halos = Table.read(halos_infile)
    subhalos = Table.read(subhalos_infile)
    
    # get the halo group IDs, masses, GroupFirstSubs, GroupNsubs, and lengths
    # for the halos and subhalos
    groupIDs, halo_logM = halos['GroupID'], halos['Group_M_Crit200']
    GroupFirstSub, GroupNsubs = halos['GroupFirstSub'], halos['GroupNsubs']
    halo_length = len(halos)
    subhalo_length = len(subhalos)
    
    # create a mask for the galaxies that will be in the sample (8261 here)
    in_sample = ((subhalos['SubhaloFlag'] == True) &
                 (subhalos['SubhaloMassStars'] >= 8.0))
    
    # create a mask for the central galaxies, based on unique indices from the
    # halo table and it's 'GroupFirstSub' column
    is_central = np.zeros(subhalo_length, dtype=bool)
    central_indices = np.unique(halos['GroupFirstSub'][halos['GroupFirstSub'] >= 0])
    is_central[central_indices] = True
    
    # create masks to label halos with various masses
    is_cluster, cluster_indices = create_mask(halo_length, groupIDs,
                                              halo_logM, 14.0, 15.0)
    is_hm_group, hm_group_indices = create_mask(halo_length, groupIDs,
                                                halo_logM, 13.5, 14.0)
    is_lm_group, lm_group_indices = create_mask(halo_length, groupIDs,
                                                halo_logM, 13.0, 13.5)
    
    # determine if subhalo is in given environment
    in_cluster = set_env(subhalo_length, cluster_indices, GroupFirstSub, GroupNsubs)
    in_hm_group = set_env(subhalo_length, hm_group_indices, GroupFirstSub, GroupNsubs)
    in_lm_group = set_env(subhalo_length, lm_group_indices, GroupFirstSub, GroupNsubs)
    
    # create masks for different populations in various environments
    cluster = (in_sample & in_cluster) # 1 BCG, 321 cluster satellites
    hm_group = (in_sample & in_hm_group) # 7 bCGs, 791 hm group satellites
    lm_group = (in_sample & in_lm_group) # 16 massive galaxies, 677 lm group satellites
    field = (in_sample & ~in_cluster & ~in_hm_group & ~in_lm_group)
    # 4667 field primaries, 1780 field satellites
    
    # create a table for the environmental information
    env = Table([subhalos['SubhaloID'], cluster, hm_group, lm_group, field],
                names=('SubhaloID', 'cluster', 'hm_group', 'lm_group', 'field'))
    
    # mask the table to entries in the sample above, and write to file
    env = env[in_sample]
    env.write(outfile)
    
    return

def download_all_mpbs(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and file, and the output directory and files
    inDir = bsPath(simName)
    mpbDir = mpbPath(simName, snapNum)
    infile = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    
    # get the subIDs of subhalos in the sample that we want SFHs for
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
    
    for subID in subIDs :
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

def primary_and_satellite_flags(simName='TNG50-1', snapNum=99, mass_min=8.0) :
    
    # define the output directory and file
    outDir = bsPath(simName)
    outfile = outDir + '/{}_{}_primary-satellite-flags.fits'.format(
        simName, snapNum)
    
    mass = np.power(10, mass_min)/1e10*cosmo.h
    
    url = ('http://www.tng-project.org/api/{}/snapshots/{}/subhalos/'.format(
        simName, snapNum) + '?mass_stars__gte={}&primary_flag='.format(mass))
    
    # get the central subhalo IDs
    central_subhalos = get(url + '1', params = {'limit':10000})
    central_ids = np.array([central_subhalos['results'][i]['id']
                            for i in range(central_subhalos['count'])])
    
    # get the satellite subhalo IDs
    satellite_subhalos = get(url + '0', params = {'limit':10000})
    satellite_ids = np.array([satellite_subhalos['results'][i]['id']
                              for i in range(satellite_subhalos['count'])])
    
    # get the total length (8806 here)
    length = len(central_ids) + len(satellite_ids)
    
    # set the primary flag at z = 0
    primary_flag = np.zeros(length, dtype=bool)
    primary_flag[:len(central_ids)] = True
    
    # create an array of all the subIDs
    subIDs = np.concatenate((central_ids, satellite_ids))
    
    # create the table and write it to file
    table = Table([subIDs, primary_flag], names=('SubhaloID', 'primary_flag'))
    table.write(outfile)
    
    return

def resave_as_hdf5(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and file, and the output file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample.fits'.format(simName, snapNum)
    redshift_file = inDir + '/snapshot_redshifts.fits'
    outfile = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    
    # open the table of subhalos in the sample that we want SFHs for
    subhalos = Table.read(infile)
    
    # get the redshifts and times (ages of the universe) for all the snapshots
    redshift_table = Table.read(redshift_file)
    redshifts = redshift_table['Redshift'].value
    times = cosmo.age(redshifts).value
    
    # check if the outfile exists, and if not, populate key information into it
    if not exists(outfile) :
        with h5py.File(outfile, 'w') as hf :
            # add information about the snapshot redshifts and times, and subIDs
            add_dataset(hf, np.arange(100), 'snapshots')
            add_dataset(hf, redshifts, 'redshifts')
            add_dataset(hf, times, 'times')
            add_dataset(hf, subhalos['SubhaloID'], 'SubhaloID')
            add_dataset(hf, subhalos['primary_flag'], 'primary_flag')
            add_dataset(hf, subhalos['cluster'], 'cluster')
            add_dataset(hf, subhalos['hm_group'], 'hm_group')
            add_dataset(hf, subhalos['lm_group'], 'lm_group')
            add_dataset(hf, subhalos['field'], 'field')
            
            # add empty arrays for various quantities to populate later
            add_dataset(hf, np.full((8260, 100), np.nan), 'subIDs')
            add_dataset(hf, np.full((8260, 100), np.nan), 'logM')
            add_dataset(hf, np.full((8260, 100), np.nan), 'Re')
            add_dataset(hf, np.full((8260, 100, 3), np.nan), 'centers')
            add_dataset(hf, np.full((8260, 100, 3), np.nan), 'UVK')
            add_dataset(hf, np.full((8260, 100), np.nan), 'SFH')
            add_dataset(hf, np.full((8260, 100), np.nan), 'SFMS')
            add_dataset(hf, np.full((8260, 100), np.nan), 'lo_SFH')
            add_dataset(hf, np.full((8260, 100), np.nan), 'hi_SFH')
            add_dataset(hf, np.full(8260, False), 'quenched')
            add_dataset(hf, np.full(8260, np.nan), 'onset_indices')
            add_dataset(hf, np.full(8260, np.nan), 'onset_times')
            add_dataset(hf, np.full(8260, np.nan), 'termination_indices')
            add_dataset(hf, np.full(8260, np.nan), 'termination_times')
    
    return

def save_mpb_values(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    
    # get the z = 0 subIDs for the galaxies in the sample
    with h5py.File(infile, 'r') as hf :
        IDs = hf['SubhaloID'][:]
    
    # loop over the galaxies in the sample, getting the relevant mpb information
    for i, subID in enumerate(IDs) :
        _, mpb_subIDs, logM, radii, centers, UVK = get_mpb_values(subID)
        
        # populate that information into the sample file
        with h5py.File(infile, 'a') as hf :
            hf['subIDs'][i, :] = mpb_subIDs
            hf['logM'][i, :] = logM
            hf['Re'][i, :] = radii
            hf['centers'][i, :, :] = centers
            hf['UVK'][i, :, :] = UVK
    
    return

def set_env(length, indices, GroupFirstSub, GroupNsubs) :
    
    in_env = np.zeros(length, dtype=bool)
    for index in indices :
        subhalo_index = GroupFirstSub[index]
        Nsubs = GroupNsubs[index]
        in_env[subhalo_index:subhalo_index+Nsubs] = True
    
    return in_env
