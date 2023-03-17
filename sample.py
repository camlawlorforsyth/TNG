
import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table, join

from core import bsPath, get

def build_final_sample(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and file, and the output file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_subhalos_catalog.fits'.format(simName, snapNum)
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
    halos_infile = inDir + '/{}_{}_halos_catalog.fits'.format(simName, snapNum)
    subhalos_infile = inDir + '/{}_{}_subhalos_catalog.fits'.format(
        simName, snapNum)
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
    cluster_primary = (in_sample & in_cluster & is_central) # 1 BCG
    cluster_satellite = (in_sample & in_cluster & ~is_central) # 321 clus sats
    
    hm_group_primary = (in_sample & in_hm_group & is_central) # 7 bCGs
    hm_group_satellite = (in_sample & in_hm_group & ~is_central) # 791 hm grp sats
    
    lm_group_primary = (in_sample & in_lm_group & is_central) # 16 massive gals
    lm_group_satellite = (in_sample & in_lm_group & ~is_central) # 677 lm grp sats
    
    field_primary = (in_sample & ~in_cluster & ~in_hm_group & # 4668 field prims
                     ~in_lm_group & is_central)
    field_satellite = (in_sample & ~in_cluster & ~in_hm_group & # 1780 field sats
                       ~in_lm_group & ~is_central)
    
    # create a table for the environmental information
    env = Table([subhalos['SubhaloID'], cluster_primary, cluster_satellite,
                 hm_group_primary, hm_group_satellite, lm_group_primary,
                 lm_group_satellite, field_primary, field_satellite],
                names=('SubhaloID', 'clus_prim', 'clus_sat', 'hm_grp_prim',
                       'hm_grp_sat', 'lm_grp_prim', 'lm_grp_sat',
                       'field_prim', 'field_sat'))
    
    # mask the table to entries in the sample above, and write to file
    env = env[in_sample]
    env.write(outfile)
    
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
    
    
    
    
    return

def set_env(length, indices, GroupFirstSub, GroupNsubs) :
    
    in_env = np.zeros(length, dtype=bool)
    for index in indices :
        subhalo_index = GroupFirstSub[index]
        Nsubs = GroupNsubs[index]
        in_env[subhalo_index:subhalo_index+Nsubs] = True
    
    return in_env
