
import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table, join

from core import bsPath, gcPath, get

def build_final_sample(simName='TNG50-1', snapNum=99) :
    
    groupsDir = gcPath(simName, snapNum)
    outDir = bsPath(simName)
    
    # read the two constituent tables
    subhalos = Table.read(groupsDir + 'subhalos_catalog_{}_{}.fits'.format(
        simName, snapNum))
    
    subhalo_flags = Table.read(outDir + '{}_{}_primary-satellite-flagIDs.fits'.format(
        simName, snapNum))
    
    # join the tables
    table = join(subhalos, subhalo_flags, keys=['SubhaloID'])
    
    # mask the resulting table based on the 'SubhaloFlag' for "real" galaxies
    table = table[table['SubhaloFlag'] == True]
    
    # save the table
    table.write(outDir + '/{}_{}_sample.fits'.format(simName, snapNum))
    
    return

def primary_and_satellite_flags(simName='TNG50-1', snapNum=99, mass_min=8.0) :
    
    outDir = bsPath(simName)
    
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
    
    length = len(central_ids) + len(satellite_ids)
    
    # set the primary flag
    primary_flag = np.zeros(length, dtype=int)
    primary_flag[:len(central_ids)] = 1
    
    all_ids = np.concatenate((central_ids, satellite_ids))
    
    table = Table([all_ids, primary_flag], names=('SubhaloID', 'primary_flag'))
    
    table.write(outDir + '{}_{}_primary-satellite-flagIDs.fits'.format(
        simName, snapNum))
    
    return
