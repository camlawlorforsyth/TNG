
import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
import illustris_python as il

from core import bsPath, gcPath

solar_Z = 0.0127 # Wiersma+ 2009, MNRAS, 399, 574; used by TNG documentation

def convert_distance_units(distances) :
    return distances/cosmo.h

def convert_mass_units(masses) :
    return np.log10(masses*1e10/cosmo.h)

def convert_metallicity_units(metallicities) :
    return np.log10(metallicities/solar_Z)

def download_catalogs(simName, snapNum) :
    # https://www.tng-project.org/data/docs/scripts/
    
    groupsDir = gcPath(simName, snapNum)
    
    # download halo properties for halos in the group catalogs
    
    # choose relevant properties
    fields = ['GroupFirstSub',
              'GroupMass',
              'GroupMassType', # in full, mini, and subbox snaps?
              'GroupNsubs',
              'Group_M_Crit200',
              'Group_M_Crit500',
              'Group_R_Crit200',
              'Group_R_Crit500']
    
    # load the given information for all the halos
    halos = il.groupcat.loadHalos(bsPath(simName), snapNum, fields=fields)
    
    # convert the total mass units to solar masses and logify the values
    halos['GroupMass'] = convert_mass_units(halos['GroupMass'])
    
    # get the stellar mass content and convert to solar masses and logify
    halos['GroupMassStars'] = halos.pop('GroupMassType') # rename
    halos['GroupMassStars'] = get_stellar_mass(halos['GroupMassStars'])
    
    # convert the mass units to solar masses and logify
    halos['Group_M_Crit200'] = convert_mass_units(halos['Group_M_Crit200'])
    halos['Group_M_Crit500'] = convert_mass_units(halos['Group_M_Crit500'])
    
    # convert the distance units to comoving units
    halos['Group_R_Crit200'] = convert_distance_units(halos['Group_R_Crit200'])
    halos['Group_R_Crit500'] = convert_distance_units(halos['Group_R_Crit500'])
    
    # create group/halo IDs
    halos['GroupID'] = np.arange(halos['count'])
    
    # save information to outfile
    outfile = groupsDir + 'halos_catalog_{}_{}.fits'.format(simName, snapNum)
    names = ['GroupID', 'GroupFirstSub', 'GroupMass', 'GroupMassStars',
             'GroupNsubs', 'Group_M_Crit200', 'Group_M_Crit500',
             'Group_R_Crit200', 'Group_R_Crit500']
    table = Table([halos[name] for name in names], names=names)
    table.write(outfile)
    
    # download subhalo properties for subhalos (galaxies) in the group catalogs
    
    # choose relevant properties
    fields = ['SubhaloFlag',                   # exclude for SubHaloFlag == 0
              'SubhaloGrNr',
              'SubhaloParent',
              'SubhaloMass',                   # in full, mini, and subbox snaps
              'SubhaloMassType',               # in full, mini, and subbox snaps
              'SubhaloMassInHalfRad',          # in full, mini, and subbox snaps
              'SubhaloMassInHalfRadType',      # in full, mini, and subbox snaps
              'SubhaloMassInRad',              # in full, mini, and subbox snaps
              'SubhaloMassInRadType',          # in full, mini, and subbox snaps
              'SubhaloMassInMaxRad',           # in full, mini, and subbox snaps
              'SubhaloMassInMaxRadType',       # in full, mini, and subbox snaps
              'SubhaloHalfmassRad',
              'SubhaloHalfmassRadType',
              'SubhaloSFRinHalfRad',           # in full, mini, and subbox snaps
              'SubhaloSFRinRad',               # in full, mini, and subbox snaps
              'SubhaloSFRinMaxRad',            # in full, mini, and subbox snaps
              'SubhaloStarMetallicityHalfRad', # in full, mini, and subbox snaps
              'SubhaloStarMetallicity',        # in full, mini, and subbox snaps
              'SubhaloStarMetallicityMaxRad',  # in full, mini, and subbox snaps
              'SubhaloStellarPhotometrics',    # in full and subbox snaps
              'SubhaloVmaxRad']
    
    # load the given information for all the subhalos
    subhalos = il.groupcat.loadSubhalos(bsPath(simName), snapNum, fields=fields)
    
    # convert the total mass units to solar masses and logify the values
    subhalos['SubhaloMass'] = convert_mass_units(subhalos['SubhaloMass'])
    subhalos['SubhaloMassInHalfRad'] = convert_mass_units(
        subhalos['SubhaloMassInHalfRad'])
    subhalos['SubhaloMassInRad'] = convert_mass_units(
        subhalos['SubhaloMassInRad'])
    subhalos['SubhaloMassInMaxRad'] = convert_mass_units(
        subhalos['SubhaloMassInMaxRad'])
    
    # get the stellar mass content and convert to solar masses and logify
    subhalos['SubhaloMassStars'] = subhalos.pop('SubhaloMassType') # rename
    subhalos['SubhaloMassStars'] = get_stellar_mass(subhalos['SubhaloMassStars'])
    
    subhalos['SubhaloMassInHalfRadStars'] = subhalos.pop(
        'SubhaloMassInHalfRadType') # rename
    subhalos['SubhaloMassInHalfRadStars'] = get_stellar_mass(
        subhalos['SubhaloMassInHalfRadStars'])
    
    subhalos['SubhaloMassInRadStars'] = subhalos.pop(
        'SubhaloMassInRadType') # rename
    subhalos['SubhaloMassInRadStars'] = get_stellar_mass(
        subhalos['SubhaloMassInRadStars'])
    
    subhalos['SubhaloMassInMaxRadStars'] = subhalos.pop(
        'SubhaloMassInMaxRadType') # rename
    subhalos['SubhaloMassInMaxRadStars'] = get_stellar_mass(
        subhalos['SubhaloMassInMaxRadStars'])
    
    # convert the distance units to comoving units
    subhalos['SubhaloHalfmassRad'] = convert_distance_units(
        subhalos['SubhaloHalfmassRad'])
    subhalos['SubhaloVmaxRad'] = convert_distance_units(
        subhalos['SubhaloVmaxRad'])
    
    # get the stellar distances and concert to comoving units
    subhalos['SubhaloHalfmassRadStars'] = subhalos.pop(
        'SubhaloHalfmassRadType') # rename
    subhalos['SubhaloHalfmassRadStars'] = get_stellar_radius(
        subhalos['SubhaloHalfmassRadStars'])
    
    # convert the metallicity units to solar metallicities and logify
    subhalos['SubhaloStarMetallicityHalfRad'] = convert_metallicity_units(
        subhalos['SubhaloStarMetallicityHalfRad'])
    subhalos['SubhaloStarMetallicity'] = convert_metallicity_units(
        subhalos['SubhaloStarMetallicity'])
    subhalos['SubhaloStarMetallicityMaxRad'] = convert_metallicity_units(
        subhalos['SubhaloStarMetallicityMaxRad'])
    
    # create subhalo IDs
    subhalos['SubhaloID'] = np.arange(subhalos['count'])
    
    # save information to outfile
    outfile = groupsDir + 'subhalos_catalog_{}_{}.fits'.format(simName, snapNum)
    names = ['SubhaloID', 'SubhaloFlag', 'SubhaloGrNr', 'SubhaloParent',
             'SubhaloMass', 'SubhaloMassStars',
             'SubhaloMassInHalfRad', 'SubhaloMassInHalfRadStars',
             'SubhaloMassInRad', 'SubhaloMassInRadStars',
             'SubhaloMassInMaxRad','SubhaloMassInMaxRadStars',
             'SubhaloHalfmassRad', 'SubhaloHalfmassRadStars',
             'SubhaloSFRinHalfRad', 'SubhaloSFRinRad', 'SubhaloSFRinMaxRad',
             'SubhaloStarMetallicityHalfRad', 'SubhaloStarMetallicity',
             'SubhaloStarMetallicityMaxRad', 'SubhaloStellarPhotometrics',
             'SubhaloVmaxRad']
    table = Table([subhalos[name] for name in names], names=names)
    table.write(outfile)
    
    return

def get_stellar_mass(MassType) :
    ptNumStars = il.snapshot.partTypeNum('stars') # 4
    Mstar = np.array(MassType)[:, ptNumStars]
    return convert_mass_units(Mstar)

def get_stellar_radius(RadType) :
    ptNumStars = il.snapshot.partTypeNum('stars') # 4
    Rstar = np.array(RadType)[:, ptNumStars]
    return convert_distance_units(Rstar)
