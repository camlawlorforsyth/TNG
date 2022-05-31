
import numpy as np

from astropy.table import Table
import h5py
import illustris_python as il

from core import add_dataset, bsPath, convert_mass_units, gcPath # , get_stars

import warnings
warnings.filterwarnings('ignore')

def download_catalogs(simName, snapNum) :
    # https://www.tng-project.org/data/docs/scripts/
    
    groupsDir = gcPath(simName, snapNum)
    
    # download halo properties for halos in the group catalogs
    '''
    # choose relevant properties
    fields = ['GroupFirstSub', 'GroupMass', 'GroupMassType', 'GroupNsubs',
              'Group_M_Crit200', 'Group_M_Crit500',
              'Group_R_Crit200', 'Group_R_Crit500']
    
    # load the given information for all the halos
    halos = il.groupcat.loadHalos(bsPath(simName), snapNum, fields=fields)
    
    # convert the mass units to solar masses
    halos['Group_M_Crit200_Msun'] = convert_mass_units(halos['Group_M_Crit200'])
    halos['Group_M_Crit500_Msun'] = convert_mass_units(halos['Group_M_Crit500'])
    
    # logify those solar masses
    halos['logMcrit200'] = np.log10(halos['Group_M_Crit200_Msun'])
    halos['logMcrit500'] = np.log10(halos['Group_M_Crit500_Msun'])
    
    # create halo IDs
    halos['haloID'] = np.arange(halos['count'])
    
    # save information to outfile
    outfile = groupsDir + 'halos_catalog_{}_{}.fits'.format(simName, snapNum)
    names = list(halos.keys())
    names = [names[-1]] + names[1:-1]
    table = Table([halos[name] for name in names], names=names)
    table.write(outfile)
    '''
    
    
    # download subhalo properties for subhalos (galaxies) in the group catalogs
    
    # choose relevant properties
    fields = ['SubhaloFlag', 'SubhaloGrNr', 'SubhaloParent',
              'SubhaloMass', 'SubhaloMassType', 
              
              'SubhaloHalfmassRad', 'SubhaloHalfmassRadType',
              
              
              'SubhaloSFRinRad',
              
              
              ]
    
    test = il.snapshot.snapPath(bsPath(simName), snapNum)
    print(test)
    
    from core import snapPath
    temp = snapPath(simName, snapNum)
    print(temp)
    
    
    max_num_snapshots = 100
    redshifts = np.zeros(max_num_snapshots, dtype='float32' )
    redshifts.fill(np.nan)
    
    for i in range(max_num_snapshots):
        h = load_snapshot_header(basePath, i)
        redshifts[i] = h['Redshift']
    
    with h5py.File('redshifts_%s.hdf5' % simname,'w') as f:
        f['redshifts'] = redshifts
    
    # load the given information for all the subhalos
    # subhalos = il.groupcat.loadSubhalos(bsPath(simName), snapNum, fields=fields)
    
    # convert the mass units to solar masses
    # subhalos['SubhaloMassType_stars_Msun'] = convert_mass_units(get_stars(subhalos['SubhaloMassType']))
    
    
    # subhalos['logmass_stars'] = np.log10(subhalos['SubhaloMassType_stars_Msun'])
    # subhalos['SubhaloHalfmassRadType_stars'] = get_stars(subhalos['SubhaloHalfmassRadType'])
    
    # subhalos = groupsDir + 'properties_subhalos_{}_{}.h5'.format(simName, snapNum)
    
    '''
    # print('Writing to ', outfile_subhalos)
    with h5py.File(outfile_subhalos, 'w') as hf:
        
        hf['Redshift'] = redshift
        add_dataset(hf, 'subID', np.arange(data_subhalos['count']))
        
        for kk, vv in data_subhalos.items() :
            if kk == 'count':
                add_dataset(hf, kk, vv)
    '''
    
    
    
    return

download_catalogs('TNG50-1', 99)
