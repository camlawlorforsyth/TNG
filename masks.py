
import numpy as np

from astropy.table import Table

from core import gcPath

solar_Z = 0.0127 # Wiersma+ 2009, MNRAS, 399, 574; used by TNG documentation

def select(simName, snapNum) :
    
    groupsDir = gcPath(simName, snapNum)
    
    # read catalogs
    halos_infile = groupsDir + 'halos_catalog_{}_{}.fits'.format(
        simName, snapNum)
    halos = Table.read(halos_infile)
    
    subhalos_infile = groupsDir + 'subhalos_catalog_{}_{}.fits'.format(
        simName, snapNum)
    subhalos = Table.read(subhalos_infile)
    
    # create a mask for the galaxies that we want to compute their SFHs for
    in_sel = ((subhalos['SubhaloFlag'] == 1) &
              (subhalos['SubhaloMassStars'] >= 8))
    subhalos['in_selection'] = in_sel
    
    # create a mask for the central galaxies, based on unique indices from the
    # halo table and it's 'GroupFirstSub' column
    subhalos['is_central'] = np.zeros(len(subhalos), dtype=bool)
    central_indices = np.unique(
        halos['GroupFirstSub'][halos['GroupFirstSub'] >= 0])
    subhalos['is_central'][central_indices] = True
    
    # create a mask for massive halos with M200 >= 13.8
    halos['is_cluster'] = np.zeros(len(halos), dtype=bool)
    cluster_indices = halos['GroupID'][halos['Group_M_Crit200'] >= 13.8]
    halos['is_cluster'][cluster_indices] = True
    
    # label subhalos in clusters
    subhalos['in_cluster'] = np.zeros(len(subhalos), dtype=bool)
    for cluster_index in cluster_indices :
        subhalo_index = halos['GroupFirstSub'][cluster_index]
        Nsubs = halos['GroupNsubs'][cluster_index]
        subhalos['in_cluster'][subhalo_index:subhalo_index+Nsubs] = True
    
    # create masks for different populations
    base = (subhalos['in_selection'] == True) # 8261 in selection
    BCG = (base & (subhalos['in_cluster'] == True) &
           (subhalos['is_central'] == True)) # 3 bCGs
    satellites = (base & (subhalos['in_cluster'] == True) &
                  (subhalos['is_central'] == False)) # 648 satellites
    field = (base & (subhalos['in_cluster'] == False) &
             (subhalos['is_central'] == True)) # 4689 field primaries
    
    # make columns based on those masks
    subhalos['BCG'] = BCG
    subhalos['satellites'] = satellites
    subhalos['field'] = field
    
    # write the good subhalo IDs to file, for downloading SFHs
    subhaloIDs = np.concatenate((subhalos['SubhaloID'][BCG].value,
                                 subhalos['SubhaloID'][satellites].value,
                                 subhalos['SubhaloID'][field].value))
    # np.savetxt(groupsDir + 'subhaloIDs_in_selection.txt', subhaloIDs, fmt='%d')
    
    # save the subhalos table to file, masked based on populations we care about
    subhalos = subhalos[BCG | satellites | field]
    subhalos.write(groupsDir + 'subhalos_catalog_{}_{}_sample.fits'.format(
        simName, snapNum))
    
    return

def transformations() :
    
    # from astropy.cosmology import Planck15 as cosmo
    # import h5py
    
    # sfh_data_file = groupsDir + 'sfhs_{}_selected.hdf5'.format(simName)
    # with h5py.File( sfh_data_file, 'r' ) as hf:
    #
    #     subID_insel = hf['subID_in_selection'][:]
    #
    #     x_masses = hf['sfh'][:]
    #     x_zhs  = hf['zh'][:]
    #
    #     lbtimes_edge = hf['lbtimes_edge'][:]
    #     lbtimes      = hf['lbtimes'][:]
    #
    #     primary_flag = hf['primary_flag'][:]
    #
    #
    # cum_logsfh = np.log10( np.sum( x_masses, axis=1 ) )
    #
    # # ====================
    #
    # import os, sys
    # root_path = '{}/HBM/'.format( os.getcwd().split('/HBM/')[0] )
    # if root_path not in sys.path: sys.path.append( root_path ) # hacky
    # way to do this, but haven't sorted out a better way yet
    #
    # hierarchical bayesian modeling == hbm?
    # from hbm_utils.transforms import sfr_to_mwa, zfrac_to_sfr, cumulate_masses
    #
    # lbtimes_edge_logyr = np.log10( lbtimes_edge )+9
    # agebins = np.array([ lbtimes_edge_logyr[:-1], lbtimes_edge_logyr[1:] ]).T
    #
    # x_sfrac = np.divide( x_masses.T, np.sum( x_masses, axis=1 )).T
    #
    # ages = sfr_to_mwa( agebins=agebins, sfrs=x_sfrac )
    #
    # x_ssfr = zfrac_to_sfr( total_mass=1, sfr_fraction=x_sfrac, agebins=agebins )
    # x_csfr = cumulate_masses( x_sfrac )
    #
    # # ====================
    #
    #
    # subhalos['has_sfh'] = np.zeros( len(subhalos) )
    # subhalos.loc[ subID_insel, 'has_sfh'] = 1
    #
    # subhalos['age'] = np.full( len(subhalos), np.nan )
    # subhalos.loc[ subID_insel, 'age'] = ages
    #
    # for tt in [0.50,0.70,0.90,0.95]:
    #     str_tt = 't{:.0f}'.format( tt*100 )
    #     subhalos[ str_tt ] = np.full( len(subhalos), np.nan )
    #     t_tts = lbtimes[ np.argmin(np.abs( x_csfr-tt ), axis=1) ]
    #     subhalos.loc[ subID_insel, str_tt ] = t_tts
    #
    # tuniv = cosmo.age(redshift).value
    # subhalos['tform'] = tuniv - subhalos['age'].values
    #
    # subhalos['cum_logsfh'] = np.full( len(subhalos), np.nan )
    # subhalos.loc[ subID_insel, 'cum_logsfh'] = cum_logsfh
    #
    # subhalos['sfh_index'] = np.full( len(subhalos), -1 )
    # subhalos.loc[ subID_insel, 'sfh_index'] = np.arange( len(subID_insel) ).astype(int)
    #
    # subhalos['primary_flag'] = np.full( len(subhalos), np.nan )
    # subhalos.loc[ subID_insel, 'primary_flag'] = primary_flag
    #
    #
    # print()
    # sel = subhalos.query('(has_sfh>0)').index.values
    # print('Number of subhalos with sfh: ', len(sel))
    #
    # sel_sat_in_cluster += ' & (has_sfh>0)'
    # sel_field          += ' & (has_sfh>0)'
    #
    # sel = subhalos.query( sel_sat_in_cluster ).index.values
    # print('  Number of subhalos with sfh in selection, satellites of clusters: ', len(sel))
    #
    # sel = subhalos.query( sel_field ).index.values
    # print('  Number of subhalos with sfh in selection, in the field: ', len(sel))
    
    return
