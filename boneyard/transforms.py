
import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
import h5py

def transformations() :
    
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
