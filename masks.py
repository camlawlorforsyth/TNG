
import os
import numpy as np

from astropy.cosmology import Planck15 as cosmo
import h5py
import pandas as pd
# import warnings
# warnings.filterwarnings('ignore')

# scale_factor = 1.0 / (1+redshift)

hh, solar_Z = cosmo.h, 0.0142 # MIST

def select(basePath, groupName, simName, snapshot) :
    
    # read catalogues
    outfile_halos = (basePath + groupName +
                     'properties_halos_{}_{}.h5'.format(simName, snapshot))
    outfile_subhalos = (basePath + groupName +
                        'properties_subhalos_{}_{}.h5'.format(simName, snapshot))
    
    with h5py.File(outfile_subhalos, 'r') as hf:
    #     print(hf.keys())
        data_subhalos = {}
        for k, v in hf.items():
            try :
                data_subhalos[k] = v[:]
            except :
                data_subhalos[k] = v
        data_subhalos['count'] = len(data_subhalos['subID'])
    
    with h5py.File(outfile_halos, 'r') as hf:
    #     print(hf.keys())
        data_halos = {}
        for k, v in hf.items():
            try :
                data_halos[k] = v[:]
            except :
                data_halos[k] = np.copy(v)
                pass
    
    # make dataframes
    # easier to query than dictionaries
    dict_subhalos = {}
    for k, v in data_subhalos.items():
        if type(v) == int: continue
        if v.ndim < 2:
            dict_subhalos[k] = v
    table_subhalos = pd.DataFrame(dict_subhalos )
    
    dict_halos = {}
    for k, v in data_halos.items():
        if type(v) == int: continue
        if v.ndim < 2:
            dict_halos[k] = v
    table_halos = pd.DataFrame(dict_halos )
    
    table_subhalos['logzsol'] = np.log10(
        table_subhalos['SubhaloStarMetallicity']/solar_Z)
    table_subhalos['log_SubhaloSFR'] = np.log10(table_subhalos['SubhaloSFR'])
    table_subhalos['log_SubhaloSFRinRad'] = np.log10(
        table_subhalos['SubhaloSFRinRad'])
    table_subhalos['log_sSubhaloSFRinRad'] = (np.log10(
        table_subhalos['SubhaloSFRinRad']) - table_subhalos['logmass_stars'])
    
    # select galaxies for which we want SFHs
    
    flag_ok = 1
    logmass_cut = (10.0, 11.8)
    
    # SFR based on star forming main sequence, fit to the data
    # quiescent is n_std*std less than main sequence
    coeffs = [0.94039153, 0.50658046]
    mx = 10.0
    std = 0.3
    n_std = 3
    sfr_cut = np.poly1d(
        coeffs)(table_subhalos['logmass_stars'] - mx) - n_std*std
    sfr_cut = np.power(10, sfr_cut)
    
    sel_q = ( logmass_cut[0] <= table_subhalos['logmass_stars'] ) &\
            ( logmass_cut[1] >= table_subhalos['logmass_stars'] ) &\
            ( table_subhalos['SubhaloSFRinRad'] <= sfr_cut ) &\
            ( table_subhalos['SubhaloFlag'] == flag_ok )
    
    table_subhalos['in_selection'] = np.zeros( len(table_subhalos) )
    table_subhalos.loc[ np.where(sel_q)[0], 'in_selection' ] = 1
    
    # select cluster vs satellite
    table_subhalos['is_central'] = np.zeros( len(table_subhalos) )
    
    # query Halo table for 'First Sub' which lists the SubHaloID of the group central
    # non-centrals have value -1, so skip the first unique entry
    idns_central = np.unique(
        table_halos.query('GroupFirstSub >= 0')['GroupFirstSub'].values)
    table_subhalos.loc[ idns_central, 'is_central'] = 1
    
    # select groups based on halo mass
    print('Number of halos:', len(table_halos))
    print()
    table_halos['is_cluster'] = np.zeros(len(table_halos))
    
    # select groups with M_Crit200_Msun > threshold
    M_Crit200_Msun_thresh = 10**(14.2)
    idns_cluster = table_halos.query(
        'Group_M_Crit200_Msun > {}'.format(M_Crit200_Msun_thresh)).index.values
    table_halos.loc[ idns_cluster, 'is_cluster'] = 1
    
    # label subhaloes in clusters
    table_subhalos['in_cluster'] = np.zeros(len(table_subhalos))
    for idn_cluster in idns_cluster:
        idn_subhalo = table_halos.loc[idn_cluster, 'GroupFirstSub']
        Nsubs = table_halos.loc[idn_cluster, 'GroupNsubs']
        table_subhalos.loc[idn_subhalo : idn_subhalo+Nsubs, 'in_cluster'] = 1
        print('Cluster {:4}, with log Group_M_Crit200_Msun = {:.3f}, has {:4} subhalos'.format(
            idn_cluster, table_halos.loc[idn_cluster, 'logMcrit'], Nsubs))
    
    print()
    print('Number of halos that are clusters: {}'.format(len(idns_cluster)))
    
    print('Number of subhalos: {}'.format(len(table_subhalos)))
    query = '(SubhaloFlag>0)'
    print('Number of subhalos, with good flag: {}'.format(
        len(table_subhalos.query(query))))
    query += ' & (is_central==1)'
    print('Number of subhalos, with good flag, is a central: {}'.format(
        len(table_subhalos.query(query))))
    query += ' & (in_selection==1)'
    print('Number of subhalos, with good flag, is a central, and in selection: {}'.format(
        len(table_subhalos.query(query))))
    
    query = '(SubhaloFlag>0)  & (is_central==0)'
    print('Number of subhalos, with good flag, that are satellites: {}'.format(
        len(table_subhalos.query(query))))
    query += ' & (in_cluster==1)'
    print('Number of subhalos, with good flag, that are satellites of a cluster: {}'.format(
        len(table_subhalos.query(query))))
    query += ' & (in_selection==1)'
    print('Number of subhalos, with good flag, that are satellites of a cluster, in selection: {}'.format(
        len(table_subhalos.query(query))))
    print()
    
    sel_base = '(in_selection==1)'
    sel_sat_in_cluster = sel_base +' & ' + '(is_central==0) & (in_cluster==1)'
    sel_field = sel_base + ' & ' + '(is_central==1)'
    
    label_sat_in_cluster = 'Quiescent satellite in a cluster'
    label_field = 'Quiescent field galaxy'
    
    # SFHs
    sfh_data_file = basePath + groupName + 'sfhs_{}_selected.hdf5'.format(simName)
    if not os.path.exists(sfh_data_file) :
        subID_selection_file = basePath + groupName + 'subIDs_in_selection.txt'
        
        print('Need to get sfhs for selected subIDs, writing list to {}:'.format(
            subID_selection_file))
        subIDs = table_subhalos.query(sel_sat_in_cluster)['subID'].values
        subIDs = np.append(subIDs, table_subhalos.query(sel_field)['subID'].values)
        np.savetxt(subID_selection_file, subIDs)
        print('python get_sfhs.py')
        print('Exiting...')
    
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
    # table_subhalos['has_sfh'] = np.zeros( len(table_subhalos) )
    # table_subhalos.loc[ subID_insel, 'has_sfh'] = 1
    #
    # table_subhalos['age'] = np.full( len(table_subhalos), np.nan )
    # table_subhalos.loc[ subID_insel, 'age'] = ages
    #
    # for tt in [0.50,0.70,0.90,0.95]:
    #     str_tt = 't{:.0f}'.format( tt*100 )
    #     table_subhalos[ str_tt ] = np.full( len(table_subhalos), np.nan )
    #     t_tts = lbtimes[ np.argmin(np.abs( x_csfr-tt ), axis=1) ]
    #     table_subhalos.loc[ subID_insel, str_tt ] = t_tts
    #
    # tuniv = cosmo.age(redshift).value
    # table_subhalos['tform'] = tuniv - table_subhalos['age'].values
    #
    # table_subhalos['cum_logsfh'] = np.full( len(table_subhalos), np.nan )
    # table_subhalos.loc[ subID_insel, 'cum_logsfh'] = cum_logsfh
    #
    # table_subhalos['sfh_index'] = np.full( len(table_subhalos), -1 )
    # table_subhalos.loc[ subID_insel, 'sfh_index'] = np.arange( len(subID_insel) ).astype(int)
    #
    # table_subhalos['primary_flag'] = np.full( len(table_subhalos), np.nan )
    # table_subhalos.loc[ subID_insel, 'primary_flag'] = primary_flag
    #
    #
    # print()
    # sel = table_subhalos.query('(has_sfh>0)').index.values
    # print('Number of subhalos with sfh: ', len(sel))
    #
    # sel_sat_in_cluster += ' & (has_sfh>0)'
    # sel_field          += ' & (has_sfh>0)'
    #
    # sel = table_subhalos.query( sel_sat_in_cluster ).index.values
    # print('  Number of subhalos with sfh in selection, satellites of clusters: ', len(sel))
    #
    # sel = table_subhalos.query( sel_field ).index.values
    # print('  Number of subhalos with sfh in selection, in the field: ', len(sel))
    
    return