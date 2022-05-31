
import os
import numpy as np

from astropy.cosmology import Planck15 as cosmo
import h5py
import illustris_python as api
# import requests

from core import add_dataset, convert_mass_units, gcPath, get, get_stars

baseURL = 'http://www.tng-project.org/api/'

def get_cutout(cutoutsPath, groupName, sub, idn, overwrite=False) :
    
    # change this to check if cutout is already downloaded
    # parameters requested for each particle in the cutout
    pp = {'stars':
          'Coordinates,GFM_InitialMass,GFM_Metallicity,GFM_StellarFormationTime'}
    
    filename = cutoutsPath + groupName + 'cutout_{}.hdf5'.format(idn)
    
    if (not os.path.isfile(filename)) | (overwrite) :
        print(' getting {}'.format(filename) )
        # get and save HDF5 cutout file
        saved_filename = get(sub['meta']['url'] +'/'+ 'cutout.hdf5', pp)
        print('{} saved'.format(saved_filename))
    else :
        print('{} reading'.format(filename))
        saved_filename = filename
        if idn % 100 == 0 :
            print('Restoring previously saved file: {}'.format(saved_filename))
    
    return saved_filename

def get_history(sub, idn, lbtimes_edge, Nt, radius=None, overwrite=False, 
                redshift=0) :
    
    cutout_filename = get_cutout(sub, idn, overwrite)
    with h5py.File(cutout_filename, 'r') as hf:
        a_form = hf['PartType4']['GFM_StellarFormationTime'][:]
        
        if radius :
            dx = hf['PartType4']['Coordinates'][:, 0] - sub['pos_x']
            dy = hf['PartType4']['Coordinates'][:, 1] - sub['pos_y']
            dz = hf['PartType4']['Coordinates'][:, 2] - sub['pos_z']
            rr = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
            # remove wind particles and cut to radius
            ww = np.where((a_form > 0) & (rr < radius))
        else :
            ww = np.where(a_form > 0) # remove wind particles
        
        metals = hf['PartType4']['GFM_Metallicity'][:]
        metals = metals[ww]
        
        mass = hf['PartType4']['GFM_InitialMass'][:]
        mass = convert_mass_units(mass[ww])
        
        # scale factor to redshift
        z_form = 1.0/a_form[ww] - 1
        
        # redshift for lookback time
        lbt_form = (cosmo.lookback_time(z_form).value -
                    cosmo.lookback_time(redshift).value)
        
        # plt.hist(age, weights=mass,bins=sfh_t_boundary)
        sfh, xx = np.histogram(lbt_form, bins=lbtimes_edge, weights = mass)
        zz, xx = np.histogram(lbt_form, bins=lbtimes_edge,
                              weights = mass*metals)
        
        # if SFH < 0, Zh = 0
        zh = np.zeros(Nt)
        mask = sfh > 0
        zh[mask] = zz[mask] / sfh[mask]
        
        # print('mean metallicity is: '+str(np.mean(metals)))
    
    return sfh, zh

def get_subhalo(baseUrl, sim_name, redshift, idn) :
    return get(baseUrl + '{}/snapshots/z={}/subhalos/{}'.format(
        sim_name, redshift, idn))

def get_subhalos(baseUrl, sim_name, snap) :
    rr = get(baseUrl)
    
    # choose simulation and download info
    names = [sim['name'] for sim in rr['simulations']]
    isim = names.index(sim_name)
    sim = get(rr['simulations'][isim]['url'])
    
    # get snapshots
    snaps = get(sim['snapshots'])
    
    snapshot = get(snaps[snap]['url'])
    subs = get(snapshot['subhalos'])
    
    return subs

def get_subhalo_from_subs(subs, idn, overwrite=False) :
    return get(subs['results'][idn]['url'])

def get_subhalo_properties(sub, fields=[]) :
    data = {}
    for field in fields :
        data[field] = sub[field]
    return data

def get_histories(simName, snapNum, redshift, basePath, groupName, params_load) :
    
    groupsDir = gcPath(simName, snapNum)
    
    infile_selection = groupsDir + 'subIDs_in_selection.txt'
    outfile = groupsDir + 'sfhs_{}_selected.hdf5'.format(simName)
    
    # check if the outfile exists and has good SFHs
    if os.path.exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            if 'sfhs' in hf.keys() :
                if np.all(~np.isnan(hf['sfhs'])) :
                    print('File already exists with all non-NaN SFHs')
                    return
    
    if not os.path.exists(outfile) :
        fields = ['SubhaloMassType',
                  'SubhaloFlag',
                  'SubhaloParent',
                  'SubhaloSFRinRad',
                  'SubhaloHalfmassRadType']
        data_subhalos = api.groupcat.loadSubhalos(fields=fields, **params_load)
        
        data_subhalos['SubhaloMassType_stars_Msun'] = convert_mass_units(
            get_stars( data_subhalos['SubhaloMassType']))
        data_subhalos['logmass_stars'] = np.log10(
            data_subhalos['SubhaloMassType_stars_Msun'] )
        data_subhalos['SubhaloHalfmassRadType_stars'] = get_stars(
            data_subhalos['SubhaloHalfmassRadType'])
        
        # select galaxies for which we want SFHs
        SubID_in_selection = np.loadtxt(infile_selection).astype(int)
        
        # choose all
        NN = data_subhalos['count']
        SubID = np.arange(NN)
        sel = np.zeros(NN, dtype=bool)
        sel[SubID_in_selection] = True
        
        Ng = len(SubID_in_selection)
    
        # number of bins in lookback time
        Nt = 500
        
        # bin boundaries, in Gyr
        tuniv = cosmo.age(redshift).value
        lbtimes_edge = np.logspace(-4, np.log10(tuniv), num = Nt+1)
        lbtimes = lbtimes_edge[:-1] + np.diff(lbtimes_edge)/2.0
        
        print('Number of galaxies in selection: {}'.format(Ng))
        print('Number of time bins: {}'.format(Nt))
        
        print('Writing to ', outfile)
        
        with h5py.File(outfile, 'w') as hf :    
            # subID
            add_dataset(hf, 'subID', SubID)
            add_dataset(hf, 'subID_in_selection', SubID[sel])
            add_dataset(hf, 'SubhaloSFRinRad',
                        data_subhalos['SubhaloSFRinRad'][sel])
            add_dataset(hf, 'SubhaloFlag', data_subhalos['SubhaloFlag'][sel])
            add_dataset(hf, 'SubhaloMassType_stars_Msun',
                        data_subhalos['SubhaloMassType_stars_Msun'][sel])
            add_dataset(hf, 'SubhaloHalfmassRadType_stars',
                        data_subhalos['SubhaloHalfmassRadType_stars'][sel])
            
            add_dataset(hf, 'zh', np.full((Ng, Nt), np.nan))
            add_dataset(hf, 'sfh', np.full((Ng, Nt), np.nan))
            
            add_dataset(hf, 'primary_flag', np.full(Ng, np.nan))
            
            # lookback time bin centers and edges
            add_dataset(hf, 'Redshift', np.array([redshift]),
                        dtype=type(redshift))
            add_dataset(hf, 'lbtimes', lbtimes )
            add_dataset(hf, 'lbtimes_edge', lbtimes_edge)
        
        with h5py.File(outfile, 'r') as hf :
            x_sfhs = hf['sfh'][:]
            x_zhs = hf['zh'][:]
            subID_in_selection = hf['subID_in_selection'][:]
            subhalfmassRads = hf['SubhaloHalfmassRadType_stars'][:]
        
        # subs = get_subhalos(baseUrl, sim_name, snap)
        
        for i, subID in enumerate(subID_in_selection) :
            if np.all(np.isnan(x_sfhs[i, :])) :
                
                # sub = get_subhalo(subs, subID, overwrite=False)
                sub = get_subhalo(baseURL, simName, redshift, subID)
                
                out = get_subhalo_properties(sub, fields=['primary_flag'])
                primary_flag = out['primary_flag']
                
                sfh, zh = get_history(sub, subID, lbtimes_edge, Nt,
                                      radius=2*subhalfmassRads[i],
                                      overwrite=False, redshift=redshift)
                
                with h5py.File(outfile, 'a') as hf :
                    hf['sfh'][i, :] = sfh
                    hf['zh'][i, :] = zh
                    hf['primary_flag'][i] = primary_flag
        
        return
