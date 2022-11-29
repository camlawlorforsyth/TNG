
import numpy as np

from astropy.cosmology import Planck15 as cosmo, z_at_value
from astropy.table import Table
import astropy.units as u
import h5py
import requests
from scipy.interpolate import interp1d

solar_Z = 0.0127 # Wiersma+ 2009, MNRAS, 399, 574; used by TNG documentation

def add_dataset(h5file, data, label, dtype=None) :
    
    # set the datatype of the new data
    if dtype is None :
        dtype = data.dtype
    
   # add the new dataset to the file
    try : # try to add the dataset using h5py's method
        h5file.create_dataset(label, data=data, shape=np.shape(data), dtype=dtype)
    except : # but add it as a dictionary entry (essentially) if that doesn't work
        h5file[label] = data
    
    return

def bsPath(simName) :
    return '{}/output'.format(simName)

def convert_distance_units(distances) :
    return distances/cosmo.h

def convert_mass_units(masses) :
    return np.log10(masses*1e10/cosmo.h)

def convert_metallicity_units(metallicities) :
    return np.log10(metallicities/solar_Z)

def cutoutPath(simName, snapNum) :
    return bsPath(simName) + '/cutouts_{:3.0f}/'.format(snapNum).replace(' ', '0')

def gcPath(simName, snapNum) :
    return bsPath(simName) + '/groups_{:3.0f}/'.format(snapNum).replace(' ', '0')

def get(path, directory=None, params=None, filename=None) :
    # https://www.tng-project.org/data/docs/api/
    
    # make HTTP GET request to path
    headers = {'api-key':'0890bad45ac29c4fdd80a1ffc7d6d27b'}
    rr = requests.get(path, params=params, headers=headers)
    
    # raise exception if response code is not HTTP SUCCESS (200)
    rr.raise_for_status()
    
    if rr.headers['content-type'] == 'application/json' :
        return rr.json() # parse json responses automatically
    
    if 'content-disposition' in rr.headers :
        if not filename :
            filename = rr.headers['content-disposition'].split('filename=')[1]
        
        with open(directory + filename, 'wb') as ff :
            ff.write(rr.content)
        return filename # return the filename string
    
    return rr

def get_ages_and_scalefactors() :
    
    # look-up table for converting scalefactors to cosmological ages
    with h5py.File('output/scalefactor_to_Gyr.hdf5', 'r') as hf :
        scalefactor, age = hf['scalefactor'][:], hf['age'][:]
    
    return scalefactor, age

def get_mpb_radii_and_centers(simName, snapNum, subID) :
    
    # define the mpb directory and file
    mpbDir = mpbPath(simName, snapNum)
    mpbfile = mpbDir + 'sublink_mpb_{}.hdf5'.format(subID)
    
    # get the snapshot numbers, mpb subIDs, stellar halfmassradii, and galaxy
    # centers
    with h5py.File(mpbfile, 'r') as hf :
        snapNums = hf['SnapNum'][:]
        mpb_subIDs = hf['SubfindID'][:]
        halfmassradii = hf['SubhaloHalfmassRadType'][:, 4]
        centers = hf['SubhaloPos'][:]
    
    # given that the mpb files are arranged such that the most recent redshift
    # (z = 0) is at the beginning of arrays, we'll flip the arrays to conform
    # to an increasing-time convention
    snapNums = np.flip(snapNums)
    mpb_subIDs = np.flip(mpb_subIDs)
    radii = np.flip(halfmassradii)
    centers = np.flip(centers, axis=0)
    
    return snapNums, mpb_subIDs, radii, centers

def get_particle_positions(simName, snapNum, snap, subID, center) :
    
    # define the mpb cutouts directory and file
    mpbcutoutDir = mpbCutoutPath(simName, snapNum)
    cutout_file = mpbcutoutDir + 'cutout_{}_{}.hdf5'.format(snap, subID)
    
    try :
        with h5py.File(cutout_file) as hf :
            dx = hf['PartType4']['Coordinates'][:, 0] - center[0]
            dy = hf['PartType4']['Coordinates'][:, 1] - center[1]
            dz = hf['PartType4']['Coordinates'][:, 2] - center[2]
            
            # conert mass units
            masses = hf['PartType4']['GFM_InitialMass'][:]*1e10/cosmo.h
            
            # formation ages are in units of scalefactor
            formation_ages = hf['PartType4']['GFM_StellarFormationTime'][:]
        
        # limit particles to those that have positive formation times
        mask = (formation_ages >= 0)
        
        return formation_ages[mask], masses[mask], dx[mask], dy[mask], dz[mask]
    
    except KeyError :
        return None, None, None, None, None

def get_particles(simName, snapNum, snap, subID, center) :
    
    # define the mpb cutouts directory and file
    mpbcutoutDir = mpbCutoutPath(simName, snapNum)
    cutout_file = mpbcutoutDir + 'cutout_{}_{}.hdf5'.format(snap, subID)
    
    try :
        with h5py.File(cutout_file) as hf :
            dx = hf['PartType4']['Coordinates'][:, 0] - center[0]
            dy = hf['PartType4']['Coordinates'][:, 1] - center[1]
            dz = hf['PartType4']['Coordinates'][:, 2] - center[2]
            
            # conert mass units
            masses = hf['PartType4']['GFM_InitialMass'][:]*1e10/cosmo.h
            
            # formation ages are in units of scalefactor
            formation_ages = hf['PartType4']['GFM_StellarFormationTime'][:]
        
        # calculate the 3D distances from the galaxy center
        rs = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
        
        # limit particles to those that have positive formation times
        mask = (formation_ages >= 0)
        ages = formation_ages[mask]
        masses = masses[mask]
        rs = rs[mask]
        
        return ages, masses, rs
    
    except KeyError :
        return None, None, None

def get_sf_particle_positions(ages, masses, dx, dy, dz, time, delta_t=100*u.Myr) :
    
    # cosmo.age(redshift) is slow for very large arrays, so we'll work in units
    # of scalefactor and convert delta_t. t_minus_delta_t is in units of redshift
    t_minus_delta_t = z_at_value(cosmo.age, time*u.Gyr - delta_t, zmax=np.inf)
    limit = 1/(1 + t_minus_delta_t) # in units of scalefactor
    
    # limit particles to those that formed within the past delta_t time
    mask = (ages >= limit)
    
    return ages[mask], masses[mask], dx[mask], dy[mask], dz[mask]

def get_sf_particles(ages, masses, rs, time, delta_t=100*u.Myr) :
    
    # cosmo.age(redshift) is slow for very large arrays, so we'll work in units
    # of scalefactor and convert delta_t. t_minus_delta_t is in units of redshift
    t_minus_delta_t = z_at_value(cosmo.age, time*u.Gyr - delta_t, zmax=np.inf)
    limit = 1/(1 + t_minus_delta_t) # in units of scalefactor
    
    # limit particles to those that formed within the past delta_t time
    mask = (ages >= limit)
    
    return ages[mask], masses[mask], rs[mask]

def get_SFH_limits(limits_dic, edges, mass) :
    
    # find the index of the corresponding mass bin
    idx = np.where((mass >= edges[:-1]) & (mass <= edges[1:]))[0][0]
    subdic = limits_dic['mass_bin_{}'.format(idx)]
    
    return subdic['lo_SFH'], subdic['hi_SFH'] # return those limits

def mpbPath(simName, snapNum) :
    return bsPath(simName) + '/mpbs_{:3.0f}/'.format(snapNum).replace(' ', '0')

def mpbCutoutPath(simName, snapNum) :
    return bsPath(simName) + '/mpb_cutouts_{:3.0f}/'.format(snapNum).replace(' ', '0')

def offsetPath(simName) :
    return '{}/postprocessing/offsets'.format(simName)

def save_lookup_table():
    
    scalefactor = np.linspace(0, 1, 1000001) # limited to [0, 1]
    age = cosmo.age(1/scalefactor - 1).value
    
    # write to file
    with h5py.File('output/scalefactor_to_Gyr.hdf5', 'w') as hf :
        add_dataset(hf, scalefactor, 'scalefactor')
        add_dataset(hf, age, 'age')
    
    # table = Table([scalefactor, age], names=('scalefactor', 'age'))
    # table.write('output/scalefactor_to_Gyr.fits')
    
    return

def snapPath(simName, snapNum) :
    return bsPath(simName) + '/snapdir_{:3.0f}/'.format(snapNum).replace(' ', '0')

def snapshot_redshifts(simName) :
    
    snaps = get('http://www.tng-project.org/api/{}/snapshots/'.format(simName))
        
    snapNums, redshifts = [], []
    for snap in snaps :
        snapNums.append(snap['number'])
        redshifts.append(snap['redshift'])
    
    table = Table([snapNums, redshifts], names=('SnapNum', 'Redshift'))
    table.write('output/snapshot_redshifts.fits')
    
    return

def test_lookup_table() :
    
    with h5py.File('output/scalefactor_to_Gyr.hdf5', 'r') as hf :
        scalefactor = hf['scalefactor'][:]
        age = hf['age'][:]
    
    # define some random scalefactors
    random_scalefactors = np.random.rand(10)
    
    # try the scipy version
    scalefactor_to_Gyr = interp1d(scalefactor, age)
    scipy_ages = scalefactor_to_Gyr(random_scalefactors)
    
    # now try the numpy version
    numpy_ages = np.interp(random_scalefactors, scalefactor, age)
    
    return

def get_test_data(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and the input file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
    # get the masses, satellite and termination times for the quenched sample
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        tsats = hf['satellite_times'][:]
        tonsets = hf['onset_times'][:]
        tterms = hf['termination_times'][:]
    
    # for testing purposes
    # test_IDs = [96806]
    test_IDs = [
        # satellites
        14, 41, 63878, 167398, 184946, 220605, 324126,
        # primaries
        # 545003, 547545, 548151, 556699, 564498,
        # 592021, 604066, 606223, 607654, 623367
        ]
    
    locs = []
    for ID in test_IDs :
        locs.append(np.where(subIDs == ID)[0][0])
    
    mask = np.full(len(subIDs), False)
    for loc in locs :
        mask[loc] = True
    
    subIDs = subIDs[mask]
    tsats = tsats[mask]
    tonsets = tonsets[mask]
    tterms = tterms[mask]
    # end of for testing purposes
    
    return redshifts, times, subIDs, tsats, tonsets, tterms
