
from os.path import exists
import numpy as np

from astropy.cosmology import FlatLambdaCDM, z_at_value
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
    return '{}'.format(simName)

def convert_distance_units(distances) :
    return distances/cosmo().h

def convert_mass_units(masses) :
    return np.log10(masses*1e10/cosmo().h)

def convert_metallicity_units(metallicities) :
    return np.log10(metallicities/solar_Z)

def cosmo() :
    return FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

def cutoutPath(simName, snapNum) :
    return bsPath(simName) + '/cutouts_{:3.0f}/'.format(snapNum).replace(' ', '0')

def determine_mass_bin_indices(masses, mass, hw=0.1, minNum=50) :
    
    mass_bin_mask = (masses >= mass - hw) & (masses <= mass + hw)
    # print(mass_bin_mask)
    
    if np.sum(mass_bin_mask) >= minNum :
        return mass_bin_mask
    else :
        return determine_mass_bin_indices(masses, mass, hw=hw+0.005, minNum=minNum)

def find_nearest(times, index_times) :
    
    indices = []
    for time in index_times :
        index = (np.abs(times - time)).argmin()
        indices.append(index)
    
    return indices

def gcPath(simName='TNG50-1', snapNum=99) :
    return 'F:/{}/groups_{:3.0f}/'.format(simName, snapNum).replace(' ', '0')

def get(path, directory=None, params=None, filename=None) :
    # https://www.tng-project.org/data/docs/api/
    
    try :
        # make HTTP GET request to path
        headers = {'api-key':'0890bad45ac29c4fdd80a1ffc7d6d27b'}
        rr = requests.get(path, params=params, headers=headers)
        
        # raise exception if response code is not HTTP SUCCESS (200)
        # rr.raise_for_status()
        if rr.status_code == 200 :
        
            if rr.headers['content-type'] == 'application/json' :
                return rr.json() # parse json responses automatically
            
            if 'content-disposition' in rr.headers :
                if not filename :
                    filename = rr.headers['content-disposition'].split('filename=')[1]
                
                with open(directory + filename, 'wb') as ff :
                    ff.write(rr.content)
                return filename # return the filename string
            
            return rr
        else :
            print('Error: {} for file {}'.format(rr.status_code, filename))
    except Exception as error :
        print('Exception: {}'.format(error))

def get_ages_and_scalefactors() :
    
    # look-up table for converting scalefactors to cosmological ages
    with h5py.File('output/scalefactor_to_Gyr.hdf5', 'r') as hf :
        scalefactor, age = hf['scalefactor'][:], hf['age'][:]
    
    return scalefactor, age

def get_late_data(basic=True) :
    
    # get the morphological metrics and the quenching episode progresses
    data = Table.read('TNG50-1/morphological_metrics_-10.5_+-1.fits')
    episode_progress = data['episode_progress'].value
    
    # mask data to the epoch of interest
    table = data.copy()
    table = table[episode_progress >= 0.75]
    
    # get parameters of interest
    late_quenched = table['quenched_status'].value
    late_sf = table['sf_status'].value
    late_below = table['below_sfms_status'].value
    late_mech = table['mechanism'].value
    late_logM = table['logM'].value
    
    # exclude redshift and stellar mass information, and mask to epoch of interest
    late_data = np.array([table['C_SF'].value, np.log10(table['R_SF'].value),
                          table['Rinner'].value, table['Router'].value]).T
    
    # mask NaN data and -99 values for Rinner and Router
    good = ((np.sum(np.isnan(late_data), axis=1) == 0) &
            (late_data[:, 2] >= 0.0) & (late_data[:, 3] >= 0.0))
    late_quenched = late_quenched[good]
    late_sf = late_sf[good]
    late_below = late_below[good]
    late_data = late_data[good]
    late_mech = late_mech[good]
    late_logM = late_logM[good]
    
    late_quenched_mech = late_mech[late_mech > 0]
    late_sf_mech = late_mech[late_mech == 0]
    late_below_mech = late_mech[late_mech == -1]
    
    late_quenched_data = late_data[late_mech > 0]
    late_sf_data = late_data[late_mech == 0]
    late_below_data = late_data[late_mech == -1]
    late_io_data = late_data[late_mech == 1]
    late_oi_data = late_data[late_mech == 3]
    late_am_data = late_data[late_mech == 5]
    
    late_quenched_logM = late_logM[late_mech > 0]
    late_sf_logM = late_logM[late_mech == 0]
    late_below_logM = late_logM[late_mech == -1]
    
    if basic :
        return (late_quenched_mech, late_sf_mech, late_quenched_data,
                late_sf_data, late_io_data, late_oi_data, late_am_data)
    else :
        return (late_quenched_mech, late_sf_mech, late_quenched_data,
                late_sf_data, late_io_data, late_oi_data, late_am_data,
                late_quenched_logM, late_sf_logM, late_below_mech,
                late_below_data, late_below_logM)

def get_mpb_values(subID, simName='TNG50-1', snapNum=99, pad=True) :
    
    # define the mpb directory and file
    mpbDir = mpbPath(simName, snapNum)
    mpbfile = mpbDir + 'sublink_mpb_{}.hdf5'.format(subID)
    
    try :
        # get the snapshot numbers, mpb subIDs, stellar masses, stellar
        # halfmassradii, and galaxy centers
        
        # given that the mpb files are arranged such that the most recent redshift
        # (z = 0) is at the beginning of arrays, we'll flip the arrays to conform
        # to an increasing-time convention
        with h5py.File(mpbfile, 'r') as hf :
            snapNums = np.flip(hf['SnapNum'][:])
            mpb_subIDs = np.flip(hf['SubfindID'][:])
            logM = np.flip(hf['SubhaloMassType'][:, 4])
            radii = np.flip(hf['SubhaloHalfmassRadType'][:, 4])
            centers = np.flip(hf['SubhaloPos'][:], axis=0)
            UBVK = np.flip(hf['SubhaloStellarPhotometrics'][:, 0:4], axis=0)
            UVK = np.delete(UBVK, 1, axis=1)
            
            # at snapshot 0, some masses are 0, which logarithms don't like
            logM[logM == 0] = 1e-10*cosmo().h
        
        # pad the arrays to all have 100 entries
        if pad and (len(snapNums) < 100) :
            snapNums_padded = np.full(100, np.nan)
            snapNums_padded[100 - len(snapNums):] = snapNums
            snapNums = snapNums_padded
            
            mpb_subIDs_padded = np.full(100, np.nan)
            mpb_subIDs_padded[100 - len(mpb_subIDs):] = mpb_subIDs
            mpb_subIDs = mpb_subIDs_padded
            
            logM_padded = np.full(100, np.nan)
            logM_padded[100 - len(logM):] = logM
            logM = logM_padded
            
            radii_padded = np.full(100, np.nan)
            radii_padded[100 - len(radii):] = radii
            radii = radii_padded
            
            centers_padded = np.full((100, 3), np.nan)
            centers_padded[100 - len(centers):] = centers
            centers = centers_padded
            
            UVK_padded = np.full((100, 3), np.nan)
            UVK_padded[100 - len(UVK):] = UVK
            UVK = UVK_padded
        
        return (snapNums.astype(int), mpb_subIDs.astype(int),
                convert_mass_units(logM), radii, centers, UVK)
    
    except KeyError :
        null = np.full(100, np.nan)
        null3d = np.full((100, 3), np.nan)
        return np.arange(100), null, null, null, null3d, null3d

def get_particle_positions(simName, snapNum, snap, subID, center) :
    
    # define the mpb cutouts directory and file
    mpbcutoutDir = mpbCutoutPath(simName, snapNum)
    cutout_file = mpbcutoutDir + 'cutout_{}_{}.hdf5'.format(snap, subID)
    
    try :
        with h5py.File(cutout_file) as hf :
            dx = hf['PartType4']['Coordinates'][:, 0] - center[0]
            dy = hf['PartType4']['Coordinates'][:, 1] - center[1]
            dz = hf['PartType4']['Coordinates'][:, 2] - center[2]
            
            # convert mass units
            masses = hf['PartType4']['GFM_InitialMass'][:]*1e10/cosmo().h
            
            # formation ages are in units of scalefactor
            formation_ages = hf['PartType4']['GFM_StellarFormationTime'][:]
        
        # limit particles to those that have positive formation times
        mask = (formation_ages > 0)
        
        return formation_ages[mask], masses[mask], dx[mask], dy[mask], dz[mask]
    
    except KeyError :
        return None, None, None, None, None

def get_particles(simName, snapNum, snap, subID, center, version='2D') :
    
    # define the mpb cutouts directory and file
    mpbcutoutDir = mpbCutoutPath(simName, snapNum)
    cutout_file = mpbcutoutDir + 'cutout_{}_{}.hdf5'.format(snap, subID)
    
    try :
        with h5py.File(cutout_file) as hf :
            dx = hf['PartType4']['Coordinates'][:, 0] - center[0]
            dy = hf['PartType4']['Coordinates'][:, 1] - center[1]
            dz = hf['PartType4']['Coordinates'][:, 2] - center[2]
            
            # convert mass units
            masses = hf['PartType4']['GFM_InitialMass'][:]*1e10/cosmo().h
            
            # formation ages are in units of scalefactor
            formation_ages = hf['PartType4']['GFM_StellarFormationTime'][:]
        
        # calculate the 3D distances from the galaxy center
        rs = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
        
        # limit particles to those that have positive formation times
        mask = (formation_ages > 0)
        
        return formation_ages[mask], masses[mask], rs[mask]
    
    except (KeyError, OSError) :
        return None, None, None

def get_quenched_data(simName='TNG50-1', snapNum=99) :
    
    # define the input files
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    
    # get relevant information for the general sample, and quenched systems
    with h5py.File(infile, 'r') as hf :
        snapshots = hf['snapshots'][:]
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        subIDfinals = hf['SubhaloID'][:].astype(int)
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        Re = hf['Re'][:]
        centers = hf['centers'][:]
        UVK = hf['UVK'][:]
        primary = hf['primary_flag'][:]
        cluster = hf['cluster'][:]
        hm_group = hf['hm_group'][:]
        lm_group = hf['lm_group'][:]
        field = hf['field'][:]
        SFHs = hf['SFH'][:]
        SFMS = hf['SFMS'][:].astype(bool) # (boolean) SFMS at each snapshot
        lo_SFH = hf['lo_SFH'][:]
        hi_SFH = hf['hi_SFH'][:]
        quenched = hf['quenched'][:]
        ionsets = hf['onset_indices'][:].astype(int)
        tonsets = hf['onset_times'][:]
        iterms = hf['termination_indices'][:].astype(int)
        tterms = hf['termination_times'][:]
    
    return (snapshots, redshifts, times, subIDs, logM, Re, centers, UVK, SFHs,
            SFMS, subIDfinals[quenched], subIDs[quenched], logM[quenched],
            SFHs[quenched], Re[quenched], centers[quenched], UVK[quenched],
            primary[quenched], cluster[quenched], hm_group[quenched],
            lm_group[quenched], field[quenched], lo_SFH[quenched],
            hi_SFH[quenched], ionsets[quenched], tonsets[quenched],
            iterms[quenched], tterms[quenched])

def get_rotation_input(simName, snapNum, snap, subID) :
    
    # define the mpb cutouts directory and file
    mpbcutoutDir = mpbCutoutPath(simName, snapNum)
    cutout_file = mpbcutoutDir + 'cutout_{}_{}.hdf5'.format(snap, subID)
    
    try :
        with h5py.File(cutout_file, 'r') as hf :
            gas_coords = hf['PartType0']['Coordinates'][:]
            gas_sfrs = hf['PartType0']['StarFormationRate'][:]
            
            # convert mass units
            gas_masses = hf['PartType0']['Masses'][:]*1e10/cosmo().h
    except KeyError :
        gas_masses, gas_sfrs, gas_coords = None, None, None
    
    try :
        with h5py.File(cutout_file, 'r') as hf :
            star_coords = hf['PartType4']['Coordinates'][:]
            star_ages = hf['PartType4']['GFM_StellarFormationTime'][:]
            
            # convert mass units
            star_gfm = hf['PartType4']['GFM_InitialMass'][:]*1e10/cosmo().h
            star_masses = hf['PartType4']['Masses'][:]*1e10/cosmo().h
        
        # limit particles to those that have positive formation times
        mask = (star_ages > 0)
        
        star_gfm, star_masses = star_gfm[mask], star_masses[mask]
        star_coords, star_ages = star_coords[mask], star_ages[mask]
    except KeyError :
        star_ages, star_gfm, star_masses, star_coords = None, None, None, None
    
    return (gas_masses, gas_sfrs, gas_coords, star_ages, star_gfm, star_masses,
            star_coords)

def get_sf_particle_positions(ages, masses, dx, dy, dz, time, delta_t=100*u.Myr) :
    
    # cosmo().age(redshift) is slow for very large arrays, so we'll work in units
    # of scalefactor and convert delta_t. t_minus_delta_t is in units of redshift
    t_minus_delta_t = z_at_value(cosmo().age, time*u.Gyr - delta_t, zmax=np.inf)
    limit = 1/(1 + t_minus_delta_t) # in units of scalefactor
    
    # limit particles to those that formed within the past delta_t time
    mask = (ages >= limit)
    
    return ages[mask], masses[mask], dx[mask], dy[mask], dz[mask]

def get_sf_particles(ages, masses, rs, time, delta_t=100*u.Myr) :
    
    # cosmo().age(redshift) is slow for very large arrays, so we'll work in units
    # of scalefactor and convert delta_t. t_minus_delta_t is in units of redshift
    t_minus_delta_t = z_at_value(cosmo().age, time*u.Gyr - delta_t, zmax=np.inf)
    limit = 1/(1 + t_minus_delta_t) # in units of scalefactor
    
    # limit particles to those that formed within the past delta_t time
    mask = (ages >= limit)
    
    return ages[mask], masses[mask], rs[mask]

def get_SFH_limits(limits_dic, edges, mass) :
    
    # find the index of the corresponding mass bin
    idx = np.where((mass >= edges[:-1]) & (mass <= edges[1:]))[0][0]
    subdic = limits_dic['mass_bin_{}'.format(idx)]
    
    return subdic['lo_SFH'], subdic['hi_SFH'] # return those limits

def mpbPath(simName='TNG50-1', snapNum=99) :
    return 'F:/{}/mpbs_{:3.0f}/'.format(simName, snapNum).replace(' ', '0')

def mpbCutoutPath(simName='TNG50-1', snapNum=99) :
    return 'F:/{}/mpb_cutouts_{:3.0f}/'.format(simName, snapNum).replace(' ', '0')

def offsetPath(simName='TNG50-1') :
    return '{}/'.format(simName)

def save_lookup_table(simName='TNG50-1'):
    
    outfile_fits = '{}/scalefactor_to_Gyr.fits'.format(simName)
    outfile_hdf5 = '{}/scalefactor_to_Gyr.hdf5'.format(simName)
    
    if (not exists(outfile_fits)) and (not exists(outfile_hdf5)) :
        
        scalefactor = np.linspace(0, 1, 1000001) # limited to [0, 1]
        age = cosmo().age(1/scalefactor - 1).value
        
        # write to fits file
        table = Table([scalefactor, age], names=('scalefactor', 'age'))
        table.write(outfile_fits)
        
        # write to hdf5 file
        with h5py.File(outfile_hdf5, 'w') as hf :
            add_dataset(hf, scalefactor, 'scalefactor')
            add_dataset(hf, age, 'age')
    
    return

def snapPath(simName, snapNum) :
    return bsPath(simName) + '/snapdir_{:3.0f}/'.format(snapNum).replace(' ', '0')

def snapshot_redshifts(simName='TNG50-1') :
    
    outfile = '{}/snapshot_redshifts.fits'.format(simName)
    
    if not exists(outfile) :
        snaps = get('http://www.tng-project.org/api/{}/snapshots/'.format(simName))
        
        snapNums, redshifts = [], []
        for snap in snaps :
            snapNums.append(snap['number'])
            redshifts.append(snap['redshift'])
        
        table = Table([snapNums, redshifts], names=('SnapNum', 'Redshift'))
        table.write(outfile)
    
    return

def surface(aa, bb, cc, dd, xs, ys) :
    # given a plane of the form aa*xx + bb*yy + cc*zz + dd = 0, solve for zz
    surf = lambda xx, yy : (-dd - aa*xx - bb*yy)/cc
    return surf(xs, ys)

def test_lookup_table() :
    
    with h5py.File('TNG50-1/scalefactor_to_Gyr.hdf5', 'r') as hf :
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

def vertex(xx, aa, hh, kk) :
    return aa*np.square(xx - hh) + kk
