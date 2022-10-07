
from os.path import exists
import numpy as np

from astropy.cosmology import Planck15 as cosmo, z_at_value
import astropy.units as u
import h5py
from scipy.signal import savgol_filter

from catalogs import convert_distance_units
from core import bsPath, get, mpbPath, mpbCutoutPath
import plotting as plt

def get_mpb_halfmassradii(simName, snapNum, subID) :
    
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
    radii = np.flip(convert_distance_units(halfmassradii))
    centers = np.flip(centers, axis=0)
    
    return snapNums, mpb_subIDs, radii, centers

def download_mpb_cutouts(simName, snapNum, subID) :
    
    # define the output directory for the mpb cutouts
    outDir = mpbCutoutPath(simName, snapNum)
    
    # define the parameters that are requested for each particle in the cutout
    params = {'stars':'Coordinates,GFM_InitialMass,GFM_StellarFormationTime'}
    
    url = 'https://www.tng-project.org/api/{}/snapshots/{}/subhalos/{}'.format(
        simName, snapNum, subID)
    
    # retrieve information about the galaxy at the redshift of interest
    sub = get(url)
    
    # save the cutout file into the output directory
    filename = 'cutout_{}_{}.hdf5'.format(sub['snap'], sub['id'])
    get(sub['meta']['url'] + '/cutout.hdf5', directory=outDir, params=params,
        filename=filename)
    
    while sub['prog_sfid'] != -1 :
        # request the full subhalo details of the progenitor by following the sublink URL
        sub = get(sub['related']['sublink_progenitor'])
        
        # check if the cutout file exists
        filename = 'cutout_{}_{}.hdf5'.format(sub['snap'], sub['id'])
        if not exists(outDir + filename) :
            # save the cutout file into the output directory
            get(sub['meta']['url'] + '/cutout.hdf5', directory=outDir,
                params=params, filename=filename)
    
    return

def compute_age_gradient(rs, ages) :
    
    if (len(ages) == 0) and (len(rs) == 0) :
        gradient = np.nan
    else :
        gradient, intercept = np.polyfit(rs, ages, 1)
    
    return gradient

def compute_sf_rms(rs) :
    
    if len(rs) == 0 :
        rms = 0.0
    else :
        rms = np.sqrt(np.mean(np.square(rs)))
    
    return rms

def get_particles(simName, snapNum, snap, subID, center, time, delta_t=100*u.Myr) :
    
    # define the mpb cutouts directory and file
    mpbcutoutDir = mpbCutoutPath(simName, snapNum)
    cutout_file = mpbcutoutDir + 'cutout_{}_{}.hdf5'.format(snap, subID)
    
    # cosmo.age(redshift) is slow for very large arrays, so we'll work in units
    # of scalefactor and convert delta_t. t_minus_delta_t is in units of redshift
    t_minus_delta_t = z_at_value(cosmo.age, time*u.Gyr - delta_t, zmax=np.inf)
    limit = 1/(1 + t_minus_delta_t) # in units of scalefactor
    
    try :
        with h5py.File(cutout_file) as hf :
            dx = hf['PartType4']['Coordinates'][:, 0] - center[0]
            dy = hf['PartType4']['Coordinates'][:, 1] - center[1]
            dz = hf['PartType4']['Coordinates'][:, 2] - center[2]
            
            # conert mass units
            masses = hf['PartType4']['GFM_InitialMass'][:]*1e10/cosmo.h
            
            # formation ages are in units of scalefactor
            formation_ages = hf['PartType4']['GFM_StellarFormationTime'][:]
        
        # calculate the 3D distances from the galaxy center, and the ages
        rs = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
        
        # limit particles to those that formed within the past delta_t time
        mask = (formation_ages >= limit)
        ages = formation_ages[mask]
        masses = masses[mask]
        rs = rs[mask]
        
        return ages, masses, rs
    
    except KeyError :
        return None, None

def determine_diagnostics(simName, snapNum, window_length, polyorder,
                          plot=False, save=False) :
    
    # define the input directory and the input file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
    # get the masses, satellite and termination times for the quenched sample
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        times = hf['times'][:]
        tsats = hf['satellite_times'][:]
        tonsets = hf['onset_times'][:]
        tterms = hf['termination_times'][:]
    
    # for testing purposes
    test_IDs = [14]
    # test_IDs = [14, 41, 63878, 167398, 184946, 220605, 324126]
    
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
    
    # loop over all the galaxies in the quenched sample
    for subID, tsat, tonset, tterm in zip(subIDs, tsats, tonsets, tterms) :
        # get the mpb snapshot numbers, subIDs, stellar halfmassradii, and
        # galaxy centers
        snapNums, mpb_subIDs, radii, centers = get_mpb_halfmassradii(
            simName, snapNum, subID)
        
        # ensure that the quotient will be real
        radii[radii == 0.0] = np.inf
        
        # now get the star particle ages and distances at each snapshot/time
        gradients = []
        rmses = []
        for time, snap, subID, center in zip(times, snapNums, mpb_subIDs,
                                             centers) :
            ages, masses, rs = get_particles(simName, snapNum, snap, subID,
                                             center, time)
            
            # only proceed if the ages and distances are intact
            if (ages is not None) and (rs is not None) :
                # now compute the age gradient at each snapshot/time
                gradient = compute_age_gradient(rs, ages, snap)
                gradients.append(gradient)
                
                # now compute the stellar rms values at each snapshot/time
                rms = compute_sf_rms(rs)
                rmses.append(rms)
            else :
                rmses.append(np.nan)
        
        # now compute the quantity zeta, which is the quotient of the RMS of the
        # radial distances of the star forming particles to the stellar half mass radii
        zeta = np.array(rmses)/radii
        
        if plot :
            # now plot the results, but limit the time axis to valid snapshots
            ts = times[len(times)-len(zeta):]
            
            # smooth the function for plotting purposes
            smoothed = savgol_filter(zeta, window_length, polyorder)
            smoothed[smoothed < 0] = 0
            
            outfile = 'output/zeta(t)/zeta_subID_{}.png'.format(subID)
            ylabel = r'$\zeta =$ RMS$_{\rm SF}/$Stellar Half Mass Radius$_{\rm SF}$'
            plt.plot_simple_multi_with_times([ts, ts], [zeta, smoothed],
                ['data', 'smoothed'], ['grey', 'k'], ['', ''], ['--', '-'], [0.5, 1],
                tsat, tonset, tterm, xlabel=r'$t$ (Gyr)', ylabel=ylabel,
                xmin=-0.1, xmax=13.8, scale='linear', save=save, outfile=outfile)
            
            '''
            ts = times[len(times)-len(gradients):]
            
            outfile = 'output/age_gradients(t)/age_gradient_subID_{}.png'.format(subID)
            ylabel = r'$\zeta =$ RMS$_{\rm SF}/$Stellar Half Mass Radius$_{\rm SF}$'
            plt.plot_simple_multi_with_times([ts, ts], [gradients, smoothed],
                ['data', 'smoothed'], ['grey', 'k'], ['', ''], ['--', '-'], [0.5, 1],
                tsat, tonset, tterm, xlabel=r'$t$ (Gyr)', ylabel=ylabel,
                xmin=-0.1, xmax=13.8, scale='linear', save=save, outfile=outfile)
            '''
    return

def concatenate() :
    
    # adapted from
    # https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    
    from PIL import Image
    
    # get the paths of the SFH, zeta, age and SFR gradient images
    paths = ['output/quenched_SFHs(t)/quenched_SFH_subID_96806.png',
             'output/zeta(t)/zeta_subID_96806.png']
    
    # open the data from those images
    images = [Image.open(image) for image in paths]
    
    # get the widths and heights to create and empty final image
    widths, heights = zip(*(i.size for i in images))
    final = Image.new('RGB', (np.sum(widths), np.max(heights)))
    
    # populate the final image with the input images
    x_offset = 0
    for im in images :
        final.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    
    # save the final image
    outfile = 'test.png'
    final.save(outfile)
    
    return
