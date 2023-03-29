
from os.path import exists
import numpy as np

import astropy.units as u
import h5py

from core import add_dataset, bsPath, get_particles, get_sf_particles

def check_for_nan_histories(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and file, and the output file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    
    # get the stellar masses and SFHs for the entire sample
    with h5py.File(infile, 'r') as hf :
        logM = hf['logM'][:] # track the stellar mass history (SMH)
        SFHs = hf['SFH'][:]
    
    # determine if any galaxies should be excluded from future analysis
    for i, (SMH, SFH) in enumerate(zip(logM, SFHs)) :
        # 19 galaxies have all NaN SMHs, and all NaN subIDs
        condition1 = np.all(np.isfinite(SMH) == False)
        
        # 47 galaxies have all 0 SFHs
        condition2 = np.all(SFH[~np.isnan(SFH)] == 0.0)
        
        # 37 galaxies are created after z = 0.5 which seems unphysical
        condition3 = np.argmax(SMH > 4.45) > 70
        
        if (condition1 | condition2 | condition3) :
            with h5py.File(infile, 'a') as hf :
                hf['exclude'][i] = True
    return

def determine_all_histories_from_cutouts(simName='TNG50-1', snapNum=99,
                                         delta_t=100*u.Myr) :
    
    # define the input directory and file, and the output file
    inDir = bsPath(simName)
    outfile = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    
    # check if the outfile exists and has good SFHs
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            if 'SFH' in hf.keys() :
                if np.all(~np.isnan(hf['SFH'])) :
                    print('File already exists with all non-NaN SFHs')
    
    # if the outfile exists, read the relevant information
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            snapshots = hf['snapshots'][:]
            times = hf['times'][:]
            subIDs = hf['subIDs'][:].astype(int)
            Res = hf['Re'][:]
            centers = hf['centers'][:]
            x_sfhs = hf['SFH'][:]
    
    # now iterate over every subID in subIDs and get the SFH
    for i, (mpb_subIDs, mpb_Res, mpb_centers) in enumerate(zip(subIDs, Res,
                                                               centers)) :
        
        # if the SFH values don't exist, determine the SFH and populate
        if np.all(np.isnan(x_sfhs[i, :])) :
            with h5py.File(outfile, 'a') as hf : # use the cutout values
                hf['SFH'][i, :] = history_from_cutouts(snapshots, times,
                    mpb_subIDs, mpb_Res, mpb_centers, simName=simName,
                    snapNum=snapNum, delta_t=delta_t)
        
        print('{}/8260 - subID {} done'.format(i, mpb_subIDs[-1]))
    
    return

def history_from_cutouts(snapshots, times, mpb_subIDs, mpb_Res, mpb_centers,
                         simName='TNG50-1', snapNum=99, delta_t=100*u.Myr) :
    
    # get the star particle ages, masses, and distances at each snapshot/time
    SFH = []
    for snap, time, subID, Re, center in zip(snapshots, times, mpb_subIDs,
                                             mpb_Res, mpb_centers) :
        
        if subID == -2147483648 : # np.nan cast into an integer, ie. for a
            SFH.append(np.nan)    # subID value that doesn't exist
        else :
            # get all particles
            ages, masses, rs = get_particles(simName, snapNum, snap, subID,
                                             center)
            
            # only proceed if the ages, masses, and distances are intact
            if (ages is not None) and (masses is not None) and (rs is not None) :
                
                # get the SF particles
                _, masses, rs = get_sf_particles(ages, masses, rs, time,
                                                 delta_t=delta_t)
                
                # now compute the SFR for particles within 2Re
                if len(masses) == 0 :
                    SFR = 0.0
                else :
                    total_mass = np.sum(masses[rs <= 2*Re])*u.solMass
                    SFR = total_mass/delta_t
                    SFR = (SFR.to(u.solMass/u.yr)).value
                SFH.append(SFR)
            else :
                SFH.append(np.nan)
    
    return SFH
