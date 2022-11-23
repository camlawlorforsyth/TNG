
from os.path import exists

import h5py

from core import bsPath, get, get_mpb_radii_and_centers, mpbCutoutPath

def download_mpb_cutouts(simName, snapNum) :
    
    # define the input directory and file
    inDir = bsPath(simName)
    # infile = inDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    infile = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    
    # define the output directory for the mpb cutouts
    outDir = mpbCutoutPath(simName, snapNum)
    
    # get the subIDs for the quenched sample
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        masses = hf['SubhaloMassStars'][:]
    
    import numpy as np
    mask = (masses > 10.07) & (masses <= 12.75)
    subIDs = subIDs[mask][np.argsort(masses[mask])]
    
    # define the parameters that are requested for each particle in the cutout
    params = {'stars':'Coordinates,GFM_InitialMass,GFM_StellarFormationTime'}
    
    for subID in subIDs :
        
        # get the mpb snapshot numbers and subIDs
        snapNums, mpb_subIDs, _, _ = get_mpb_radii_and_centers(
            simName, snapNum, subID)
        
        for snap, mpb_subID in zip(snapNums, mpb_subIDs) :
            
            # define the URL for the galaxy at the redshift of interest
            url = 'https://www.tng-project.org/api/{}/snapshots/{}/subhalos/{}'.format(
                simName, snap, mpb_subID)
            
            # save the cutout file into the output directory if it doesn't exist
            filename = 'cutout_{}_{}.hdf5'.format(snap, mpb_subID)
            if not exists(outDir + filename) :
                get(url + '/cutout.hdf5', directory=outDir, params=params,
                    filename=filename)
    
    return
