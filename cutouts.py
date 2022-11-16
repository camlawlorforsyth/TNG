
from os.path import exists

import h5py

from core import bsPath, get, mpbCutoutPath

def download_mpb_cutouts(simName, snapNum) :
    
    # define the input directory and file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
    # define the output directory for the mpb cutouts
    outDir = mpbCutoutPath(simName, snapNum)
    
    # get the subIDs for the quenched sample
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
    
    # define the parameters that are requested for each particle in the cutout
    params = {'stars':'Coordinates,GFM_InitialMass,GFM_StellarFormationTime'}
    
    for subID in subIDs :
        url = 'https://www.tng-project.org/api/{}/snapshots/{}/subhalos/{}'.format(
            simName, snapNum, subID)
        
        # retrieve information about the galaxy at the redshift of interest
        sub = get(url)
        
        # save the cutout file into the output directory
        filename = 'cutout_{}_{}.hdf5'.format(sub['snap'], sub['id'])
        get(sub['meta']['url'] + '/cutout.hdf5', directory=outDir,
            params=params, filename=filename)
        
        while sub['prog_sfid'] != -1 :
            # request the full subhalo details of the progenitor by following
            # the sublink URL
            sub = get(sub['related']['sublink_progenitor'])
            
            # check if the cutout file exists
            filename = 'cutout_{}_{}.hdf5'.format(sub['snap'], sub['id'])
            if not exists(outDir + filename) :
                # save the cutout file into the output directory
                get(sub['meta']['url'] + '/cutout.hdf5', directory=outDir,
                    params=params, filename=filename)
    
    return
