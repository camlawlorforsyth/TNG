
import numpy as np

import h5py

from core import add_dataset, bsPath, get, mpbCutoutPath

def determine_mpb_cutouts_to_download(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and file, and output file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    outfile = inDir + '/{}_{}_mpb_cutouts_to_download.hdf5'.format(simName, snapNum)
    
    # get relevant information for the general sample
    with h5py.File(infile, 'r') as hf :
        snaps = hf['snapshots'][:]
        subIDs = hf['subIDs'][:].astype(int)
    
    list_of_subIDs, list_of_snaps = [], []
    for row in subIDs :
        valid = (row > 0)
        for ID, snap in zip(row[valid], snaps[valid]):
            list_of_snaps.append(snap)
            list_of_subIDs.append(ID)
    
    with h5py.File(outfile, 'w') as hf :
        add_dataset(hf, np.array(list_of_snaps), 'list_of_snaps')
        add_dataset(hf, np.array(list_of_subIDs), 'list_of_subIDs')
    
    return

def download_mpb_cutouts(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and file, and output directory for the mpb cutouts
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_mpb_cutouts_to_download.hdf5'.format(simName, snapNum)
    outDir = mpbCutoutPath(simName, snapNum)
    
    # get the mpb cutouts to download
    with h5py.File(infile, 'r') as hf :
        list_of_snaps = hf['list_of_snaps'][:]
        list_of_subIDs = hf['list_of_subIDs'][:]
    
    # define the parameters that are requested for each particle in the cutout
    params = {'gas':'Coordinates,Density,ElectronAbundance,GFM_Metallicity,InternalEnergy,Masses,StarFormationRate',
              'stars':'Coordinates,GFM_InitialMass,GFM_Metallicity,GFM_StellarFormationTime,Masses,StellarHsml'}
    
    # loop over all the required mpb cutouts
    for snap, subID in zip(list_of_snaps, list_of_subIDs) :
        
        # define the URL for the galaxy at the redshift of interest
        url = 'https://www.tng-project.org/api/{}/snapshots/{}/subhalos/{}'.format(
            simName, snap, subID)
        
        # save the cutout file into the output directory if it doesn't exist
        filename = 'cutout_{}_{}.hdf5'.format(snap, subID)
        # if not exists(outDir + filename) :
        get(url + '/cutout.hdf5', directory=outDir, params=params,
                filename=filename)
        
        print('{} done'.format(filename))
    
    return
