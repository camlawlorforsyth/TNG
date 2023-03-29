
from os.path import exists
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

def download_mpb_cutouts(simName='TNG50-1', snapNum=99, gas=False) :
    
    # define the input directory and file, and output directory for the mpb cutouts
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_mpb_cutouts_to_download.hdf5'.format(simName, snapNum)
    outDir = mpbCutoutPath(simName, snapNum)
    
    # get the mpb cutouts to download
    with h5py.File(infile, 'r') as hf :
        list_of_snaps = hf['list_of_snaps'][:]
        list_of_subIDs = hf['list_of_subIDs'][:]
    
    # define the parameters that are requested for each particle in the cutout
    if gas :
        params = {'stars':'GFM_InitialMass,Masses,GFM_StellarFormationTime,Coordinates',
                  'gas':'Masses,StarFormationRate,Coordinates'}
    else :
        params = {'stars':'Coordinates,GFM_InitialMass,GFM_StellarFormationTime'}
    
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

def estimate_remaining_time(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and file, and output directory for the mpb cutouts
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_mpb_cutouts_to_download.hdf5'.format(simName, snapNum)
    outDir = mpbCutoutPath(simName, snapNum)
    
    # get the mpb cutouts to download
    with h5py.File(infile, 'r') as hf :
        list_of_snaps = hf['list_of_snaps'][:]
        list_of_IDs = hf['list_of_subIDs'][:]
    
    # check if the files exist
    to_download, downloaded = len(list_of_snaps), 0
    for snap, subID in zip(list_of_snaps, list_of_IDs) :
        if exists(outDir + 'cutout_{}_{}.hdf5'.format(snap, subID)) :
            downloaded +=1
    
    # determine the remaining number of files to download, and use an estimate
    # for the download rate
    remaining = to_download - downloaded
    rate = 2000 # ~2000 files per hour
    
    print('{} files remaining, at ~2000 files/hr -> {:.2f} hr remaining'.format(
        remaining, remaining/rate))
    
    return
