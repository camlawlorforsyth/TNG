
import numpy as np

from astropy.table import Table
import requests

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

def mpbPath(simName, snapNum) :
    return bsPath(simName) + '/mpbs_{:3.0f}/'.format(snapNum).replace(' ', '0')

def mpbCutoutPath(simName, snapNum) :
    return bsPath(simName) + '/mpb_cutouts_{:3.0f}/'.format(snapNum).replace(' ', '0')

def offsetPath(simName) :
    return '{}/postprocessing/offsets'.format(simName)

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
