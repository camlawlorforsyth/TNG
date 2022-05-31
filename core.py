
import numpy as np

from astropy.cosmology import Planck15 as cosmo
import illustris_python as il
import requests

def add_dataset(h5file, data, label, dtype=None) :
    
    # set the datatype of the new data
    if dtype is None :
        dtype = data.dtype
    
    # try to add the dataset using h5py's method
    try :
        h5file.create_dataset(label, data=data, shape=np.shape(data), dtype=dtype)
        # hx[:] = data
    except : # but add it as a dictionary (essentially) if that doesn't work
        h5file[label] = data
    
    return # h5file

def bsPath(simName) :
    return '{}/output'.format(simName)

def convert_mass_units(masses) :
    return masses*1e10/cosmo.h

def cutoutPath(simName, snapNum) :
    return bsPath(simName) + '/cutouts_{:3.0f}/'.format(snapNum).replace(' ', '0')

def gcPath(simName, snapNum) :
    return bsPath(simName) + '/groups_{:3.0f}/'.format(snapNum).replace(' ', '0')

def get(path, params=None) :
    # https://www.tng-project.org/data/docs/api/
    
    # make HTTP GET request to path
    headers = {'api-key':'0890bad45ac29c4fdd80a1ffc7d6d27b'}
    rr = requests.get(path, params=params, headers=headers)
    
    # raise exception if response code is not HTTP SUCCESS (200)
    rr.raise_for_status()
    
    if rr.headers['content-type'] == 'application/json' :
        return rr.json() # parse json responses automatically
    
    if 'content-disposition' in rr.headers :
        filename = rr.headers['content-disposition'].split('filename=')[1]
        with open(filename, 'wb') as ff :
            ff.write(rr.content)
        return filename # return the filename string
    
    return rr

def get_stars(xx) :
    itype_stars = il.snapshot.partTypeNum('stars') # 4
    x_stars = np.array(xx)[:, itype_stars]
    return x_stars

def offsetPath(simName) :
    return '{}/postprocessing/offsets'.format(simName)

def snapPath(simName, snapNum) :
    return bsPath(simName) + '/snapdir_{:3.0f}/'.format(snapNum).replace(' ', '0')






