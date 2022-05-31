
import os
# import numpy as np

# import astropy.constants as c
# from astropy.cosmology import Planck15 as cosmo
# from astropy.table import Table
# import astropy.units as u
# import h5py

import catalogs # old name was get_prop_for_selection
# import masks # old name was make_selections
# import sfhs # old name was get_sfhs
# import plotting as plt

import core

def premain(simName, snapNum) :
    # ensure the output directories are available
    
    for path in [core.gcPath(simName, snapNum),
                 core.cutoutPath(simName, snapNum),
                 core.snapPath(simName, snapNum),
                 core.offsetPath(simName)] :
        os.makedirs(path, exist_ok=True) 
    
    return

def main(simName, snapNum, redshift) :
    
    # START
    
    # download halo and subhalo properties ("fields") for halos and subhalos
    # from the group catalogs
    catalogs.download_catalogs(simName, snapNum)
    
    
    
    
    # NOTE - NEED TO WORK ON THE BELOW
    
    # select subhalos (ie. subIDs) based on SFR, logMass, environment
    # masks.select()
    
    # get SFHs from cutouts for the selected subhalos
    # sfhs.get_histories(simName, snapNum, redshift, outDir, groupName, params_load)
    
    # END
    
    return

# main('TNG50-1', 99, 0)
