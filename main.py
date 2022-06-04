
import os

import catalogs
import masks
import sfhs

import core

def premain(simName, snapNum) :
    
    # ensure the output directories are available
    for path in [core.gcPath(simName, snapNum),
                 core.cutoutPath(simName, snapNum),
                 core.snapPath(simName, snapNum),
                 core.offsetPath(simName)] :
        os.makedirs(path, exist_ok=True) 
    
    return

def main(simName, snapNum, redshift=0.0) :
    
    # download halo and subhalo properties ("fields") for halos and subhalos
    # from the group catalogs
    catalogs.download_catalogs(simName, snapNum)
    
    # select subhalos (ie. subIDs) based on SFR, logMass, environment
    masks.select(simName, snapNum)
    
    # get SFHs from cutouts for the selected subhalos
    sfhs.get_histories(simName, snapNum, redshift)
    
    return
