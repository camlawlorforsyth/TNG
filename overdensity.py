
import numpy as np

import h5py

from core import add_dataset, convert_distance_units

def compute_delta(simName='TNG50-1', snapNum=99) :
    
    infile = '{}/{}_{}_sample(t).hdf5'.format(simName, simName, snapNum)
    outfile = '{}/{}_{}_overdensity(t).hdf5'.format(simName, simName, snapNum)
    
    with h5py.File(infile, 'r') as hf :
        snaps = hf['snapshots'][:]
        # subIDfinals = hf['SubhaloID'][:]
        centers = hf['centers'][:]
    
    # volume = 4/3*np.pi*np.power(1*u.Mpc, 3)
    
    delta = np.full((8260, 100), np.nan)
    
    # loop over every snapshot
    for snap in snaps :
        
        centers_at_snap = centers[:, snap, :] # get the positions at the snapshot
        
        Ngals = []
        # loop over every galaxy
        for center in centers_at_snap :
            offsets = centers_at_snap - center
            
            # get the radial distance
            rs = np.sqrt(np.square(offsets[:, 0]) + np.square(offsets[:, 1]) +
                         np.square(offsets[:, 2]))
            
            # convert that distance to kpc
            rs = convert_distance_units(rs)
            
            # get the number of galaxies within 1 Mpc in any direction
            within_1_Mpc = rs[rs < 1000]
            Ngal = len(within_1_Mpc) - 1 # -1 accounts for the galaxy itself
            
            # error checking for the galaxies with no positions
            if Ngal == -1 :
                Ngals.append(np.nan)
            else :
                Ngals.append(Ngal)
        Ngals = np.array(Ngals)
        
        delta[:, snap] = Ngals/np.nanmean(Ngals) - 1
        print('{} done'.format(snap))
    
    with h5py.File(outfile, 'w') as hf :
        add_dataset(hf, delta, 'delta')
    
    return

with h5py.File('TNG50-1/TNG50-1_99_overdensity(t).hdf5', 'r') as hf :
    delta = hf['delta'][:]

with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
    snaps = hf['snapshots'][:]
    redshifts = hf['redshifts'][:]
    subIDfinals = hf['SubhaloID'][:]

for snap, zz, d1, d2, d3 in zip(snaps, redshifts, delta[65], delta[100], delta[134]) :
    print('{:2}  {:6.2f} {:7.2f}  {:7.2f}  {:7.2f}'.format(snap, zz, d1, d2, d3))


