
import numpy as np

from astropy.table import Table

from core import gcPath

def determine_sample(simName, snapNum) :
    
    groupsDir = gcPath(simName, snapNum)
    
    # read catalogs
    halos_infile = groupsDir + 'halos_catalog_{}_{}.fits'.format(
        simName, snapNum)
    halos = Table.read(halos_infile)
    
    subhalos_infile = groupsDir + 'subhalos_catalog_{}_{}.fits'.format(
        simName, snapNum)
    subhalos = Table.read(subhalos_infile)
    
    # create a mask for the galaxies that we want to compute their SFHs for
    in_sel = ((subhalos['SubhaloFlag'] == 1) &
              (subhalos['SubhaloMassStars'] >= 8))
    subhalos['in_selection'] = in_sel
    
    # create a mask for the central galaxies, based on unique indices from the
    # halo table and it's 'GroupFirstSub' column
    subhalos['is_central'] = np.zeros(len(subhalos), dtype=bool)
    central_indices = np.unique(
        halos['GroupFirstSub'][halos['GroupFirstSub'] >= 0])
    subhalos['is_central'][central_indices] = True
    
    # create a mask for massive halos with M200 >= 13.8
    halos['is_cluster'] = np.zeros(len(halos), dtype=bool)
    cluster_indices = halos['GroupID'][halos['Group_M_Crit200'] >= 13.8]
    halos['is_cluster'][cluster_indices] = True
    
    # label subhalos in clusters
    subhalos['in_cluster'] = np.zeros(len(subhalos), dtype=bool)
    for cluster_index in cluster_indices :
        subhalo_index = halos['GroupFirstSub'][cluster_index]
        Nsubs = halos['GroupNsubs'][cluster_index]
        subhalos['in_cluster'][subhalo_index:subhalo_index+Nsubs] = True
    
    # create masks for different populations
    base = (subhalos['in_selection'] == True) # 8261 in selection
    BCG = (base & (subhalos['in_cluster'] == True) &
           (subhalos['is_central'] == True)) # 3 bCGs
    satellites = (base & (subhalos['in_cluster'] == True) &
                  (subhalos['is_central'] == False)) # 648 satellites
    field = (base & (subhalos['in_cluster'] == False) &
             (subhalos['is_central'] == True)) # 4689 field primaries
    
    # make columns based on those masks
    subhalos['BCG'] = BCG
    subhalos['satellites'] = satellites
    subhalos['field'] = field
    
    # write the good subhalo IDs to file, for downloading SFHs
    subhaloIDs = np.concatenate((subhalos['SubhaloID'][BCG].value,
                                 subhalos['SubhaloID'][satellites].value,
                                 subhalos['SubhaloID'][field].value))
    # np.savetxt(groupsDir + 'subhaloIDs_in_selection.txt', subhaloIDs, fmt='%d')
    
    # save the subhalos table to file, masked based on populations we care about
    subhalos = subhalos[BCG | satellites | field]
    subhalos.write(groupsDir + 'subhalos_catalog_{}_{}_sample.fits'.format(
        simName, snapNum))
    
    return
