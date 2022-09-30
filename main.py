
import os

import catalogs
import core
import quenched
import sample
import satellite_time
import sfhs
import sfms

def premain(simName, snapNum) :
    
    # ensure the output directories are available
    for path in [core.gcPath(simName, snapNum),
                 core.cutoutPath(simName, snapNum),
                 core.snapPath(simName, snapNum),
                 core.offsetPath(simName)] :
        os.makedirs(path, exist_ok=True) 
    
    return

def main(simName='TNG50-1', snapNum=99) :
    
    # define the edges for each mass bin
    if (simName == 'TNG50-1') and (snapNum == 99) :
        # to have >~800 galaxies in each bin for decent statistics
        # numGals_per_bin = [834, 830, 828, 828, 824, 825, 824, 822, 823, 822]
        mass_bin_edges = [8, 8.118, 8.252, 8.415, 8.577, 8.751,
                          8.96, 9.183, 9.51, 10.07, 12.75]
    
    # set the parameters for the Savitzky-Golay filter which will smooth the data
    window_length, polyorder = 15, 3 # good starting values
    
    # download halo and subhalo properties ("fields") for halos and subhalos
    # from the group catalogs
    # creates halos_catalog_simName_snapNum.fits
    # creates subhalos_catalog_simName_snapNum.fits
    catalogs.download_catalogs(simName, snapNum)
    
    # determine the primary and satellite subhalos with Mstar >= 10^8 Mdot
    # creates simName_snapNum_primary-satellite-flagIDs.fits
    sample.primary_and_satellite_flags(simName, snapNum)
    
    # build the final sample
    # creates simName_snapNum_sample.fits
    sample.build_final_sample(simName, snapNum)
    
    # get SFHs from cutouts for the selected subhalos
    # creates simName_snapNum_sample_SFHs.hdf5
    sfhs.download_all_cutouts(simName, snapNum)
    sfhs.determine_all_histories(simName, snapNum)
    
    # determine the SFMS at z = 0 and use the SFMS to set the SFH limits to
    # subsequently determine the quenched population
    # creates simName_snapNum_SFMS_SFH_limits.pkl
    sfms.check_SFMS_and_limits(simName, snapNum, mass_bin_edges, window_length,
                               polyorder)
    
    # determine which systems are quenched by our definition
    # appends to simName_snapNum_sample_SFHs.hdf5
    # creates simName_snapNum_quenched_SFHs.hdf5
    quenched.determine_quenched_systems(simName, snapNum, mass_bin_edges,
                                        window_length, polyorder)
    quenched.reduce_SFH_to_quenched_systems(simName, snapNum)
    
    # determine the 'primary_flag' as a function of time for selected subhalos
    # appends to simName_snapNum_quenched_SFHs.hdf5
    satellite_time.add_primary_flags(simName, snapNum)
    satellite_time.determine_satellite_time(simName, snapNum)
    
    return
