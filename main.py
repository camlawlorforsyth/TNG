
import os

# import age_gradients
import catalogs
# import concatenate
import core
import cutouts
import psi
import quenched
import sample
import satellite_time
import sfhs
import sfms
import xi
import zeta

def premain(simName, snapNum) :
    
    # ensure the output directories are available
    for path in [core.gcPath(simName, snapNum),
                 core.cutoutPath(simName, snapNum),
                 core.snapPath(simName, snapNum),
                 core.offsetPath(simName)] :
        os.makedirs(path, exist_ok=True) 
    
    return

def main(simName='TNG50-1', snapNum=99, hw=0.1, minNum=50, kernel=2) :
    
    # get the redshifts for each snapshot
    core.snapshot_redshifts(simName=simName)
    
    # create a look-up table for converting scalefactors to cosmological ages
    core.save_lookup_table(simName=simName)
    
    # download halo and subhalo properties ("fields") for halos and subhalos
    # from the group catalogs
    # creates halos_catalog_simName_snapNum.fits
    # creates subhalos_catalog_simName_snapNum.fits
    catalogs.download_catalogs(simName=simName, snapNum=snapNum)
    
    # determine the primary and satellite subhalos with Mstar >= 10^8 at z = 0
    # creates simName_snapNum_primary-satellite-flagIDs.fits
    sample.primary_and_satellite_flags(simName=simName, snapNum=snapNum)
    
    # build the final sample
    # creates simName_snapNum_sample.fits
    sample.build_final_sample(simName, snapNum)
    
    # get SFHs from the Donnari/Pillipech catalog for the selected subhalos
    # creates simName_snapNum_sample_SFHs(t).hdf5
    sfhs.download_all_mpbs(simName=simName, snapNum=snapNum)
    sfhs.determine_all_histories_from_catalog(simName=simName, snapNum=snapNum)
    
    # determine the SFMS at each snapshot
    # appends to simName_snapNum_sample_SFHs(t).hdf5
    sfms.compute_SFMS_percentile_limits(simName=simName, snapNum=snapNum)
    sfms.determine_SFMS(simName=simName, snapNum=snapNum)
    
    # determine which systems are quenched by our definition
    # appends to simName_snapNum_sample_SFHs(t).hdf5
    quenched.determine_comparison_systems_relative(simName=simName,
        snapNum=snapNum, hw=hw, minNum=minNum)
    quenched.determine_quenched_systems_relative(simName=simName,
        snapNum=snapNum, kernel=kernel)
    
    # save required MPB cutouts of quenched and control galaxies
    cutouts.determine_mpb_cutouts_to_download(simName=simName, snapNum=snapNum)
    cutouts.download_mpb_cutouts(simName=simName, snapNum=snapNum)
    
    # determine the 'primary_flag' as a function of time for selected subhalos
    # appends to simName_snapNum_quenched_SFHs(t).hdf5
    satellite_time.add_primary_flags(simName, snapNum)
    satellite_time.determine_satellite_time(simName, snapNum)
    
    # compute various star formation related quantities of interest through time
    zeta.zeta_for_sample(simName, snapNum)
    xi.xi_for_sample(simName, snapNum)
    psi.determine_psi(simName, snapNum)
    
    # save plots together - this can be deleted later
    # concatenate.concatenate_diagnostics(subIDs, masses)
    
    # revisit the age gradients after first investigating properties
    # of star formation in the quenched sample
    # age_gradients.determine_age_gradients(simName, snapNum, kernel=kernel)
    
    return
