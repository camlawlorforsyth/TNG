
import os

# import age_gradients
import catalogs
# import concatenate
import core
import cutouts
import diagnostics
# import psi
import quenched
import sample
import satellite_time
import sfhs
import sfms
# import xi
# import zeta

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
    # creates simName_snapNum_halos_catalog.fits
    # creates simName_snapNum_subhalos_catalog.fits
    catalogs.download_catalogs(simName=simName, snapNum=snapNum)
    
    # determine the primary and satellite subhalos with Mstar >= 10^8 at z = 0,
    # the z = 0 environments, build the final sample, and download mpb files
    # creates simName_snapNum_primary-satellite-flagIDs.fits
    # creates simName_snapNum_env.fits
    # creates simName_snapNum_sample.fits
    # creates simName_snapNum_sample.hdf5
    # appends to simName_snapNum_sample.hdf5
    sample.primary_and_satellite_flags(simName=simName, snapNum=snapNum)
    sample.determine_environment(simName=simName, snapNum=snapNum)
    sample.build_final_sample(simName=simName, snapNum=snapNum)
    sample.resave_as_hdf5(simName=simName, snapNum=snapNum)
    sample.download_all_mpbs(simName=simName, snapNum=snapNum)
    sample.save_mpb_values(simName=simName, snapNum=snapNum)
    
    # save required mpb cutouts for all subhalos
    cutouts.determine_mpb_cutouts_to_download(simName=simName, snapNum=snapNum)
    cutouts.download_mpb_cutouts(simName=simName, snapNum=snapNum)
    
    # get SFHs from the mpb cutouts for all subhalos
    # appends to simName_snapNum_sample(t).hdf5
    sfhs.determine_all_histories_from_cutouts(simName=simName, snapNum=snapNum)
    sfhs.check_for_nan_histories(simName=simName, snapNum=snapNum)
    
    # determine the SFMS at each snapshot
    # appends to simName_snapNum_sample(t).hdf5
    sfms.compute_SFMS_percentile_limits(simName=simName, snapNum=snapNum)
    sfms.determine_SFMS(simName=simName, snapNum=snapNum)
    
    # determine which systems are quenched by our definition
    # appends to simName_snapNum_sample(t).hdf5
    quenched.determine_comparison_systems_relative(simName=simName,
        snapNum=snapNum, hw=hw, minNum=minNum)
    quenched.determine_quenched_systems_relative(simName=simName,
        snapNum=snapNum, kernel=kernel)
    
    # determine the 'primary_flag' as a function of time for selected subhalos
    # appends to simName_snapNum_quenched_SFHs(t).hdf5
    satellite_time.add_primary_flags(simName, snapNum)
    satellite_time.determine_satellite_time(simName, snapNum)
    
    # compute various star formation related quantities of interest through time
    diagnostics.find_control_sf_sample()
    diagnostics.calculate_required_radial_profiles()
    diagnostics.calculate_required_morphological_parameters()
    # diagnostics.determine_diagnostics()                    # to delete
    # diagnostics.determine_diagnostics_for_matched_sample() # to delete
    
    # zeta.zeta_for_sample(simName, snapNum) # to delete
    # xi.xi_for_sample(simName, snapNum)     # to delete
    # psi.determine_psi(simName, snapNum)    # to delete
    
    # save plots together - this can be deleted later
    # concatenate.concatenate_diagnostics(subIDs, masses)
    
    # revisit the age gradients after first investigating properties
    # of star formation in the quenched sample
    # age_gradients.determine_age_gradients(simName, snapNum, kernel=kernel)
    
    return
