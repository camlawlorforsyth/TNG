
import os

import catalogs
import classification
import comprehensive_view
import core
import cutouts
import metrics
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
    # creates and appends to simName_snapNum_sample.hdf5
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
    # appends to simName_snapNum_primary_flags(t).hdf5
    satellite_time.get_all_primary_flags(simName, snapNum)
    satellite_time.determine_satellite_time(simName, snapNum)
    
    # compute various star formation related quantities of interest through time
    # creates locations_mask_control_sf.npy
    # creates locations_mask_quenched.npy
    # creates and appends to simName_snapNum_massive_radial_profiles(t).hdf5
    # creates and appends to simName_snapNum_massive_C_SF(t).hdf5
    # creates and appends to simName_snapNum_massive_R_SF(t).hdf5
    # creates and appends to simName_snapNum_massive_Rinner(t).hdf5
    # creates and appends to simName_snapNum_massive_Router(t).hdf5
    metrics.find_control_sf_sample()
    metrics.calculate_required_radial_profiles()
    metrics.calculate_required_morphological_metrics()
    
    # prepare the morphological metrics for classification
    # creates morphological_metrics_-10.5_+-1.fits
    metrics.prepare_morphological_metrics_for_classification()
    
    
    # comprehensive_view.
    
    
    return

def plots() :
    
    # creates SFHs.pdf
    quenched.save_SFH_with_quenched_galaxy_plot()
    
    # creates SFMS_z0_with_quenched.pdf
    quenched.save_SFMS_with_quenched_galaxies_plot()
    
    # creates SMF.pdf
    quenched.save_SMF_with_quenched_galaxies_plot()
    
    # creates postage_stamps_and_sSFR_profiles.pdf
    # comprehensive_view.
    
    # doesn't create any saved plot
    # metrics.save_mass_distribution_plots()
    
    # creates metric_example.pdf
    metrics.save_morphological_metric_example_evolution_plot()
    
    # creates metric_global_evolution.pdf and metric_global_evolution_amb.pdf
    metrics.save_morphological_metric_global_evolution_plot()
    
    # creates metric_plane_evolution.pdf
    metrics.save_CSF_and_RSF_evolution_plot()
    
    # creates morphological_metrics_interactive_figure_1.html and
    metrics.save_plotly_plots() # morphological_metrics_interactive_figure_2.html
    
    # creates classification_boundaries.pdf
    classification.save_classification_boundaries_plot()
    classification.save_purity_completness_plot()
    
    # creates time_comparison.pdf
    satellite_time.save_time_comparison_plot()
    
    return
