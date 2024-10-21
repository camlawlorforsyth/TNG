
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

'''
import numpy as np; import h5py; import plotting as plt
from scipy.optimize import curve_fit
# fit linear curves
def linear(xx, aa, bb) :
    return aa*xx + bb

with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
    redshifts = hf['redshifts'][:]
    logM = hf['logM'][:, -1]
    quenched_status = hf['quenched'][:]
    tonsets = hf['onset_times'][:]
    tterms = hf['termination_times'][:]
    ionsets = hf['onset_indices'][:]

mask = quenched_status & (logM >= 9.5)
logM = logM[mask]
tonsets = tonsets[mask]
tterms = tterms[mask]
ionsets = ionsets[mask]
z_onsets = redshifts[ionsets.astype(int)]

delta_t = tterms - tonsets

# get the quenching mechanisms
with h5py.File('TNG50-1/TNG50-1_99_mechanism.hdf5', 'r') as hf :
    io = hf['inside-out'][:] # 103
    oi = hf['outside-in'][:] # 109
    uni = hf['uniform'][:]   # 8
    amb = hf['ambiguous'][:] # 58
mechs = np.sum(np.array([1*io, 3*oi, 5*uni, 5*amb]).T, axis=1)[mask]
color = np.full(mechs.shape, 'm')
color[mechs == 3] = 'r'

xs = [logM[mechs == 1], logM[mechs == 3]]
ys = [delta_t[mechs == 1], delta_t[mechs == 3]]
colors = [color[mechs == 1], color[mechs == 3]]

io_mass = logM[mechs == 1]
io_delta_t = delta_t[mechs == 1] # see code below for reasoning as to why these limits
io_popt, _ = curve_fit(linear, io_mass[io_mass > 10.15], io_delta_t[io_mass > 10.15])
io_mini, io_maxi = np.min(io_mass[io_mass > 10.15]), np.max(io_mass[io_mass > 10.15])
io_temp = np.linspace(io_mini, io_maxi, 1000)
io_fit = linear(io_temp, *io_popt)

oi_mass = logM[mechs == 3]
oi_delta_t = delta_t[mechs == 3] # in short, due to number of galaxies per stellar mass bin
oi_popt, _ = curve_fit(linear, oi_mass[oi_mass < 10.3], oi_delta_t[oi_mass < 10.3])
oi_mini, oi_maxi = np.min(oi_mass[oi_mass < 10.3]), np.max(oi_mass[oi_mass < 10.3])
oi_temp = np.linspace(oi_mini, oi_maxi, 1000)
oi_fit = linear(oi_temp, *oi_popt)

plt.plot_scatter_multi(xs, ys, colors, ['inside-out', 'outside-in'],
    ['o', 'o'], [1, 1],
    xlabel=r'$\log{(M_{*}/{\rm M}_{\odot})}$',
    ylabel=r'$\Delta t_{\rm quench}/{\rm Gyr}$')
plt.plot_simple_multi([io_temp, oi_temp], [io_fit, oi_fit],
    ['inside-out', 'outside-in'], ['m', 'r'], ['', ''], ['-', '-'], [1, 1],
    xlabel=r'$\log{(M_{*}/{\rm M}_{\odot})}$',
    ylabel=r'$\Delta t_{\rm quench}$ (Gyr)', xmin=9.43, xmax=11.35, ymin=0, ymax=6.22,
    figsizewidth=7.10000594991006, figsizeheight=9.095321710253218/2)

# plt.plot_scatter_multi([z_onsets[mechs == 1], z_onsets[mechs == 3]],
#                        ys, colors, ['inside-out', 'outside-in'],
#     ['o', 'o'], [1, 1],
#     xlabel=r'$z_{\rm onset}$',
#     ylabel=r'$\Delta t_{\rm quench}/{\rm Gyr}$')

oi_logM = np.sort(logM[mechs == 3])
# oi_logM_edges = oi_logM[::20]
oi_logM_edges = np.linspace(oi_logM[0], oi_logM[-1], 11)
oi_los = np.full(oi_logM_edges.shape[0] - 1, np.nan)
oi_meds = np.full(oi_logM_edges.shape[0] - 1, np.nan)
oi_his = np.full(oi_logM_edges.shape[0] - 1, np.nan)
for i, (first, second) in enumerate(zip(oi_logM_edges, oi_logM_edges[1:])) :
    mask = (logM >= first) & (logM < second) & (mechs == 3)
    if np.sum(mask) >= 7 :
        lo, med, hi = np.percentile(delta_t[mask], [16, 50, 84])
        oi_los[i] = lo
        oi_meds[i] = med
        oi_his[i] = hi

io_logM = np.sort(logM[mechs == 1])
# io_logM_edges = io_logM[::20]
io_logM_edges = np.linspace(io_logM[0], io_logM[-1], 11)
io_los = np.full(io_logM_edges.shape[0] - 1, np.nan)
io_meds = np.full(io_logM_edges.shape[0] - 1, np.nan)
io_his = np.full(io_logM_edges.shape[0] - 1, np.nan)
for i, (first, second) in enumerate(zip(io_logM_edges, io_logM_edges[1:])) :
    mask = (logM >= first) & (logM < second) & (mechs == 1)
    if np.sum(mask) >= 7 :
        lo, med, hi = np.percentile(delta_t[mask], [16, 50, 84])
        io_los[i] = lo
        io_meds[i] = med
        io_his[i] = hi

io_xs = io_logM_edges[:-1] + np.diff(io_logM_edges)/2
io_los_popt, _ = curve_fit(linear, io_xs[io_los > 0], io_los[io_los > 0])
io_los_fit = linear(io_xs, *io_los_popt)
io_los_fit[np.isnan(io_los)] = np.nan
io_meds_popt, _ = curve_fit(linear, io_xs[io_meds > 0], io_meds[io_meds > 0])
io_meds_fit = linear(io_xs, *io_meds_popt)
io_meds_fit[np.isnan(io_meds)] = np.nan
io_his_popt, _ = curve_fit(linear, io_xs[io_his > 0], io_his[io_his > 0])
io_his_fit = linear(io_xs, *io_his_popt)
io_his_fit[np.isnan(io_his)] = np.nan

oi_xs = oi_logM_edges[:-1] + np.diff(oi_logM_edges)/2
oi_los_popt, _ = curve_fit(linear, oi_xs[oi_los > 0], oi_los[oi_los > 0])
oi_los_fit = linear(oi_xs, *oi_los_popt)
oi_los_fit[np.isnan(oi_los)] = np.nan
oi_meds_popt, _ = curve_fit(linear, oi_xs[oi_meds > 0], oi_meds[oi_meds > 0])
oi_meds_fit = linear(oi_xs, *oi_meds_popt)
oi_meds_fit[np.isnan(oi_meds)] = np.nan
oi_his_popt, _ = curve_fit(linear, oi_xs[oi_his > 0], oi_his[oi_his > 0])
oi_his_fit = linear(oi_xs, *oi_his_popt)
oi_his_fit[np.isnan(oi_his)] = np.nan

plt.plot_filledbetween_multi(io_xs, oi_xs, io_los_fit, io_his_fit, io_meds_fit,
    oi_los_fit, oi_his_fit, oi_meds_fit, loc=2,
    xlabel=r'$\log{(M_{*}/{\rm M}_{\odot})}$',
    ylabel=r'$\Delta t_{\rm quench}$ (Gyr)', xmin=9.43, xmax=11.35, ymin=0, ymax=6.22,
    figsizewidth=7.10000594991006, figsizeheight=9.095321710253218/2,
    outfile='delta_t_quench_vs_logM_fit.png', save=False)

# fix the data for plotting purposes
oi_los[-1] = oi_los[-2]
oi_his[-1] = oi_his[-2]
io_los = np.hstack([io_los, io_los[-1][None]])
io_his = np.hstack([io_his, io_his[-1][None]])

plt.plot_steps_multi(io_logM_edges, oi_logM_edges, io_los, io_his, io_meds,
    oi_los, oi_his, oi_meds, loc=2, xlabel=r'$\log{(M_{*}/{\rm M}_{\odot})}$',
    ylabel=r'$\Delta t_{\rm quench}$ (Gyr)', xmin=9.43, xmax=11.35, ymin=0, ymax=6.22,
    figsizewidth=3.35224200913242, figsizeheight=9.095321710253218/3,
    outfile='delta_t_quench_vs_logM.pdf', save=False)
'''
