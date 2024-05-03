
from os.path import exists
import numpy as np

from astropy.table import Table
import astropy.units as u
import h5py
from scipy.ndimage import gaussian_filter1d, gaussian_filter

from core import (add_dataset, bsPath, determine_mass_bin_indices, find_nearest,
                  get_quenched_data, get_rotation_input, get_sf_particle_positions)
import plotting as plt
from projection import calculate_MoI_tensor, rotation_matrix_from_MoI_tensor

def comp_prop_plots_for_sample(simName='TNG50-1', snapNum=99, hw=0.1,
                               minNum=50, nelson2021version=True, save=False) :
    
    # define the input and output directories, and the helper file
    inDir = bsPath(simName)
    helper_file = inDir + '/{}_{}_profiles(t).hdf5'.format(simName, snapNum)
    outDir = inDir + '/figures/comprehensive_plots/'
    
    # get relevant information for the general sample, and quenched systems
    (snapshots, redshifts, times, subIDs, logM, Re, centers, UVK, SFHs, SFMS,
     q_subIDfinals, q_subIDs, q_logM, q_SFHs, q_Re, q_centers, q_UVK, q_primary,
     q_cluster, q_hm_group, q_lm_group, q_field, q_lo_SFH, q_hi_SFH,
     q_ionsets, q_tonsets, q_iterms, q_tterms) = get_quenched_data(
         simName=simName, snapNum=snapNum)
    
    # get basic information for the UVK and radial profile plots
    UVK_X_cent, UVK_Y_cent, UVK_contour, UVK_levels = determine_UVK_contours(UVK)
    (radial_edges, mids,
     spatial_edges, XX, YY, X_cent, Y_cent) = set_basic_info_for_plots()
    
    # set which version to use
    if nelson2021version :
        label = r'sSFR (yr$^{-1}$)'
    else :
        label = (r'$\log(\dot{\Sigma}_{\rm *, 100~Myr}/' +
                 r'{\rm M}_{\odot}~{\rm yr}^{-1}~{\rm kpc}^{-2})$')
    
    st = 559 # 6 # 0
    end = 560 # 7 # 1577
    # loop through the quenched galaxies in the sample
    for i, (q_subfin, q_subID, q_logM, q_SFH, q_rad, q_cent, q_UVK_ind, q_lo, q_hi,
         q_ionset, q_tonset, q_iterm, q_tterm) in enumerate(zip(q_subIDfinals[st:end],
             q_subIDs[st:end], q_logM[st:end], q_SFHs[st:end], q_Re[st:end],
             q_centers[st:end], q_UVK[st:end], q_lo_SFH[st:end], q_hi_SFH[st:end],
             q_ionsets[st:end], q_tonsets[st:end], q_iterms[st:end], q_tterms[st:end])) :
        
        # print('{}/1576 - attempting subID {}'.format(i, q_subfin)) # 1577 total
        
        # work through the snapshots from onset until termination, but using
        # tterm isn't very illustrative, so use alternative times based on the
        # final snapshot being 75% of the quenching mechanism duration
        index_times = np.array([q_tonset,
                                q_tonset + 0.375*(q_tterm - q_tonset),
                                q_tonset + 0.75*(q_tterm - q_tonset)])
        indices = find_nearest(times, index_times)
        
        try :
            outfile = outDir + 'subID_{}_prev.png'.format(q_subfin)
            # if not exists(outfile) :
            # create the comprehensive plot for the quenched galaxy
            comprehensive_plot(q_subfin, q_subID, q_logM, q_SFH, q_rad,
                               q_cent, q_UVK_ind, q_lo, q_hi, redshifts,
                               times, subIDs, logM, Re, centers, UVK, SFHs,
                               SFMS, q_tonset, q_iterm, q_tterm, indices,
                               radial_edges, mids, spatial_edges, XX, YY,
                               X_cent, Y_cent, UVK_X_cent, UVK_Y_cent,
                               UVK_contour, UVK_levels, label, helper_file,
                               outfile, nelson2021version=nelson2021version,
                               save=save)
            
            # create the proposal plot for the quenched galaxy
            # proposal_plot(q_subfin, q_subID, q_logM, q_rad, q_cent,
            #               indices, redshifts, times, subIDs, logM, Re,
            #               centers, SFHs, SFMS, outfile)
            
        except :
            pass
    
    return

def comprehensive_plot(q_subfin, q_subID, q_logM, q_SFH, q_rad, q_cent,
                       q_UVK_ind, q_lo_SFH, q_hi_SFH, redshifts, times, subIDs,
                       logM, Re, centers, UVK, SFHs, SFMS, q_tonset,
                       q_iterm, q_tterm, snaps, radial_edges, mids,
                       spatial_edges, XX, YY, X_cent, Y_cent, UVK_X_cent,
                       UVK_Y_cent, UVK_contour, UVK_levels, label, helper_file,
                       outfile, save=False, nelson2021version=True) :
    
    # get the quenched galaxy's UVK positions
    UVK_snaps = q_UVK_ind[np.concatenate((snaps, [q_iterm]))]
    UVK_snaps_xs = UVK_snaps[:, 1] - UVK_snaps[:, 2]
    UVK_snaps_ys = UVK_snaps[:, 0] - UVK_snaps[:, 1]
    
    hists, contours, levels, subtitles = [], [], [], []
    mains, los, meds, his = [], [], [], []
    # loop over the specified snapshots
    for snap in snaps :
        
        subtitle = (r'$z = {:.2f}$'.format(redshifts[snap]) + ', ' +
                    r'$\Delta t_{\rm since~onset} = $' +
                    '{:.2f} Gyr'.format(times[snap] - times[snaps][0]))
        subtitles.append(subtitle)
        
        # get required information for the spatial plots
        hist, contour, level = spatial_plot_info(times[snap], snap,
            q_subID[snap], q_cent[snap], q_rad[snap], spatial_edges, 100*u.Myr,
            nelson2021version=nelson2021version)
        hists.append(hist)
        contours.append(contour)
        levels.append(level)
        
        # get required information for the radial plots
        main, lo, med, hi = radial_plot_info(subIDs, logM, SFMS, snap,
            q_subID[snap], q_logM[snap], helper_file, nelson2021version=True)
        mains.append(main)
        los.append(lo)
        meds.append(med)
        his.append(hi)
    
    # find the colorbar limits for the spatial plots
    vmin, vmax = determine_limits_spatial(hists)
    
    # find the y-axis limits for the radial plots
    ymin_b, ymax_b = determine_limits_radial(mains, los, meds, his)
    
    # get required information for the SFH and SMH plots
    sm = gaussian_filter1d(q_SFH, 2)
    lo_sm = gaussian_filter1d(q_lo_SFH, 2)
    hi_sm = gaussian_filter1d(q_hi_SFH, 2)
    SMH = gaussian_filter1d(q_logM, 2) # stellar mass history
    
    mtitle = (r'subID$_{z = 0}$' + ' {}'.format(q_subfin) +
              r', $\log{(M/{\rm M}_{\odot})}_{z = 0} = $' 
              + '{:.2f}'.format(q_logM[-1]))
    
    plt.plot_comprehensive_mini(
        np.log10(hists[0]), np.log10(hists[1]), np.log10(hists[2]),
        contours[0], contours[1], contours[2], levels[0], levels[1], levels[2],
        mids, np.log10(mains[0]), np.log10(mains[1]), np.log10(mains[2]),
        np.log10(los[0]), np.log10(los[1]), np.log10(los[2]),
        np.log10(meds[0]), np.log10(meds[1]), np.log10(meds[2]),
        np.log10(his[0]), np.log10(his[1]), np.log10(his[2]),
        XX, YY, X_cent, Y_cent, vmin=np.log10(vmin), vmax=np.log10(vmax),
        ymin=np.log10(ymin_b), ymax=np.log10(ymax_b), save=False,
        outfile='postage_stamps_and_sSFR_profiles.pdf')
    
    '''
    plt.plot_comprehensive_plot(
        subtitles[0], subtitles[1], subtitles[2],
        hists[0], hists[1], hists[2],
        contours[0], contours[1], contours[2],
        levels[0], levels[1], levels[2],
        mids,
        mains[0], mains[1], mains[2],
        los[0], los[1], los[2],
        meds[0], meds[1], meds[2],
        his[0], his[1], his[2],
        XX, YY,
        X_cent, Y_cent,
        times, sm, lo_sm, hi_sm, q_tonset, q_tterm,
        times[snaps[1]], times[snaps[2]],
        SMH,
        UVK_X_cent, UVK_Y_cent, UVK_contour, UVK_levels,
        UVK_snaps_xs, UVK_snaps_ys,
        xlabel_t=r'$\Delta x$ ($R_{\rm e}$)',
        ylabel_t=r'$\Delta z$ ($R_{\rm e}$)',
        xlabel_b=r'$r/R_{\rm e}$',
        ylabel_b=label,
        vlabel=label,
        vmin=vmin, vmax=vmax,
        xlabel_SFH=r'$t$ (Gyr)',
        ylabel_SFH=r'SFR (${\rm M}_{\odot}$ yr$^{-1}$)',
        xlabel_SMH=r'$t$ (Gyr)',
        ylabel_SMH=r'$\log{(M/{\rm M}_{\odot})}$',
        xlabel_UVK=r'$V - K$',
        ylabel_UVK=r'$U - V$',
        mtitle=mtitle,
        xmin_t=-5, xmax_t=5, ymin_t=-5, ymax_t=5,
        xmin_b=0, xmax_b=5, ymin_b=ymin_b, ymax_b=ymax_b,
        xmin_SFH=-0.1, xmax_SFH=13.8, xmin_SMH=-0.1, xmax_SMH=13.8,
        xmin_UVK=0.5, xmax_UVK=3.4, ymin_UVK=-0.5, ymax_UVK=1.75,
        outfile=outfile, save=save)
    '''
    
    return

def convert_csv_mechanism_to_hdf5() :
    
    tt = Table.read('TNG50-1/comprehensive_plots_visual_inspection.csv')
    IDs = np.array(tt['subID'])
    mechanism = np.array(tt['mechanism'])
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        subIDfinals = hf['SubhaloID'][:]
    
    outside_in = np.full(8260, False)
    inside_out = np.full(8260, False)
    uniform = np.full(8260, False)
    ambiguous = np.full(8260, False)
    for i, subID in enumerate(subIDfinals) :
        in_table = np.where(IDs == subID)[0]
        if len(in_table) > 0 :
            loc = in_table[0]
            mech = mechanism[loc]
            
            if mech == 'outside-in' :
                outside_in[i] = True
            if mech == 'inside-out' :
                inside_out[i] = True
            if mech == 'uniform' :
                uniform[i] = True
            if mech == 'ambiguous' :
                ambiguous[i] = True
    
    with h5py.File('TNG50-1/TNG50-1_99_mechanism.hdf5', 'w') as hf :
        add_dataset(hf, outside_in, 'outside-in')
        add_dataset(hf, inside_out, 'inside-out')
        add_dataset(hf, uniform, 'uniform')
        add_dataset(hf, ambiguous, 'ambiguous')
    
    return

def determine_limits_radial(mains, los, meds, his) :
    
    # convert lists to arrays, and flatten those arrays
    mains = np.array(mains).flatten()
    los = np.array(los).flatten()
    meds = np.array(meds).flatten()
    his = np.array(his).flatten()
    
    # create array of everything
    full = np.concatenate((mains, los, meds, his))
    
    # mask out zeros, which will create issues when taking the logarithm
    full[full == 0.0] = np.nan
    
    # find appropriate limits based on the extremes
    # ymin = np.power(10, np.floor(np.log10(np.nanmin(full))))
    # ymax = np.power(10, np.ceil(np.log10(np.nanmax(full))))
    ymin = np.nanmin(full)/2
    ymax = 2*np.nanmax(full)
    
    return ymin, ymax

def determine_limits_spatial(histograms) :
    
    # convert lists to arrays, and create array of everything, flattened
    full = np.array(histograms).flatten()
    
    # mask zeros
    full[full == 0.0] = np.nan
    
    # find appropriate limits based on the 3 sigma percentiles
    # vmin, vmax = np.nanmin(full), np.nanmax(full)
    vmin, vmax = np.nanpercentile(full, [1, 99])
    
    return vmin, vmax

def determine_UVK_contours(UVK) :
    
    # define basic information for the UVK contours
    UVK_edges_x = np.linspace(0.5, 3.4, 21)
    UVK_edges_y = np.linspace(-0.5, 1.75, 21)
    UVK_x_centers = UVK_edges_x[:-1] + np.diff(UVK_edges_x)/2
    UVK_y_centers = UVK_edges_y[:-1] + np.diff(UVK_edges_y)/2
    UVK_X_cent, UVK_Y_cent = np.meshgrid(UVK_x_centers, UVK_y_centers)
    
    UVK_z0 = UVK[:, -1, :]
    UVK_xs = UVK_z0[:, 1] - UVK_z0[:, 2]
    UVK_ys = UVK_z0[:, 0] - UVK_z0[:, 1]
    
    UVK_hist, _, _ = np.histogram2d(UVK_xs, UVK_ys, bins=(UVK_edges_x,
                                                          UVK_edges_y))
    UVK_contour = gaussian_filter(UVK_hist.T, 0.4)
    
    UVK_vals = np.sort(UVK_hist.flatten())
    UVK_vals = UVK_vals[UVK_vals > 0]
    UVK_levels = np.percentile(UVK_vals, [16, 50, 84, 95, 99])
    
    return UVK_X_cent, UVK_Y_cent, UVK_contour, UVK_levels

def radial_plot_info(subIDs, logM, SFMS, snap, q_subID, q_mass,
                     helper_file, nelson2021version=True) :
    
    # open the helper file to access the profiles
    with h5py.File(helper_file, 'r') as hf :
        SFR_profiles = hf['SFR_profiles'][:]
        mass_profiles = hf['mass_profiles'][:]
        area_profiles = hf['area_profiles'][:]
    
    # get values at the snapshot
    subIDs_at_snap = subIDs[:, snap]
    logM_at_snap = logM[:, snap]
    SFMS_at_snap = SFMS[:, snap]
    
    SFR_profiles_at_snap = SFR_profiles[:, snap, :]
    mass_profiles_at_snap = mass_profiles[:, snap, :]
    area_profiles_at_snap = area_profiles[:, snap, :]
    
    # create a mask for the SFMS galaxy masses at that snapshot
    SFMS_at_snap_masses_mask = np.where(SFMS_at_snap > 0,
                                        logM_at_snap, False)
    
    # find galaxies in a similar mass range as the galaxy, but that
    # are on the SFMS at that snapshot
    mass_bin = determine_mass_bin_indices(SFMS_at_snap_masses_mask,
        q_mass, hw=0.1, minNum=50)
    
    # mask the profiles to the control sample
    control_SFR_profiles = SFR_profiles_at_snap[mass_bin]
    control_mass_profiles = mass_profiles_at_snap[mass_bin]
    control_area_profiles = area_profiles_at_snap[mass_bin]
    
    # find the location of the quenched galaxy
    loc = np.where(subIDs_at_snap == q_subID)[0][0]
    
    if nelson2021version :
        # sSFR in each annulus, units of yr^-1
        main = SFR_profiles_at_snap[loc]/mass_profiles_at_snap[loc]
        comparison = control_SFR_profiles/control_mass_profiles
    else :
        # SFR surface density in each annulus, units of Mdot/yr/kpc^2
        main = SFR_profiles_at_snap[loc]/area_profiles_at_snap[loc]
        comparison = control_SFR_profiles/control_area_profiles
    
    # get the percentiles
    lo, med, hi = np.nanpercentile(comparison, [16, 50, 84], axis=0)
    
    return main, lo, med, hi

def set_basic_info_for_plots() :
    
    # define the center points of the radial bins
    radial_edges = np.linspace(0, 5, 21)
    mids = []
    for start, end in zip(radial_edges, radial_edges[1:]) :
        mids.append(0.5*(start + end))
    
    # define basic information for the 2D histograms
    spatial_edges = np.linspace(-5, 5, 61)
    XX, YY = np.meshgrid(spatial_edges, spatial_edges)
    
    hist_centers = spatial_edges[:-1] + np.diff(spatial_edges)/2
    X_cent, Y_cent = np.meshgrid(hist_centers, hist_centers)
    
    return radial_edges, mids, spatial_edges, XX, YY, X_cent, Y_cent

def spatial_plot_info(time, snap, mpbsubID, center, Re, edges, delta_t,
                      nelson2021version=True, sfr_map=False) :
    
    # get all particles
    (gas_masses, gas_sfrs, gas_coords, star_ages, star_gfm, star_masses,
     star_coords) = get_rotation_input('TNG50-1', 99, snap, mpbsubID)
    
    # determine the rotation matrix
    rot = rotation_matrix_from_MoI_tensor(calculate_MoI_tensor(
        gas_masses, gas_sfrs, gas_coords, star_ages, star_masses, star_coords,
        Re, center))
    
    # reproject the coordinates using the face-on projection
    dx, dy, dz = np.matmul(np.asarray(rot['face-on']), (star_coords-center).T)
    
    # don't project using face-on version
    # dx, dy, dz = (star_coords - center).T
    
    # get the SF particles
    _, sf_masses, sf_dx, sf_dy, sf_dz = get_sf_particle_positions(
        star_ages, star_gfm, dx, dy, dz, time, delta_t=delta_t)
    
    # create 2D histograms of the particles and SF particles
    hh, _, _ = np.histogram2d(dx/Re, dy/Re, bins=(edges, edges),
                              weights=star_gfm)
    hh = hh.T
    
    hh_sf, _, _ = np.histogram2d(sf_dx/Re, sf_dy/Re, bins=(edges, edges),
                                 weights=sf_masses)
    hh_sf = hh_sf.T
    
    # determine the area of a single pixel
    area = np.square((edges[1] - edges[0])*Re)
    
    if nelson2021version :
        hist = hh_sf/(delta_t.to(u.yr).value)/hh
    else :
        hist = hh_sf/(delta_t.to(u.yr).value)/area
    
    if sfr_map :
        hist = hh_sf/(delta_t.to(u.yr).value)
    
    # set up the contours
    # adapted from https://stackoverflow.com/questions/26351621
    vals = np.sort(hh.flatten())
    vals = vals[vals > 0]
    levels = np.percentile(vals, [50, 84, 95, 99])
    
    return hist, gaussian_filter(hh, 0.6), levels
