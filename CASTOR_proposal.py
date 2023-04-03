
from os.path import exists
import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
import astropy.units as u
import h5py
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from pypdf import PdfWriter

from core import (add_dataset, bsPath, determine_mass_bin_indices, find_nearest,
                  get, get_mpb_values, get_particles, get_particle_positions,
                  get_quenched_data, get_rotation_input,
                  get_sf_particles, get_sf_particle_positions)
import plotting as plt
from projection import calculate_MoI_tensor, rotation_matrix_from_MoI_tensor

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# from slack import send_message

def comp_prop_plots_for_sample(simName='TNG50-1', snapNum=99, hw=0.1, minNum=50) :
    
    # define the input and output directories, and the helper file
    inDir = bsPath(simName)
    helper_file = inDir + '/{}_{}_comprehensive_plots_helper.hdf5'.format(
        simName, snapNum)
    outDir = inDir + '/figures/comprehensive_plots/'
    
    # write an empty file which will hold all the computed radial profiles
    if not exists(helper_file) :
        hf = h5py.File(helper_file, 'w')
        hf.close()
    
    # get relevant information for the general sample, and quenched systems
    (snapshots, redshifts, times, subIDs, logM, Re, centers, UVK, SFHs, SFMS,
     q_subIDfinals, q_subIDs, q_logM, q_SFHs, q_Re, q_centers, q_UVK, q_primary,
     q_cluster, q_hm_group, q_lm_group, q_field, q_lo_SFH, q_hi_SFH,
     q_ionsets, q_tonsets, q_iterms, q_tterms) = get_quenched_data(
         simName=simName, snapNum=snapNum)
    
    st = 0
    end = 1577
    # loop through the quenched galaxies in the sample
    for i, (q_subfin, q_subID, q_logM, q_SFH, q_rad, q_cent, q_UVK_ind, q_lo, q_hi,
         q_ionset, q_tonset, q_iterm, q_tterm) in enumerate(zip(q_subIDfinals[st:end],
             q_subIDs[st:end], q_logM[st:end], q_SFHs[st:end], q_Re[st:end],
             q_centers[st:end], q_UVK[st:end], q_lo_SFH[st:end], q_hi_SFH[st:end],
             q_ionsets[st:end], q_tonsets[st:end], q_iterms[st:end], q_tterms[st:end])) :
        
        print('{}/1576 - attempting subID {}'.format(i, q_subfin)) # 1577 total
        
        # work through the snapshots from onset until termination, but using
        # tterm isn't very illustrative, so use alternative times based on the
        # final snapshot being 75% of the quenching mechanism duration
        index_times = np.array([q_tonset,
                                q_tonset + 0.375*(q_tterm - q_tonset),
                                q_tonset + 0.75*(q_tterm - q_tonset)])
        indices = find_nearest(times, index_times)
        
        try :
            outfile = outDir + 'subID_{}.pdf'.format(q_subfin)
            # if not exists(outfile) :
            # create the comprehensive plot for the quenched galaxy
            comprehensive_plot(q_subfin, q_subID, q_logM, q_SFH, q_rad, q_cent,
                               q_UVK_ind, q_lo, q_hi, redshifts, times, subIDs,
                               logM, Re, centers, UVK, SFHs, SFMS, q_ionset,
                               q_tonset, q_iterm, q_tterm, indices, outfile,
                               save=True, nelson2021version=True)
            
            # if i % 100 == 0 :
            #     msg = '{}/1576 - subID {} done'.format(i, q_subfin)
            #     send_message(msg)
            
            # create the proposal plot for the quenched galaxy
            # proposal_plot(q_subfin, q_subID, q_logM, q_rad, q_cent, indices,
            #               redshifts, times, subIDs, logM, Re, centers, SFHs,
            #               SFMS, outfile)
        except :
            # print(q_subfin)
            pass
    
    return

def comprehensive_plot(q_subfin, q_subID, q_logM, q_SFH, q_rad, q_cent,
                       q_UVK_ind, q_lo_SFH, q_hi_SFH, redshifts, times, subIDs,
                       logM, Re, centers, UVK, SFHs, SFMS, q_ionset, q_tonset,
                       q_iterm, q_tterm, snaps, outfile, save=False,
                       nelson2021version=True) :
    
    UVK_X_cent, UVK_Y_cent, UVK_contour, UVK_levels = UVK_contours(UVK)
    
    # get the quenched galaxy's UVK positions
    UVK_snaps = q_UVK_ind[np.concatenate((snaps, [q_iterm]))]
    UVK_snaps_xs = UVK_snaps[:, 1] - UVK_snaps[:, 2]
    UVK_snaps_ys = UVK_snaps[:, 0] - UVK_snaps[:, 1]
    
    # define basic information for the 2D histograms
    spatial_edges = np.linspace(-5, 5, 61)
    XX, YY = np.meshgrid(spatial_edges, spatial_edges)
    
    hist_centers = spatial_edges[:-1] + np.diff(spatial_edges)/2
    X_cent, Y_cent = np.meshgrid(hist_centers, hist_centers)
    
    # define the center points of the radial bins
    radial_edges = np.linspace(0, 5, 21)
    mids = []
    for start, end in zip(radial_edges, radial_edges[1:]) :
        mids.append(0.5*(start + end))
    
    # set which version to use
    if nelson2021version :
        label = r'sSFR (yr$^{-1}$)'
    else :
        label = r'$\log(\dot{\Sigma}_{\rm *, 100~Myr}/{\rm M}_{\odot}~{\rm yr}^{-1}~{\rm kpc}^{-2})$'
    
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
        helper_file = 'TNG50-1/TNG50-1_99_comprehensive_plots_helper.hdf5'
        if exists(helper_file) :
            
            # get the radial profiles for the quenched galaxy and the
            # comparison control sample
            with h5py.File(helper_file, 'r') as hf :
                main = hf['Snapshot_{}/subID_{}_main'.format(
                    snap, q_subID[snap])][:]
                comparison = hf['Snapshot_{}/subID_{}_control'.format(
                    snap, q_subID[snap])][:]
            
            # get the percentiles
            lo, med, hi = np.nanpercentile(comparison, [16, 50, 84], axis=0)
            
            # append those values to the lists
            mains.append(main)
            los.append(lo)
            meds.append(med)
            his.append(hi)
        else :
            save_radial_plot_info(times, subIDs, logM, Re, centers, SFMS,
                snap, q_subID, q_logM, q_rad, q_cent, radial_edges, 100*u.Myr,
                nelson2021version=nelson2021version)
    
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
    
    return

def determine_limits_radial(mains, los, meds, his) :
    
    # convert lists to arrays, and flatten those arrays
    mains = np.array(mains).flatten()
    los = np.array(los).flatten()
    meds = np.array(meds).flatten()
    his = np.array(his).flatten()
    
    # create array of everything
    full = np.concatenate((mains, los, meds, his))
    
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

def determine_radial_profile(simName, snapNum, time, subID, center, snap, Re,
                             edges, delta_t=100*u.Myr, nelson2021version=False) :
    
    # open the corresponding cutouts and get their particles
    ages, masses, rs = get_particles(simName, snapNum, snap, subID, center)
    
    if rs is not None :
        # mask all particles to within 5Re
        rs = rs/Re
        ages, masses, rs = ages[rs <= 5], masses[rs <= 5], rs[rs <= 5]
        
        # find the total mass and area (in kpc^2) in each annulus
        mass_in_annuli, areas = [], []
        for start, end in zip(edges, edges[1:]) :
            mass = np.sum(masses[(rs >= start) & (rs < end)])
            
            if mass == 0.0 :
                mass_in_annuli.append(np.nan)
            else :
                mass_in_annuli.append(mass)
            
            area = np.pi*(np.square(end*Re) - np.square(start*Re))
            areas.append(area)
        mass_in_annuli, areas = np.array(mass_in_annuli), np.array(areas)
        
        # get the SF particles
        _, masses, rs = get_sf_particles(ages, masses, rs, time,
                                         delta_t=delta_t)
        
        # find the SF mass in each annulus
        SF_mass_in_annuli = []
        for start, end in zip(edges, edges[1:]) :
            total_mass = np.sum(masses[(rs >= start) & (rs < end)])
            
            if total_mass == 0.0 :
                SF_mass_in_annuli.append(np.nan)
            else :
                SF_mass_in_annuli.append(total_mass)
        SF_mass_in_annuli = np.array(SF_mass_in_annuli)
        
        delta_t = (delta_t.to(u.yr).value)
        
        if nelson2021version :
            # sSFR in each annulus
            final = SF_mass_in_annuli/delta_t/mass_in_annuli # units of yr^-1
        else :
            # SFR surface density in each annulus
            final = SF_mass_in_annuli/delta_t/areas # units of Mdot/yr/kpc^2
    else :
        final = np.full(20, np.nan)
    
    return final

def merge_comp_plots(simName='TNG50-1', snapNum=99) :
    
    # define the input and output directories and files
    base = bsPath(simName)
    infile = base + '/TNG50-1_99_sample(t).hdf5'
    inDir = base + '/figures/comprehensive_plots/'
    outDir = base + '/figures/comprehensive_plots_merged/'
    
    # get information about the quenched sample
    with h5py.File(infile, 'r') as hf :
        subIDfinals = hf['SubhaloID'][:]
        logM = hf['logM'][:]
        quenched = hf['quenched'][:]
    
    # get the subIDfinals, 
    quenched_subIDs = subIDfinals[quenched]
    quenched_logM_z0 = logM[:, -1][quenched]
    
    # now sort the subIDs and stellar masses
    sort = np.argsort(quenched_logM_z0)
    quenched_subIDs_sorted = quenched_subIDs[sort]
    quenched_logM_z0_sorted = quenched_logM_z0[sort]
    
    # set the mass bin edges to group quenched galaxies of a simiarl mass
    edges = [8.00, 8.25, 8.50, 8.75, 9.00, 9.50, 10.00, 11.00, 13.00]
    # 356, 276, 255, 186, 226, 111, 144, 23 quenched galaxies per bin
    
    # loop over every bin
    for start, end in zip(edges, edges[1:]) :
        mass_bin = ((quenched_logM_z0_sorted >= start) &
                    (quenched_logM_z0_sorted < end))
        
        # create a merged object
        merger = PdfWriter()
        
        # loop over every quenched galaxy in the mass bin
        for subID in quenched_subIDs_sorted[mass_bin] :
            merger.append(inDir + 'subID_{}.pdf'.format(subID))
        
        # write the merged file and close it
        outfile = outDir + 'merged_{:.2f}_{:.2f}.pdf'.format(start, end)
        merger.write(outfile)
        merger.close()
    
    table = Table([quenched_subIDs_sorted, quenched_logM_z0_sorted],
                  names=('subID', 'logM'))
    table.write('comprehensive_plots_visual_inspection.csv', overwrite=False)
    
    return

def proposal_plot(q_subfin, q_subID, q_mass, q_radii, q_center, snaps,
                  redshifts, times, subIDs, logM, Re, centers, SFHs, SFMS,
                  outfile, nelson2021version=True) :
    
    spatial_edges = np.linspace(-5, 5, 61)
    XX, YY = np.meshgrid(spatial_edges, spatial_edges)
    
    # define the center points of the radial bins
    radial_edges = np.linspace(0, 5, 21)
    mids = []
    for start, end in zip(radial_edges, radial_edges[1:]) :
        mids.append(0.5*(start + end))
    
    if nelson2021version :
        label = r'sSFR (yr$^{-1}$)'
        ymin_b, ymax_b = 1e-12, 1e-9
    else :
        label = r'$\log(\dot{\Sigma}_{\rm *, 100~Myr}/{\rm M}_{\odot}~{\rm yr}^{-1}~{\rm kpc}^{-2})$'
        ymin_b, ymax_b = 1e-5, 1
    
    dfs, hists, fwhms, titles = [], [], [], []
    mains, los, meds, his = [], [], [], []
    
    for snap in snaps :
        
        resolution = (0.15*u.arcsec)/cosmo.arcsec_per_kpc_proper(
            redshifts[snap])
        res = resolution/(q_radii[snap]*u.kpc)
        fwhms.append(res)
        
        title = (r'$z = {:.2f}$'.format(redshifts[snap]) + ', ' +
                  r'$\Delta t_{\rm since~onset} = $' +
                  '{:.2f} Gyr'.format(times[snap] - times[snaps][0]))
        titles.append(title)
        
        df, hist = spatial_plot(times[snap], snap, q_subID[snap],
            q_center[snap], q_radii[snap], spatial_edges, 100*u.Myr,
            nelson2021version=nelson2021version)
        dfs.append(df)
        hists.append(hist)
        
        main, lo, med, hi = radial_plot(times, subIDs, logM, Re, centers, SFMS,
            snap, q_subID, q_mass, q_radii, q_center, radial_edges, 100*u.Myr,
            nelson2021version=nelson2021version)
        mains.append(main)
        los.append(lo)
        meds.append(med)
        his.append(hi)
    
    plt.plot_CASTOR_proposal(titles[0], titles[1], titles[2],
                             dfs[0], dfs[1], dfs[2],
                             hists[0], hists[1], hists[2],
                             fwhms[0], fwhms[1], fwhms[2],
                             mids,
                             mains[0], mains[1], mains[2],
                             los[0], los[1], los[2],
                             meds[0], meds[1], meds[2],
                             his[0], his[1], his[2],
                             XX, YY,
                             label=label,
                             xlabel_t=r'$\Delta x$ ($R_{\rm e}$)',
                             ylabel_t=r'$\Delta z$ ($R_{\rm e}$)',
                             xlabel_b=r'$r/R_{\rm e}$',
                             ylabel_b=label,
                             xmin_t=-5, xmax_t=5, ymin_t=-5, ymax_t=5,
                             xmin_b=0, xmax_b=5, ymin_b=ymin_b, ymax_b=ymax_b,
                             outfile=outfile, save=True)
    
    return

def save_radial_plot_info(times, subIDs, logM, Re, centers, SFMS, snap, 
                          q_subID, q_mass, q_radii, q_center, edges, delta_t,
                          nelson2021version=False) :
    
    helper_file = 'TNG50-1/TNG50-1_99_comprehensive_plots_helper.hdf5'
    
    # get values at the snapshot
    time = times[snap]
    subIDs_at_snap = subIDs[:, snap]
    logM_at_snap = logM[:, snap]
    Re_at_snap = Re[:, snap]
    centers_at_snap = centers[:, snap, :]
    SFMS_at_snap = SFMS[:, snap]
    
    # and for the quenched galaxy as well
    q_ID = q_subID[snap]
    mass = q_mass[snap]
    Re = q_radii[snap]
    cent = q_center[snap]
    
    # calculate the radial profile for the quenched galaxy
    main_profile = determine_radial_profile('TNG50-1', 99, time, q_ID, cent,
        snap, Re, edges, delta_t=delta_t, nelson2021version=nelson2021version)
    
    # create a mask for the SFMS galaxy masses at that snapshot
    SFMS_at_snap_masses_mask = np.where(SFMS_at_snap > 0,
                                        logM_at_snap, False)
    
    # find galaxies in a similar mass range as the galaxy, but that
    # are on the SFMS at that snapshot
    mass_bin = determine_mass_bin_indices(SFMS_at_snap_masses_mask,
        mass, hw=0.1, minNum=50)
    
    # create an empty array that will hold all of the radial profiles
    profiles = np.full((len(subIDs[mass_bin]), 20), np.nan)
    
    # loop over all the comparison galaxies and populate into the array
    control_IDs = subIDs_at_snap[mass_bin]
    control_logM = logM_at_snap[mass_bin]
    control_Re = Re_at_snap[mass_bin]
    control_centers = centers_at_snap[mass_bin]
    
    for i, (ID, mass, radius, center) in enumerate(zip(control_IDs,
        control_logM, control_Re, control_centers)) :
        profiles[i, :] = determine_radial_profile('TNG50-1', 99, time, ID,
            center, snap, radius, edges, delta_t=delta_t,
            nelson2021version=nelson2021version)
    
    # save the determined radial profiles to the helper file
    with h5py.File(helper_file, 'a') as hf :
        add_dataset(hf, main_profile, 'Snapshot_{}/subID_{}_main'.format(
            snap, q_ID))
        add_dataset(hf, profiles, 'Snapshot_{}/subID_{}_control'.format(
            snap, q_ID))
    
    return

def spatial_plot_info(time, snap, mpbsubID, center, Re, edges, delta_t,
                      nelson2021version=False, fast=True) :
    
    # get all particles
    (gas_masses, gas_sfrs, gas_coords, star_ages, star_gfm, star_masses,
     star_coords) = get_rotation_input('TNG50-1', 99, snap, mpbsubID)
    
    # determine the rotation matrix
    rot = rotation_matrix_from_MoI_tensor(calculate_MoI_tensor(
        gas_masses, gas_sfrs, gas_coords, star_ages, star_masses, star_coords,
        Re, center))
    
    # reproject the coordinates using the face-on projection
    dx, dy, dz = np.matmul(np.asarray(rot['face-on']), (star_coords-center).T)
    
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
    
    # set up the contours
    # adapted from https://stackoverflow.com/questions/26351621
    vals = np.sort(hh.flatten())
    vals = vals[vals > 0]
    levels = np.percentile(vals, [50, 84, 95, 99])
    
    return hist, gaussian_filter(hh, 0.6), levels

def UVK_contours(UVK) :
    
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
