
from os.path import exists
import numpy as np

from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
import astropy.units as u
import h5py
from scipy.ndimage import gaussian_filter1d, gaussian_filter

from core import (bsPath, determine_mass_bin_indices, find_nearest, get,
                  get_mpb_values, get_particles, get_particle_positions,
                  get_quenched_data, get_rotation_input,
                  get_sf_particles, get_sf_particle_positions)
import plotting as plt
from projection import calculate_MoI_tensor, rotation_matrix_from_MoI_tensor

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def determine_sigmadot(simName, snapNum, time, subID, center, snap, Re, edges,
                       delta_t=100*u.Myr, nelson2021version=False) :
    
    # open the corresponding cutouts and get their particles
    ages, masses, rs = get_particles(simName, snapNum, snap, subID, center)
    
    if rs is not None :
        # mask all particles to within 5Re
        rs = rs/Re
        ages, masses, rs = ages[rs <= 5], masses[rs <= 5], rs[rs <= 5]
        
        # find the total mass and area (in kpc^2) in each annulus
        mass_in_annuli, areas = [], []
        for start, end in zip(edges, edges[1:]) :
            mass = np.sum(masses[(rs > start) & (rs <= end)])
            mass_in_annuli.append(mass)
            
            area = np.pi*(np.square(end*Re) - np.square(start*Re))
            areas.append(area)
        mass_in_annuli, areas = np.array(mass_in_annuli), np.array(areas)
        
        # get the SF particles
        _, masses, rs = get_sf_particles(ages, masses, rs, time, delta_t=delta_t)
        
        # find the SF mass in each annulus
        SF_mass_in_annuli = []
        for start, end in zip(edges, edges[1:]) :
            total_mass = np.sum(masses[(rs > start) & (rs <= end)])
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
        final = np.nan
    
    return final

def comp_prop_plots_for_sample(simName='TNG50-1', snapNum=99, hw=0.1, minNum=50) :
    
    # get relevant information for the general sample, and quenched systems
    (snapshots, redshifts, times, subIDs, logM, Re, centers, UVK, SFHs, SFMS,
     q_subIDfinals, q_subIDs, q_logM, q_SFHs, q_Re, q_centers, q_UVK, q_primary,
     q_cluster, q_hm_group, q_lm_group, q_field, q_lo_SFH, q_hi_SFH,
     q_ionsets, q_tonsets, q_iterms, q_tterms) = get_quenched_data(
         simName=simName, snapNum=snapNum)
    
    st = 6 # 0
    end = 7 # 1577
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
        
        '''
        # check if the cutout files exist
        exist_array = []
        files = []
        for snap, subID in zip(indices, q_subID[indices]) :
            filename = 'F:/TNG50-1/mpb_cutouts_099/cutout_{}_{}.hdf5'.format(snap, subID)
            files.append(filename)
            file_exists = exists(filename)
            if not file_exists :
                missing += 1
            exist_array.append(file_exists)
        proceed = np.all(exist_array)
        '''
        
        outfile = 'TNG50-1/figures/comprehensive_plots/subID_{}.pdf'.format(q_subfin)
        if not exists(outfile) :
            # create the comprehensive plot for the quenched galaxy
            comprehensive_plot(q_subfin, q_subID, q_logM, q_SFH, q_rad, q_cent,
                               q_UVK_ind, q_lo, q_hi,
                               redshifts, times, subIDs, logM, Re, centers, UVK,
                               SFHs, SFMS, q_ionset, q_tonset, q_iterm, q_tterm,
                               indices, outfile, save=False)
        
        # create the proposal plot for the quenched galaxy
        # proposal_plot(q_subfin, q_subID, q_logM, q_rad, q_cent, indices,
        #               redshifts, times, subIDs, logM, Re, centers, SFHs,
        #               SFMS, outfile)
    
    return

def comprehensive_plot(q_subfin, q_subID, q_logM, q_SFH, q_rad, q_cent, q_UVK_ind,
                       q_lo_SFH, q_hi_SFH,
                       redshifts, times, subIDs, logM, Re, centers, UVK,
                       SFHs, SFMS, q_ionset, q_tonset, q_iterm, q_tterm,
                       snaps, outfile, nelson2021version=True, save=False) :
    
    UVK_z0 = UVK[:, -1, :]
    UVK_xs = UVK_z0[:, 1] - UVK_z0[:, 2]
    UVK_ys = UVK_z0[:, 0] - UVK_z0[:, 1]
    
    UVK_snaps = q_UVK_ind[np.concatenate((snaps, [q_iterm]))]
    UVK_snaps_xs = UVK_snaps[:, 1] - UVK_snaps[:, 2]
    UVK_snaps_ys = UVK_snaps[:, 0] - UVK_snaps[:, 1]
    
    table = Table([UVK_xs, UVK_ys], names=('V-K', 'U-V'))
    UVK_df = table.to_pandas()
    
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
    
    dfs, hists, titles = [], [], []
    mains, los, meds, his = [], [], [], []
    
    for snap in snaps :
        
        title = (r'$z = {:.2f}$'.format(redshifts[snap]) + ', ' +
                  r'$\Delta t_{\rm since~onset} = $' +
                  '{:.2f} Gyr'.format(times[snap] - times[snaps][0]))
        titles.append(title)
        
        df, hist = spatial_plot(times[snap], snap, q_subID[snap], q_cent[snap],
                                q_rad[snap], spatial_edges, 100*u.Myr,
                                nelson2021version=nelson2021version)
        dfs.append(df)
        hists.append(hist)
        
        main, lo, med, hi = radial_plot(times, subIDs, logM, Re, centers, SFMS, snap,
                                        q_subID, q_logM, q_rad, q_cent,
                                        radial_edges, 100*u.Myr,
                                        nelson2021version=nelson2021version)
        mains.append(main)
        los.append(lo)
        meds.append(med)
        his.append(hi)
    
    sm = gaussian_filter1d(q_SFH, 2)
    lo_sm = gaussian_filter1d(q_lo_SFH, 2)
    hi_sm = gaussian_filter1d(q_hi_SFH, 2)
    SMH = gaussian_filter1d(q_logM, 2) # stellar mass history
    
    mtitle = (r'subID$_{z = 0}$' + ' {}'.format(q_subfin) +
              r', $\log{(M/{\rm M}_{\odot})}_{z = 0} = $' 
              + '{:.2f}'.format(q_logM[-1]))
    
    plt.plot_comprehensive_plot(
        titles[0], titles[1], titles[2],
        dfs[0], dfs[1], dfs[2],
        hists[0], hists[1], hists[2],
        mids,
        mains[0], mains[1], mains[2],
        los[0], los[1], los[2],
        meds[0], meds[1], meds[2],
        his[0], his[1], his[2],
        XX, YY,
        times, sm, lo_sm, hi_sm, q_tonset, q_tterm,
        times[snaps[1]], times[snaps[2]],
        SMH,
        UVK_df, UVK_snaps_xs, UVK_snaps_ys,
        xlabel_t=r'$\Delta x$ ($R_{\rm e}$)',
        ylabel_t=r'$\Delta z$ ($R_{\rm e}$)',
        xlabel_b=r'$r/R_{\rm e}$',
        ylabel_b=label,
        vlabel=label,
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
        xmin_UVK=1.75, xmax_UVK=3.4, ymin_UVK=-0.5, ymax_UVK=1.75,
        outfile=outfile, save=save)
    
    return

def proposal_plot(q_subfin, q_subID, q_mass, q_radii, q_center, snaps,
                  redshifts, times, subIDs, logM, Re, centers, SFHs, SFMS, outfile,
                  nelson2021version=True) :
    
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
        
        resolution = (0.15*u.arcsec)/cosmo.arcsec_per_kpc_proper(redshifts[snap])
        res = resolution/(q_radii[snap]*u.kpc)
        fwhms.append(res)
        
        title = (r'$z = {:.2f}$'.format(redshifts[snap]) + ', ' +
                  r'$\Delta t_{\rm since~onset} = $' +
                  '{:.2f} Gyr'.format(times[snap] - times[snaps][0]))
        titles.append(title)
        
        df, hist = spatial_plot(times[snap], snap, q_subID[snap], q_center[snap],
                                q_radii[snap], spatial_edges, 100*u.Myr,
                                nelson2021version=nelson2021version)
        dfs.append(df)
        hists.append(hist)
        
        main, lo, med, hi = radial_plot(times, subIDs, logM, Re, centers, SFMS, snap,
                                        q_subID, q_mass, q_radii, q_center,
                                        radial_edges, 100*u.Myr,
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
                             outfile=outfile, save=False)
    
    return

def radial_plot(times, subIDs, logM, Re, centers, SFMS, snap,
                q_subID, q_mass, q_radii, q_center,
                edges, delta_t, nelson2021version=False) :
    
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
    
    # calculate the SFR surface density for the quenched galaxy
    SFR_density = determine_sigmadot('TNG50-1', 99, time, q_ID, cent, snap, Re, edges,
                                     delta_t=delta_t, nelson2021version=nelson2021version)
    
    # create a mask for the SFMS galaxy masses at that snapshot
    SFMS_at_snap_masses_mask = np.where(SFMS_at_snap > 0,
                                        logM_at_snap, False)
    
    # find galaxies in a similar mass range as the galaxy, but that
    # are on the SFMS at that snapshot
    mass_bin = determine_mass_bin_indices(SFMS_at_snap_masses_mask,
        mass, hw=0.1, minNum=50)
    
    # create an empty array that will hold all of the SFR surface densities
    SFR_densities = np.full((len(subIDs[mass_bin]), 20), np.nan)
    
    # loop over all the comparison galaxies and populate into the array
    control_IDs = subIDs_at_snap[mass_bin]
    control_logM = logM_at_snap[mass_bin]
    control_Re = Re_at_snap[mass_bin]
    control_centers = centers_at_snap[mass_bin]
    
    for i, (ID, mass, radius, center) in enumerate(zip(control_IDs,
        control_logM, control_Re, control_centers)) :
        SFR_densities[i, :] = determine_sigmadot('TNG50-1', 99, time,
            ID, center, snap, radius, edges, delta_t=delta_t,
            nelson2021version=nelson2021version)
    
    lo, med, hi = np.nanpercentile(SFR_densities, [16, 50, 84], axis=0)
    
    return SFR_density, lo, med, hi

def spatial_plot(time, snap, mpbsubID, center, Re, edges, delta_t,
                 nelson2021version=False, fast=True) :
    
    # get all particles
    ages, masses, dx, dy, dz = get_particle_positions('TNG50-1',
        99, snap, mpbsubID, center)
    
    # get the SF particles
    _, sf_masses, sf_dx, sf_dy, sf_dz = get_sf_particle_positions(
        ages, masses, dx, dy, dz, time, delta_t=delta_t)
    
    # create 2D histograms of the particles and SF particles
    hh, _, _ = np.histogram2d(dx/Re, dz/Re, bins=(edges, edges))
    hh = hh.T
    
    hh_sf, _, _ = np.histogram2d(sf_dx/Re, sf_dz/Re, bins=(edges, edges))
    hh_sf = hh_sf.T
    
    # determine the area of a single pixel
    area = np.square((edges[1] - edges[0])*Re)
    
    if nelson2021version :
        hist = hh_sf/(delta_t.to(u.yr).value)/hh
    else :
        hist = hh_sf/(delta_t.to(u.yr).value)/area
    
    # define an array of random numbers to select ~1000 particles
    np.random.seed(0)
    length = len(dx)
    rand = np.random.random(length)
    
    lim = 1000/length
    
    table = Table([rand, dx/Re, dz/Re, masses],
                  names=('rand', 'dx', 'dz', 'masses'))
    if fast :
        table = table[table['rand'] < lim] # use ~1000 of the particles
    df = table.to_pandas()
    
    return df, hist
