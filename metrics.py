
from os.path import exists
import numpy as np

import astropy.units as u
from astropy.table import Table
import h5py
import pandas as pd
import plotly.express as px
import plotly.io as pio
from pypdf import PdfWriter
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.optimize import curve_fit
# from scipy.stats import anderson_ksamp, ks_2samp
# from scipy.interpolate import CubicSpline

# from concatenate import concat_horiz
from core import (add_dataset, find_nearest, get_late_data,
                  get_particle_positions, get_particles, get_sf_particles,
                  vertex)
import plotting as plt

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def calculate_required_morphological_metrics(delta_t=100*u.Myr, version='2D',
                                             quenched_pop=True, threshold=-10.5,
                                             slope=1) :
    
    # define the input and output files
    infile = 'TNG50-1/TNG50-1_99_sample(t).hdf5'
    profiles_file = 'TNG50-1/TNG50-1_99_massive_radial_profiles(t).hdf5'
    C_SF_outfile = 'TNG50-1/TNG50-1_99_massive_C_SF(t).hdf5'
    R_SF_outfile = 'TNG50-1/TNG50-1_99_massive_R_SF(t).hdf5'
    Rinner_outfile = 'TNG50-1/TNG50-1_99_massive_Rinner(t)_revised_{}_+{}.hdf5'.format(
        threshold, slope)
    Router_outfile = 'TNG50-1/TNG50-1_99_massive_Router(t)_revised_{}_-{}.hdf5'.format(
        threshold, slope)
    
    # get relevant information for the general sample
    with h5py.File(infile, 'r') as hf :
        times = hf['times'][:] # for determining active SF populations
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        Res = hf['Re'][:]
        centers = hf['centers'][:]
    
    # select massive galaxies
    mask = (logM[:, -1] >= 9.5) # 1666 entries, but len(mask) = 8260
    
    # mask all attributes to select only the massive population
    subIDs = subIDs[mask]     # (1666, 100)
    logM = logM[mask]         # (1666, 100)
    Res = Res[mask]           # (1666, 100)
    centers = centers[mask]   # (1666, 100, 3)
    
    # use presaved arrays to determine exactly which galaxies need to have
    # their morphological metrics computed
    unique_quenched = np.load('TNG50-1/locations_mask_quenched.npy') # (1666, 100), sum 4695
    unique_sf = np.load('TNG50-1/locations_mask_control_sf.npy')     # (1666, 100), sum 27763
    
    # write empty files which will hold the morphological metrics
    if not exists(C_SF_outfile) :
        with h5py.File(C_SF_outfile, 'w') as hf :
            add_dataset(hf, np.full((1666, 100), np.nan), 'C_SF')
    if not exists(R_SF_outfile) :
        with h5py.File(R_SF_outfile, 'w') as hf :
            add_dataset(hf, np.full((1666, 100), np.nan), 'R_SF')
    if not exists(Rinner_outfile) :
        with h5py.File(Rinner_outfile, 'w') as hf :
            add_dataset(hf, np.full((1666, 100), np.nan), 'Rinner')
    if not exists(Router_outfile) :
        with h5py.File(Router_outfile, 'w') as hf :
            add_dataset(hf, np.full((1666, 100), np.nan), 'Router')
    
    # read in the required arrays
    with h5py.File(C_SF_outfile, 'r') as hf :
        # 4686 non-NaN quenchers, expect 4695 -> 9 missing
        C_SF_array = hf['C_SF'][:] # 26339 non-NaN SFers, expect 27763 -> 1424 missing
    with h5py.File(R_SF_outfile, 'r') as hf :
        # 4686 non-NaN quenchers -> 9 missing
        R_SF_array = hf['R_SF'][:] # 26339 non-NaN SFers -> 1424 missing
    with h5py.File(Rinner_outfile, 'r') as hf :
        # 4692 non-NaN quenchers -> 3 missing
        Rinner_array = hf['Rinner'][:] # 26339 non-NaN SFers -> 1424 missing
    with h5py.File(Router_outfile, 'r') as hf :
        # 4692 non-NaN quenchers -> 3 missing
        Router_array = hf['Router'][:] # 26339 non-NaN SFers -> 1424 missing
    with h5py.File(profiles_file, 'r') as hf :
        radial_bin_centers = hf['midpoints'][:] # units of Re
        sSFR_profiles = hf['sSFR_profiles'][:] # shape (1666, 100, 20)
    
    if quenched_pop :
        # work on the quenched population first
        for row in range(1666) :
            for snap in range(100) :
                if unique_quenched[row, snap] :
                    # compute Rinner and Router
                    Rinner = calculate_Rinner(radial_bin_centers,
                                              sSFR_profiles[row, snap],
                                              threshold=threshold, slope=slope)
                    Router = calculate_Router(radial_bin_centers,
                                              sSFR_profiles[row, snap],
                                              threshold=threshold, slope=-slope)
                    
                    # get all particles, for the other metrics
                    ages, masses, dx, dy, dz = get_particle_positions(
                        'TNG50-1', 99, snap, subIDs[row, snap],
                        centers[row, snap])
                    
                    # only proceed if the ages, masses, and distances are intact
                    if ((ages is not None) and (masses is not None) and
                        (dx is not None) and (dy is not None) and
                        (dz is not None)) :
                        
                        if version == '2D' :
                            rs = np.sqrt(np.square(dx) + np.square(dy))
                        else :
                            rs = np.sqrt(np.square(dx) + np.square(dy) +
                                         np.square(dz))
                        
                        # get the SF particles
                        _, sf_masses, sf_rs = get_sf_particles(ages, masses,
                            rs, times[snap], delta_t=delta_t)
                        
                        # compute C_SF and R_SF
                        C_SF = calculate_C_SF(sf_masses, sf_rs, Res[row, snap])
                        R_SF = calculate_R_SF(masses, rs, sf_masses, sf_rs)
                    else :
                        C_SF = np.nan
                        R_SF = np.nan
                    
                    # don't replace existing values
                    if np.isnan(C_SF_array[row, snap]) :
                        with h5py.File(C_SF_outfile, 'a') as hf :
                            hf['C_SF'][row, snap] = C_SF
                    if np.isnan(R_SF_array[row, snap]) :
                        with h5py.File(R_SF_outfile, 'a') as hf :
                            hf['R_SF'][row, snap] = R_SF
                    if np.isnan(Rinner_array[row, snap]) :
                        with h5py.File(Rinner_outfile, 'a') as hf :
                            hf['Rinner'][row, snap] = Rinner
                    if np.isnan(Router_array[row, snap]) :
                        with h5py.File(Router_outfile, 'a') as hf :
                            hf['Router'][row, snap] = Router
            print('{} done'.format(row))
    else :
        # work on the control SF population next
        for row in range(1666) :
            for snap in range(100) :
                if unique_sf[row, snap] :
                    # compute Rinner and Router
                    Rinner = calculate_Rinner(radial_bin_centers,
                                              sSFR_profiles[row, snap],
                                              threshold=threshold, slope=slope)
                    Router = calculate_Router(radial_bin_centers,
                                              sSFR_profiles[row, snap],
                                              threshold=threshold, slope=-slope)
                    
                    # get all particles, for the other metrics
                    ages, masses, dx, dy, dz = get_particle_positions(
                        'TNG50-1', 99, snap, subIDs[row, snap],
                        centers[row, snap])
                    
                    # only proceed if the ages, masses, and distances are intact
                    if ((ages is not None) and (masses is not None) and
                        (dx is not None) and (dy is not None) and
                        (dz is not None)) :
                        
                        if version == '2D' :
                            rs = np.sqrt(np.square(dx) + np.square(dy))
                        else :
                            rs = np.sqrt(np.square(dx) + np.square(dy) +
                                         np.square(dz))
                        
                        # get the SF particles
                        _, sf_masses, sf_rs = get_sf_particles(ages, masses,
                            rs, times[snap], delta_t=delta_t)
                        
                        # compute C_SF and R_SF
                        C_SF = calculate_C_SF(sf_masses, sf_rs, Res[row, snap])
                        R_SF = calculate_R_SF(masses, rs, sf_masses, sf_rs)
                    else :
                        C_SF = np.nan
                        R_SF = np.nan
                    
                    # don't replace existing values
                    if np.isnan(C_SF_array[row, snap]) :
                        with h5py.File(C_SF_outfile, 'a') as hf :
                            hf['C_SF'][row, snap] = C_SF
                    if np.isnan(R_SF_array[row, snap]) :
                        with h5py.File(R_SF_outfile, 'a') as hf :
                            hf['R_SF'][row, snap] = R_SF
                    if np.isnan(Rinner_array[row, snap]) :
                        with h5py.File(Rinner_outfile, 'a') as hf :
                            hf['Rinner'][row, snap] = Rinner
                    if np.isnan(Router_array[row, snap]) :
                        with h5py.File(Router_outfile, 'a') as hf :
                            hf['Router'][row, snap] = Router
            print('{} done'.format(row))
    
    return

def calculate_required_radial_profiles(simName='TNG50-1', snapNum=99,
                                       delta_t=100*u.Myr, quenched_pop=True) :
    
    # define the input and output files
    infile = 'TNG50-1/TNG50-1_99_sample(t).hdf5'
    outfile = 'TNG50-1/TNG50-1_99_massive_radial_profiles(t).hdf5'
    
    # get relevant information for the general sample
    with h5py.File(infile, 'r') as hf :
        times = hf['times'][:] # for determining active SF populations
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        Res = hf['Re'][:]
        centers = hf['centers'][:]
    
    # select massive galaxies
    mask = (logM[:, -1] >= 9.5) # 1666 entries, but len(mask) = 8260
    
    # mask all attributes to select only the massive population
    subIDs = subIDs[mask]     # (1666, 100)
    logM = logM[mask]         # (1666, 100)
    Res = Res[mask]           # (1666, 100)
    centers = centers[mask]   # (1666, 100, 3)
    
    # use presaved arrays to determine exactly which galaxies need to have
    # their radial profiles computed
    unique_quenched = np.load('TNG50-1/locations_mask_quenched.npy') # (1666, 100), sum 4695
    unique_sf = np.load('TNG50-1/locations_mask_control_sf.npy')     # (1666, 100), sum 27763
    
    # write an empty file which will hold the computed radial profiles
    if not exists(outfile) :
        
        # define the edges and center points of the radial bins
        edges = np.linspace(0, 5, 21)
        mids = []
        for start, end in zip(edges, edges[1:]) :
            mids.append(0.5*(start + end))
        
        # populate the helper file to be filled later
        with h5py.File(outfile, 'w') as hf :
            add_dataset(hf, edges, 'edges')
            add_dataset(hf, np.array(mids), 'midpoints')
            add_dataset(hf, np.full((1666, 100, 20), np.nan), 'sSFR_profiles')
    
    # read in the required arrays
    with h5py.File(outfile, 'r') as hf :
        edges = hf['edges'][:]
        sSFR_profiles = hf['sSFR_profiles'][:]
    
    if quenched_pop :
        # work on the quenched population first
        for row in range(1666) :
            for snap in range(100) :
                if unique_quenched[row, snap] :
                    
                    # get the SFR and mass profiles
                    SFR_profile, mass_profile, _, _, _ = determine_radial_profiles(
                        'TNG50-1', 99, snap, times[snap], subIDs[row, snap],
                        Res[row, snap], centers[row, snap], edges, 20,
                        delta_t=delta_t)
                    
                    # don't replace existing values
                    if np.all(np.isnan(sSFR_profiles[row, snap, :])) :
                        # produces 93853 non-NaN values, 47 less than the
                        # expected 4695*20 = 93900
                        with h5py.File(outfile, 'a') as hf :
                            hf['sSFR_profiles'][row, snap, :] = SFR_profile/mass_profile
            print('{} done'.format(row))
    else :
        # work on the control SF population next
        for row in range(1666) :
            for snap in range(100) :
                if unique_sf[row, snap] :
                    
                    # get the SFR and mass profiles
                    SFR_profile, mass_profile, _, _, _ = determine_radial_profiles(
                        'TNG50-1', 99, snap, times[snap], subIDs[row, snap],
                        Res[row, snap], centers[row, snap], edges, 20,
                        delta_t=delta_t)
                    
                    # don't replace existing values
                    if np.all(np.isnan(sSFR_profiles[row, snap, :])) :
                        # produces 526755 non-NaN values, 28505 less than the
                        # expected 27763*20 = 555260
                        with h5py.File(outfile, 'a') as hf :
                            hf['sSFR_profiles'][row, snap, :] = SFR_profile/mass_profile
            print('{} done'.format(row))
    
    return

def calculate_C_SF(sf_masses, sf_rs, Re) :
    
    sf_mass_within_1kpc, _, sf_mass = compute_C_SF(sf_masses, sf_rs, Re)
    
    return sf_mass_within_1kpc/sf_mass

def calculate_R_SF(masses, rs, sf_masses, sf_rs) :
    
    _, R50, _, _, sf_R50, _ = compute_R_SF(masses, rs, sf_masses, sf_rs)
    
    return sf_R50/R50

def calculate_Rinner(bin_centers, profile, threshold=-10.5, slope=1) :
    
    if np.all(profile == 0) :
        Rinner = -99
    elif np.all(np.isnan(profile)) :
        Rinner = -49
    else :
        # take the logarithm of the profile for easier manipulation
        profile = np.log10(profile)
        # plt.plot_simple_dumb(bin_centers, profile, xlabel=r'$r/R_{\rm e}$',
        #                      ylabel='sSFR')
        
        # for NaNs, set the value in each bin to be below the minimum valid
        # value appearing in the profile
        mask = ~np.isfinite(profile)
        profile[mask] = np.min(profile[np.isfinite(profile)]) - 0.5
        
        # take the derivative of the profile, using the bin centers as x values
        deriv = np.gradient(profile, bin_centers)
        
        # find the locations where the derivative is more than our desired slope
        locs = np.where(deriv >= slope)[0]
        
        if len(locs) > 0 :
            # loop through every location, and check to see if the profile is
            for loc in np.flip(locs) : # always less than the threshold value
                if np.all(profile[:loc+1] <= threshold) : # before that location
                    Rinner = bin_centers[loc] # if so, return that location
                    break
                else :
                    Rinner = 0
        else :         # if the derivative is never more than the desired slope,
            Rinner = 0 # simply return the innermost radius
    
    return Rinner

def calculate_Router(bin_centers, profile, threshold=-10.5, slope=-1) :
    
    if np.all(profile == 0) :
        Router = -99
    elif np.all(np.isnan(profile)) :
        Router = -49
    else :
        # take the logarithm of the profile for easier manipulation
        profile = np.log10(profile)
        # plt.plot_simple_dumb(bin_centers, profile, xlabel=r'$r/R_{\rm e}$',
        #                      ylabel='sSFR')
        
        # for NaNs, set the value in each bin to be below the minimum valid
        # value appearing in the profile
        mask = ~np.isfinite(profile)
        profile[mask] = np.min(profile[np.isfinite(profile)]) - 0.5
        
        # take the derivative of the profile, using the bin centers as x values
        deriv = np.gradient(profile, bin_centers)
        
        # find the locations where the derivative is less than our desired slope
        locs = np.where(deriv <= slope)[0]
        
        if len(locs) > 0 :
            # loop through every location, and check to see if the profile is
            for loc in locs : # always less than the threshold value beyond
                if np.all(profile[loc:] <= threshold) : # that location
                    Router = bin_centers[loc] # if so, return that location
                    break
                else :
                    Router = 5
        else :         # if the derivative is never less than the desired slope,
            Router = 5 # simply return the outermost radius
    
    return Router

def compute_C_SF(sf_masses, sf_rs, Re) :
    # determine the concentration of SF
    
    if (len(sf_masses) == 0) and (len(sf_rs) == 0) :
        sf_mass_within_1kpc, sf_mass_within_tenthRe = np.nan, np.nan
        sf_mass = np.nan
    else :
        sf_mass_within_1kpc = np.sum(sf_masses[sf_rs <= 1.0])
        sf_mass_within_tenthRe = np.sum(sf_masses[sf_rs <= 0.1*Re])
        sf_mass = np.sum(sf_masses)
    
    return sf_mass_within_1kpc, sf_mass_within_tenthRe, sf_mass

def compute_R_SF(masses, rs, sf_masses, sf_rs) :
    # determine the disk size ratio
    
    if (len(masses) == 0) and (len(rs) == 0) :
        stellar_half_radius = np.nan
        stellar_tenth_radius, stellar_ninetieth_radius = np.nan, np.nan
    else :
        sort_order = np.argsort(rs)
        masses = masses[sort_order]
        rs = rs[sort_order]
        stellar_tenth_radius = np.interp(0.1*np.sum(masses),
                                         np.cumsum(masses), rs)
        stellar_half_radius = np.interp(0.5*np.sum(masses),
                                        np.cumsum(masses), rs)
        stellar_ninetieth_radius = np.interp(0.9*np.sum(masses),
                                             np.cumsum(masses), rs)
    
    if (len(sf_masses) == 0) and (len(sf_rs) == 0) :
        sf_half_radius = np.nan
        sf_tenth_radius, sf_ninetieth_radius = np.nan, np.nan
    else :
        sort_order = np.argsort(sf_rs)
        sf_masses = sf_masses[sort_order]
        sf_rs = sf_rs[sort_order]
        sf_tenth_radius = np.interp(0.1*np.sum(sf_masses),
                                    np.cumsum(sf_masses), sf_rs)
        sf_half_radius = np.interp(0.5*np.sum(sf_masses),
                                   np.cumsum(sf_masses), sf_rs)
        sf_ninetieth_radius = np.interp(0.9*np.sum(sf_masses),
                                   np.cumsum(sf_masses), sf_rs)
    
    return (stellar_tenth_radius, stellar_half_radius, stellar_ninetieth_radius,
            sf_tenth_radius, sf_half_radius, sf_ninetieth_radius)

def determine_radial_profiles(simName, snapNum, snap, time, subID, Re, center,
                              edges, length, delta_t=100*u.Myr) :
    
    # open the corresponding cutouts and get their particles
    ages, masses, rs = get_particles(simName, snapNum, snap, subID, center)
    
    if (ages is not None) and (masses is not None) and (rs is not None) :
        # mask all particles to within the maximum radius (default 5Re)
        rs = rs/Re
        max_Re = edges[-1]
        ages = ages[rs <= max_Re]
        masses = masses[rs <= max_Re]
        rs = rs[rs <= max_Re]
        
        # find the total mass and area (in kpc^2) in each annulus
        mass_profile, area_profile, nParticles = [], [], []
        for start, end in zip(edges, edges[1:]) :
            mass_in_bin = masses[(rs >= start) & (rs < end)]
            mass = np.sum(mass_in_bin)
            
            nParticles.append(len(mass_in_bin))
            mass_profile.append(mass)
            
            area = np.pi*(np.square(end*Re) - np.square(start*Re))
            area_profile.append(area)
        
        # convert lists to arrays
        nParticles, mass_profile = np.array(nParticles), np.array(mass_profile)
        area_profile = np.array(area_profile)
        
        # get the SF particles
        _, SF_masses, SF_rs = get_sf_particles(ages, masses, rs, time,
                                               delta_t=delta_t)
        
        # find the SF mass in each annulus
        SF_mass_profile, nSFparticles = [], []
        for start, end in zip(edges, edges[1:]) :
            SF_mass_in_bin = SF_masses[(SF_rs >= start) & (SF_rs < end)]
            SF_mass = np.sum(SF_mass_in_bin)
            
            nSFparticles.append(len(SF_mass_in_bin))
            SF_mass_profile.append(SF_mass)
        
        # convert lists to arrays
        nSFparticles = np.array(nSFparticles)
        SF_mass_profile = np.array(SF_mass_profile)
        
        # convert the length of time to years
        SFR_profile = SF_mass_profile/(delta_t.to(u.yr).value)
    else :
        full_vals = np.full(length, np.nan)
        SFR_profile, mass_profile, area_profile = full_vals, full_vals, full_vals
        nParticles, nSFparticles = full_vals, full_vals
    
    return SFR_profile, mass_profile, area_profile, nParticles, nSFparticles

def find_control_sf_sample(plot_SFHs=False) :
    
    infile = 'TNG50-1/TNG50-1_99_sample(t).hdf5'
    
    # get basic information about the sample, where some parameters are a
    # function of time
    with h5py.File(infile, 'r') as hf :
        redshifts = hf['redshifts'][:]
        times = hf['times'][:] # for determining active SF populations
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        SFHs = hf['SFH'][:]
        SFMS = hf['SFMS'][:].astype(bool) # (boolean) SFMS at each snapshot
        ionsets = hf['onset_indices'][:].astype(int)
        tonsets = hf['onset_times'][:]
        iterms = hf['termination_indices'][:].astype(int)
        tterms = hf['termination_times'][:]
        quenched = hf['quenched'][:]
    
    # select massive galaxies
    mask = (logM[:, -1] >= 9.5) # 1666 entries, but len(mask) = 8260
    
    # mask all attributes to select only the massive population
    subIDs = subIDs[mask]     # (1666, 100)
    logM = logM[mask]         # (1666, 100)
    ionsets = ionsets[mask]   # (1666)
    tonsets = tonsets[mask]   # (1666)
    iterms = iterms[mask]     # (1666)
    tterms = tterms[mask]     # (1666)
    SFHs = SFHs[mask]         # (1666, 100)
    SFMS = SFMS[mask]         # (1666, 100)
    quenched = quenched[mask] # (1666)
    
    if not exists('TNG50-1/locations_mask_quenched.npy') :
        # create a simple mask to select all quenching episode locations
        unique_quenched = np.full((1666, 100), False)
        for i, (status, ionset, iterm) in enumerate(zip(quenched, ionsets, iterms)) :
            if status :
                unique_quenched[i, ionset:iterm+1] = True
        np.save('TNG50-1/locations_mask_quenched.npy', unique_quenched) # sum = 4695 = 4417 + 278
    
    # get parameters for quenching sample
    quenched_subIDs = subIDs[:, -1][quenched]
    quenched_logM = logM[quenched]
    quenched_SFHs = SFHs[quenched]
    quenched_ionsets = ionsets[quenched]
    quenched_tonsets = tonsets[quenched]
    quenched_iterms = iterms[quenched]
    quenched_tterms = tterms[quenched]
    
    # loop through all quenched galaxies
    N_always_on_SFMS_array = []
    N_similar_masses = []
    N_finals = []
    unique_sf = np.full((1666, 100), False)
    all_quenched_subIDs = []
    all_control_subIDs = []
    all_episode_progresses = []
    all_redshifts = []
    all_masses = []
    for (quenched_subID, quenched_mass, quenched_SFH,
         ionset, tonset, iterm, tterm) in zip(quenched_subIDs,
         quenched_logM, quenched_SFHs, quenched_ionsets, quenched_tonsets,
         quenched_iterms, quenched_tterms) :
        
        # get the stellar mass of the quenched galaxy at onset and termination
        quenched_logM_onset = quenched_mass[ionset]
        quenched_logM_term = quenched_mass[iterm]
        
        # find galaxies that are on the SFMS from onset until termination
        always_on_SFMS = np.all(SFMS[:, ionset:iterm+1] > 0, axis=1) # (1666)
        N_always_on_SFMS = np.sum(always_on_SFMS)
        N_always_on_SFMS_array.append(N_always_on_SFMS)
        
        # compare stellar masses for all galaxies, looking for small differences
        similar_mass = (np.abs(logM[:, ionset] - quenched_logM_onset) <= 0.1) # (1666)
        N_similar_mass = np.sum(similar_mass)
        N_similar_masses.append(N_similar_mass)
        
        # create a final mask where both conditions are true
        final = (similar_mass & always_on_SFMS) # (1666)
        N_final = np.sum(final)
        N_finals.append(N_final)
        
        if N_final > 0 : # 13 galaxies don't have any
            # create a simple mask to select all unique locations
            unique_sf[np.argwhere(final), ionset:iterm] = True
            
            # determine the length of the quenching episode
            episode_duration_Gyr = tterm - tonset
            
            # loop over every control SF galaxy for the quenched galaxy
            for loc in np.argwhere(final) :
                control_sf_subIDfinal = subIDs[loc, -1][0] # get the SF subID
                for snap in range(ionset, iterm+1) :
                    # determine the progress through the quenching episode
                    # at each snapshot within the episode
                    episode_progress = (times[snap] - tonset)/episode_duration_Gyr
                    
                    # determine the redshift and stellar mass at the snapshot
                    redshift_at_snap = redshifts[snap]
                    mass = logM[loc, snap][0]
                    
                    # append those values for future use
                    all_quenched_subIDs.append(quenched_subID)
                    all_control_subIDs.append(control_sf_subIDfinal)
                    all_episode_progresses.append(episode_progress)
                    all_redshifts.append(redshift_at_snap)
                    all_masses.append(mass)
        else : # append placeholders for the 13 galaxies without control SFs
            all_quenched_subIDs.append(quenched_subID)
            all_control_subIDs.append(-1)
            all_episode_progresses.append(-1.0)
            all_redshifts.append(-1.0)
            all_masses.append(-1.0)
        
        # plot companion galaxies relative to the quenched galaxy (requires one
        # always-on-the-SFMS similar mass companion)
        if (N_final > 0) and plot_SFHs : # 13 galaxies don't have any
            
            # prepare input arrays for plotting
            xs = [times]*(1 + N_final)
            ys = [gaussian_filter1d(quenched_SFH, 2)]
            for SFH in SFHs[final] :
                ys.append(gaussian_filter1d(SFH, 2))
            labels = ['quenched'] + ['']*N_final
            colors = ['k'] + ['grey']*N_final
            markers = ['']*(1 + N_final)
            styles = ['-'] + ['--']*N_final
            alphas = [1] + [0.5]*N_final
            
            outfile = 'compare_with_control_SFMS_galaxies/subID_{}.png'.format(quenched_subID)
            plt.plot_simple_multi_with_times(xs, ys, labels, colors, markers,
                styles, alphas, np.nan, tonset, tterm, None, None,
                scale='linear', xmin=-0.1, xmax=13.8, xlabel=r'$t$ (Gyr)',
                ylabel=r'SFR (${\rm M}_{\odot}$ yr$^{-1}$)',
                outfile=outfile, save=True)
            
            mass_diff_onset = np.abs(logM[:, ionset][final] - quenched_logM_onset)
            mass_diff_term = np.abs(logM[:, iterm][final] - quenched_logM_term)
            print(np.median(mass_diff_onset), np.median(mass_diff_term))
    
    # if not exists('all_control_sf_points.fits') :
    #     t = Table([all_quenched_subIDs, all_control_subIDs,
    #                all_episode_progresses, all_redshifts, all_masses],
    #               names=('quenched_subID', 'control_subID',
    #                      'episode_progress', 'redshift', 'logM'))
    #     t.write('all_control_sf_points.fits')
    
    if not exists('TNG50-1/locations_mask_control_sf.npy') :
        np.save('TNG50-1/locations_mask_control_sf.npy', unique_sf) # sum = 27763
    
    if not exists('TNG50-1/check-N_always_on_SFMS-during-quenching-episode.fits') :
        t = Table([quenched_subIDs, quenched_logM[np.arange(278), quenched_ionsets],
                   quenched_ionsets, quenched_iterms,
                   quenched_iterms - quenched_ionsets + 1, tonsets[quenched],
                   tterms[quenched], tterms[quenched] - tonsets[quenched],
                   N_always_on_SFMS_array, N_similar_masses, N_finals,
                   (quenched_iterms - quenched_ionsets + 1)*N_finals],
                  names=('subID', 'logM_onset', 'ionset', 'iterm',
                          'episode_duration_snaps', 'tonset', 'tterm',
                          'episode_duration_Gyr', 'N_always_on_SFMS',
                          'N_similar_mass_01dex', 'N_final', 'N_compare'))
        t.write('TNG50-1/check-N_always_on_SFMS-during-quenching-episode.fits')
    
    return

def perform_masking(data, mechanism, status, episode_progress, column,
                    population, epoch) :
    
    # select the population
    pop_mask = (mechanism == population) & status
    
    # copy the episode progress and morphological metrics
    pop_episode_progress = episode_progress.copy()[pop_mask]
    pop_data = data.copy()[pop_mask]
    
    # define the metric of interest
    metric = pop_data[:, column]
    
    # mask the data based on valid values
    valid = (~np.isnan(metric) & (metric >= 0.0))
    metric = metric[valid]
    pop_episode_progress = pop_episode_progress[valid]
    
    if column == 1 :
        metric = np.log10(metric)
        plotting_mask = (metric >= -1.3) & (metric <= 1.3)
        metric = metric[plotting_mask]
        pop_episode_progress = pop_episode_progress[plotting_mask]
    
    if epoch == 0 :
        epoch_mask = (pop_episode_progress <= 0.25)
    elif epoch == 1 :
        epoch_mask = (pop_episode_progress > 0.25) & (pop_episode_progress <= 0.5)
    elif epoch == 2 :
        epoch_mask = (pop_episode_progress > 0.5) & (pop_episode_progress <= 0.75)
    elif epoch == 3 :
        epoch_mask = (pop_episode_progress > 0.75)
    
    return metric[epoch_mask], np.ones(len(metric[epoch_mask]))/len(metric[epoch_mask])

def prepare_morphological_metrics_for_classification(threshold=-10.5, slope=1) :
    
    outfile = 'TNG50-1/morphological_metrics_{}_+-{}.fits'.format(threshold, slope)
    
    # get basic information about the sample, where some parameters are a
    # function of time
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        redshifts = hf['redshifts'][:]
        times = hf['times'][:] # for determining active SF populations
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        SFMS = hf['SFMS'][:].astype(bool) # (boolean) SFMS at each snapshot
        ionsets = hf['onset_indices'][:].astype(int)
        tonsets = hf['onset_times'][:]
        iterms = hf['termination_indices'][:].astype(int)
        tterms = hf['termination_times'][:]
        quenched = hf['quenched'][:]
    
    # read in the required arrays
    with h5py.File('TNG50-1/TNG50-1_99_massive_C_SF(t).hdf5', 'r') as hf :
        C_SFs = hf['C_SF'][:]
    with h5py.File('TNG50-1/TNG50-1_99_massive_R_SF(t).hdf5', 'r') as hf :
        R_SFs = hf['R_SF'][:]
    Rinner_infile = 'TNG50-1/TNG50-1_99_massive_Rinner(t)_revised_{}_+{}.hdf5'.format(
        threshold, slope)
    with h5py.File(Rinner_infile, 'r') as hf :
        Rinners = hf['Rinner'][:]
    Router_infile ='TNG50-1/TNG50-1_99_massive_Router(t)_revised_{}_-{}.hdf5'.format(
        threshold, slope)
    with h5py.File(Router_infile, 'r') as hf :
        Routers = hf['Router'][:]
    
    # get the quenching mechanisms
    with h5py.File('TNG50-1/TNG50-1_99_mechanism.hdf5', 'r') as hf :
        io = hf['inside-out'][:] # 103
        oi = hf['outside-in'][:] # 109
        uni = hf['uniform'][:]   # 8
        amb = hf['ambiguous'][:] # 58
    mechs = np.sum(np.array([1*io, 3*oi, 5*uni, 5*amb]).T, axis=1)
    
    # select massive galaxies
    mask = (logM[:, -1] >= 9.5) # 1666 entries, but len(mask) = 8260
    
    # mask all attributes to select only the massive population
    subIDs = subIDs[mask]     # (1666, 100)
    logM = logM[mask]         # (1666, 100)
    ionsets = ionsets[mask]   # (1666)
    tonsets = tonsets[mask]   # (1666)
    iterms = iterms[mask]     # (1666)
    tterms = tterms[mask]     # (1666)
    SFMS = SFMS[mask]         # (1666, 100)
    quenched = quenched[mask] # (1666)
    mechs = mechs[mask]       # (1666)
    
    # get parameters for quenching sample
    quenched_subIDs = subIDs[:, -1][quenched]
    quenched_logM = logM[quenched]
    quenched_ionsets = ionsets[quenched]
    quenched_tonsets = tonsets[quenched]
    quenched_iterms = iterms[quenched]
    quenched_tterms = tterms[quenched]
    quenched_mechs = mechs[quenched]
    
    # loop through all quenched galaxies
    all_quenched_subIDs = []
    all_quenched_status = []
    all_sf_status = []
    all_control_subIDs = []
    all_episode_progresses = []
    all_redshifts = []
    all_masses = []
    all_C_SFs = []
    all_R_SFs = []
    all_Rinners = []
    all_Routers = []
    all_mechs = []
    all_mechs_for_quenched_reference = []
    for (quenched_subID, quenched_mass, ionset, tonset, iterm, tterm, 
         quenched_mech) in zip(quenched_subIDs, quenched_logM,
        quenched_ionsets, quenched_tonsets, quenched_iterms, quenched_tterms,
        quenched_mechs) :
        
        # find the location of the quenched galaxy within the subsample
        location = (np.argwhere(subIDs[:, 99] == quenched_subID))[0][0]
        
        for snap in range(ionset, iterm+1) :
            # get the relevant quantities
            episode_progress = (times[snap] - tonset)/(tterm - tonset)
            redshift_at_snap = redshifts[snap]
            mass = quenched_mass[snap]
            C_SF = C_SFs[location, snap]
            R_SF = R_SFs[location, snap]
            Rinner = Rinners[location, snap]
            Router = Routers[location, snap]
            
            # append values
            all_quenched_subIDs.append(quenched_subID)
            all_quenched_status.append(True)
            all_sf_status.append(False)
            all_control_subIDs.append(-1)
            all_episode_progresses.append(episode_progress)
            all_redshifts.append(redshift_at_snap)
            all_masses.append(mass)
            all_C_SFs.append(C_SF)
            all_R_SFs.append(R_SF)
            all_Rinners.append(Rinner)
            all_Routers.append(Router)
            all_mechs.append(quenched_mech)
            all_mechs_for_quenched_reference.append(-1)
        
        # get the stellar mass of the quenched galaxy at onset
        quenched_logM_onset = quenched_mass[ionset]
        
        # find galaxies that are on the SFMS from onset until termination
        always_on_SFMS = np.all(SFMS[:, ionset:iterm+1] > 0, axis=1) # (1666)
        
        # compare stellar masses for all galaxies, looking for small differences
        similar_mass = (np.abs(logM[:, ionset] - quenched_logM_onset) <= 0.1) # (1666)
        
        # create a final mask where both conditions are true
        final = (similar_mass & always_on_SFMS) # (1666)
        N_final = np.sum(final)
        
        if N_final > 0 : # 13 galaxies don't have any
            # loop over every control SF galaxy for the quenched galaxy
            for loc in np.argwhere(final) :
                control_sf_subIDfinal = subIDs[loc, -1][0] # get the SF subID
                for snap in range(ionset, iterm+1) :
                    # get the relevant quantities
                    episode_progress = (times[snap] - tonset)/(tterm - tonset)
                    redshift_at_snap = redshifts[snap]
                    mass = logM[loc, snap][0]
                    C_SF = C_SFs[loc, snap][0]
                    R_SF = R_SFs[loc, snap][0]
                    Rinner = Rinners[loc, snap][0]
                    Router = Routers[loc, snap][0]
                    
                    # append values
                    all_quenched_subIDs.append(quenched_subID)
                    all_quenched_status.append(False)
                    all_sf_status.append(True)
                    all_control_subIDs.append(control_sf_subIDfinal)
                    all_episode_progresses.append(episode_progress)
                    all_redshifts.append(redshift_at_snap)
                    all_masses.append(mass)
                    all_C_SFs.append(C_SF)
                    all_R_SFs.append(R_SF)
                    all_Rinners.append(Rinner)
                    all_Routers.append(Router)
                    all_mechs.append(0)
                    all_mechs_for_quenched_reference.append(quenched_mech)
    
    if not exists(outfile) :
        t = Table([all_quenched_subIDs, all_quenched_status, all_sf_status,
                   all_control_subIDs, all_episode_progresses,
                   all_redshifts, all_masses, all_C_SFs, all_R_SFs,
                   all_Rinners, all_Routers, all_mechs,
                   all_mechs_for_quenched_reference],
                  names=('quenched_subID', 'quenched_status', 'sf_status',
                         'control_subID', 'episode_progress', 'redshift',
                         'logM', 'C_SF', 'R_SF', 'Rinner', 'Router',
                         'mechanism', 'quenched_comparison_mechanism'))
        t.write(outfile)
    
    return

def save_CSF_and_RSF_evolution_plot() :
    
    textwidth = 7.10000594991006
    textheight = 9.095321710253218
    
    table = Table.read('TNG50-1/morphological_metrics_-10.5_+-1.fits')
    quenched = table['quenched_status']
    progress = table['episode_progress']
    mechanism = table['mechanism']
    C_SF = table['C_SF']
    R_SF = table['R_SF']
    
    xmin, xmax = -0.03, 1.03
    ymin, ymax = -1.2, 1.5
    xbins = np.around(np.arange(np.around(xmin, 1), np.around(xmax + 0.1, 1), 0.1), 1)
    ybins = np.around(np.arange(np.around(ymin, 1), np.around(ymax + 0.1, 1), 0.1), 1)
    
    xlabel = r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$'
    ylabel = r'$R_{\rm SF} = \log{(R_{\rm e,SF}/R_{\rm e})}$'
    
    # define basic information for the contours
    x_centers, y_centers = xbins[:-1] + 0.05, ybins[:-1] + 0.05
    X_cent, Y_cent = np.meshgrid(x_centers, y_centers)
    
    progress_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
    windows = ['early', 'early-mid', 'mid-late', 'late']
    xs = []
    ys = []
    for lo, hi, window in zip(progress_thresholds, progress_thresholds[1:],
                              windows) :
        progress_mask = (progress >= lo) & (progress < hi)
        if hi == 1.0 :
            progress_mask = (progress >= lo) & (progress <= hi)
        
        # mask the quantities to the progress window of interest
        window_C_SF = C_SF.copy()[progress_mask]
        window_R_SF = R_SF.copy()[progress_mask]
        window_mech = mechanism.copy()[progress_mask]
        window_quenched = quenched.copy()[progress_mask]
        
        sf_mask = (window_mech == 0.0) & ~window_quenched
        threshold = np.sum(window_quenched)/np.sum(sf_mask)
        
        select = (np.random.rand(len(sf_mask)) <= threshold)
        
        CSFs = [window_C_SF[select],
                window_C_SF[window_mech == 1.0],
                window_C_SF[window_mech == 3.0]]
        RSFs = [np.log10(window_R_SF[select]),
                np.log10(window_R_SF[window_mech == 1.0]),
                np.log10(window_R_SF[window_mech == 3.0])]
        # title = r'{} <= episode progress < {}'.format(lo, hi)
        # plt.plot_scatter_with_hists(xs, ys, ['r', 'm'],
        #     ['outside-in', 'inside-out'], ['s', 's'], [0.2, 0.2],
        #     xlabel=xlabel, ylabel=ylabel, title=window,
        #     xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        #     loc=1, xbins=xbins, ybins=ybins)
        
        xs.append(CSFs)
        ys.append(RSFs)
        
        '''
        contours, levels = [], []
        for mech in [3.0, 1.0] :
            hist, _, _ = np.histogram2d(window_C_SF[window_mech == mech],
                np.log10(window_R_SF[window_mech == mech]),
                bins=(xbins, ybins))
            contour = gaussian_filter(hist.T, 0.7)
            
            vals = np.sort(hist.flatten())
            level = np.percentile(vals[vals > 0], [50, 84, 95])
            
            contours.append(contour)
            levels.append(level)
        
        plt.plot_contour_with_hists(CSFs, RSFs, ['m', 'r'], ['s', 's'], [0.2, 0.2],
            X_cent, Y_cent, contours, levels, xlabel=xlabel, ylabel=ylabel,
            title=window, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            loc=1, xbins=xbins, ybins=ybins)
        '''
    
    colors = ['k', 'm', 'r']
    markers = ['o', 's', 's']
    alphas = [0.2, 0.2, 0.2]
    labels = ['SF', 'inside-out', 'outside-in']
    plt.quad_scatter(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2],
        xs[3], ys[3], colors, markers, alphas, labels, windows,
        xlabel=r'$C_{\rm SF}$', ylabel=ylabel, xmin=0, xmax=1, ymin=-1,
        ymax=1.5, figsizeheight=textheight/3, figsizewidth=textwidth,
        save=False, outfile='metric_plane_evolution.pdf')
    
    return

def save_mass_distribution_plots() :
    
    table = Table.read('TNG50-1/morphological_metrics_-10.5_+-1.fits')
    quenched_status = table['quenched_status'].value
    sf_status = table['sf_status'].value
    mechanism = table['mechanism'].value
    quenched_comparison_mechanism = table['quenched_comparison_mechanism'].value
    logM = table['logM'].value
    
    bins = np.around(np.arange(9.2, 11.75, 0.1), 1)
    bins = [bins, bins, bins]
    
    # check the mass distributions for the control SF galaxies
    sf_logM = logM[sf_status]
    quenched_comparison_mechanism = quenched_comparison_mechanism[sf_status]
    
    data = [sf_logM[quenched_comparison_mechanism == 1.0],
            sf_logM[quenched_comparison_mechanism == 3.0],
            sf_logM[quenched_comparison_mechanism == 5.0]]
    weights = [np.ones(len(datum))/len(datum) for datum in data]
    
    plt.histogram_multi(data, r'$\log{(M_{*}/{\rm M}_{\odot})}$', ['darkmagenta', 'darkred', 'darkorange'],
        ['-', '-', '-'], ['inside-out comparison', 'outside-in comparison',
         'ambiguous comparison'], bins, weights, xmin=9.2, xmax=11.7, ymax=0.18)
    
    # check the corresponding distributions for the quenched galaxies
    quenched_logM = logM[quenched_status]
    mechanism = mechanism[quenched_status]
    
    data = [quenched_logM[mechanism == 1.0], quenched_logM[mechanism == 3.0],
            quenched_logM[mechanism == 5.0]]
    weights = [np.ones(len(datum))/len(datum) for datum in data]
    
    plt.histogram_multi(data, r'$\log{(M_{*}/{\rm M}_{\odot})}$', ['m', 'r', 'orange'],
        ['-', '-', '-'], ['inside-out', 'outside-in', 'ambiguous'], bins,
        weights, xmin=9.2, xmax=11.7, ymax=0.18)
    
    return

def save_morphological_metric_example_evolution_plot() :
    
    colwidth = 3.35224200913242
    # textwidth = 7.10000594991006
    textheight = 9.095321710253218
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
    
    table = Table.read('TNG50-1/morphological_metrics_-10.5_+-1.fits')
    
    # select the rows for the subID of interest
    table = table[np.where(table['quenched_subID'].value == 198186)]
    
    # select quenched galaxy's rows
    table = table[table['quenched_status'].value == True]
    
    zs = table['redshift'].value
    
    locs = np.full(len(table), -1)
    for i, zz in enumerate(zs) :
        locs[i] = np.where(redshifts == zz)[0][0]
    
    xs = times[locs]
    
    CSF = table['C_SF'].value
    RSF = table['R_SF'].value
    Rinner = table['Rinner'].value
    Router = table['Router'].value
    
    plt.plot_simple_multi_secax([xs, xs, xs, xs], [CSF, RSF, Rinner, Router],
        [r'$C_{\rm SF}$', r'$R_{\rm SF}$', r'$R_{\rm inner}/R_{\rm e}$',
         r'$R_{\rm outer}/R_{\rm e}$'], ['k', 'k', 'r', 'r'], ['', '', '', ''],
        ['-', '--', ':', '-.'], [1, 1, 1, 1], 2, xlabel=r'$t$ (Gyr)',
        ylabel=r'$C_{\rm SF}$ and $R_{\rm SF}$',
        seclabel=r'$R_{\rm inner}$ and $R_{\rm outer}$', xmin=4.71, xmax=6.62,
        ymin=0, ymax=1.4, secmin=-0.1, secmax=5.1,
        figsizeheight=textheight/3, figsizewidth=colwidth,
        save=False, outfile='metric_example.pdf')
    
    return

def save_morphological_metric_global_evolution_plot() :
    
    textwidth = 7.10000594991006
    textheight = 9.095321710253218
    
    infile = 'TNG50-1/morphological_metrics_-10.5_+-1.fits'
    data = Table.read(infile)

    quenched = data['quenched_status'].value
    episode_progress = data['episode_progress'].value
    mechanism = data['mechanism'].value
    
    # exclude redshift and stellar mass information
    data = np.array([data['C_SF'].value, data['R_SF'].value,
                     data['Rinner'].value, data['Router'].value]).T
    
    CSF_sf_1st, wi1 = perform_masking(data, mechanism, ~quenched, episode_progress, 0, 0, 0)
    CSF_sf_2nd, wi2 = perform_masking(data, mechanism, ~quenched, episode_progress, 0, 0, 1)
    CSF_sf_3rd, wi3 = perform_masking(data, mechanism, ~quenched, episode_progress, 0, 0, 2)
    CSF_sf_4th, wi4 = perform_masking(data, mechanism, ~quenched, episode_progress, 0, 0, 3)
    
    CSF_io_1st, wi5 = perform_masking(data, mechanism, quenched, episode_progress, 0, 1, 0)
    CSF_io_2nd, wi6 = perform_masking(data, mechanism, quenched, episode_progress, 0, 1, 1)
    CSF_io_3rd, wi7 = perform_masking(data, mechanism, quenched, episode_progress, 0, 1, 2)
    CSF_io_4th, wi8 = perform_masking(data, mechanism, quenched, episode_progress, 0, 1, 3)
    
    CSF_oi_1st, wi9 = perform_masking(data, mechanism, quenched, episode_progress, 0, 3, 0)
    CSF_oi_2nd, wi10 = perform_masking(data, mechanism, quenched, episode_progress, 0, 3, 1)
    CSF_oi_3rd, wi11 = perform_masking(data, mechanism, quenched, episode_progress, 0, 3, 2)
    CSF_oi_4th, wi12 = perform_masking(data, mechanism, quenched, episode_progress, 0, 3, 3)
    
    RSF_sf_1st, wi13 = perform_masking(data, mechanism, ~quenched, episode_progress, 1, 0, 0)
    RSF_sf_2nd, wi14 = perform_masking(data, mechanism, ~quenched, episode_progress, 1, 0, 1)
    RSF_sf_3rd, wi15 = perform_masking(data, mechanism, ~quenched, episode_progress, 1, 0, 2)
    RSF_sf_4th, wi16 = perform_masking(data, mechanism, ~quenched, episode_progress, 1, 0, 3)
    
    RSF_io_1st, wi17 = perform_masking(data, mechanism, quenched, episode_progress, 1, 1, 0)
    RSF_io_2nd, wi18 = perform_masking(data, mechanism, quenched, episode_progress, 1, 1, 1)
    RSF_io_3rd, wi19 = perform_masking(data, mechanism, quenched, episode_progress, 1, 1, 2)
    RSF_io_4th, wi20 = perform_masking(data, mechanism, quenched, episode_progress, 1, 1, 3)
    
    RSF_oi_1st, wi21 = perform_masking(data, mechanism, quenched, episode_progress, 1, 3, 0)
    RSF_oi_2nd, wi22 = perform_masking(data, mechanism, quenched, episode_progress, 1, 3, 1)
    RSF_oi_3rd, wi23 = perform_masking(data, mechanism, quenched, episode_progress, 1, 3, 2)
    RSF_oi_4th, wi24 = perform_masking(data, mechanism, quenched, episode_progress, 1, 3, 3)
    
    Rinner_sf_1st, wi25 = perform_masking(data, mechanism, ~quenched, episode_progress, 2, 0, 0)
    Rinner_sf_2nd, wi26 = perform_masking(data, mechanism, ~quenched, episode_progress, 2, 0, 1)
    Rinner_sf_3rd, wi27 = perform_masking(data, mechanism, ~quenched, episode_progress, 2, 0, 2)
    Rinner_sf_4th, wi28 = perform_masking(data, mechanism, ~quenched, episode_progress, 2, 0, 3)
    
    Rinner_io_1st, wi29 = perform_masking(data, mechanism, quenched, episode_progress, 2, 1, 0)
    Rinner_io_2nd, wi30 = perform_masking(data, mechanism, quenched, episode_progress, 2, 1, 1)
    Rinner_io_3rd, wi31 = perform_masking(data, mechanism, quenched, episode_progress, 2, 1, 2)
    Rinner_io_4th, wi32 = perform_masking(data, mechanism, quenched, episode_progress, 2, 1, 3)
    
    Rinner_oi_1st, wi33 = perform_masking(data, mechanism, quenched, episode_progress, 2, 3, 0)
    Rinner_oi_2nd, wi34 = perform_masking(data, mechanism, quenched, episode_progress, 2, 3, 1)
    Rinner_oi_3rd, wi35 = perform_masking(data, mechanism, quenched, episode_progress, 2, 3, 2)
    Rinner_oi_4th, wi36 = perform_masking(data, mechanism, quenched, episode_progress, 2, 3, 3)
    
    Router_sf_1st, wi37 = perform_masking(data, mechanism, ~quenched, episode_progress, 3, 0, 0)
    Router_sf_2nd, wi38 = perform_masking(data, mechanism, ~quenched, episode_progress, 3, 0, 1)
    Router_sf_3rd, wi39 = perform_masking(data, mechanism, ~quenched, episode_progress, 3, 0, 2)
    Router_sf_4th, wi40 = perform_masking(data, mechanism, ~quenched, episode_progress, 3, 0, 3)
    
    Router_io_1st, wi41 = perform_masking(data, mechanism, quenched, episode_progress, 3, 1, 0)
    Router_io_2nd, wi42 = perform_masking(data, mechanism, quenched, episode_progress, 3, 1, 1)
    Router_io_3rd, wi43 = perform_masking(data, mechanism, quenched, episode_progress, 3, 1, 2)
    Router_io_4th, wi44 = perform_masking(data, mechanism, quenched, episode_progress, 3, 1, 3)
    
    Router_oi_1st, wi45 = perform_masking(data, mechanism, quenched, episode_progress, 3, 3, 0)
    Router_oi_2nd, wi46 = perform_masking(data, mechanism, quenched, episode_progress, 3, 3, 1)
    Router_oi_3rd, wi47 = perform_masking(data, mechanism, quenched, episode_progress, 3, 3, 2)
    Router_oi_4th, wi48 = perform_masking(data, mechanism, quenched, episode_progress, 3, 3, 3)
    
    h1 = [CSF_sf_1st, CSF_io_1st, CSF_oi_1st]
    h2 = [CSF_sf_2nd, CSF_io_2nd, CSF_oi_2nd]
    h3 = [CSF_sf_3rd, CSF_io_3rd, CSF_oi_3rd]
    h4 = [CSF_sf_4th, CSF_io_4th, CSF_oi_4th]
    
    h5 = [RSF_sf_1st, RSF_io_1st, RSF_oi_1st]
    h6 = [RSF_sf_2nd, RSF_io_2nd, RSF_oi_2nd]
    h7 = [RSF_sf_3rd, RSF_io_3rd, RSF_oi_3rd]
    h8 = [RSF_sf_4th, RSF_io_4th, RSF_oi_4th]
    
    h9 = [Rinner_sf_1st, Rinner_io_1st, Rinner_oi_1st]
    h10 = [Rinner_sf_2nd, Rinner_io_2nd, Rinner_oi_2nd]
    h11 = [Rinner_sf_3rd, Rinner_io_3rd, Rinner_oi_3rd]
    h12 = [Rinner_sf_4th, Rinner_io_4th, Rinner_oi_4th]
    
    h13 = [Router_sf_1st, Router_io_1st, Router_oi_1st]
    h14 = [Router_sf_2nd, Router_io_2nd, Router_oi_2nd]
    h15 = [Router_sf_3rd, Router_io_3rd, Router_oi_3rd]
    h16 = [Router_sf_4th, Router_io_4th, Router_oi_4th]
    
    w1 = [wi1, wi5, wi9]
    w2 = [wi2, wi6, wi10]
    w3 = [wi3, wi7, wi11]
    w4 = [wi4, wi8, wi12]
    
    w5 = [wi13, wi17, wi21]
    w6 = [wi14, wi18, wi22]
    w7 = [wi15, wi19, wi23]
    w8 = [wi16, wi20, wi24]
    
    w9 = [wi25, wi29, wi33]
    w10 = [wi26, wi30, wi34]
    w11 = [wi27, wi31, wi35]
    w12 = [wi28, wi32, wi36]
    
    w13 = [wi37, wi41, wi45]
    w14 = [wi38, wi42, wi46]
    w15 = [wi39, wi43, wi47]
    w16 = [wi40, wi44, wi48]
    
    plt.histogram_large(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13,
        h14, h15, h16, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13,
        w14, w15, w16, ['k', 'm', 'r'], ['SF', 'inside-out', 'outside-in'],
        ['early', 'early-mid', 'mid-late', 'late'],
        ylabel1=r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
        ylabel2=r'$R_{\rm SF} = \log{(R_{\rm e, SF}/R_{\rm e})}$',
        ylabel3=r'$R_{\rm inner}/R_{\rm e}$',
        ylabel4=r'$R_{\rm outer}/R_{\rm e}$',
        figsizewidth=textwidth, figsizeheight=textheight, loc=2,
        outfile='metric_global_evolution.pdf', save=False)
    
    CSF_am_1st, wa1 = perform_masking(data, mechanism, quenched, episode_progress, 0, 5, 0)
    CSF_am_2nd, wa2 = perform_masking(data, mechanism, quenched, episode_progress, 0, 5, 1)
    CSF_am_3rd, wa3 = perform_masking(data, mechanism, quenched, episode_progress, 0, 5, 2)
    CSF_am_4th, wa4 = perform_masking(data, mechanism, quenched, episode_progress, 0, 5, 3)
    
    RSF_am_1st, wa5 = perform_masking(data, mechanism, quenched, episode_progress, 1, 5, 0)
    RSF_am_2nd, wa6 = perform_masking(data, mechanism, quenched, episode_progress, 1, 5, 1)
    RSF_am_3rd, wa7 = perform_masking(data, mechanism, quenched, episode_progress, 1, 5, 2)
    RSF_am_4th, wa8 = perform_masking(data, mechanism, quenched, episode_progress, 1, 5, 3)
    
    Rinner_am_1st, wa9 = perform_masking(data, mechanism, quenched, episode_progress, 2, 5, 0)
    Rinner_am_2nd, wa10 = perform_masking(data, mechanism, quenched, episode_progress, 2, 5, 1)
    Rinner_am_3rd, wa11 = perform_masking(data, mechanism, quenched, episode_progress, 2, 5, 2)
    Rinner_am_4th, wa12 = perform_masking(data, mechanism, quenched, episode_progress, 2, 5, 3)
    
    Router_am_1st, wa13 = perform_masking(data, mechanism, quenched, episode_progress, 3, 5, 0)
    Router_am_2nd, wa14 = perform_masking(data, mechanism, quenched, episode_progress, 3, 5, 1)
    Router_am_3rd, wa15 = perform_masking(data, mechanism, quenched, episode_progress, 3, 5, 2)
    Router_am_4th, wa16 = perform_masking(data, mechanism, quenched, episode_progress, 3, 5, 3)
    
    plt.histogram_large([CSF_am_1st], [CSF_am_2nd], [CSF_am_3rd], [CSF_am_4th],
        [RSF_am_1st], [RSF_am_2nd], [RSF_am_3rd], [RSF_am_4th], [Rinner_am_1st],
        [Rinner_am_2nd], [Rinner_am_3rd], [Rinner_am_4th], [Router_am_1st],
        [Router_am_2nd], [Router_am_3rd], [Router_am_4th], [wa1], [wa2], [wa3],
        [wa4], [wa5], [wa6], [wa7], [wa8], [wa9], [wa10], [wa11], [wa12],
        [wa13], [wa14], [wa15], [wa16], ['orange'], ['ambiguous'],
        ['early', 'early-mid', 'mid-late', 'late'],
        ylabel1=r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
        ylabel2=r'$R_{\rm SF} = \log{(R_{\rm e, SF}/R_{\rm e})}$',
        ylabel3=r'$R_{\rm inner}/R_{\rm e}$',
        ylabel4=r'$R_{\rm outer}/R_{\rm e}$',
        figsizewidth=textwidth, figsizeheight=textheight, loc=2,
        outfile='metric_global_evolution_amb.pdf', save=False)
    
    return

def save_plotly_plots() :
    
    CSF_plotly = '<i>C</i><sub>SF</sub> = SFR<sub><1 kpc</sub>/SFR<sub>total</sub>'
    RSF_plotly = '<i>R</i><sub>SF</sub> = log(<i>R</i><sub>e, SF</sub>/<i>R</i><sub>e</sub>)'
    Rinner_plotly = '<i>R</i><sub>inner</sub>/<i>R</i><sub>e</sub>'
    Router_plotly = '<i>R</i><sub>outer</sub>/<i>R</i><sub>e</sub>'
    
    (y_quenched, y_sf, X_quenched, X_sf, X_io, X_oi, X_amb, logM_quenched,
     logM_sf) = get_late_data(include_mass=True)
    
    # select the same number of SF galaxies as quenching galaxies
    select = (np.random.rand(len(y_sf)) <= len(y_quenched)/len(y_sf))
    X_sf_it = X_sf[select]
    y_sf_it = y_sf[select]
    logM_sf_it = logM_sf[select]
    
    X_final = np.concatenate([X_quenched, X_sf_it])
    y_final = np.concatenate([y_quenched, y_sf_it])
    logM_final = np.concatenate([logM_quenched, logM_sf_it])
    
    # create a column with the mechanisms as strings
    mech = np.full(y_final.shape[0], '', dtype=object)
    mech[y_final == 0] = 'star forming'
    mech[y_final == 1] = 'inside-out'
    mech[y_final == 3] = 'outside-in'
    mech[y_final == 5] = 'ambiguous'
    
    # determine the size of the points for use in plotly
    mini, stretch = 1.5, 20 # define the minimum size and the maximum stretch
    logM_min, logM_max = np.min(logM_final), np.max(logM_final)
    diff = (logM_max - logM_min)/2
    logM_fit_vals = np.array([logM_min, logM_min + diff, logM_max])
    size_fit_vals = np.array([1, np.sqrt(stretch), stretch])*mini
    # adapted from https://stackoverflow.com/questions/12208634
    popt, _ = curve_fit(lambda xx, aa: vertex(xx, aa, logM_fit_vals[0], mini),
                        logM_fit_vals, size_fit_vals) # fit the curve
    size = vertex(logM_final, popt[0], logM_fit_vals[0], mini) # get the size for the points
    
    # create a pandas dataframe for future use with plotly
    df = pd.DataFrame(X_final, columns=['C_SF', 'R_SF', 'Rinner', 'Router'])
    df['logM'] = logM_final
    df['signature'] = mech
    df['size'] = size
    
    # use the browser to view the plot when testing
    pio.renderers.default = 'browser'
    
    # create the plotly figure for C_SF/R_SF/Rinner
    fig = px.scatter_3d(df, x='C_SF', y='R_SF', z='Rinner', color='signature',
        size='size', hover_name='signature',
        hover_data={'C_SF':':.3f', 'R_SF':':.3f', 'Rinner':':.3f', 'Router':':.3f',
            'signature':False, 'size':False, 'logM':':.3f'}, size_max=np.max(size),
        color_discrete_sequence=['orange', 'magenta', 'red', 'black'],
        opacity=0.6, range_x=[-0.1, 1.1], range_y=[-1.1, 1.1], range_z=[-0.1, 5.1])
    
    # update the axes labels and font sizes, and aesthetic template
    fig.update_layout(scene=dict(xaxis_title=CSF_plotly, 
        yaxis_title=RSF_plotly, zaxis_title=Rinner_plotly,
        xaxis=dict(titlefont=dict(size=25)), yaxis=dict(titlefont=dict(size=25)),
        zaxis=dict(titlefont=dict(size=25))), template='plotly_white',
        legend=dict(font=dict(size=30)))
    fig.show()
    # fig.write_html('morphological_metrics_interactive_figure_1.html')
    
    # create the plotly figure for C_SF/R_SF/Router
    fig = px.scatter_3d(df, x='C_SF', y='R_SF', z='Router', color='signature',
        size='size', hover_name='signature',
        hover_data={'C_SF':':.3f', 'R_SF':':.3f', 'Rinner':':.3f', 'Router':':.3f',
            'signature':False, 'size':False, 'logM':':.3f'}, size_max=np.max(size),
        color_discrete_sequence=['orange', 'magenta', 'red', 'black'],
        opacity=0.6, range_x=[-0.1, 1.1], range_y=[-1.1, 1.1], range_z=[-0.1, 5.1])
    
    # update the axes labels and font sizes, and aesthetic template
    fig.update_layout(scene=dict(xaxis_title=CSF_plotly,
        yaxis_title=RSF_plotly, zaxis_title=Router_plotly,
        xaxis=dict(titlefont=dict(size=25)), yaxis=dict(titlefont=dict(size=25)),
        zaxis=dict(titlefont=dict(size=25))), template='plotly_white',
        legend=dict(font=dict(size=30)))
    fig.show()
    # fig.write_html('morphological_metrics_interactive_figure_2.html')
    
    return
