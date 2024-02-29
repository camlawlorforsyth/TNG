
from os.path import exists
import numpy as np

import astropy.units as u
from astropy.table import Table
import h5py
from pypdf import PdfWriter
from scipy.ndimage import gaussian_filter1d
# from scipy.stats import anderson_ksamp, ks_2samp
# from scipy.interpolate import CubicSpline

# from concatenate import concat_horiz
from core import (add_dataset, find_nearest, get_particle_positions,
                  get_particles, get_sf_particles)
import plotting as plt

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def calculate_required_morphological_parameters(delta_t=100*u.Myr, version='2D',
                                                quenched_pop=True) :
    
    # define the input and output files
    infile = 'TNG50-1/TNG50-1_99_sample(t).hdf5'
    profiles_file = 'TNG50-1/TNG50-1_99_massive_radial_profiles(t).hdf5'
    C_SF_outfile = 'TNG50-1/TNG50-1_99_massive_C_SF(t).hdf5'
    R_SF_outfile = 'TNG50-1/TNG50-1_99_massive_R_SF(t).hdf5'
    Rinner_outfile = 'TNG50-1/TNG50-1_99_massive_Rinner(t).hdf5'
    Router_outfile = 'TNG50-1/TNG50-1_99_massive_Router(t).hdf5'
    
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
    # their morphological parameters computed
    unique_quenched = np.load('locations_mask_quenched.npy') # (1666, 100), sum 4695
    unique_sf = np.load('locations_mask_control_sf.npy')     # (1666, 100), sum 27763
    
    # write empty files which will hold the parameters
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
        C_SF_array = hf['C_SF'][:]  # 26339 non-NaN SFers, expect 27763 -> 1424 missing
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
                    # get all particles
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
                        
                        # compute the morphological parameters
                        C_SF = calculate_C_SF(sf_masses, sf_rs, Res[row, snap])
                        R_SF = calculate_R_SF(masses, rs, sf_masses, sf_rs)
                        Rinner = calculate_Rinner(radial_bin_centers,
                                                  sSFR_profiles[row, snap])
                        Router = calculate_Router(radial_bin_centers,
                                                  sSFR_profiles[row, snap])
                    else :
                        C_SF = np.nan
                        R_SF = np.nan
                        Rinner = np.nan
                        Router = np.nan
                    
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
                    # get all particles
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
                        
                        # compute the morphological parameters
                        C_SF = calculate_C_SF(sf_masses, sf_rs, Res[row, snap])
                        R_SF = calculate_R_SF(masses, rs, sf_masses, sf_rs)
                        Rinner = calculate_Rinner(radial_bin_centers,
                                                  sSFR_profiles[row, snap])
                        Router = calculate_Router(radial_bin_centers,
                                                  sSFR_profiles[row, snap])
                    else :
                        C_SF = np.nan
                        R_SF = np.nan
                        Rinner = np.nan
                        Router = np.nan
                    
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
    unique_quenched = np.load('locations_mask_quenched.npy') # (1666, 100), sum 4695
    unique_sf = np.load('locations_mask_control_sf.npy')     # (1666, 100), sum 27763
    
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






def compare_diagnostics_and_radii() :
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        times = hf['times'][:]
        logM = hf['logM'][:, -1]
        Res = hf['Re'][:]
        tonsets = hf['onset_times'][:]
        tterms = hf['termination_times'][:]
        quenched = hf['quenched'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_diagnostics(t).hdf5', 'r') as hf :
        sf_mass_within_1kpc = hf['sf_mass_within_1kpc'][:]
        sf_mass_within_tenthRe = hf['sf_mass_within_tenthRe'][:]
        sf_mass_tot = hf['sf_mass'][:]
        R10s = hf['R10'][:]
        R50s = hf['R50'][:]
        R90s = hf['R90'][:]
        sf_R10s = hf['sf_R10'][:]
        sf_R50s = hf['sf_R50'][:]
        sf_R90s = hf['sf_R90'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_diagnostics(t)_2d.hdf5', 'r') as hf :
        sf_mass_within_1kpc_2d = hf['sf_mass_within_1kpc'][:]
        sf_mass_within_tenthRe_2d = hf['sf_mass_within_tenthRe'][:]
        sf_mass_tot_2d = hf['sf_mass'][:]
        R10s_2d = hf['R10'][:]
        R50s_2d = hf['R50'][:]
        R90s_2d = hf['R90'][:]
        sf_R10s_2d = hf['sf_R10'][:]
        sf_R50s_2d = hf['sf_R50'][:]
        sf_R90s_2d = hf['sf_R90'][:]
    
    # create various metrics of interest
    xis = sf_mass_within_1kpc/sf_mass_tot
    alt_xis = sf_mass_within_tenthRe/sf_mass_tot
    
    R10_zetas = sf_R10s/R10s
    zetas = sf_R50s/R50s
    R90_zetas = sf_R90s/R90s
    
    xis_2d = sf_mass_within_1kpc_2d/sf_mass_tot_2d
    alt_xis_2d = sf_mass_within_tenthRe_2d/sf_mass_tot_2d
    
    R10_zetas_2d = sf_R10s_2d/R10s_2d
    zetas_2d = sf_R50s_2d/R50s_2d
    R90_zetas_2d = sf_R90s_2d/R90s_2d
    
    plt.plot_simple_dumb(xis, xis_2d, xlabel='xi', ylabel='xi_2d',
                         xmin=-0.05, xmax=1.05, ymin=-0.05, ymax=1.05)
    plt.plot_simple_dumb(alt_xis, alt_xis_2d, xlabel='alt xi', ylabel='alt xi_2d',
                          xmin=-0.05, xmax=1.05, ymin=-0.05, ymax=1.05)
    plt.plot_simple_dumb(R10_zetas, R10_zetas_2d, xlabel='R10 zeta', ylabel='R10 zeta_2d',
                         scale='log', xmin=0.05, xmax=80, ymin=0.05, ymax=80)
    plt.plot_simple_dumb(zetas, zetas_2d, xlabel='zeta', ylabel='zeta_2d',
                         scale='log', xmin=0.05, xmax=80, ymin=0.05, ymax=80)
    plt.plot_simple_dumb(R90_zetas, R90_zetas_2d, xlabel='R90 zeta', ylabel='R90 zeta_2d',
                         scale='log', xmin=0.05, xmax=80, ymin=0.05, ymax=80)
    
    # determine the snapshots that correspond to 75% through the quenching event
    imajorities = np.array(find_nearest(times, tonsets + 0.75*(tterms - tonsets)))
    
    # create a mask for the massive quenched population
    mask = quenched & (logM >= 9.5)
    
    # mask quantities
    Res = Res[mask]
    imajorities = imajorities[mask]
    R10s = R10s[mask]
    R50s = R50s[mask]
    R90s = R90s[mask]
    sf_R10s = sf_R10s[mask]
    sf_R50s = sf_R50s[mask]
    sf_R90s = sf_R90s[mask]
    R10s_2d = R10s_2d[mask]
    R50s_2d = R50s_2d[mask]
    R90s_2d = R90s_2d[mask]
    sf_R10s_2d = sf_R10s_2d[mask]
    sf_R50s_2d = sf_R50s_2d[mask]
    sf_R90s_2d = sf_R90s_2d[mask]
    
    # collapse 2D arrays to 1D
    firstDim = np.arange(278)
    Res = Res[firstDim, imajorities]
    R10s = R10s[firstDim, imajorities]
    R50s = R50s[firstDim, imajorities]
    R90s = R90s[firstDim, imajorities]
    sf_R10s = sf_R10s[firstDim, imajorities]
    sf_R50s = sf_R50s[firstDim, imajorities]
    sf_R90s = sf_R90s[firstDim, imajorities]
    R10s_2d = R10s_2d[firstDim, imajorities]
    R50s_2d = R50s_2d[firstDim, imajorities]
    R90s_2d = R90s_2d[firstDim, imajorities]
    sf_R10s_2d = sf_R10s_2d[firstDim, imajorities]
    sf_R50s_2d = sf_R50s_2d[firstDim, imajorities]
    sf_R90s_2d = sf_R90s_2d[firstDim, imajorities]
    
    plt.plot_simple_dumb(sf_R90s/R90s, sf_R90s/R50s,
                         xlabel=r'$R_{\rm 90, SF, 3D}/R_{\rm 90, 3D}$',
                         ylabel=r'$R_{\rm 90, SF, 3D}/R_{\rm 50, 3D}$',
                         xmin=0.03, xmax=200, ymin=0.03, ymax=200,
                         scale='log', label='data')
    
    plt.plot_simple_dumb(sf_R90s_2d/R90s_2d, sf_R90s_2d/R50s_2d,
                         xlabel=r'$R_{\rm 90, SF, 2D}/R_{\rm 90, 2D}$',
                         ylabel=r'$R_{\rm 90, SF, 2D}/R_{\rm 50, 2D}$',
                         xmin=0.03, xmax=200, ymin=0.03, ymax=200,
                         scale='log', label='data')
    
    plt.plot_simple_dumb(R50s, sf_R10s/R50s, xlabel=r'$R_{\rm 50, 3D}$ (ckpc)',
                         ylabel=r'$R_{\rm 10, SF, 3D}/R_{\rm 50, 3D}$',
                         xmin=0.2, xmax=9, ymin=0.01, ymax=10)
    
    plt.plot_simple_dumb(R50s_2d, sf_R10s_2d/R50s_2d, xlabel=r'$R_{\rm 50, 2D}$ (ckpc)',
                         ylabel=r'$R_{\rm 10, SF, 2D}/R_{\rm 50, 2D}$',
                         xmin=0.2, xmax=9, ymin=0.01, ymax=10)
    
    plt.plot_simple_dumb(R50s, sf_R90s/R50s, xlabel=r'$R_{\rm 50, 3D}$ (ckpc)',
                         ylabel=r'$R_{\rm 90, SF, 3D}/R_{\rm 50, 3D}$',
                         xmin=0.2, xmax=9, ymin=0.1, ymax=100)
    
    plt.plot_simple_dumb(R50s_2d, sf_R90s_2d/R50s_2d, xlabel=r'$R_{\rm 50, 2D}$ (ckpc)',
                         ylabel=r'$R_{\rm 90, SF, 2D}/R_{\rm 50, 2D}$',
                         xmin=0.2, xmax=9, ymin=0.1, ymax=100)
    
    plt.plot_simple_dumb(R10s, R10s_2d, xlabel=r'$R_{\rm 10, 3D}$ (ckpc)',
                         ylabel=r'$R_{\rm 10, 2D}$ (ckpc)', xmin=0.08, xmax=1.5,
                         ymin=0.08, ymax=1.5, scale='log', label='data')
    
    plt.plot_simple_dumb(R50s, R50s_2d, xlabel=r'$R_{\rm 50, 3D}$ (ckpc)',
                         ylabel=r'$R_{\rm 50, 2D}$ (ckpc)', xmin=0.3, xmax=10,
                         ymin=0.3, ymax=10, scale='log', label='data', loc=2)
    
    plt.plot_simple_dumb(Res, R50s_2d, xlabel=r'$R_{\rm e, 3D}$ (ckpc)',
                         ylabel=r'$R_{\rm 50, 2D}$ (ckpc)', xmin=0.3, xmax=10,
                         ymin=0.3, ymax=10, scale='log', label='data', loc=2)
    
    plt.plot_simple_dumb(R90s, R90s_2d, xlabel=r'$R_{\rm 90, 3D}$ (ckpc)',
                         ylabel=r'$R_{\rm 90, 2D}$ (ckpc)', xmin=0.8, xmax=100,
                         ymin=0.8, ymax=100, scale='log', label='data')
    
    plt.plot_simple_dumb(Res, R50s, xlabel=r'$R_{\rm e, 3D}$ (ckpc)',
                         ylabel=r'$R_{\rm 50, 3D}$ (ckpc)', xmin=0.3, xmax=10,
                         ymin=0.3, ymax=10, scale='log', label='data')
    
    plt.plot_simple_dumb(R50s, R10s/R50s, xlabel=r'$R_{\rm 50, 3D}$ (ckpc)',
                         ylabel=r'$R_{\rm 10, 3D}/R_{\rm 50, 3D}$',
                         xmin=0.2, xmax=9, ymin=0, ymax=0.6)
    
    plt.plot_simple_dumb(R50s_2d, R10s_2d/R50s_2d, xlabel=r'$R_{\rm 50, 2D}$ (ckpc)',
                         ylabel=r'$R_{\rm 10, 2D}/R_{\rm 50, 2D}$',
                         xmin=0.2, xmax=9, ymin=0, ymax=0.6)
    
    plt.plot_simple_dumb(R50s, R90s/R50s, xlabel=r'$R_{\rm 50, 3D}$ (ckpc)',
                         ylabel=r'$R_{\rm 90, 3D}/R_{\rm 50, 3D}$',
                         xmin=0.3, xmax=10, ymin=1, ymax=25)
    
    plt.plot_simple_dumb(R50s_2d, R90s_2d/R50s_2d, xlabel=r'$R_{\rm 50, 2D}$ (ckpc)',
                         ylabel=r'$R_{\rm 90, 2D}/R_{\rm 50, 2D}$',
                         xmin=0.3, xmax=10, ymin=1, ymax=25)
    
    plt.histogram(R90s/R50s, r'$R_{\rm 90, 3D}/R_{\rm 50, 3D}$', bins=15)
    
    plt.histogram(R90s_2d/R50s_2d, r'$R_{\rm 90, 2D}/R_{\rm 50, 2D}$', bins=15)
    
    # 2 galaxies have extremely large R50 values - why?
    # selection = (R50s < 10)
    # ratio = R90s[selection]/R50s[selection]
    # ratio_below_5 = np.sum(ratio <= 5) # 192 out of 276
    
    # 2 galaxies have extremely large R50_2d values - why?
    # selection_2d = (R50s_2d < 10)
    # ratio_2d = R90s_2d[selection_2d]/R50s_2d[selection_2d]
    # ratio_below_5_2d = np.sum(ratio_2d <= 5) # 169 out of 276
    
    return

def calculate_C_SF(sf_masses, sf_rs, Re) :
    
    sf_mass_within_1kpc, _, sf_mass = compute_C_SF(sf_masses, sf_rs, Re)
    
    return sf_mass_within_1kpc/sf_mass

def calculate_R_SF(masses, rs, sf_masses, sf_rs) :
    
    _, R50, _, _, sf_R50, _ = compute_R_SF(masses, rs, sf_masses, sf_rs)
    
    return sf_R50/R50

def calculate_Rinner(bin_centers, profile, threshold=-10, slope=1) :
    
    if np.all(profile == 0) :
        Rinner = -99
    elif np.all(np.isnan(profile)) :
        Rinner = -49
    elif np.all(profile[:4] == 0) :
        Rinner = bin_centers[3]
    else :
        # we want to move inwards from 1 Re, so mask the bin centers and
        # profiles, and also take the logarithm of the profile for easier
        # manipulation
        bin_centers = bin_centers[:4]
        profile = np.log10(profile[:4])
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
    elif np.all(profile[4:] == 0) :
        Router = bin_centers[4]
    else :
        # we want to move outwards from 1 Re, so mask the bin centers and
        # profiles, and also take the logarithm of the profile for easier
        # manipulation
        bin_centers = bin_centers[4:]
        profile = np.log10(profile[4:])
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

def determine_morphological_parameters(delta_t=100*u.Myr, version='2D') :
    
    infile = 'TNG50-1/TNG50-1_99_sample(t).hdf5'
    profiles_file = 'TNG50-1/TNG50-1_99_profiles(t).hdf5'
    # C_SF_outfile = 'TNG50-1/C_SF(t).hdf5' # for massive quenched galaxies
    # R_SF_outfile = 'TNG50-1/R_SF(t).hdf5' # for massive quenched galaxies
    # R_inner_outfile = 'TNG50-1/R_inner(t).hdf5' # for massive quenched galaxies
    # R_outer_outfile = 'TNG50-1/R_outer(t).hdf5' # for massive quenched galaxies
    outfile = 'TNG50-1/morphological_parameters_quenched.fits'
    
    # get basic information about the sample, where some parameters are a
    # function of time
    with h5py.File(infile, 'r') as hf :
        redshifts = hf['redshifts'][:]
        times = hf['times'][:] # for determining active SF populations
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        Res = hf['Re'][:]
        centers = hf['centers'][:]
        SFHs = hf['SFH'][:]
        ionsets = hf['onset_indices'][:].astype(int)
        tonsets = hf['onset_times'][:]
        iterms = hf['termination_indices'][:].astype(int)
        tterms = hf['termination_times'][:]
        quenched = hf['quenched'][:]
    
    # !!! January 29, 2024 -> see if this can be computed "on the fly," or is
    # it truly better to save out to an array? this is a large file, and
    # reading it in is slow every time the code has to be run
    # get radial profile information, as a function of time
    with h5py.File(profiles_file, 'r') as hf :
        radial_bin_centers = hf['mids'][:] # units of Re
        mass_profiles = hf['mass_profiles'][:] # shape (8260, 100, 20)
        SFR_profiles = hf['SFR_profiles'][:]   # shape (8260, 100, 20)
    
    # select the quenched galaxies
    mask = (logM[:, -1] >= 9.5) & quenched # 278 entries, but len(mask) = 8260
    
    # mask all attributes to select only the quenched population
    subIDs = subIDs[mask]
    logM = logM[mask]
    Res = Res[mask]
    centers = centers[mask]
    ionsets = ionsets[mask]
    tonsets = tonsets[mask]
    iterms = iterms[mask]
    tterms = tterms[mask]
    SFHs = SFHs[mask]
    sSFR_radial_profiles = (SFR_profiles/mass_profiles)[mask] # shape (278, 100, 20)
    
    imids = np.full(278, -1)
    SFR_onset_fractions_at_midpoint = np.full(278, np.nan)
    for i, (subIDfinal, SFH, ionset, iterm) in enumerate(zip(
            subIDs[:, -1], SFHs, ionsets, iterms)) :
        
        SFH = gaussian_filter1d(SFH, 2) # smooth the SFH, as we've typically done
        
        # to create SKIRT input:
        # ilate = ionset + np.where(SFH[ionset:]/np.max(SFH[ionset:]) < 0.25)[0][0]
        # but note this does not work for subIDs 514274, 656524, 657979, 680429,
        # as they don't ever dip below 25% of SFR_onset (ie. reduce by 75% of SFR_onset)
        
        # !!! January 29, 2024
        # subID 43 wasn't used to create SKIRT synthetic imaging, for whatever
        # reason? -> try to track this down
        
        # for testing of support vector machines, look for drop of 50% of SFR_onset
        SFR_onset = SFH[ionset]
        imid = ionset + np.where(SFH[ionset:]/SFR_onset < 0.5)[0][0]
        
        # record the fraction of SFR_onset at that point
        fraction_of_SFR_onset = SFH[imid]/SFR_onset
        
        # record the midpoint snapshot for future use, and the SFR fraction
        # at midpoint
        imids[i] = imid
        SFR_onset_fractions_at_midpoint[i] = fraction_of_SFR_onset
    
    # check the distribution -> low values around ~0.3? really?
    # plt.histogram(SFR_onset_fractions_at_midpoint,
    #               r'${\rm SFR}_{\rm onset}$ fraction at midpoint')
    # sort = np.argsort(SFR_onset_fractions_at_midpoint)
    # print(subIDs[:, -1][sort])
    # NOTE: by visual inspection, values with low (~0.3) SFR_onset fractions
    # quench very rapidly, which makes sense (e.g., subIDs 362994, 167434, etc.)
    # plt.plot_simple_dumb(tterms - tonsets, SFR_onset_fractions_at_midpoint,
    #     xlabel = 'Quenching Episode Length (Gyr)',
    #     ylabel = r'Fraction of ${\rm SFR}_{\rm onset}$ at midpoint')
    
    # for subIDfinal, ionset, imid, iterm, check in zip(subIDs[:, -1], ionsets,
    #     imids, iterms, imids<=iterms) :
    #     print(subIDfinal, ionset, imid, iterm, check)
    
    # determine the redshifts that correspond to the midway points
    z_mids = redshifts[imids]
    # plt.histogram(z_mids, r'$z$')
    
    # get subIDs, stellar masses, sizes, centers midway through quenching
    firstDim = np.arange(278)
    subIDs_midway = subIDs[firstDim, imids]
    logM_midway = logM[firstDim, imids]
    Re_midway = Res[firstDim, imids] # ckpc/h
    centers_midway = centers[firstDim, imids]
    sSFR_radial_profiles_midway = sSFR_radial_profiles[firstDim, imids]
    
    # !!! generalize to loop through the quenching episode for every galaxy,
    # as opposed to just at the midway point
    # loop over every quenched galaxy's midway point, calculating morphological
    # parameters
    C_SFs = np.full(278, np.nan)
    R_SFs = np.full(278, np.nan)
    Rinners = np.full(278, np.nan)
    Routers = np.full(278, np.nan)
    for i, (subIDfinal, imid, sub, mass, rad, cent, profile) in enumerate(zip(
        subIDs[:, -1], imids, subIDs_midway, logM_midway, Re_midway,
        centers_midway, sSFR_radial_profiles_midway)) :
        
        # get all particles
        ages, masses, dx, dy, dz = get_particle_positions('TNG50-1', 99, imid,
            sub, cent)
        
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
            _, sf_masses, sf_rs = get_sf_particles(ages, masses, rs,
                times[imid], delta_t=delta_t)
            
            # compute the morphological parameters
            C_SF = calculate_C_SF(sf_masses, sf_rs, rad)
            R_SF = calculate_R_SF(masses, rs, sf_masses, sf_rs)
            Rinner = calculate_Rinner(radial_bin_centers, profile)
            Router = calculate_Router(radial_bin_centers, profile)
        else :
            C_SF = np.nan
            R_SF = np.nan
            Rinner = np.nan
            Router = np.nan
        
        C_SFs[i] = C_SF
        R_SFs[i] = R_SF
        Rinners[i] = Rinner
        Routers[i] = Router
    
    table = Table([subIDs[:, -1], z_mids, SFR_onset_fractions_at_midpoint,
                   logM_midway, C_SFs, R_SFs, Rinners, Routers],
                  names=('subID', 'redshift', 'SFR_onset fraction', 'logM',
                         'C_SF', 'R_SF', 'Rinner', 'Router'))
    table.write(outfile)
    
    # work on SF matched mass sample, and compute morph. parameters for them
    # as well, in a similar manner as above
    
    return

def determine_diagnostics(delta_t=100*u.Myr, version='2D') :
    
    infile = 'TNG50-1/TNG50-1_99_sample(t).hdf5'
    
    if version == '2D' :
        outfile = 'TNG50-1/TNG50-1_99_diagnostics(t)_2d.hdf5'
    else :
        outfile = 'TNG50-1/TNG50-1_99_diagnostics(t).hdf5'
    
    with h5py.File(infile, 'r') as hf :
        snaps = hf['snapshots'][:]
        times = hf['times'][:]
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:, -1]
        Res = hf['Re'][:]
        centers = hf['centers'][:]
        ionsets = hf['onset_indices'][:]
        tonsets = hf['onset_times'][:]
        iterms = hf['termination_indices'][:]
        tterms = hf['termination_times'][:]
        quenched = hf['quenched'][:]
    
    # get snapshots corresponding to 25%, 50%, and 75% through the quenching episode
    iminorities = np.array(find_nearest(times, tonsets + 0.25*(tterms - tonsets)))
    ihalfways = np.array(find_nearest(times, tonsets + 0.5*(tterms - tonsets)))
    imajorities = np.array(find_nearest(times, tonsets + 0.75*(tterms - tonsets)))
    
    if not exists(outfile) :
        with h5py.File(outfile, 'w') as hf :
            add_dataset(hf, np.full((8260, 100), np.nan), 'sf_mass_within_1kpc')
            add_dataset(hf, np.full((8260, 100), np.nan), 'sf_mass_within_tenthRe')
            add_dataset(hf, np.full((8260, 100), np.nan), 'sf_mass')
            
            add_dataset(hf, np.full((8260, 100), np.nan), 'R10')
            add_dataset(hf, np.full((8260, 100), np.nan), 'R50')
            add_dataset(hf, np.full((8260, 100), np.nan), 'R90')
            add_dataset(hf, np.full((8260, 100), np.nan), 'sf_R10')
            add_dataset(hf, np.full((8260, 100), np.nan), 'sf_R50')
            add_dataset(hf, np.full((8260, 100), np.nan), 'sf_R90')
    
    with h5py.File(outfile, 'r') as hf :
        x_within_1kpc = hf['sf_mass_within_1kpc'][:]
        x_within_tenthRe = hf['sf_mass_within_tenthRe'][:]
        x_sf_mass = hf['sf_mass'][:]
        
        x_R10 = hf['R10'][:]
        x_R50 = hf['R50'][:]
        x_R90 = hf['R90'][:]
        x_sf_R10 = hf['sf_R10'][:]
        x_sf_R50 = hf['sf_R50'][:]
        x_sf_R90 = hf['sf_R90'][:]
    
    # loop over every galaxy in the sample
    for i, (mpb_subIDs, mass, mpb_Res, mpb_centers, use, ionset, iminority,
        ihalfway, imajority, iterm) in enumerate(zip(subIDs, logM, Res,
        centers, quenched, ionsets, iminorities, ihalfways, imajorities,
        iterms)) :
            
        # only compute for quenched galaxies with sufficient stellar mass,
        # and don't replace existing values
        if (use and (mass >= 9.5) and np.all(np.isnan(x_within_1kpc[i])) and
            np.all(np.isnan(x_within_tenthRe)) and
            np.all(np.isnan(x_sf_mass[i])) and
            np.all(np.isnan(x_R10[i])) and np.all(np.isnan(x_R50[i])) and
            np.all(np.isnan(x_R90[i])) and np.all(np.isnan(x_sf_R10[i])) and
            np.all(np.isnan(x_sf_R50[i])) and np.all(np.isnan(x_sf_R90[i]))) :
            
            # find the indices 1 Gyr before and after the quenching episode
            # start, end = find_nearest(times, [tonset-1, tterm+1])
            
            sf_masses_within_1kpc, sf_masses_within_tenthRe = [], []
            sf_masses_total = []
            R10s, R50s, R90s = [], [], []
            sf_R10s, sf_R50s, sf_R90s = [], [], []
            # loop over all the snapshots
            for snap, time, subID, Re, center in zip(snaps, times,
                mpb_subIDs, mpb_Res, mpb_centers) :
                
                # if the snapshot is between 1 Gyr before the onset time
                # and 1 Gyr after the termination time, compute the
                # diagnostics
                # if (start <= snap <= end) :
                if snap in [ionset, iminority, ihalfway, imajority, iterm] :
                    
                    # get all particles
                    ages, masses, dx, dy, dz = get_particle_positions(
                        'TNG50-1', 99, snap, subID, center)
                    
                    # only proceed if the ages, masses, and distances are
                    # intact
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
                            rs, time, delta_t=delta_t)
                        
                        # compute the diagnostics
                        (sf_mass_within_1kpc, sf_mass_within_tenthRe,
                         sf_mass) = compute_C_SF(sf_masses, sf_rs, Re)
                        
                        (R10, R50, R90,
                         sf_R10, sf_R50, sf_R90) = compute_R_SF(masses, rs,
                            sf_masses, sf_rs)
                    else : # otherwise append NaNs
                        sf_mass_within_1kpc = np.nan
                        sf_mass_within_tenthRe = np.nan
                        sf_mass = np.nan
                        
                        R10, R50, R90 = np.nan, np.nan, np.nan
                        sf_R10, sf_R50, sf_R90 = np.nan, np.nan, np.nan
                    
                    sf_masses_within_1kpc.append(sf_mass_within_1kpc)
                    sf_masses_within_tenthRe.append(sf_mass_within_tenthRe)
                    sf_masses_total.append(sf_mass)
                    
                    R10s.append(R10)
                    R50s.append(R50)
                    R90s.append(R90)
                    sf_R10s.append(sf_R10)
                    sf_R50s.append(sf_R50)
                    sf_R90s.append(sf_R90)
                else :
                    sf_masses_within_1kpc.append(np.nan)
                    sf_masses_within_tenthRe.append(np.nan)
                    sf_masses_total.append(np.nan)
                    
                    R10s.append(np.nan)
                    R50s.append(np.nan)
                    R90s.append(np.nan)
                    sf_R10s.append(np.nan)
                    sf_R50s.append(np.nan)
                    sf_R90s.append(np.nan)
            
            # populate the values in the output file
            with h5py.File(outfile, 'a') as hf :
                hf['sf_mass_within_1kpc'][i] = sf_masses_within_1kpc
                hf['sf_mass_within_tenthRe'][i] = sf_masses_within_tenthRe
                hf['sf_mass'][i] = sf_masses_total
                
                hf['R10'][i] = R10s
                hf['R50'][i] = R50s
                hf['R90'][i] = R90s
                hf['sf_R10'][i] = sf_R10s
                hf['sf_R50'][i] = sf_R50s
                hf['sf_R90'][i] = sf_R90s
        
        print('{} done'.format(mpb_subIDs[-1]))
    
    return

def determine_diagnostics_for_matched_sample(version='2D') :
    
    if version == '2D' :
        outfile = 'TNG50-1/TNG50-1_99_diagnostics(t)_2d.hdf5'
    else :
        outfile = 'TNG50-1/TNG50-1_99_diagnostics(t).hdf5'
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        snaps = hf['snapshots'][:]
        times = hf['times'][:]
        subIDs = hf['subIDs'][:].astype(int)
        Res = hf['Re'][:]
        centers = hf['centers'][:]
        # ionsets = hf['onset_indices'][:]
        # tonsets = hf['onset_times'][:]
        # iterms = hf['termination_indices'][:]
        # tterms = hf['termination_times'][:]
        comparison = hf['comparison'][:]
        i0s = hf['i0s'][:].astype(int)
        i25s = hf['i25s'][:].astype(int)
        i50s = hf['i50s'][:].astype(int)
        i75s = hf['i75s'][:].astype(int)
        i100s = hf['i100s'][:].astype(int)
    
    with h5py.File(outfile, 'r') as hf :
        x_within_1kpc = hf['sf_mass_within_1kpc'][:]
        x_within_tenthRe = hf['sf_mass_within_tenthRe'][:]
        x_sf_mass = hf['sf_mass'][:]
        
        x_R10 = hf['R10'][:]
        x_R50 = hf['R50'][:]
        x_R90 = hf['R90'][:]
        x_sf_R10 = hf['sf_R10'][:]
        x_sf_R50 = hf['sf_R50'][:]
        x_sf_R90 = hf['sf_R90'][:]
    
    # loop over everything in the sample
    for loc, ionset, iminority, ihalfway, imajority, iterm in zip(
        np.arange(8260)[comparison], i0s[comparison], i25s[comparison],
        i50s[comparison], i75s[comparison], i100s[comparison]) :
        
        # loop over all the snapshots
        for snap in snaps :
            if snap in [ionset, iminority, ihalfway, imajority, iterm] :
                
                # don't overwrite existing values
                if (np.isnan(x_within_1kpc[loc, snap]) and
                    np.isnan(x_within_tenthRe[loc, snap]) and
                    np.isnan(x_sf_mass[loc, snap]) and
                    np.isnan(x_R10[loc, snap]) and
                    np.isnan(x_R50[loc, snap]) and
                    np.isnan(x_R90[loc, snap]) and
                    np.isnan(x_sf_R10[loc, snap]) and
                    np.isnan(x_sf_R50[loc, snap]) and
                    np.isnan(x_sf_R90[loc, snap])) :
                    
                    # get the subID and parameters for the comparison galaxy
                    subID = subIDs[loc, snap]
                    Re = Res[loc, snap]
                    center = centers[loc, snap]
                    
                    # get all particles
                    ages, masses, dx, dy, dz = get_particle_positions(
                        'TNG50-1', 99, snap, subID, center)
                    
                    # only proceed if the ages, masses, and distances are
                    # intact
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
                            rs, times[snap], delta_t=100*u.Myr)
                        
                        # compute the diagnostics
                        (sf_mass_within_1kpc, sf_mass_within_tenthRe,
                         sf_mass) = compute_C_SF(sf_masses, sf_rs, Re)
                        
                        (R10, R50, R90,
                         sf_R10, sf_R50, sf_R90) = compute_R_SF(masses, rs,
                            sf_masses, sf_rs)
                        
                        # populate the values in the output file
                        with h5py.File(outfile, 'a') as hf :
                            hf['sf_mass_within_1kpc'][loc, snap] = sf_mass_within_1kpc
                            hf['sf_mass_within_tenthRe'][loc, snap] = sf_mass_within_tenthRe
                            hf['sf_mass'][loc, snap] = sf_mass
                            
                            hf['R10'][loc, snap] = R10
                            hf['R50'][loc, snap] = R50
                            hf['R90'][loc, snap] = R90
                            hf['sf_R10'][loc, snap] = sf_R10
                            hf['sf_R50'][loc, snap] = sf_R50
                            hf['sf_R90'][loc, snap] = sf_R90
        
        print('{} done'.format(subIDs[loc, -1]))
    
    return

def find_average_diagnostic_values() :
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        times = hf['times'][:]
        subIDfinals = hf['SubhaloID'][:]
        logM = hf['logM'][:, -1]
        ionsets = hf['onset_indices'][:].astype(int)
        tonsets = hf['onset_times'][:]
        iterms = hf['termination_indices'][:].astype(int)
        tterms = hf['termination_times'][:]
        quenched = hf['quenched'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_diagnostics(t).hdf5', 'r') as hf :
        xis = hf['xi'][:]
        zetas = hf['zeta'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_mechanism.hdf5', 'r') as hf :
        outside_in = hf['outside-in'][:] # 109
        inside_out = hf['inside-out'][:] # 103
        uniform = hf['uniform'][:]       # 8
        ambiguous = hf['ambiguous'][:]   # 58
    
    oi_xi_full, oi_xi_first, oi_xi_second = [], [], []
    oi_zeta_full, oi_zeta_first, oi_zeta_second = [], [], []
    
    io_xi_full, io_xi_first, io_xi_second = [], [], []
    io_zeta_full, io_zeta_first, io_zeta_second = [], [], []
    
    uni_xi_full, uni_xi_first, uni_xi_second = [], [], []
    uni_zeta_full, uni_zeta_first, uni_zeta_second = [], [], []
    # loop over every galaxy in the sample
    for (subIDfinal, mass, use, ionset, tonset, iterm, tterm, xi, zeta, oi, io,
         uni, am) in zip(subIDfinals, logM, quenched, ionsets, tonsets, iterms,
                         tterms, xis, zetas, outside_in, inside_out, uniform,
                         ambiguous) :
            
            if use and (mass >= 9.5) :
                
                # split the results into a first half and a second half
                mid = find_nearest(times, [0.5*(tonset + tterm)])[0]
                
                if oi :
                    for val in xi[ionset:iterm+1] :
                        oi_xi_full.append(val)
                    for val in zeta[ionset:iterm+1] :
                        oi_zeta_full.append(val)
                    
                    for val in xi[ionset:mid+1] :
                        oi_xi_first.append(val)
                    for val in zeta[ionset:mid+1] :
                        oi_zeta_first.append(val)
                    for val in xi[mid:iterm+1] :
                        oi_xi_second.append(val)
                    for val in zeta[mid:iterm+1] :
                        oi_zeta_second.append(val)
                
                if io :
                    for val in xi[ionset:iterm+1] :
                        io_xi_full.append(val)
                    for val in zeta[ionset:iterm+1] :
                        io_zeta_full.append(val)
                    
                    for val in xi[ionset:mid+1] :
                        io_xi_first.append(val)
                    for val in zeta[ionset:mid+1] :
                        io_zeta_first.append(val)
                    for val in xi[mid:iterm+1] :
                        io_xi_second.append(val)
                    for val in zeta[mid:iterm+1] :
                        io_zeta_second.append(val)
                
                if uni :
                    for val in xi[ionset:iterm+1] :
                        uni_xi_full.append(val)
                    for val in zeta[ionset:iterm+1] :
                        uni_zeta_full.append(val)
                    
                    for val in xi[ionset:mid+1] :
                        uni_xi_first.append(val)
                    for val in zeta[ionset:mid+1] :
                        uni_zeta_first.append(val)
                    for val in xi[mid:iterm+1] :
                        uni_xi_second.append(val)
                    for val in zeta[mid:iterm+1] :
                        uni_zeta_second.append(val)
    
    # compare distributions in the first half compared to the second half,
    # and compared to the entire duration
    # print(np.percentile(oi_xi_first, [16, 50, 84]))
    # print(np.nanpercentile(oi_xi_second, [16, 50, 84]))
    # print(np.percentile(oi_zeta_first, [16, 50, 84]))
    # print(np.nanpercentile(oi_zeta_second, [16, 50, 84]))
    # print()
    # print(np.percentile(io_xi_first, [16, 50, 84]))
    # print(np.nanpercentile(io_xi_second, [16, 50, 84]))
    # print(np.percentile(io_zeta_first, [16, 50, 84]))
    # print(np.nanpercentile(io_zeta_second, [16, 50, 84]))
    # print()
    # print(np.percentile(uni_xi_first, [16, 50, 84]))
    # print(np.nanpercentile(uni_xi_second, [16, 50, 84]))
    # print(np.percentile(uni_zeta_first, [16, 50, 84]))
    # print(np.nanpercentile(uni_zeta_second, [16, 50, 84]))
    # plt.histogram_multi([oi_xi_full, oi_xi_first, oi_xi_second],
    #     r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['full', 'first half', 'second half'], title='outside-in')
    # plt.histogram_multi([oi_zeta_full, oi_zeta_first, oi_zeta_second],
    #     r'$R_{\rm SF} = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['full', 'first half', 'second half'], title='outside-in')
    # plt.histogram_multi([io_xi_full, io_xi_first, io_xi_second],
    #     r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['full', 'first half', 'second half'], title='inside-out')
    # plt.histogram_multi([io_zeta_full, io_zeta_first, io_zeta_second],
    #     r'$R_{\rm SF} = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['full', 'first half', 'second half'], title='inside-out')
    # plt.histogram_multi([uni_xi_full, uni_xi_first, uni_xi_second],
    #     r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['full', 'first half', 'second half'], title='uniform')
    # plt.histogram_multi([uni_zeta_full, uni_zeta_first, uni_zeta_second],
    #     r'$R_{\rm SF} = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['full', 'first half', 'second half'], title='uniform')
    
    # compare the distributions amongst different quenching mechanisms
    # oi_xi_weight = np.ones(len(oi_xi_full))/len(oi_xi_full)
    # io_xi_weight = np.ones(len(io_xi_full))/len(io_xi_full)
    # uni_xi_weight = np.ones(len(uni_xi_full))/len(uni_xi_full)
    # plt.histogram_multi([oi_xi_full, io_xi_full, uni_xi_full],
    #     r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['outside-in', 'inside-out', 'uniform'],
    #     [oi_xi_weight, io_xi_weight, uni_xi_weight])
    # oi_zeta_weight = np.ones(len(oi_zeta_full))/len(oi_zeta_full)
    # io_zeta_weight = np.ones(len(io_zeta_full))/len(io_zeta_full)
    # uni_zeta_weight = np.ones(len(uni_zeta_full))/len(uni_zeta_full)
    # plt.histogram_multi([oi_zeta_full, io_zeta_full, uni_zeta_full],
    #     r'$R_{\rm SF} = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['outside-in', 'inside-out', 'uniform'],
    #     [oi_zeta_weight, io_zeta_weight, uni_zeta_weight])
    
    '''
    # C_SF and R_SF values are from different distributions when comparing
    # outside-in to inside-out
    # print(ks_2samp(oi_xi_full, io_xi_full))
    # print(ks_2samp(oi_zeta_full, io_zeta_full))
    # print(anderson_ksamp([oi_xi_full, io_xi_full]))
    # print(anderson_ksamp([oi_zeta_full, io_zeta_full]))
    
    # C_SF and R_SF values are from different distributions when comparing
    # outside-in to uniform
    # print(ks_2samp(oi_xi_full, uni_xi_full))
    # print(ks_2samp(oi_zeta_full, uni_zeta_full))
    # print(anderson_ksamp([oi_xi_full, uni_xi_full]))
    # print(anderson_ksamp([oi_zeta_full, uni_zeta_full]))
    
    # C_SF and R_SF values are from different distributions when comparing
    # inside-out to uniform, but are more similar than the other comparisons
    # print(ks_2samp(io_xi_full, uni_xi_full))
    # print(ks_2samp(io_zeta_full, uni_zeta_full))
    # print(anderson_ksamp([io_xi_full, uni_xi_full]))
    # print(anderson_ksamp([io_zeta_full, uni_zeta_full]))
    '''
    
    # find expected values based on the distributions
    # print(np.nanpercentile(oi_xi_full, 50), np.nanpercentile(oi_zeta_full, 50))
    # print(np.nanpercentile(io_xi_full, 50), np.nanpercentile(io_zeta_full, 50))
    # print(np.percentile(uni_xi_full, 50), np.percentile(uni_zeta_full, 50))
    
    return

def find_matched_sample() :
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        times = hf['times'][:]
        # subIDfinals = hf['SubhaloID'][:]
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        # Re = hf['Re'][:]
        SFMS = hf['SFMS'][:].astype(bool) # (boolean) SFMS at each snapshot
        ionsets = hf['onset_indices'][:].astype(int)
        tonsets = hf['onset_times'][:]
        iterms = hf['termination_indices'][:].astype(int)
        tterms = hf['termination_times'][:]
        quenched = hf['quenched'][:]
        # cluster = hf['cluster'][:]
        # hmGroup = hf['hm_group'][:]
        # lmGroup = hf['lm_group'][:]
        # field = hf['field'][:]
    
    # get snapshots corresponding to 25%, 50%, and 75% through the quenching episode
    iminorities = np.array(find_nearest(times, tonsets + 0.25*(tterms - tonsets)))
    ihalfways = np.array(find_nearest(times, tonsets + 0.5*(tterms - tonsets)))
    imajorities = np.array(find_nearest(times, tonsets + 0.75*(tterms - tonsets)))
    
    # envs = np.array([4*cluster, 3*hmGroup, 2*lmGroup, field]).T
    # envs = np.sum(envs, axis=1)
    
    empty = np.full(8260, np.nan)
    all_locs = empty.copy()
    i0s, i25s, i50s = empty.copy(), empty.copy(), empty.copy()
    i75s, i100s = empty.copy(), empty.copy()
    # loop over every galaxy in the sample
    for i, (mpb_subIDs, mpb_logM, use, start, early, middle, snap,
        end) in enumerate(zip(subIDs, logM, quenched, ionsets, iminorities,
        ihalfways, imajorities, iterms)) :
        
        if use and (mpb_logM[-1] >= 9.5) :
            
            # get values at the snapshot
            mass = mpb_logM[snap]
            logM_at_snap = logM[:, snap]
            SFMS_at_snap = SFMS[:, snap]
            
            # find galaxies of a similar mass as the galaxy, but that are on
            # the SFMS at that snapshot
            comparisons = ((SFMS_at_snap > 0) & (logM_at_snap != mass) &
                           (np.abs(logM_at_snap - mass) <= 0.1) & ~quenched)
            
            # find the indices that correpsond to the comparison galaxies
            locs = np.nonzero(comparisons)[0]
            # numComparisons = np.sum(comparisons)
            
            diffs = np.abs(logM_at_snap[locs] - mass)
            
            # only 1 galaxy has 0 comparison galaxies
            # if numComparisons > 0 :
            best = np.argsort(diffs) # sort by the difference in stellar mass
            
            # check that the comparison galaxy isn't already included in
            # the comparison locations
            if locs[best[0]] not in all_locs :
                all_locs[i] = locs[best[0]]
            elif locs[best[1]] not in all_locs :
                all_locs[i] = locs[best[1]]
            elif locs[best[2]] not in all_locs :
                all_locs[i] = locs[best[2]]
            elif locs[best[3]] not in all_locs :
                all_locs[i] = locs[best[3]]
            elif locs[best[4]] not in all_locs :
                all_locs[i] = locs[best[4]]
            elif locs[best[5]] not in all_locs :
                all_locs[i] = locs[best[5]]
            elif locs[best[6]] not in all_locs :
                all_locs[i] = locs[best[6]]
            else :
                all_locs[i] = locs[best[7]]
            
            loc = all_locs[i].astype(int)
            
            # place the corresponding snapshots into relevant arrays
            i0s[loc] = start
            i25s[loc] = early
            i50s[loc] = middle
            i75s[loc] = snap
            i100s[loc] = end
    
    # get the locations as ints for indexing
    all_locs = all_locs[np.isfinite(all_locs)].astype(int)
    
    # create a new mask which will set the comparison galaxies
    comparison = np.full(8260, False)
    comparison[all_locs] = True
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'a') as hf :
        add_dataset(hf, comparison, 'comparison')
        add_dataset(hf, i0s, 'i0s')
        add_dataset(hf, i25s, 'i25s')
        add_dataset(hf, i50s, 'i50s')
        add_dataset(hf, i75s, 'i75s')
        add_dataset(hf, i100s, 'i100s')
    
    return

def find_control_sf_sample(plot_SFHs=False) :
    # !!! February 27th, 2024
    # this should (effectively) replace the function find_matched_sample above
    
    '''
    imids = np.array([35, 67, 44, 49, 81, 86, 68, 61, 50, 23, # maybe save
                      60, 60, 59, 57, 48, 65, 54, 34, 34, 48, # imids to file,
                      70, 50, 31, 36, 94, 70, 44, 41, 67, 47, # that way they
                      44, 64, 37, 35, 44, 41, 37, 56, 44, 78, # can be accessed
                      32, 41, 66, 31, 73, 42, 66, 56, 53, 87, # in a better
                       7, 57, 76, 54, 43, 45, 71, 66, 59, 36, # way, cause
                      37, 77, 53, 67, 55, 57, 42, 36, 82, 60, # this isn't
                      39, 58, 60, 61, 50, 47, 96, 87, 50, 42, # ideal right now
                      58, 46, 66, 54, 41, 62, 77, 57, 67, 79,
                      64, 84, 54, 47, 61, 56, 57, 65, 51, 59,
                      42, 57, 42, 85, 61, 36, 29, 53, 57, 54,
                      63, 43, 52, 89, 60, 43, 44, 49, 44, 84,
                      95, 51, 54, 45, 71, 57, 42, 42, 50, 92,
                      52, 48, 40, 70, 58, 61, 52, 44, 78, 80,
                      50, 46, 50, 60, 26, 82, 41, 73, 69, 55,
                      69, 65, 62, 76, 71, 48, 47, 86, 68, 46,
                      74, 65, 70, 44, 52, 35, 58, 68, 45, 40,
                      24, 72, 85, 48, 90, 54, 58, 39, 68, 55,
                      44, 81, 77, 70, 44, 60, 51, 44, 54, 66,
                      54, 54, 45, 29, 75, 70, 47, 61, 44, 62,
                      86, 52, 50, 35, 48, 75, 53, 71, 93, 85,
                      96, 51, 77, 37, 36, 55, 77, 71, 72, 76,
                      80, 41, 38, 38, 64, 28, 69, 84, 53, 96,
                      64, 56, 87, 58, 69, 70, 39, 77, 77, 71,
                      66, 63, 69, 85, 60, 72, 55, 81, 68, 49,
                      91, 75, 59, 59, 65, 94, 89, 83, 92, 63,
                      49, 74, 97, 79, 97, 96, 79, 81, 75, 83,
                      93, 60, 96, 71, 77, 85, 70, 71])
    '''
    
    infile = 'TNG50-1/TNG50-1_99_sample(t).hdf5'
    
    # get basic information about the sample, where some parameters are a
    # function of time
    with h5py.File(infile, 'r') as hf :
        redshifts = hf['redshifts'][:]
        times = hf['times'][:] # for determining active SF populations
        subIDs = hf['subIDs'][:].astype(int)
        # subIDfinals = hf['SubhaloID'][:]
        logM = hf['logM'][:]
        # Res = hf['Re'][:]
        # centers = hf['centers'][:]
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
    # Res = Res[mask]           # (1666, 100)
    # centers = centers[mask]   # (1666, 100, 3)
    ionsets = ionsets[mask]   # (1666)
    tonsets = tonsets[mask]   # (1666)
    iterms = iterms[mask]     # (1666)
    tterms = tterms[mask]     # (1666)
    SFHs = SFHs[mask]         # (1666, 100)
    SFMS = SFMS[mask]         # (1666, 100)
    quenched = quenched[mask] # (1666)
    
    if not exists('locations_mask_quenched.npy') :
        # create a simple mask to select all quenching episode locations
        unique_quenched = np.full((1666, 100), False)
        for i, (status, ionset, iterm) in enumerate(zip(quenched, ionsets, iterms)) :
            if status :
                unique_quenched[i, ionset:iterm+1] = True
        np.save('locations_mask_quenched.npy', unique_quenched) # sum = 4695 = 4417 + 278
    
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
    
    if not exists('all_control_sf_points.fits') :
        t = Table([all_quenched_subIDs, all_control_subIDs,
                   all_episode_progresses, all_redshifts, all_masses],
                  names=('quenched_subID', 'control_subID',
                         'episode_progress', 'redshift', 'logM'))
        t.write('all_control_sf_points.fits')
    
    if not exists('locations_mask_control_sf.npy') :
        np.save('locations_mask_control_sf.npy', unique_sf) # sum = 27763
    
    if not exists('check-N_always_on_SFMS-during-quenching-episode.fits') :
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
        t.write('check-N_always_on_SFMS-during-quenching-episode.fits')
    
    return

def get_locations_of_potential_starbursts() :
    
    outfile = 'TNG50-1/TNG50-1_99_potential_starburst_locations.hdf5'
    
    with h5py.File('TNG50-1/TNG50-1_99_midpoints.hdf5', 'r') as hf :
        imids = hf['middle_indices'][:].astype(int)
    
    with h5py.File('TNG50-1/TNG50-1_99_diagnostics(t).hdf5', 'r') as hf :
        xis = hf['xi'][:]
        zetas = hf['zeta'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_matched_sample_locations.hdf5', 'r') as hf :
        locs = hf['locations'][:, 0].astype(int) # only use the first location
                                                 # for simplicity
    
    starburst = np.full(8260, False)
    starburst_snaps = np.full(8260, np.nan)
    # loop over everything in the sample
    for loc, snap in zip(locs, imids) :
        
        # only use the valid entries
        if ((snap >= 0) and (loc >= 0) and (xis[loc, snap] >= 0.6) and
            (np.log10(zetas[loc, snap]) <= 0.0)) :
            starburst[loc] = True
            starburst_snaps[loc] = snap
    
    with h5py.File(outfile, 'w') as hf :
        add_dataset(hf, starburst, 'starburst')
        add_dataset(hf, starburst_snaps, 'starburst_snaps')
    
    return

def get_values_for_plotting(comparison_dict, quenched_dict, xmetric, ymetric,
                            index, mech, version, logx=False, logy=True) :
    
    labels = {'xis':r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
              'alt_xis':r'$C_{\rm SF} = {\rm SFR}_{<0.1~R_{\rm e}}/{\rm SFR}_{\rm total}$',
              'zetas':r'$\log{(R_{\rm SF} = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}})}$',
              'R10_zetas':r'$\log{(R_{\rm SF} = R_{{\rm 10}_{*, {\rm SF}}}/R_{{\rm 10}_{*, {\rm total}}})}$',
              'R90_zetas':r'$\log{(R_{\rm SF} = R_{{\rm 90}_{*, {\rm SF}}}/R_{{\rm 90}_{*, {\rm total}}})}$'}
    
    if version == 'outside-in' :
        version = 4
        colours = ['k', 'r']
    elif version == 'inside-out' :
        version = 3
        colours = ['k', 'm']
    elif version == 'uniform' :
        version = 2
        colours = ['k', 'cyan']
    elif version == 'ambiguous' :
        version = 1
        colours = ['k', 'b']
    
    comparison_x = comparison_dict[xmetric][index]
    comparison_y = comparison_dict[ymetric][index]
    
    q_x = quenched_dict[xmetric][index][mech == version]
    q_y = quenched_dict[ymetric][index][mech == version]
    
    if logx :
        xs = [np.log10(comparison_x), np.log10(q_x)]
    else :
        xs = [comparison_x, q_x]
    
    if logy :
        ys = [np.log10(comparison_y), np.log10(q_y)]
    else :
        ys = [comparison_y, q_y]
    
    return xs, ys, colours, labels[xmetric], labels[ymetric]

def merge_diagnostic_plots_based_on_mechanism() :
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        subIDfinals = hf['SubhaloID'][:]
        logM = hf['logM'][:, -1]
        quenched = hf['quenched'][:]
    
    # 278 total galaxies with logM >= 9.5
    with h5py.File('TNG50-1/TNG50-1_99_mechanism.hdf5', 'r') as hf :
        outside_in = hf['outside-in'][:] # 109
        inside_out = hf['inside-out'][:] # 103
        uniform = hf['uniform'][:]       # 8
        ambiguous = hf['ambiguous'][:]   # 58
    
    inDir = 'TNG50-1/figures/diagnostics(t)/'
    
    mask = (logM >= 9.5) & quenched
    subIDfinals = subIDfinals[mask]
    outside_in = outside_in[mask]
    inside_out = inside_out[mask]
    uniform = uniform[mask]
    ambiguous = ambiguous[mask]
    logM = logM[mask]
    
    sort = np.argsort(logM)
    subIDfinals = subIDfinals[sort]
    outside_in = outside_in[sort]
    inside_out = inside_out[sort]
    uniform = uniform[sort]
    ambiguous = ambiguous[sort]
    logM = logM[sort]

    merger = PdfWriter()
    for subID in subIDfinals[outside_in] :
        merger.append(inDir + 'subID_{}.pdf'.format(subID))
    outfile = 'TNG50-1/figures/diagnostics(t)_merged_mechanism/outside_in.pdf'
    merger.write(outfile)
    merger.close()
    
    merger = PdfWriter()
    for subID in subIDfinals[inside_out] :
        merger.append(inDir + 'subID_{}.pdf'.format(subID))
    outfile = 'TNG50-1/figures/diagnostics(t)_merged_mechanism/inside_out.pdf'
    merger.write(outfile)
    merger.close()
    
    merger = PdfWriter()
    for subID in subIDfinals[uniform] :
        merger.append(inDir + 'subID_{}.pdf'.format(subID))
    outfile = 'TNG50-1/figures/diagnostics(t)_merged_mechanism/uniform.pdf'
    merger.write(outfile)
    merger.close()
    
    merger = PdfWriter()
    for subID in subIDfinals[ambiguous] :
        merger.append(inDir + 'subID_{}.pdf'.format(subID))
    outfile = 'TNG50-1/figures/diagnostics(t)_merged_mechanism/ambiguous.pdf'
    merger.write(outfile)
    merger.close()
    
    return

def plot_diskRatios_vs_concentrations() :
    
    outDir = 'TNG50-1/figures/zeta_vs_xi/'
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        times = hf['times'][:]
        # subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:, -1]
        ionsets = hf['onset_indices'][:].astype(int)
        tonsets = hf['onset_times'][:]
        iterms = hf['termination_indices'][:].astype(int)
        tterms = hf['termination_times'][:]
        quenched = hf['quenched'][:]
        # cluster = hf['cluster'][:]
        # hmGroup = hf['hm_group'][:]
        # lmGroup = hf['lm_group'][:]
        # field = hf['field'][:]
        comparison = hf['comparison'][:]
        i0s = hf['i0s'][:].astype(int)
        i25s = hf['i25s'][:].astype(int)
        i50s = hf['i50s'][:].astype(int)
        i75s = hf['i75s'][:].astype(int)
        i100s = hf['i100s'][:].astype(int)
    
    # get snapshots corresponding to 25%, 50%, and 75% through the quenching episode
    iminorities = np.array(find_nearest(times, tonsets + 0.25*(tterms - tonsets)))
    ihalfways = np.array(find_nearest(times, tonsets + 0.5*(tterms - tonsets)))
    imajorities = np.array(find_nearest(times, tonsets + 0.75*(tterms - tonsets)))
    
    with h5py.File('TNG50-1/TNG50-1_99_diagnostics(t)_2d.hdf5', 'r') as hf :
        sf_mass_within_1kpc = hf['sf_mass_within_1kpc'][:]
        sf_mass_within_tenthRe = hf['sf_mass_within_tenthRe'][:]
        sf_mass_tot = hf['sf_mass'][:]
        R10s = hf['R10'][:]
        R50s = hf['R50'][:]
        R90s = hf['R90'][:]
        sf_R10s = hf['sf_R10'][:]
        sf_R50s = hf['sf_R50'][:]
        sf_R90s = hf['sf_R90'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_mechanism.hdf5', 'r') as hf :
        outside_in = hf['outside-in'][:] # 109
        inside_out = hf['inside-out'][:] # 103
        uniform = hf['uniform'][:]       # 8
        ambiguous = hf['ambiguous'][:]   # 58
    
    # create various metrics of interest
    xis = sf_mass_within_1kpc/sf_mass_tot
    alt_xis = sf_mass_within_tenthRe/sf_mass_tot
    
    R10_zetas = sf_R10s/R10s
    zetas = sf_R50s/R50s
    R90_zetas = sf_R90s/R90s
    
    # get values for the comparison sample
    firstDim = np.arange(278)
    
    i0s = i0s[comparison]
    i25s = i25s[comparison]
    i50s = i50s[comparison]
    i75s = i75s[comparison]
    i100s = i100s[comparison]
    
    c_xis = xis[comparison]
    c_alt_xis = alt_xis[comparison]
    c_R10_zetas = R10_zetas[comparison]
    c_zetas = zetas[comparison]
    c_R90_zetas = R90_zetas[comparison]
    
    c_xis = np.array([c_xis[firstDim, i0s], c_xis[firstDim, i25s],
                      c_xis[firstDim, i50s], c_xis[firstDim, i75s],
                      c_xis[firstDim, i100s]])
    c_alt_xis = np.array([c_alt_xis[firstDim, i0s], c_alt_xis[firstDim, i25s],
                          c_alt_xis[firstDim, i50s], c_alt_xis[firstDim, i75s],
                          c_alt_xis[firstDim, i100s]])
    c_R10_zetas = np.array([c_R10_zetas[firstDim, i0s],
                            c_R10_zetas[firstDim, i25s],
                            c_R10_zetas[firstDim, i50s],
                            c_R10_zetas[firstDim, i75s],
                            c_R10_zetas[firstDim, i100s]])
    c_zetas = np.array([c_zetas[firstDim, i0s], c_zetas[firstDim, i25s],
                        c_zetas[firstDim, i50s], c_zetas[firstDim, i75s],
                        c_zetas[firstDim, i100s]])
    c_R90_zetas = np.array([c_R90_zetas[firstDim, i0s],
                            c_R90_zetas[firstDim, i25s],
                            c_R90_zetas[firstDim, i50s],
                            c_R90_zetas[firstDim, i75s],
                            c_R90_zetas[firstDim, i100s]])
    comparison_dict = {'xis':c_xis, 'alt_xis':c_alt_xis,
                       'R10_zetas':c_R10_zetas, 'zetas':c_zetas,
                       'R90_zetas':c_R90_zetas}
    
    # get the quenching mechanisms
    mech = np.array([4*outside_in, 3*inside_out, 2*uniform, ambiguous]).T
    mech = np.sum(mech, axis=1)
    
    # select the quenched galaxies
    mask = (logM >= 9.5) & quenched # 278 entries, but len(mask) = 8260
    
    ionsets = ionsets[mask]
    iminorities = iminorities[mask]
    ihalfways = ihalfways[mask]
    imajorities = imajorities[mask]
    iterms = iterms[mask]
    
    xis = xis[mask]
    alt_xis = alt_xis[mask]
    R10_zetas = R10_zetas[mask]
    zetas = zetas[mask]
    R90_zetas = R90_zetas[mask]
    
    # get values for the quenched sample
    q_xis = np.array([xis[firstDim, ionsets], xis[firstDim, iminorities],
                      xis[firstDim, ihalfways], xis[firstDim, imajorities],
                      xis[firstDim, iterms]])
    q_alt_xis = np.array([alt_xis[firstDim, ionsets],
                          alt_xis[firstDim, iminorities],
                          alt_xis[firstDim, ihalfways],
                          alt_xis[firstDim, imajorities],
                          alt_xis[firstDim, iterms]])
    q_R10_zetas = np.array([R10_zetas[firstDim, ionsets],
                            R10_zetas[firstDim, iminorities],
                            R10_zetas[firstDim, ihalfways],
                            R10_zetas[firstDim, imajorities],
                            R10_zetas[firstDim, iterms]])
    q_zetas = np.array([zetas[firstDim, ionsets],
                        zetas[firstDim, iminorities],
                        zetas[firstDim, ihalfways],
                        zetas[firstDim, imajorities],
                        zetas[firstDim, iterms]])
    q_R90_zetas = np.array([R90_zetas[firstDim, ionsets],
                            R90_zetas[firstDim, iminorities],
                            R90_zetas[firstDim, ihalfways],
                            R90_zetas[firstDim, imajorities],
                            R90_zetas[firstDim, iterms]])
    quenched_dict = {'xis':q_xis, 'alt_xis':q_alt_xis, 'R10_zetas':q_R10_zetas,
                     'zetas':q_zetas, 'R90_zetas':q_R90_zetas}
    
    version = 'outside-in' # 'outside-in', 'inside-out', 'uniform', 'ambiguous'
    x_metric = 'xis' # 'xis', 'alt_xis', 'R10_zetas'
    y_metric = 'zetas' # 'R10_zetas', 'zetas', 'R90_zetas'
    
    # xmin, xmax, ymin, ymax, logx, logy = None, None, None, None, 0, 1
    xmin, xmax, ymin, ymax, logx, logy = -0.03, 1.03, -1.2, 1.5, 0, 1 # R_SF vs C_SF
    # xmin, xmax, ymin, ymax, logx, logy = -0.7, 1.9, -1.2, 1.6, 1, 1 # R_SF vs R10
    # xmin, xmax, ymin, ymax, logx, logy = -1.5, 0.7, -1.2, 1.6, 1, 1 # R_SF vs R90
    
    xbins = np.arange(np.around(xmin, 1), np.around(xmax, 1), 0.1)
    ybins = np.arange(np.around(ymin, 1), np.around(ymax, 1), 0.1)
    
    paths = []
    # for progress, amt in zip([0, 1, 2, 3, 4], ['0', '25', '50', '75', '100']) :
    for progress, amt in zip([3], ['75']) :
        
        xs, ys, colours, xlabel, ylabel = get_values_for_plotting(
            comparison_dict, quenched_dict, x_metric, y_metric, progress,
            mech[mask], version, logx=logx, logy=logy)
        
        title = '{}% through the quenching event'.format(amt)
        outfile = outDir + version + '_{}.png'.format(amt)
        paths.append(outfile)
        
        plt.plot_scatter_with_hists(xs, ys, colours, ['control', version],
            ['o', 's'], [0.3, 0.5], xlabel=xlabel, ylabel=ylabel, title=title,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            outfile=outfile, save=False, loc=1, xbins=xbins, ybins=ybins)
        
        '''
        xs_extra, ys_extra, colours_extra, _, _ = get_values_for_plotting(
            comparison_dict, quenched_dict, x_metric, y_metric, progress,
            mech[mask], 'inside-out', logx=logx, logy=logy)
        
        xs.append(xs_extra[1])
        ys.append(ys_extra[1])
        colours.append(colours_extra[1])
        
        plt.plot_scatter_with_hists(xs, ys, colours, ['control', version, 'inside-out'],
            ['o', 's', 's'], [0.3, 0.5, 0.5], xlabel=xlabel, ylabel=ylabel,
            title=title, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            outfile='D:/Desktop/figure.png', save=False, loc=1,
            xbins=xbins, ybins=ybins)
        '''
    
    # concat_horiz(paths, outfile = outDir + version + '.png')
    
    return

def save_diagnostic_plots() :
    
    outDir = 'TNG50-1/figures/diagnostics(t)/'
    outfile = 'TNG50-1/TNG50-1_99_diagnostics(t).hdf5'
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        # snaps = hf['snapshots'][:]
        # redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        subIDfinals = hf['SubhaloID'][:]
        logM = hf['logM'][:, -1]
        tonsets = hf['onset_times'][:]
        tterms = hf['termination_times'][:]
        quenched = hf['quenched'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_satellite_times.hdf5', 'r') as hf :
        tsats = hf['tsats'][:]
        # tsat_indices = hf['tsat_indices'][:]
    
    with h5py.File(outfile, 'r') as hf :
        xis = hf['xi'][:]
        zetas = hf['zeta'][:]
    
    labels = [r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
              r'$R_{\rm SF} = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}}$']
    
    # loop over every galaxy in the sample
    for subIDfinal, mass, use, tonset, tterm, tsat, xi, zeta in zip(subIDfinals,
        logM, quenched, tonsets, tterms, tsats, xis, zetas) :
            
            # only save plots for the galaxies which have computed diagnostics
            if use and (mass >= 9.5) :
                
                title = (r'subID$_{z = 0}$' + ' {}'.format(subIDfinal) +
                         r', $\log{(M/{\rm M}_{\odot})}_{z = 0} = $' 
                         + '{:.2f}'.format(mass))
                
                # plot the results
                outfile = outDir + 'subID_{}.pdf'.format(subIDfinal)
                plt.plot_simple_multi_with_times([times, times], [xi, zeta],
                    labels, ['k', 'r'], ['', ''], ['-', '-'], [1, 1],
                    tsat, tonset, tterm, xlabel=r'$t$ (Gyr)', ylabel='metric',
                    xmin=0, xmax=14, scale='linear', title=title,
                    outfile=outfile, save=True)
                # print('subID {} done'.format(subIDfinal))
    
    return

def plot_profiles_with_derivatives() :
    
    profiles_file = 'TNG50-1/TNG50-1_99_profiles(t).hdf5'
    
    with h5py.File('TNG50-1/TNG50-1_99_mechanism.hdf5', 'r') as hf :
        outside_in = hf['outside-in'][:] # 109
        inside_out = hf['inside-out'][:] # 103
        uniform = hf['uniform'][:]       # 8
        ambiguous = hf['ambiguous'][:]   # 58
    
    # get the quenching mechanisms
    mechs = np.array([4*outside_in, 3*inside_out, 2*uniform, ambiguous]).T
    mechs = np.sum(mechs, axis=1)
    
    with h5py.File(profiles_file, 'r') as hf :
        # edges = hf['edges'][:]
        mids = hf['mids'][:]
        mass_profiles = hf['mass_profiles'][:]
        SFR_profiles = hf['SFR_profiles'][:]
        area_profiles = hf['area_profiles'][:]
        # nParticles = hf['nParticles'][:].astype(int)
        # nSFparticles = hf['nSFparticles'][:].astype(int)
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        times = hf['times'][:]
        subIDs = hf['subIDs'][:, -1].astype(int)
        logM = hf['logM'][:, -1]
        # Res = hf['Re'][:]
        # ionsets = hf['onset_indices'][:].astype(int)
        tonsets = hf['onset_times'][:]
        # iterms = hf['termination_indices'][:].astype(int)
        tterms = hf['termination_times'][:]
        quenched = hf['quenched'][:]
        # cluster = hf['cluster'][:]
        # hmGroup = hf['hm_group'][:]
        # lmGroup = hf['lm_group'][:]
        # field = hf['field'][:]
        comparison = hf['comparison'][:]
        # i0s = hf['i0s'][:].astype(int)
        # i25s = hf['i25s'][:].astype(int)
        # i50s = hf['i50s'][:].astype(int)
        i75s = hf['i75s'][:].astype(int)
        # i100s = hf['i100s'][:].astype(int)
    
    firstDim = np.arange(278)
    # xs = np.linspace(0, 5, 101)
    
    # get snapshots corresponding to 25%, 50%, and 75% through the quenching episode
    # iminorities = np.array(find_nearest(times, tonsets + 0.25*(tterms - tonsets)))
    # ihalfways = np.array(find_nearest(times, tonsets + 0.5*(tterms - tonsets)))
    imajorities = np.array(find_nearest(times, tonsets + 0.75*(tterms - tonsets)))
    
    # select the quenched galaxies
    quenched = (logM >= 9.5) & quenched # 278 entries, but len(mask) = 8260
    q_masses = logM[quenched]
    mechs = mechs[quenched]
    q_subIDs = subIDs[quenched]
    imajorities = imajorities[quenched]
    q_profiles = ((SFR_profiles/mass_profiles)[quenched])[firstDim, imajorities]
    q_SFR_profiles = (SFR_profiles[quenched])[firstDim, imajorities]
    q_area_profiles = (area_profiles[quenched])[firstDim, imajorities]
    q_SF_mass_surface_density_profiles = q_SFR_profiles*1e8/q_area_profiles/1e6 # solMass/pc^2
    # q_Res = (Res[quenched])[firstDim, imajorities]
    
    Rtruncs = np.full(278, np.nan)
    # Rtruncs_alt = np.full(278, np.nan)
    for i, (profile, SF_surface_density, mech, sub, mass) in enumerate(zip(
        q_profiles, q_SF_mass_surface_density_profiles, mechs, q_subIDs,
        q_masses)) :
        
        # get the sSFR profile normalized by the value at 1 (stellar) Re
        # normalized = np.interp(xs, mids, profile)/np.interp(1, mids, profile)
        # try with a factor of 10 decrease for outside-in quenchers
        # diff = (normalized - 0.1)[21:] # index 20 is 1 Re
        # locs = np.where(diff <= 0)[0] + 21
        # try with a factor of 10 decrease for inside-out quenchers
        # diff = (normalized - 0.1)[:20] # index 20 is 1 Re
        # locs = np.where(diff <= 0)[0]
        # if len(locs) > 0 :
        #     loc = locs[0]
        #     Rtruncs[i] = xs[loc]
        
        if mech == 4 :
            Rtrunc = sSFR_profile_and_deriv(i, sub, mass, mids, profile,
                                            threshold=-11, version='outside-in',
                                            show_plot=True)
            Rtruncs[i] = Rtrunc
    
    
    
    # 30 galaxies with radius = 5, including one profile of all zeros
    # plt.histogram_multi([Rtruncs[mechs == 4], Rtruncs[mechs == 3]],
    #     r'$R_{\rm truncation}/R_{\rm e}$', ['r', 'm'], ['-', '-'],
    #     ['outside-in', 'inside-out'],
    #     [np.linspace(0, 5, 21), np.linspace(0, 1, 5)], loc=2)
    
    '''
    # get values for the comparison sample
    c_subIDs = subIDs[comparison]
    c_masses = logM[comparison]
    i75s = i75s[comparison]
    c_profiles = ((SFR_profiles/mass_profiles)[comparison])[firstDim, i75s]
    c_SFR_profiles = (SFR_profiles[comparison])[firstDim, imajorities]
    c_area_profiles = (area_profiles[comparison])[firstDim, imajorities]
    c_SF_mass_surface_density_profiles = c_SFR_profiles*1e8/c_area_profiles/1e6 # solMass/pc^2
    # c_Res = (Res[comparison])[firstDim, i75s]
    
    st = 0
    end = 10
    for i, (profile, SF_surface_density, sub, mass) in enumerate(zip(
        (c_profiles)[st:end], (c_SF_mass_surface_density_profiles)[st:end],
        (c_subIDs)[st:end], (c_masses)[st:end])) :
        
        sSFR_profile_and_deriv(sub, mass, mids, profile)
    
    # plt.histogram(c_Rtruncs, r'$R_{\rm truncation}/R_{\rm e}$',
    #               bins=np.linspace(0, 5, 21))
    '''
    return

def sSFR_profile_and_deriv(index, sub, mass, mids, profile, threshold=-11,
                           version='inside-out', show_plot=False) :
    
    if np.all(profile == 0) :
        radius = 5
    else :
        profile = np.log10(profile)
        mask = ~np.isfinite(profile)
        profile[mask] = np.min(profile[np.isfinite(profile)]) - 0.5
        
        deriv = np.gradient(profile, mids)
        # cs = CubicSpline(mids, profile)
        # cs_deriv = cs(xs, 1)
        
        if version == 'inside-out' :
            a = 1 # placeholder
        else :
            locs = np.where(deriv <= -1)[0]
            if len(locs) > 0 :
                for loc in locs :
                    if np.all(profile[loc:] <= threshold) :
                        radius = mids[loc]
                        break
                    else :
                        radius = 5
            else :
                radius = 5
    
    '''
    if show_plot and (~np.all(profile ==0)) and (radius == 5) :
        title = ('index {}, '.format(index) +
                 r'subID$_{z = 0}$' + ' {}'.format(sub) +
                 r', $\log{(M/{\rm M}_{\odot})}_{z = 0} = $'  +
                 '{:.2f}'.format(mass))
        xlabel = r'$r/R_{\rm e}$'
        ylabel = r'$f(x) = \log({\rm sSFR}/{\rm yr}^{-1})$'
        seclabel = r"$f~'(x)$"
        
        xmin = 0
        xmax = 5
        ymin = np.min(profile) - 0.1
        ymax = np.max(profile) + 0.1
        
        seclim = 2 #max(np.max(np.abs(cs_deriv)), np.max(np.abs(deriv)))
        
        # update colors for main sSFR profile to denote values
        # that were exactly 0.0 before logging
        gtr_zero_xs = mids[~mask]
        gtr_zero_ys = profile[~mask]
        zero_xs = mids[mask]
        zero_ys = profile[mask]
        
        xvals = [gtr_zero_xs, zero_xs, mids]
        yvals = [gtr_zero_ys, zero_ys, deriv]
        labels = ['', '', r"$f~'$"]
        colors = ['k', 'grey', 'grey']
        markers = ['o', 'o', '']
        styles = ['', '', '--']
        alphas = [1, 1, 1]
        
        plt.plot_simple_multi_secax(xvals, yvals, labels,
            colors, markers, styles, alphas, 2, xlabel=xlabel,
            ylabel=ylabel, seclabel=seclabel, title=title,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            secmin=-seclim, secmax=seclim, xscale='linear',
            yscale='linear', secscale='linear',
            outfile='subID_{}.png'.format(sub), save=False)
    '''
    return radius

def merge_comp_plots_incorrect_onset_times() :
    
    # outside_in = [28, 30, 34, 39, 63879, 63898, 96778, 96781, 96801, 117275,
    #               143904, 167420, 167421, 184944, 253873, 282790, 294879, 324126, 355734, 623367]
    # inside_out = [4, 167398, 275550, 294875, 362994, 404818, 434356, 445626, 447914, 450916,
    #               480803, 503987, 507294, 515296, 526879, 539667, 545703, 567607, 572328, 576516,
    #               576705, 580907, 584724, 588399, 588831, 607654, 609710, 625281, 637199, 651449]
    # uniform =[117306, 457431]
    ambiguous = [2, 7, 23, 96764, 96767, 96772, 184932, 242789, 253865, 406941,
                  418335, 475619, 568646, 592021, 634631, 657979, 671746, 708459]
    
    merger = PdfWriter()
    for subID in ambiguous :
        merger.append('TNG50-1/figures/comprehensive_plots/subID_{}.pdf'.format(subID))
    merger.write('ambiguous_incorrect_onset_times.pdf')
    merger.close()
    
    return

# plot_profiles_with_derivatives()

# determine_morphological_parameters() # to be deleted -> superceded by better function
