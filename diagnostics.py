
from os.path import exists
import numpy as np

import astropy.units as u
import h5py
from pypdf import PdfWriter
# from scipy.stats import anderson_ksamp, ks_2samp
# from scipy.interpolate import CubicSpline

# from concatenate import concat_horiz
from core import (add_dataset, find_nearest, get_particle_positions,
                  get_sf_particles)
import plotting as plt

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

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

def compute_xi(sf_masses, sf_rs, Re) :
    
    if (len(sf_masses) == 0) and (len(sf_rs) == 0) :
        sf_mass_within_1kpc, sf_mass_within_tenthRe = np.nan, np.nan
        sf_mass = np.nan
    else :
        sf_mass_within_1kpc = np.sum(sf_masses[sf_rs <= 1.0])
        sf_mass_within_tenthRe = np.sum(sf_masses[sf_rs <= 0.1*Re])
        sf_mass = np.sum(sf_masses)
    
    return sf_mass_within_1kpc, sf_mass_within_tenthRe, sf_mass

def compute_zeta(masses, rs, sf_masses, sf_rs) :
    
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
                         sf_mass) = compute_xi(sf_masses, sf_rs, Re)
                        
                        (R10, R50, R90,
                         sf_R10, sf_R50, sf_R90) = compute_zeta(masses, rs,
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
                         sf_mass) = compute_xi(sf_masses, sf_rs, Re)
                        
                        (R10, R50, R90,
                         sf_R10, sf_R50, sf_R90) = compute_zeta(masses, rs,
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
    #     r'$\xi = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['full', 'first half', 'second half'], title='outside-in')
    # plt.histogram_multi([oi_zeta_full, oi_zeta_first, oi_zeta_second],
    #     r'$\zeta = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['full', 'first half', 'second half'], title='outside-in')
    # plt.histogram_multi([io_xi_full, io_xi_first, io_xi_second],
    #     r'$\xi = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['full', 'first half', 'second half'], title='inside-out')
    # plt.histogram_multi([io_zeta_full, io_zeta_first, io_zeta_second],
    #     r'$\zeta = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['full', 'first half', 'second half'], title='inside-out')
    # plt.histogram_multi([uni_xi_full, uni_xi_first, uni_xi_second],
    #     r'$\xi = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['full', 'first half', 'second half'], title='uniform')
    # plt.histogram_multi([uni_zeta_full, uni_zeta_first, uni_zeta_second],
    #     r'$\zeta = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['full', 'first half', 'second half'], title='uniform')
    
    # compare the distributions amongst different quenching mechanisms
    # oi_xi_weight = np.ones(len(oi_xi_full))/len(oi_xi_full)
    # io_xi_weight = np.ones(len(io_xi_full))/len(io_xi_full)
    # uni_xi_weight = np.ones(len(uni_xi_full))/len(uni_xi_full)
    # plt.histogram_multi([oi_xi_full, io_xi_full, uni_xi_full],
    #     r'$\xi = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['outside-in', 'inside-out', 'uniform'],
    #     [oi_xi_weight, io_xi_weight, uni_xi_weight])
    # oi_zeta_weight = np.ones(len(oi_zeta_full))/len(oi_zeta_full)
    # io_zeta_weight = np.ones(len(io_zeta_full))/len(io_zeta_full)
    # uni_zeta_weight = np.ones(len(uni_zeta_full))/len(uni_zeta_full)
    # plt.histogram_multi([oi_zeta_full, io_zeta_full, uni_zeta_full],
    #     r'$\zeta = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}}$',
    #     ['k', 'r', 'b'], ['-', '--', '--'],
    #     ['outside-in', 'inside-out', 'uniform'],
    #     [oi_zeta_weight, io_zeta_weight, uni_zeta_weight])
    
    '''
    # xi and zeta values are from different distributions when comparing
    # outside-in to inside-out
    # print(ks_2samp(oi_xi_full, io_xi_full))
    # print(ks_2samp(oi_zeta_full, io_zeta_full))
    # print(anderson_ksamp([oi_xi_full, io_xi_full]))
    # print(anderson_ksamp([oi_zeta_full, io_zeta_full]))
    
    # xi and zeta values are from different distributions when comparing
    # outside-in to uniform
    # print(ks_2samp(oi_xi_full, uni_xi_full))
    # print(ks_2samp(oi_zeta_full, uni_zeta_full))
    # print(anderson_ksamp([oi_xi_full, uni_xi_full]))
    # print(anderson_ksamp([oi_zeta_full, uni_zeta_full]))
    
    # xi and zeta values are from different distributions when comparing
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
    
    labels = {'xis':r'$\xi = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
              'alt_xis':r'$\xi = {\rm SFR}_{<0.1~R_{\rm e}}/{\rm SFR}_{\rm total}$',
              'zetas':r'$\log{(\zeta = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}})}$',
              'R10_zetas':r'$\log{(\zeta = R_{{\rm 10}_{*, {\rm SF}}}/R_{{\rm 10}_{*, {\rm total}}})}$',
              'R90_zetas':r'$\log{(\zeta = R_{{\rm 90}_{*, {\rm SF}}}/R_{{\rm 90}_{*, {\rm total}}})}$'}
    
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
    xmin, xmax, ymin, ymax, logx, logy = -0.03, 1.03, -1.2, 1.5, 0, 1 # zeta vs xi
    # xmin, xmax, ymin, ymax, logx, logy = -0.7, 1.9, -1.2, 1.6, 1, 1 # zeta vs R10
    # xmin, xmax, ymin, ymax, logx, logy = -1.5, 0.7, -1.2, 1.6, 1, 1 # zeta vs R90
    
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
    
    labels = [r'$\xi = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
              r'$\zeta = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}}$']
    
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
