
from os.path import exists
import numpy as np

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import h5py
from scipy.ndimage import gaussian_filter1d

from core import (add_dataset, bsPath, determine_mass_bin_indices, find_nearest,
                  get_mpb_radii_and_centers, get_particles, get_sf_particles)
from diagnostics import determine_radial_profiles
import plotting as plt

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

def calculate_all_radial_profiles(simName='TNG50-1', snapNum=99,
                                  delta_t=100*u.Myr) :
    
    # define the input and output files
    infile = 'TNG50-1/TNG50-1_99_sample(t).hdf5'
    outfile = 'TNG50-1/TNG50-1_99_massive_radial_profiles(t).hdf5'
    
    # get relevant information for the general sample
    with h5py.File(infile, 'r') as hf :
        snapshots = hf['snapshots'][:]
        times = hf['times'][:]
        subIDs = hf['subIDs'][:].astype(int)
        Re = hf['Re'][:]
        centers = hf['centers'][:]
        SFMS = hf['SFMS'][:].astype(bool) # (boolean) SFMS at each snapshot
        quenched = hf['quenched'][:]
        ionsets = hf['onset_indices'][:].astype(int)
        iterms = hf['termination_indices'][:].astype(int)
    
    # find the total number of radial profiles that will be calculated
    # print(np.sum(SFMS) + np.sum((iterms + 1 - ionsets)*quenched)) # 482308
    
    # write an empty file which will hold all the computed radial profiles
    if not exists(outfile) :
        
        # define the edges and center points of the radial bins
        edges = np.linspace(0, 5, 21)
        mids = []
        for start, end in zip(edges, edges[1:]) :
            mids.append(0.5*(start + end))
        
        # populate the helper file with empty arrays to be populated later
        full_vals = np.full((8260, 100, len(mids)), np.nan)
        with h5py.File(outfile, 'w') as hf :
            add_dataset(hf, edges, 'edges')
            add_dataset(hf, np.array(mids), 'mids')
            add_dataset(hf, full_vals, 'SFR_profiles')
            add_dataset(hf, full_vals, 'mass_profiles')
            add_dataset(hf, full_vals, 'area_profiles')
            add_dataset(hf, full_vals, 'nParticles')
            add_dataset(hf, full_vals, 'nSFparticles')
    
    with h5py.File(outfile, 'r') as hf :
        edges = hf['edges'][:]
        length = len(hf['mids'][:])
        SFR_profiles = hf['SFR_profiles'][:]
        mass_profiles = hf['mass_profiles'][:]
        area_profiles = hf['area_profiles'][:]
        nParticles = hf['nParticles'][:]
        nSFparticles = hf['nSFparticles'][:]
    
    count = 1
    # loop over the galaxies in the sample, determining the radial profiles
    # for the SFMS galaxies at all snapshots, and for the quenched galaxies
    # between the onset and termination of quenching, inclusive
    for i, (mpb_subIDs, mpb_Res, mpb_centers, mpb_SFMS, use, ionset_val,
            iterm_val) in enumerate(
                zip(subIDs, Re, centers, SFMS, quenched, ionsets, iterms)) :
        
        for j, (snap, time, subID, radius, center, SFMS_val) in enumerate(
                zip(snapshots, times, mpb_subIDs, mpb_Res, mpb_centers, mpb_SFMS)) :
            
            # don't replace existing values
            if (np.all(np.isnan(SFR_profiles[i, j, :])) and
                np.all(np.isnan(mass_profiles[i, j, :])) and
                np.all(np.isnan(area_profiles[i, j, :])) and
                np.all(np.isnan(nParticles[i, j, :])) and
                np.all(np.isnan(nSFparticles[i, j, :]))) :
                
                # if the galaxy is on the SFMS at the snapshot then proceed
                if SFMS_val :
                    (SFR_profile, mass_profile, area_profile, nParticle_profile,
                     nSFparticle_profile) = determine_radial_profiles(simName,
                        snapNum, snap, time, subID, radius, center, edges,
                        length, delta_t=delta_t)
                    
                    with h5py.File(outfile, 'a') as hf :
                        hf['SFR_profiles'][i, j, :] = SFR_profile
                        hf['mass_profiles'][i, j, :] = mass_profile
                        hf['area_profiles'][i, j, :] = area_profile
                        hf['nParticles'][i, j, :] = nParticle_profile
                        hf['nSFparticles'][i, j, :] = nSFparticle_profile
                
                # if the galaxy is a quenched galaxy and the snapshot is between
                # the onset and termination of quenching, inclusive
                if use and (ionset_val <= snap <= iterm_val) :
                    (SFR_profile, mass_profile, area_profile, nParticle_profile,
                     nSFparticle_profile) = determine_radial_profiles(simName,
                        snapNum, snap, time, subID, radius, center, edges,
                        length, delta_t=delta_t)
                    
                    with h5py.File(outfile, 'a') as hf :
                        hf['SFR_profiles'][i, j, :] = SFR_profile
                        hf['mass_profiles'][i, j, :] = mass_profile
                        hf['area_profiles'][i, j, :] = area_profile
                        hf['nParticles'][i, j, :] = nParticle_profile
                        hf['nSFparticles'][i, j, :] = nSFparticle_profile
            
            # print to console for visual inspection, after 10% of every snapshot
            if count % 826 == 0.0 :
                print('{}/826000 done'.format(count))
            
            # iterate the counter
            count += 1
    
    return

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

def check_differences_between_diagnostic_methods() :
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        times = hf['times'][:]
        logM = hf['logM'][:, -1]
        tonsets = hf['onset_times'][:]
        tterms = hf['termination_times'][:]
        quenched = hf['quenched'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_diagnostics(t).hdf5', 'r') as hf :
        xis = hf['xi'][:]
        zetas = hf['zeta'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_diagnostics(t)_revised.hdf5', 'r') as hf :
        sf_mass_within_1kpc = hf['sf_mass_within_1kpc'][:]
        sf_mass_tot = hf['sf_mass'][:]
        R50s = hf['R50'][:]
        sf_R50s = hf['sf_R50'][:]
    
    # check the values for the quenched sample
    for (mass, use, tonset, tterm, xi, zeta, sf_within_1kpc, sf_mass,
         R50, sf_R50) in zip(logM, quenched, tonsets, tterms, xis, zetas,
        sf_mass_within_1kpc, sf_mass_tot, R50s, sf_R50s) :
            
            if use and (mass >= 9.5) :
                start, end = find_nearest(times, [tonset-1, tterm+1])
                
                old_xi = xi[start:end+1]
                new_xi = sf_within_1kpc[start:end+1]/sf_mass[start:end+1]
                print(new_xi)
                print()
                
                old_zeta = zeta[start:end+1]
                new_zeta = sf_R50[start:end+1]/R50[start:end+1]
    
    with h5py.File('TNG50-1/TNG50-1_99_midpoints.hdf5', 'r') as hf :
        imids = hf['middle_indices'][:].astype(int)
    
    with h5py.File('TNG50-1/TNG50-1_99_matched_sample_locations.hdf5', 'r') as hf :
        locs = hf['locations'][:, 0].astype(int)
    
    # check the values for the comparison sample
    for loc, snap in zip(locs, imids) :
        if (snap >= 0) and (loc >= 0) : 
            
            old_xi = xis[loc, snap]
            new_xi = sf_mass_within_1kpc[loc, snap]/sf_mass_tot[loc, snap]
            
            old_zeta = zetas[loc, snap]
            new_zeta = sf_R50s[loc, snap]/R50s[loc, snap]
    
    return

def diagnostics_at_snaps(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and the input file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    quenched_file = inDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
    # open basic info for all galaxies
    with h5py.File(infile, 'r') as hf :
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        # SFHs = hf['SFH'][:]
        SFMS = hf['SFMS'][:] # 6337 galaxies
        subIDs = hf['SubhaloID'][:]
        masses = hf['SubhaloMassStars'][:]
        # quenched = hf['quenched_mask'][:]
    
    # open basic info for the quenched galaxies
    with h5py.File(quenched_file, 'r') as hf :
        q_subIDs = hf['SubhaloID'][:]
        q_masses = hf['SubhaloMassStars'][:]
        onset_indices = hf['onset_indices'][:]
        tonsets = hf['onset_times'][:]
        term_indices = hf['termination_indices'][:]
        tterms = hf['termination_times'][:]
        # satellite = hf['satellite'][:] # once I've determined the satellite systems
    
    satellite = np.array([2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1,
                          2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2,
                          2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2,
                          2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2,
                          2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 0, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 0, 2,
                          2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 1, 2, 2, 1, 0, 2, 2,
                          2, 2, 2, 2, 1, 2, 1, 0, 1, 2, 0, 2, 2, 2, 0, 2, 1, 1,
                          2, 1, 2, 0, 2, 0, 2, 0, 2, 2, 2, 0, 2, 2, 0, 2, 2, 0,
                          2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2])
    
    # print(len(satellite[satellite == 0]))
    
    massive_sat = (q_masses > 10.07) & (satellite == 1) # massive galaxies have cutouts downloaded
    
    q_subIDs = q_subIDs[massive_sat]
    onset_indices = onset_indices[massive_sat]
    tonsets = tonsets[massive_sat]
    term_indices = term_indices[massive_sat]
    tterms = tterms[massive_sat]
    q_masses = q_masses[massive_sat]
    
    # loop over all the massive quenched galaxies
    for (q_subID, q_mass, onset_index,
         tonset, term_index, tterm) in zip(q_subIDs, q_masses, onset_indices,
                                           tonsets, term_indices, tterms) :
        
        # print(q_subID, q_mass, onset_index, tonset, term_index, tterm)
                                           
        # using tterm isn't very illustrative
        index_times = np.array([tonset, tonset + 0.5*(tterm - tonset), tterm])
        indices = find_nearest(times, index_times)
        
        # use alternative times based on the final snapshot being 75% of the
        # quenching mechanism duration
        # index_times = np.array([tonset,
        #                         tonset + 0.375*(tterm - tonset),
        #                         tonset + 0.75*(tterm - tonset)])
        # indices = find_nearest(times, index_times)
        
        snaps = indices
        
        # print(times[snaps])
        
        # get basic MPB info about the quenched galaxy
        (_, q_mpb_subIDs, _, q_centers) = get_mpb_radii_and_centers(
            simName, snapNum, q_subID)
        
        xi_fxn, zeta_fxn = [], []
        for snap in snaps :
            xi = determine_xi(simName, snapNum, redshifts[snap], times[snap],
                              snap, int(q_mpb_subIDs[snap]), q_centers[snap])
            xi_fxn.append(xi)
            
            zeta = determine_zeta(simName, snapNum, times[snap], snap,
                                  int(q_mpb_subIDs[snap]), q_centers[snap])
            zeta_fxn.append(zeta)
        
        # now find control galaxies of a similar mass in a small mass bin
        mass_bin = determine_mass_bin_indices(masses[SFMS], q_mass,
                                              hw=0.10, minNum=50)
        
        # find the subIDs for those control galaxies
        control = subIDs[SFMS][mass_bin]
        
        # create empty arrays that will hold all of the xi and zeta metrics
        control_xis = np.full((len(control), 3), np.nan)
        control_zetas = np.full((len(control), 3), np.nan)
        
        # loop over all the comparison galaxies and populate into the array
        for i, ID in enumerate(control) :
            
            # get basic MPB info
            (_, mpb_subIDs, _, centers) = get_mpb_radii_and_centers(
                simName, snapNum, ID)
            
            # determine xi and zeta for the given snapshots
            control_xi_fxn, control_zeta_fxn = [], []
            for snap in snaps :
                
                xi = determine_xi(simName, snapNum, redshifts[snap], times[snap],
                                  snap, int(mpb_subIDs[snap]), centers[snap])
                control_xi_fxn.append(xi)
                
                zeta = determine_zeta(simName, snapNum, times[snap], snap,
                                      int(mpb_subIDs[snap]), centers[snap])
                control_zeta_fxn.append(zeta)
            
            # place those values into the empty arrays from above
            control_xis[i, :] = control_xi_fxn
            control_zetas[i, :] = control_zeta_fxn
        
        np.savez('diagnostics_subID_{}.npz'.format(q_subID),
                 xi=xi_fxn, zeta=zeta_fxn,
                 control_xis=control_xis, control_zetas=control_zetas)
    
    return

def check_diagnostic_evolution_scatter() :
    
    for subID in [5, 13, 14, 96763, 167398, 324126] : # 242789
        file = np.load('TNG50-1/temp/diagnostics_subID_{}.npz'.format(subID))
        
        xi = file['xi']
        zeta = file['zeta']
        
        xis = file['control_xis']
        zetas = file['control_zetas']
        
        # print(xi)
        # print(zeta)
        # print()
        # print(xis)
        # print(zetas)
        
        good_mask = ~(np.isnan(xis) | np.isnan(zetas))
        for i, row in enumerate(good_mask) :
            if np.sum(row) < 3 :
                good_mask[i] = np.array([False, False, False]) 
        
        xis = np.where(good_mask, xis, np.nan)
        zetas = np.where(good_mask, zetas, np.nan)
        
        
        
        xlabel = r'$\xi = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$'
        ylabel = r'$\zeta = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}}$'
        
        # rand = np.random.random(len(xis))
        # mask = rand < 0.1
        plt.plot_lines(xi, zeta, xis, zetas, label='subID {}'.format(subID),
                        xlabel=xlabel, ylabel=ylabel)
    
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

def recover_paper_plots() :
    
    table = Table.read('TNG50-1/morphological_metrics_-10.5_+-1.fits')
    
    # select only the quenched population
    quenched = table['quenched_status']
    table = table[quenched]
    
    # get the unique subIDs that make up the quenched sample
    subIDs = np.unique(table['quenched_subID'])
    
    # create a new table that will contain the information for the previously
    # defined "75% through the quenching episode" time points, with the
    # associated morphological measures
    new_table = Table(names=table.colnames)
    new_table_alt = Table(names=table.colnames)
    
    for subID in subIDs :
        masked_table = table[table['quenched_subID'] == subID]
        episode_progress = masked_table['episode_progress'].data
        
        index = np.abs(episode_progress - 0.75).argmin()
        new_table.add_row(masked_table[index])
        
        index_alt = np.where(episode_progress - 0.75 >= 0.0)[0][0]
        new_table_alt.add_row(masked_table[index_alt])
    
    new_table.write('TNG50-1/morphological_metrics_quenched_paper-draft-time-points.fits')
    new_table_alt.write('TNG50-1/morphological_metrics_quenched_at-least-75-percent.fits')
    
    return

def compare_with_paper_draft_plots() :
    
    files = ['TNG50-1/morphological_metrics_quenched_paper-draft-time-points.fits',
             'TNG50-1/morphological_metrics_quenched_at-least-75-percent.fits']
    
    xmin, xmax = -0.03, 1.03
    ymin, ymax = -1.2, 1.5
    xbins = np.arange(np.around(xmin, 1), np.around(xmax, 1), 0.1)
    ybins = np.arange(np.around(ymin, 1), np.around(ymax, 1), 0.1)
    
    xlabel = r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$'
    ylabel = r'$\log{(R_{\rm SF} = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}})}$'
    title = '~75% through the quenching event'
    
    for file in files :
        table = Table.read(file)
        mech = table['mechanism']
        C_SF = table['C_SF']
        R_SF = table['R_SF']
        # Rinner = table['Rinner']
        # Router = table['Router']
        
        xs = [C_SF[mech == 3.0], C_SF[mech == 1.0]]
        ys = [np.log10(R_SF[mech == 3.0]), np.log10(R_SF[mech == 1.0])]
        plt.plot_scatter_with_hists(xs, ys, ['r', 'm'],
            ['outside-in', 'inside-out'], ['s', 's'], [0.5, 0.5],
            xlabel=xlabel, ylabel=ylabel, title=title,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            loc=1, xbins=xbins, ybins=ybins)
    
    return

def check_oi_large_Router_values() :
    
    # get basic information about the sample, where some parameters are a
    # function of time
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        # redshifts = hf['redshifts'][:]
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
    with h5py.File('TNG50-1/TNG50-1_99_massive_Rinner(t).hdf5', 'r') as hf :
        Rinners = hf['Rinner'][:]
    with h5py.File('TNG50-1/TNG50-1_99_massive_Router(t).hdf5', 'r') as hf :
        Routers = hf['Router'][:]
    with h5py.File('TNG50-1/TNG50-1_99_massive_radial_profiles(t).hdf5', 'r') as hf :
        radial_bin_centers = hf['midpoints'][:] # units of Re
        sSFR_profiles = hf['sSFR_profiles'][:] # shape (1666, 100, 20)
    
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
    for (quenched_subID, quenched_mass, ionset, tonset, iterm, tterm, 
         quenched_mech) in zip(quenched_subIDs, quenched_logM,
        quenched_ionsets, quenched_tonsets, quenched_iterms, quenched_tterms,
        quenched_mechs) :
        
        # select only OI quenched galaxies
        if quenched_mech == 3 :
            
            # find the location of the OI quenched galaxy within the subsample
            location = (np.argwhere(subIDs[:, 99] == quenched_subID))[0][0]
            
            for snap in range(ionset, iterm+1) :
                episode_progress = (times[snap] - tonset)/(tterm - tonset)
                
                if episode_progress >= 0.65 :
                    
                    Router = Routers[location, snap]
                    
                    # if quenched_subID == 623367 :
                    #     plt.plot_simple_dumb(radial_bin_centers,
                    #         np.log10(sSFR_profiles[location, snap]),
                    #         xlabel=r'$r/R_{\rm e}$', ylabel=r'sSFR',
                    #         ymin=-12.7, ymax=-9.6)
                    
                    if Router == 5.0 :
                        
                        Rinner = Rinners[location, snap]
                        # redshift_at_snap = redshifts[snap]
                        mass = quenched_mass[snap]
                        
                        print('{:6} {:3} {:6.2f} {:5.2f} {:5.2f}'.format(
                            quenched_subID, snap, mass, episode_progress, Rinner))
                        
                        outfile = '{}_{:.2f}_{:.2f}.png'.format(
                            quenched_subID, mass, episode_progress)
                        plt.plot_simple_dumb(radial_bin_centers,
                            np.log10(sSFR_profiles[location, snap]),
                            xlabel=r'$r/R_{\rm e}$', ylabel=r'sSFR',
                            outfile=outfile, save=True)
    
    return

def check_morphological_metric_distributions(threshold=-10.5, slope=1) :
    
    infile = 'TNG50-1/morphological_metrics_{}_+-{}.fits'.format(threshold, slope)
    data = Table.read(infile)

    quenched = data['quenched_status'].value
    episode_progress = data['episode_progress'].value
    mechanism = data['mechanism'].value
    
    # exclude redshift and stellar mass information
    data = np.array([data['C_SF'].value, data['R_SF'].value,
                     data['Rinner'].value, data['Router'].value]).T
    
    colors = ['b', 'm', 'r', 'k']
    styles = ['-', '-', '-', '-']
    labels = ['early', 'early-mid', 'mid-late', 'late']
    bins = [20, 20, 20, 20]
    columns = [0, 1, 2, 3]
    hlabels = [r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
               r'$\log{(R_{\rm SF} = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}})}$',
               r'$R_{\rm inner}/R_{\rm e}$', r'$R_{\rm outer}/R_{\rm e}$']
    params = ['CSF', 'RSF', 'Rinner', 'Router']
    ymaxs = [0.5, 0.25, 1, 1]
    
    for col, hlabel, param, ymax in zip(columns, hlabels, params, ymaxs) :
        for pop, status, title in zip([0, 1, 3, 5],
            [~quenched, quenched, quenched, quenched],
            ['sf', 'inside-out', 'outside-in', 'ambiguous']) :
            
            # select the population
            pop_mask = (mechanism == pop) & status
            
            # copy the episode progress and morphological parameters
            pop_episode_progress = episode_progress.copy()[pop_mask]
            pop_data = data.copy()[pop_mask]
            
            # define the metric of interest
            metric = pop_data[:, col]
            
            # mask the data based on valid values
            valid = ~np.isnan(metric)
            metric = metric[valid]
            pop_episode_progress = pop_episode_progress[valid]
            
            valid = (metric >= 0.0)
            metric = metric[valid]
            pop_episode_progress = pop_episode_progress[valid]
            
            xmin, xmax = None, None
            loc = 0
            if col == 1 :
                metric = np.log10(metric)
                xmin, xmax = -1.3, 1.3
                plotting_mask = (metric >= xmin) & (metric <= xmax)
                metric = metric[plotting_mask]
                pop_episode_progress = pop_episode_progress[plotting_mask]
            if col in [2, 3] :
                xmin, xmax = -0.1, 5.1
            if col == 3 :
                loc = 2
            
            values = [metric[pop_episode_progress <= 0.25],
                      metric[(pop_episode_progress > 0.25) & (pop_episode_progress <= 0.5)],
                      metric[(pop_episode_progress > 0.5) & (pop_episode_progress <= 0.75)],
                      metric[pop_episode_progress > 0.75]]
            weights = [np.ones(len(value))/len(value) for value in values]
            
            plt.histogram_multi(values, hlabel, colors, styles, labels, bins,
                weights, xmin=xmin, xmax=xmax, ymax=ymax, title=title, loc=loc,
                outfile='{}_{}_distribution.png'.format(param, title),
                save=False)
    
    return

def check_paper_draft_Figure_6_plane() :
    
    table = Table.read('TNG50-1/morphological_metrics_-10.5_+-1.fits')
    quenched_status = table['quenched_status'].value
    sf_status = table['sf_status'].value
    episode_progress = table['episode_progress'].value
    mechanism = table['mechanism'].value
    
    quenched_mask = (episode_progress > 0.9) & (mechanism > 0) & quenched_status
    sf_mask = (episode_progress > 0.9) & (mechanism == 0) & sf_status
    
    # exclude redshift and stellar mass information
    data = np.array([table['C_SF'].value, table['R_SF'].value,
                     table['Rinner'].value, table['Router'].value]).T
    
    quenched_C_SF = data[:, 0][quenched_mask]
    quenched_R_SF = np.log10(data[:, 1][quenched_mask])
    sf_C_SF = data[:, 0][sf_mask]
    sf_R_SF = np.log10(data[:, 1][sf_mask])
    
    xs = [quenched_C_SF, sf_C_SF]
    ys = [quenched_R_SF, sf_R_SF]
    colors = ['r', 'b']
    labels = ['quenched', 'sf']
    markers = ['o', 's']
    alphas = [0.1, 0.01]
    xlabel = r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$'
    ylabel = r'$\log{(R_{\rm SF} = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}})}$'
    xmin, xmax, ymin, ymax = -0.03, 1.03, -1.2, 1.5
    plt.plot_scatter_with_hists(xs, ys, colors, labels, markers, alphas,
        xlabel=xlabel, ylabel=ylabel, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    
    return






# functions that were previously in psi.py
def compute_nabla_psi(redshift, masses, rs, radius, snap, delta_t=100*u.Myr) :
    
    if (len(rs) == 0) and (len(masses) == 0) :
        gradient = np.nan
    else :
        
        # some particles have distances of 0, so we need to increase their
        # distances by a small amount, equal to the distance of the next most
        # inner star particle, for taking the logarithm
        if len(rs) > 1 :
            rs[rs == 0.0] = np.sort(rs)[1]
        else :
            rs[rs == 0.0] = 0.001
        
        # scale the radius by the effective radius
        rs = np.log10(rs/radius)
        
        # define the shell edges, based on the region that's the most physically
        # interesting
        edges = np.linspace(-1, 0.5, 11)
        
        # now determine the middle points in those shells, and the total
        # SFR within that shell/volume
        centers, psis = [], []
        for first, second in zip(edges, edges[1:]) :
            
            # determine the middle points
            center = np.mean([first, second])
            centers.append(center)
            
            # limit the star particles to the given shell
            mask = (rs > first) & (rs <= second)
            
            # if there are star particles that are within that radius range
            if np.sum(mask) > 0 :
                
                # determine the total mass formed within that bin
                masses_in_bin = masses[mask]
                total_mass = np.sum(masses_in_bin)*u.solMass
                
                # determine the total volume in the shell, in physical units
                outer_r = np.power(10, second*radius)/(1 + redshift)/cosmo.h
                inner_r = np.power(10, first*radius)/(1 + redshift)/cosmo.h
                volume = 4/3*np.pi*(np.power(outer_r, 3) -
                                    np.power(inner_r, 3))*(u.kpc**3)
                
                # determine the SFR within that bin, and also the SFR density
                SFR = (total_mass/delta_t).to(u.solMass/u.yr)
                psi = SFR/volume
                psis.append(psi.value) # ensure that the
                # SFR densities are unitless for future masking, but note
                # that they have units of solMass/yr/kpc^3
            else :
                psis.append(np.nan)
        
        # mask out the nan values
        centers = np.array(centers)[~np.isnan(psis)]
        psis = np.log10(psis)[~np.isnan(psis)]
        
        # if we have sufficient data, then compute the gradient
        if (len(centers) > 1) and (len(psis) > 1) :
            gradient, intercept = np.polyfit(centers, psis, 1)
        else :
            gradient = np.nan
    
    return gradient

def determine_psi(simName, snapNum, delta_t=100*u.Myr, plot=False, save=False) :
    
    # define the output directory
    outDir = 'output/psi(t)/'
    
    '''
    # define the input directory and file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
    # get the masses, satellite and termination times for the quenched sample
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        tsats = hf['satellite_times'][:]
        tonsets = hf['onset_times'][:]
        tterms = hf['termination_times'][:]
    '''
    
    from core import get_test_data
    redshifts, times, subIDs, tsats, tonsets, tterms = get_test_data()
    
    # loop over all the galaxies in the quenched sample
    for subID, tsat, tonset, tterm in zip(subIDs, tsats, tonsets, tterms) :
        # get the mpb snapshot numbers, subIDs, stellar halfmassradii, and
        # galaxy centers
        snapNums, mpb_subIDs, radii, centers = get_mpb_radii_and_centers(
            simName, snapNum, subID)
        
        # limit the time axis to valid snapshots
        ts = times[len(times)-len(snapNums):]
        
        # now get the star particle ages, masses, and distances at each
        # snapshot/time
        gradients = []
        for redshift, time, snap, mpbsubID, center, Re in zip(redshifts,
            ts, snapNums, mpb_subIDs, centers, radii) :
                            
            # get all particles
            ages, masses, rs = get_particles(simName, snapNum, snap, mpbsubID,
                                             center)
            
            # only proceed if the ages, masses, and distances are intact
            if (ages is not None) and (masses is not None) and (rs is not None) :
                
                # get the SF particles
                _, masses, rs = get_sf_particles(ages, masses, rs, time,
                                                 delta_t=delta_t)
                
                # now compute the SFR density gradient at each snapshot/time
                gradient = compute_nabla_psi(redshift, masses, rs, Re, snap,
                                             delta_t=delta_t)
                gradients.append(gradient)
            else :
                gradients.append(np.nan)
        
        if plot :
            # smooth the function for plotting purposes
            smoothed = gaussian_filter1d(gradients, 2)
            
            # determine the y-axis limits based on minima +/- 1 Gyr around
            # the quenching episode
            window = np.where((ts >= tonset - 1) & (ts <= tterm + 1))
            lo, hi = np.nanmin(smoothed[window]) - 1, np.nanmax(smoothed[window]) + 1
            
            outfile = outDir + 'SFR_density_gradient_subID_{}.png'.format(subID)
            ylabel = r'$\nabla \left[ \log (\psi/M_{\odot}~{\rm yr}^{-1}~{\rm kpc}^{-3})/\log (r/R_{\rm e}) \right]$'
            plt.plot_simple_multi_with_times([ts, ts], [gradients, smoothed],
                ['data', 'smoothed'], ['grey', 'k'], ['', ''], ['--', '-'], [0.5, 1],
                tsat, tonset, tterm, xlabel=r'$t$ (Gyr)', ylabel=ylabel,
                xmin=0, xmax=14,
                scale='linear', save=save, outfile=outfile)
    
    return

# functions that were previously in xi.py
def determine_xi(simName, snapNum, redshift, time, snap, mpbsubID, center,
                 delta_t=100*u.Myr) :
    
    # get all particles
    ages, masses, rs = get_particles(simName, snapNum, snap, mpbsubID,
                                     center)
    
    # only proceed if the ages, masses, and distances are intact
    if (ages is not None) and (masses is not None) and (rs is not None) :
        
        # get the SF particles
        _, masses, rs = get_sf_particles(ages, masses, rs, time,
                                         delta_t=delta_t)
        
        # now compute the ratio of the SFR density within 1 kpc
        # relative to the total SFR
        xi = compute_xi(redshift, masses, rs)
    else :
        xi = np.nan
    
    return xi

def determine_xi_fxn(simName, snapNum, redshifts, times, subID, 
                     delta_t=100*u.Myr) :
    
    # get the mpb snapshot numbers, subIDs, stellar halfmassradii, and
    # galaxy centers
    snapNums, mpb_subIDs, _, centers = get_mpb_radii_and_centers(
        simName, snapNum, subID)
    
    # limit the time axis to valid snapshots
    ts = times[len(times)-len(snapNums):]
    
    # now get the star particle ages, masses, and distances at each
    # snapshot/time
    xis = []
    for redshift, time, snap, mpbsubID, center in zip(redshifts, ts,
        snapNums, mpb_subIDs, centers) :
                        
        xi = determine_xi(redshift, time, snap, mpbsubID, center)
        xis.append(xi)
    
    return ts, xis

def save_xi_for_sample(simName, snapNum) :

    # define the input directory and file for the sample, and the output file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    outfile = inDir + '/{}_{}_highMassGals_xi(t).hdf5'.format(
        simName, snapNum)
    
    # get basic information for the sample of primary and satellite systems
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        masses = hf['SubhaloMassStars'][:]
    
    # limit the sample to the highest mass objects
    subIDs = subIDs[(masses > 10.07) & (masses <= 12.75)]
    
    # add empty xi(t) into the HDF5 file to populate later
    if not exists(outfile) :
        with h5py.File(outfile, 'w') as hf :
            if 'subID' not in hf.keys() :
                add_dataset(hf, subIDs, 'SubhaloID')
            if 'xi(t)' not in hf.keys() :
                add_dataset(hf, np.full((len(subIDs), len(times)), np.nan),
                            'xi(t)')
    
    # if the outfile exists, read the relevant information
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            subIDs = hf['SubhaloID'][:]
            x_xi = hf['xi(t)'][:]
    
    # now iterate over every subID in subIDs and get xi(t)
    for i, subID in enumerate(subIDs) :
        
        # if xi(t) doesn't exist for the galaxy, populate the values
        if np.all(np.isnan(x_xi[i, :])) :
            
            _, xi = determine_xi_fxn(simName, snapNum, redshifts, times, subID,
                                     delta_t=100*u.Myr)
            start_index = len(x_xi[i, :]) - len(xi)
            
            # append the determined values for xi(t) into the outfile
            with h5py.File(outfile, 'a') as hf :
                hf['xi(t)'][i, start_index:] = xi
        
        print('{} - {} done'.format(i, subID))
    
    return

def xi_for_sample(simName, snapNum, delta_t=100*u.Myr, plot=False, save=False) :
    
    # define the output directory
    outDir = 'output/xi(t)/'
    
    '''
    # define the input directory and file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
    # get the masses, satellite and termination times for the quenched sample
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        tsats = hf['satellite_times'][:]
        tonsets = hf['onset_times'][:]
        tterms = hf['termination_times'][:]
    '''
    
    # get attributes for the test data
    from core import get_test_data
    redshifts, times, subIDs, tsats, tonsets, tterms = get_test_data()
    
    # get relevant information for the larger sample of 8260 galaxies
    infile = bsPath(simName) + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    with h5py.File(infile, 'r') as hf :
        # sample_subIDs = hf['SubhaloID'][:]
        sample_masses = hf['SubhaloMassStars'][:]
    mask = (sample_masses > 10.07) & (sample_masses <= 12.75)
    # sample_subIDs = sample_subIDs[mask]
    sample_masses = sample_masses[mask]
    
    # get information about high mass galaxies and their xi parameter
    infile = bsPath(simName) + '/{}_{}_highMassGals_xi(t).hdf5'.format(
        simName, snapNum)
    with h5py.File(infile, 'r') as hf :
        highMass_subIDs = hf['SubhaloID'][:]
        highMass_xis = hf['xi(t)'][:]
    
    # loop over all the galaxies in the quenched sample
    for subID, tsat, tonset, tterm in zip(subIDs, tsats, tonsets, tterms) :
        
        # ts, xis = determine_xi_fxn(simName, snapNum, subID, redshifts, times,
        #                            delta_t=delta_t)
        
        # find the corresponding index for the galaxy
        loc = np.where(highMass_subIDs == subID)[0][0]
        
        # find galaxies in a similar mass range as the galaxy
        mass = sample_masses[loc]
        mass_bin = determine_mass_bin_indices(sample_masses, mass, halfwidth=0.05)
        
        # use the xi values for those comparison galaxies to determine percentiles
        comparison_xis = highMass_xis[mass_bin]
        lo_xi, hi_xi = np.nanpercentile(comparison_xis, [16, 84], axis=0)
        lo_xi = gaussian_filter1d(lo_xi, 2)
        hi_xi = gaussian_filter1d(hi_xi, 2)
        
        # get xi for the galaxy
        xis = highMass_xis[loc]
        
        if plot :            
            # smooth the function for plotting purposes
            smoothed = gaussian_filter1d(xis, 2)
            
            outfile = outDir + 'xi_subID_{}.png'.format(subID)
            ylabel = r'$\xi = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$'
            plt.plot_simple_multi_with_times(
                [times, times, times, times],
                [xis, smoothed, lo_xi, hi_xi],
                ['data', 'smoothed', 'lo, hi', ''],
                ['grey', 'k', 'lightgrey', 'lightgrey'],
                ['', '', '', ''],
                ['--', '-', '-.', '-.'],
                [0.5, 1, 1, 1],
                tsat, tonset, tterm, xlabel=r'$t$ (Gyr)', ylabel=ylabel,
                xmin=0, xmax=14, ymin=0, ymax=1,
                scale='linear', save=save, outfile=outfile)
    
    return

# functions that were previously in zeta.py
def scatter() :
    
    simName = 'TNG50-1'
    snapNum = 99
    
    # get information about high mass galaxies and their xi parameter
    infile = bsPath(simName) + '/{}_{}_highMassGals_xi(t).hdf5'.format(
        simName, snapNum)
    with h5py.File(infile, 'r') as hf :
        highMass_subIDs = hf['SubhaloID'][:]
        highMass_xis = hf['xi(t)'][:]
    
    # get information about high mass galaxies and their zeta parameter
    infile = bsPath(simName) + '/{}_{}_highMassGals_zeta(t).hdf5'.format(
        simName, snapNum)
    with h5py.File(infile, 'r') as hf :
        highMass_subIDs = hf['SubhaloID'][:]
        highMass_zetas = hf['zeta(t)'][:]
    
    return

def compute_sf_rms(rs) :
    
    if len(rs) == 0 :
        rms = 0.0
    else :
        rms = np.sqrt(np.mean(np.square(rs)))
    
    return rms

def determine_zeta(simName, snapNum, time, snap, mpbsubID, center,
                   delta_t=100*u.Myr) :
    
    # get all particles
    ages, masses, rs = get_particles(simName, snapNum, snap, mpbsubID,
                                     center)
    
    # only proceed if the ages, masses, and distances are intact
    if (ages is not None) and (masses is not None) and (rs is not None) :
        
        # find the stellar half mass radius for all particles
        stellar_halfmass_radius = compute_halfmass_radius(masses, rs)
        
        # get the SF particles
        _, sf_masses, sf_rs = get_sf_particles(ages, masses, rs, time,
                                               delta_t=delta_t)
        
        # find the stellar half mass radius for SF particles
        sf_halfmass_radius = compute_halfmass_radius(sf_masses, sf_rs)
        
        # now compute the ratio of the half mass radius of the SF
        # particles to the half mass radius of all particles
        zeta = sf_halfmass_radius/stellar_halfmass_radius
    else :
        zeta = np.nan
    
    return zeta

def determine_zeta_fxn(simName, snapNum, times, subID, delta_t=100*u.Myr) :
    
    # get the mpb snapshot numbers, subIDs, stellar halfmassradii, and
    # galaxy centers
    snapNums, mpb_subIDs, _, centers = get_mpb_radii_and_centers(
        simName, snapNum, subID)
    
    # limit the time axis to valid snapshots
    ts = times[len(times)-len(snapNums):]
    
    # now get the star particle ages, masses, and distances at each
    # snapshot/time
    zetas = []
    for time, snap, mpbsubID, center in zip(ts, snapNums, mpb_subIDs, centers) :
        
        zeta = determine_zeta(simName, snapNum, time, snap, mpbsubID, center,
                              delta_t=delta_t)
        zetas.append(zeta)
    
    return ts, zetas

def save_zeta_for_sample(simName, snapNum) :

    # define the input directory and file for the sample, and the output file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    outfile = inDir + '/{}_{}_highMassGals_zeta(t).hdf5'.format(
        simName, snapNum)
    
    # get basic information for the sample of primary and satellite systems
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        times = hf['times'][:]
        masses = hf['SubhaloMassStars'][:]
    
    # limit the sample to the highest mass objects
    subIDs = subIDs[(masses > 10.07) & (masses <= 12.75)]
    
    # add empty zeta(t) into the HDF5 file to populate later
    if not exists(outfile) :
        with h5py.File(outfile, 'w') as hf :
            if 'subID' not in hf.keys() :
                add_dataset(hf, subIDs, 'SubhaloID')
            if 'zeta(t)' not in hf.keys() :
                add_dataset(hf, np.full((len(subIDs), len(times)), np.nan),
                            'zeta(t)')
    
    # if the outfile exists, read the relevant information
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            subIDs = hf['SubhaloID'][:]
            x_zeta = hf['zeta(t)'][:]
    
    # now iterate over every subID in subIDs and get zeta(t)
    for i, subID in enumerate(subIDs) :
        
        # if zeta(t) doesn't exist for the galaxy, populate the values
        if np.all(np.isnan(x_zeta[i, :])) :
            
            _, zeta = determine_zeta_fxn(simName, snapNum, times, subID,
                                         delta_t=100*u.Myr)
            start_index = len(x_zeta[i, :]) - len(zeta)
            
            # append the determined values for zeta(t) into the outfile
            with h5py.File(outfile, 'a') as hf :
                hf['zeta(t)'][i, start_index:] = zeta
        
        print('{} - {} done'.format(i, subID))
    
    return

def zeta_for_sample(simName, snapNum, delta_t=100*u.Myr, plot=False, save=False) :
    
    # define the output directory
    outDir = 'output/zeta(t)/'
    
    '''
    # define the input directory and file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_quenched_SFHs(t).hdf5'.format(simName, snapNum)
    
    # get the masses, satellite and termination times for the quenched sample
    with h5py.File(infile, 'r') as hf :
        subIDs = hf['SubhaloID'][:]
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        tsats = hf['satellite_times'][:]
        tonsets = hf['onset_times'][:]
        tterms = hf['termination_times'][:]
    '''
    
    # get attributes for the test data
    from core import get_test_data
    _, times, test_subIDs, tsats, tonsets, tterms = get_test_data()
    
    # get relevant information for the larger sample of 8260 galaxies
    infile = bsPath(simName) + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    with h5py.File(infile, 'r') as hf :
        # sample_subIDs = hf['SubhaloID'][:]
        sample_masses = hf['SubhaloMassStars'][:]
    mask = (sample_masses > 10.07) & (sample_masses <= 12.75)
    # sample_subIDs = sample_subIDs[mask]
    sample_masses = sample_masses[mask]
    
    # get information about high mass galaxies and their zeta parameter
    infile = bsPath(simName) + '/{}_{}_highMassGals_zeta(t).hdf5'.format(
        simName, snapNum)
    with h5py.File(infile, 'r') as hf :
        highMass_subIDs = hf['SubhaloID'][:]
        highMass_zetas = hf['zeta(t)'][:]
    
    # loop over all the galaxies in the quenched sample
    for subID, tsat, tonset, tterm in zip(test_subIDs, tsats, tonsets, tterms) :
        
        # ts, zetas = determine_zeta_fxn(simName, snapNum, times, subID,
        #                                delta_t=delta_t)
        
        # find the corresponding index for the galaxy
        loc = np.where(highMass_subIDs == subID)[0][0]
        
        # find galaxies in a similar mass range as the galaxy
        mass = sample_masses[loc]
        mass_bin = determine_mass_bin_indices(sample_masses, mass, halfwidth=0.05)
        
        # use the zeta values for those comparison galaxies to determine percentiles
        comparison_zetas = highMass_zetas[mass_bin]
        lo_zeta, hi_zeta = np.nanpercentile(comparison_zetas, [16, 84], axis=0)
        lo_zeta = gaussian_filter1d(lo_zeta, 2)
        hi_zeta = gaussian_filter1d(hi_zeta, 2)
        
        # get zeta for the galaxy
        zetas = highMass_zetas[loc]
        
        if plot :
            # smooth the function for plotting purposes
            smoothed = gaussian_filter1d(zetas, 2)
            
            outfile = outDir + 'zeta_subID_{}.png'.format(subID)
            ylabel = r'$\zeta = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}}$'
            plt.plot_simple_multi_with_times(
                [times, times, times, times],
                [zetas, smoothed, lo_zeta, hi_zeta],
                ['data', 'smoothed', 'lo, hi', ''],
                ['grey', 'k', 'lightgrey', 'lightgrey'],
                ['', '', '', ''],
                ['--', '-', '-.', '-.'],
                [0.5, 1, 1, 1],
                tsat, tonset, tterm, xlabel=r'$t$ (Gyr)', ylabel=ylabel,
                xmin=0, xmax=14, ymin=0, ymax=6,
                scale='linear', save=save, outfile=outfile)
    
    return
