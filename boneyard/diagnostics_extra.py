
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
