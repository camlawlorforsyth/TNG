
from os.path import exists
import numpy as np

import astropy.units as u
import h5py

from core import (add_dataset, bsPath, determine_mass_bin_indices, find_nearest)
from diagnostics import determine_radial_profiles
from xi import determine_xi
from zeta import determine_zeta
import plotting as plt

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









