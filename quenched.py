
from os.path import exists
import numpy as np

import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from core import add_dataset, bsPath, determine_mass_bin_indices
import plotting as plt

def determine_comparison_systems_relative(simName='TNG50-1', snapNum=99,
                                          hw=0.1, minNum=50) :
    
    # define the input directory and the input file for the SFHs
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    
    # open the SFHs for the sample of candidate primary and satellite systems
    with h5py.File(infile, 'r') as hf :
        logM = hf['logM'][:] # track the stellar mass history (SMH)
        SFHs = hf['SFH'][:]
        SFMS = hf['SFMS'][:].astype(bool) # (boolean) SFMS at each snapshot
        exclude = hf['exclude'][:] # 103 galaxies should be excluded, see sfhs.py
    
    # loop through the galaxies in the sample
    for i, (SMH, SFH, use) in enumerate(zip(logM, SFHs, ~exclude)) :
        if use :
            # find the first index with a mass greater than 10^4.45 (minimum of
            # all valid values across all times)
            init = np.argmax(SMH > 4.45)
            
            # the SFMS array is mostly NaN for the first few snapshots, so
            # start at either the init snapshot or snapshot 4 (equal to z = 10)
            init = max(init, 4)
            
            # pad the front of the lo and hi arrays with NaNs
            lo_SFH = np.full(init, np.nan).tolist()
            hi_SFH = np.full(init, np.nan).tolist()
            
            # work through the snapshots
            for snap, mass in zip(np.arange(init, 100), SMH[init:]) :
                
                # get values at the snapshot
                SFHs_at_snap = SFHs[:, snap]
                logM_at_snap = logM[:, snap]
                SFMS_at_snap = SFMS[:, snap]
                
                # create a mask for the SFMS galaxy masses at that snapshot
                SFMS_at_snap_masses_mask = np.where(SFMS_at_snap > 0,
                                                    logM_at_snap, False)
                
                # find galaxies in a similar mass range as the galaxy, but that
                # are on the SFMS at that snapshot
                mass_bin = determine_mass_bin_indices(SFMS_at_snap_masses_mask,
                    mass, hw=hw, minNum=minNum)
                
                # get the percentiles for those SFHs
                lo, hi = np.nanpercentile(SFHs_at_snap[mass_bin], [2.5, 97.5])
                
                # append those values
                lo_SFH.append(lo)
                hi_SFH.append(hi)
            
            # save those limits
            with h5py.File(infile, 'a') as hf :
                hf['lo_SFH'][i, :] = np.array(lo_SFH)
                hf['hi_SFH'][i, :] = np.array(hi_SFH)
    
    return

def determine_quenched_systems_relative(simName='TNG50-1', snapNum=99, kernel=2) :
    
    # define the input directory and the input file for the SFHs
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    
    # open the SFHs for the sample of candidate primary and satellite systems
    with h5py.File(infile, 'r') as hf :
        times = hf['times'][:]
        subIDfinals = hf['SubhaloID'][:]
        SFHs = hf['SFH'][:]
        lo_SFH = hf['lo_SFH'][:]
        hi_SFH = hf['hi_SFH'][:]
        exclude = hf['exclude'][:] # 103 galaxies should be excluded, see sfhs.py
    
    # loop through the galaxies in the sample
    for i, (subID, SFH, use, lo, hi) in enumerate(zip(subIDfinals, SFHs,
                                                      ~exclude, lo_SFH, hi_SFH)) :
        if use :
            # smooth the SFH of the specific galaxy
            smoothed = gaussian_filter1d(SFH, kernel)
            
            # smooth the curves
            lo_sm = gaussian_filter1d(lo, kernel)
            hi_sm = gaussian_filter1d(hi, kernel) # only for plotting
            
            # lo_diff is positive before quenching, then negative after
            lo_diff = smoothed - lo_sm
            
            # find the first element that is non NaN
            smoothed_idx = np.argmax(np.isfinite(smoothed) == True)
            lo_idx = np.argmax(np.isfinite(lo_diff) == True)
            start_index = np.max([smoothed_idx, lo_idx])
            
            # find the index where the galaxy is first totally under the lo curve
            termination_index = np.argmax(lo_diff[start_index:] < 0) + start_index
            # or use np.where(lo_diff[start_lim:] < 0)[0][0] + start_lim
            
            # these are the criteria for the galaxy to be quenched
            if (np.all(lo_diff[start_index:termination_index] >= 0) and
                np.all(lo_diff[termination_index:] <= 0)) :
                
                # find the peaks of the smoothed curve, before quenching finishes
                maxima, props = find_peaks(smoothed[:termination_index],
                                           height=0.4*np.nanmax(smoothed))
                
                if len(maxima) > 0 : # there must be at least 1 maximum
                    
                    # use the final peak as the onset of quenching
                    onset_index = maxima[-1]
                    
                    # define the onset and termination times
                    tonset = times[onset_index]
                    tterm = times[termination_index]
                    
                    # if the galaxy is quenched, change the value in the mask,
                    # and update the onset and termination indices and times
                    with h5py.File(infile, 'a') as hf :
                        hf['quenched'][i] = True
                        hf['onset_indices'][i] = onset_index
                        hf['onset_times'][i] = tonset
                        hf['termination_indices'][i] = termination_index
                        hf['termination_times'][i] = tterm
                    
                    # define the outfile name for the plot
                    outfile = ('{}/figures/quenched_SFHs(t)/'.format(simName) +
                               'quenched_SFH_subID_{}.png'.format(subID))
                    
                    plt.plot_simple_multi_with_times([times, times, times],
                        [smoothed, lo_sm, hi_sm], ['smoothed', 'lo, hi', ''],
                        ['k', 'grey', 'grey'], ['', '', ''],
                        ['-', '-.', '-.'], [1, 0.8, 0.8], np.nan, tonset, tterm,
                        xlabel=r'$t$ (Gyr)', ylabel=r'SFR ($M_{\odot}$ yr$^{-1}$)',
                        xmin=-0.1, xmax=13.8, scale='linear', outfile=outfile,
                        save=True, loc=0)
    
    return

def quenched_example() :
    
    colwidth = 3.35224200913242
    # textwidth = 7.10000594991006
    textheight = 9.095321710253218
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        # subIDs = hf['subIDs'][:].astype(int)
        times = hf['times'][:]
        subIDfinals = hf['SubhaloID'][:].astype(int)
        logM = hf['logM'][:]
        SFHs = hf['SFH'][:]
        SFMS = hf['SFMS'][:].astype(bool)
        lo_SFHs = hf['lo_SFH'][:]
        hi_SFHs = hf['hi_SFH'][:]
        tonsets = hf['onset_times'][:]
        tterms = hf['termination_times'][:]
    
    loc = 1115   # subID 198186 at z = 0
    sfloc = 3109 # subID 537941 at z = 0
    
    xs = [times, times, times]
    ys = [gaussian_filter1d(SFHs[loc], 2),
          gaussian_filter1d(lo_SFHs[loc], 2),
          gaussian_filter1d(hi_SFHs[loc], 2)]
    sf_ys = [gaussian_filter1d(SFHs[sfloc], 2),
             gaussian_filter1d(lo_SFHs[sfloc], 2),
             gaussian_filter1d(hi_SFHs[sfloc], 2)]
    labels = ['quenched', r'$\pm 2 \sigma$', '']
    sf_labels = ['SF', r'$\pm 2 \sigma$', '']
    colors = ['k', 'grey', 'grey']
    styles = ['-', '-.', '-.']
    sf_styles = ['--', '-.', '-.']
    alphas = [1, 1, 1]
    
    # plt.plot_simple_multi_with_times(xs, ys, labels, colors, markers, styles,
    #     alphas, np.nan, tonsets[loc], tterms[loc], [np.nan], [''], scale='linear',
    #     xlabel=r'$t$ (Gyr)', ylabel=r'SFR (${\rm M}_{\odot}$ yr$^{-1}$)',
    #     xmin=0, xmax=13.8, ymin=0)
    
    plt.double_simple_with_times(xs, sf_ys, colors, sf_styles, alphas, sf_labels,
        xs, ys, colors, styles, alphas, labels, tonsets[loc], tterms[loc],
        xlabel=r'$t$ (Gyr)', ylabel=r'SFR (${\rm M}_{\odot}$ yr$^{-1}$)',
        xmin=0, xmax=13.8, ymin=0, ymax=10, figsizeheight=textheight/2,
        figsizewidth=colwidth, save=False, outfile='SFHs.pdf')
    
    '''
    # find SFMS galaxies with a similar stellar mass at onset
    mass = logM[1115, 43] # ionset = 43 for subID 198186
    comparisons = (np.abs(logM[:, 43] - mass) <= 0.1) & (SFMS[:, 43]) & (SFMS[:, 99])
    for subIDfinal, SFH, lo, hi in zip(subIDfinals[comparisons],
        SFHs[comparisons], lo_SFHs[comparisons], hi_SFHs[comparisons]) :
        ys = [gaussian_filter1d(SFH, 2), gaussian_filter1d(lo, 2),
              gaussian_filter1d(hi, 2)]
        ymax = np.nanmax([ys[0], ys[1], ys[2]])
        plt.plot_simple_multi_with_times(xs, ys, labels, colors, markers, styles,
            alphas, np.nan, -5, -4, [np.nan], [''], scale='linear',
            xlabel=r'$t$ (Gyr)', ylabel=r'SFR (${\rm M}_{\odot}$ yr$^{-1}$)',
            xmin=0, xmax=13.8, ymin=0, ymax=ymax, save=True,
            outfile='find_subID_198186_comparison/comparison_{}.png'.format(
                subIDfinal))
    '''
    
    return

def quenched_galaxies_on_SFMS() :
    
    colwidth = 3.35224200913242
    # textwidth = 7.10000594991006
    textheight = 9.095321710253218
    
    # get the stellar masses and SFHs for the entire sample
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        logM = hf['logM'][:]
        SFHs = hf['SFH'][:]
        SFMS = hf['SFMS'][:]
        quenched = hf['quenched'][:]
    
    # the edges, 16th, and 84th percentiles will be loaded from the helper file
    with h5py.File('TNG50-1/TNG50-1_99_SFMS_helper.hdf5', 'r') as helper :
        centers = helper['Snapshot_99/edges'][:-1] + 0.1
        los = helper['Snapshot_99/los'][:]
        his = helper['Snapshot_99/his'][:]
    
    # limit the selections to the massive population
    mass_mask = (logM[:, 99] >= 9.5)
    SFMS_mask = (SFMS[:, 99].astype(bool))[mass_mask]
    SFRs = SFHs[:, 99][mass_mask]
    masses = logM[:, 99][mass_mask]
    quenched = quenched[mass_mask]
    sf_mask = (~SFMS_mask) & (~quenched)
    
    # define a mask for the quiescent population
    q_mask = (SFRs == 0)
    
    # update values in the array to prepare for taking the logarithm
    SFRs[q_mask] = 0.00066861
    SFRs = np.log10(SFRs)
    
    # perturb the quiescent values
    SFRs[q_mask] = np.log10(0.00066861) - np.random.rand(np.sum(q_mask))
    
    # create three populations: quenched, SF, and SFMS
    quenched_masses, quenched_SFRs = masses[quenched], SFRs[quenched]
    sf_masses, sf_SFRs = masses[sf_mask], SFRs[sf_mask]
    SFMS_masses, SFMS_SFRs = masses[SFMS_mask], SFRs[SFMS_mask]
    
    # plot the SFMS with those percentiles
    outfile = 'SFMS_z0_with_quenched_new.pdf'
    plt.plot_scatter_multi_with_bands(SFMS_masses, SFMS_SFRs,
        quenched_masses, quenched_SFRs, sf_masses, sf_SFRs, centers, los, his,
        xlabel=r'$\log(M_{*}/{\rm M}_{\odot})$',
        ylabel=r'$\log({\rm SFR}/{\rm M}_{\odot}~{\rm yr}^{-1}$)',
        xmin=9.5, xmax=12, ymin=-4.3, ymax=2,
        figsizeheight=textheight/2, figsizewidth=colwidth,
        outfile=outfile, save=False)
    
    return

def schechter_log(logM, Mstar, alpha, phi) :
    return np.log(10)*np.exp(-np.power(10, logM - Mstar))*np.power(
        10, (logM - Mstar)*(alpha + 1))*phi

def schechter_double_log(logM, Mstar, alpha1, alpha2, phi1, phi2) :
    return np.log(10)*np.exp(-np.power(10, logM - Mstar))*(
        phi1*np.power(10, (logM - Mstar)*(alpha1 + 1)) +
        phi2*np.power(10, (logM - Mstar)*(alpha2 + 1)))

def quenched_mass_distribution_z0() :
    
    colwidth = 3.35224200913242
    # textwidth = 7.10000594991006
    textheight = 9.095321710253218
    
    # get the stellar masses and SFHs for the entire sample
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        logM = hf['logM'][:, 99]
        # SFHs = hf['SFH'][:]
        # SFMS = hf['SFMS'][:]
        quenched = hf['quenched'][:]
    
    mask = (logM >= 9.5)
    quenched = quenched[mask]
    logM = logM[mask]
    
    edges = np.arange(9.5, 12, 0.25)
    xs = edges[:-1] + np.diff(edges)/2
    
    # plt.histogram(logM[quenched], r'$\log(M_{*}/{\rm M}_{\odot})$', bins=edges)
    
    vals, _ = np.histogram(logM[quenched], bins=edges)
    volume = np.power(35/0.6774, 3)
    
    data = vals/volume/0.25
    
    # create an array of stellar masses for plotting the SMF
    logM = np.linspace(9.5, 12, 100)
    
    # fit values from Baldry+ 2012 for blue and red z < 0.06 galaxies
    # baldry_sf = schechter_log(logM, 10.72, -1.45, 0.71e-3)
    baldry = schechter_double_log(logM, 10.72, -0.45, -1.45, 3.25e-3, 0.08e-3)
    renorm = [np.nan]*100
    
    plt.plot_simple_multi([xs, logM, logM],
        [np.log10(data), np.log10(baldry), np.log10(renorm)],
        ['quenched', 'Baldry et al. (2012)', ''],
        ['k', 'k', 'k'], ['o', '', ''], ['', '-', '--'],
        [1, 1, 1], xlabel=r'$\log(M_{*}/{\rm M}_{\odot})$',
        ylabel=r'$\log{({\rm number~density}/{\rm dex}^{-1}~{\rm Mpc}^{-3})}$',
        xmin=9.5, xmax=11.75, ymin=-5, ymax=-2, loc=3,
        figsizeheight=textheight/3, figsizewidth=colwidth, save=False,
        outfile='SMF.pdf')
    
    return
