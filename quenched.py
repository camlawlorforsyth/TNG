
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
