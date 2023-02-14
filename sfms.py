
import pickle
import numpy as np

import h5py
from scipy.signal import savgol_filter

from core import add_dataset, bsPath
import plotting as plt

def check_SFMS_and_limits(simName, snapNum) :
    
    # define the input directory and the input file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    
    # open the SFHs for the sample of candidate primary and satellite systems
    with h5py.File(infile, 'r') as hf :
        # times = hf['times'][:]
        # masses = hf['SubhaloMassStars'][:]
        SFHs = hf['SFH'][:]
    
    # select the SFHs corresponding to the SFMS at z = 0, and write a mask to file
    with h5py.File(infile, 'a') as hf :
        if 'SFMS' not in hf.keys() :
            add_dataset(hf, SFHs[:, -1] > 0.001, 'SFMS')
    
    # check the sSFR distribution in different mass bins
    # check_sSFR(times, masses, SFHs, mass_bin_edges)
    
    # check the SFMS at z = 0 in different mass bins
    # check_SFMS_at_z0(masses, SFHs, mass_bin_edges)
    
    # compute the SFR limits for the SFMS at z = 0 in different mass bins
    # compute_SFMS_limits(simName, snapNum, masses, SFHs, mass_bin_edges,
    #                     window_length, polyorder)
    
    # check the limits and save plots to file
    # check_SFMS_limits(simName, snapNum, times, mass_bin_edges)
    
    return

def check_sSFR(times, masses, SFHs, mass_bin_edges) :
        
    for lo, hi in zip(mass_bin_edges[:-1], mass_bin_edges[1:]) :
        
        # define the subsample for the mass range/bin of interest
        mask = (masses >= lo) & (masses < hi)
        subsample_SFHs = SFHs[mask][:]
        subsample_masses = masses[mask]
        
        # define an empty array to hold the sSFRs
        sSFRs = np.full((len(subsample_SFHs), len(times)), np.nan)
        
        # set a lower limit for the SFH so that we can take the log more easily
        subsample_SFHs[subsample_SFHs == 0.0] = 1e-6
        
        # compute the sSFRs for the sample in the mass bin
        for i in range(len(subsample_SFHs)) :
            sSFRs[i] = np.log10(subsample_SFHs[i, :]) - subsample_masses[i]
        
        # create an array of those times, and make a 1D array for the sSFRs
        xs = np.array(list(times)*len(sSFRs))
        ys = sSFRs.flatten()
        
        # mask out infinite values, coming from np.log10(0)
        good = np.isfinite(ys)
        
        # plot the 2D histograms
        title = '{}'.format(lo) + r'$< \log(M_{*}/M_{\odot}) <$' + '{}'.format(hi)
        plt.histogram_2d(xs[good], ys[good], bins=[np.arange(0, 14.5, 0.5), 20],
                         xlabel=r'$t$ (Gyr)',
                         ylabel=r'$\log({\rm sSFR} / {\rm yr}^{-1}$)',
                         title=title, xmin=0, xmax=13.8, ymin=-19, ymax=-8)
    
    return

def check_SFMS_at_z0(masses, SFHs, mass_bin_edges) :
    
    for i, (lo, hi) in enumerate(zip(mass_bin_edges[:-1], mass_bin_edges[1:])) :
        
        # define the subsample for the mass range/bin of interest
        mask = (masses >= lo) & (masses < hi)
        subsample_SFHs = SFHs[mask][:]
        subsample_masses = masses[mask]
        
        # define an empty array to hold the SFRs at z = 0, and a mask for the
        # quiescent population
        SFRs_at_z0 = np.full(len(subsample_SFHs), np.nan)
        q_mask = subsample_SFHs[:, -1] == 0
        
        # populate that array with the corresponding SFRs, separating the
        # quiescent population from the star forming population
        SFRs_at_z0[q_mask] = -5 - np.random.rand(np.sum(q_mask))
        SFRs_at_z0[~q_mask] = np.log10(subsample_SFHs[:, -1][~q_mask])
        
        # plot and save the SFMS in each mass bin
        outfile = 'output/SFMS(t)/SFMS_z0_massBin_{}.png'.format(i)
        plt.plot_scatter(subsample_masses, SFRs_at_z0, 'k', '', 'o',
                         xlabel=r'$\log(M_{*}/M_{\odot})$',
                         ylabel=r'$\log(SFR/M_{\odot}$ yr$^{-1}$)',
                         save=True, outfile=outfile)
    
    return

def compute_SFMS_limits(simName, snapNum, masses, SFHs, mass_bin_edges,
                        window_length, polyorder, logSF_tresh=-3) :
    
    # define the output directory and the output file
    outDir = bsPath(simName)
    outfile = outDir + '/{}_{}_SFMS_SFH_limits(t).pkl'.format(simName, snapNum)
    
    dictionary = {}
    for i, (lo, hi) in enumerate(zip(mass_bin_edges[:-1], mass_bin_edges[1:])) :
        
        # define the subsample for the mass range/bin of interest
        mask = (masses >= lo) & (masses < hi)
        subsample_SFHs = SFHs[mask][:]
        
        # define a mask for the star forming main sequence (at z = 0)
        # population (which was verified in `check_SFMS_at_z0`), and select
        # those SFHs
        SFMS_at_z0_mask = np.log10(subsample_SFHs[:, -1]) > logSF_tresh
        SFMS_at_z0 = subsample_SFHs[SFMS_at_z0_mask]
        
        # use the SFMS population to define the SFR +/- 2 sigma limits
        lo_SFH, hi_SFH = np.percentile(SFMS_at_z0, [2.5, 97.5], axis=0)
        
        # smooth the limits so they are less choppy
        lo_SFH = savgol_filter(lo_SFH, window_length, polyorder)
        lo_SFH[lo_SFH < 0] = 0
        
        hi_SFH = savgol_filter(hi_SFH, window_length, polyorder)
        hi_SFH[hi_SFH < 0] = 0
        
        # now create the appropriate label and store info into a dictionary
        label = 'mass_bin_{}'.format(i)
        dictionary[label] = {}
        dictionary[label]['lo'] = lo
        dictionary[label]['hi'] = hi
        dictionary[label]['lo_SFH'] = lo_SFH
        dictionary[label]['hi_SFH'] = hi_SFH
    
    # save the dictionary to a pickle object file for future use
    with open(outfile, 'wb') as file :
        pickle.dump(dictionary, file)
    
    return

def check_SFMS_limits(simName, snapNum, times, mass_bin_edges, save=False) :
    
    # define the output directory and the output file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_SFMS_SFH_limits(t).pkl'.format(simName, snapNum)
    
    # open the dictionary containing the SFH limits
    with open(infile, 'rb') as file :
        dic = pickle.load(file)
    
    # iterate over the mass bins and save plots of the limits for reference
    for i, key in enumerate(dic.keys()) :
        lo, hi = dic[key]['lo_SFH'], dic[key]['hi_SFH']
        
        # plot and save the SFH limits in each mass bin
        outfile = 'output/SFH_limits(t)/SFMS_SFH_limits_massBin_{}.png'.format(i)
        plt.plot_simple_multi([times, times], [lo, hi],
                              ['lo, hi', ''], ['grey', 'grey'], ['', ''],
                              ['-', '-'], [1, 1],
                              xlabel=r'$t$ (Gyr)',
                              ylabel=r'SFR ($M_{\odot}$ yr$^{-1}$)',
                              xmin=-0.1, xmax=13.8, scale='linear',
                              outfile=outfile, save=save)
    
    return
