
from os.path import exists
import numpy as np

import astropy.units as u
import h5py

from core import bsPath, get_particles, get_sf_particles

def check_for_nan_histories(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and file, and the output file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    
    # get the stellar masses and SFHs for the entire sample
    with h5py.File(infile, 'r') as hf :
        logM = hf['logM'][:] # track the stellar mass history (SMH)
        SFHs = hf['SFH'][:]
    
    # determine if any galaxies should be excluded from future analysis
    for i, (SMH, SFH) in enumerate(zip(logM, SFHs)) :
        # 19 galaxies have all NaN SMHs, and all NaN subIDs
        condition1 = np.all(np.isfinite(SMH) == False)
        
        # 47 galaxies have all 0 SFHs
        condition2 = np.all(SFH[~np.isnan(SFH)] == 0.0)
        
        # 37 galaxies are created after z = 0.5 which seems unphysical
        condition3 = np.argmax(SMH > 4.45) > 70
        
        if (condition1 | condition2 | condition3) :
            with h5py.File(infile, 'a') as hf :
                hf['exclude'][i] = True
    return

def determine_all_histories_from_cutouts(simName='TNG50-1', snapNum=99,
                                         delta_t=100*u.Myr) :
    
    # define the input directory and file, and the output file
    inDir = bsPath(simName)
    outfile = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    
    # check if the outfile exists and has good SFHs
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            if 'SFH' in hf.keys() :
                if np.all(~np.isnan(hf['SFH'])) :
                    print('File already exists with all non-NaN SFHs')
    
    # if the outfile exists, read the relevant information
    if exists(outfile) :
        with h5py.File(outfile, 'r') as hf :
            snapshots = hf['snapshots'][:]
            times = hf['times'][:]
            subIDs = hf['subIDs'][:].astype(int)
            Res = hf['Re'][:]
            centers = hf['centers'][:]
            x_sfhs = hf['SFH'][:]
    
    # now iterate over every subID in subIDs and get the SFH
    for i, (mpb_subIDs, mpb_Res, mpb_centers) in enumerate(zip(subIDs, Res,
                                                               centers)) :
        
        # if the SFH values don't exist, determine the SFH and populate
        if np.all(np.isnan(x_sfhs[i, :])) :
            with h5py.File(outfile, 'a') as hf : # use the cutout values
                hf['SFH'][i, :] = history_from_cutouts(snapshots, times,
                    mpb_subIDs, mpb_Res, mpb_centers, simName=simName,
                    snapNum=snapNum, delta_t=delta_t)
        
        print('{}/8260 - subID {} done'.format(i, mpb_subIDs[-1]))
    
    return

def history_from_cutouts(snapshots, times, mpb_subIDs, mpb_Res, mpb_centers,
                         simName='TNG50-1', snapNum=99, delta_t=100*u.Myr) :
    
    # get the star particle ages, masses, and distances at each snapshot/time
    SFH = []
    for snap, time, subID, Re, center in zip(snapshots, times, mpb_subIDs,
                                             mpb_Res, mpb_centers) :
        
        if subID == -2147483648 : # np.nan cast into an integer, ie. for a
            SFH.append(np.nan)    # subID value that doesn't exist
        else :
            # get all particles
            ages, masses, rs = get_particles(simName, snapNum, snap, subID,
                                             center)
            
            # only proceed if the ages, masses, and distances are intact
            if (ages is not None) and (masses is not None) and (rs is not None) :
                
                # get the SF particles
                _, masses, rs = get_sf_particles(ages, masses, rs, time,
                                                 delta_t=delta_t)
                
                # now compute the SFR for particles within 2Re
                if len(masses) == 0 :
                    SFR = 0.0
                else :
                    total_mass = np.sum(masses[rs <= 2*Re])*u.solMass
                    SFR = total_mass/delta_t
                    SFR = (SFR.to(u.solMass/u.yr)).value
                SFH.append(SFR)
            else :
                SFH.append(np.nan)
    
    return SFH










from scipy.ndimage import gaussian_filter1d
from core import find_nearest
import plotting as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
    times = hf['times'][:]
    subIDs = hf['subIDs'][:].astype(int)
    subIDfinals = hf['SubhaloID'][:]
    logM = hf['logM'][:, -1]
    SMHs = hf['logM'][:]
    SFHs = hf['SFH'][:]
    lo_SFHs = hf['lo_SFH'][:]
    hi_SFHs = hf['hi_SFH'][:]
    quenched = hf['quenched'][:]
    ionsets = hf['onset_indices'][:].astype(int)
    tonsets = hf['onset_times'][:]
    iterms = hf['termination_indices'][:].astype(int)
    tterms = hf['termination_times'][:]
    # comparison = hf['comparison'][:]
    
    Res = hf['Re'][:]
    centers = hf['centers'][:]

quenched = (quenched & (logM >= 9.5))
i75s = np.array(find_nearest(times, tonsets + 0.75*(tterms - tonsets)))

np.random.seed(0)

plot = False
outDir = 'TNG50-1/figures/quenched_SFHs(t)_logM-gtr-9.5/'

seventyfives = []
for i, (subIDfinal, subIDhist, ReHist, centerHist, SMH, SFH, lo, hi, ionset, tonset,
    iterm, tterm, i75, val) in enumerate(zip(subIDfinals[quenched],
    subIDs[quenched], Res[quenched], centers[quenched], SMHs[quenched],
    SFHs[quenched], lo_SFHs[quenched], hi_SFHs[quenched], ionsets[quenched],
    tonsets[quenched], iterms[quenched], tterms[quenched], i75s[quenched],
    np.random.rand(278))) :
    
    SFH = gaussian_filter1d(SFH, 2)
    lo = gaussian_filter1d(lo, 2)
    hi = gaussian_filter1d(hi, 2)
    
    # defined relative to only the onset of quenching
    onset_SFR = SFH[ionset]
    SFH_after_peak = SFH[ionset:]
    drop75_list = np.where(SFH_after_peak - 0.25*onset_SFR <= 0)[0]
    
    if subIDfinal not in [43, 514274, 656524, 657979, 680429] : # len(drop75_list) > 0
        
        # the first snapshot where the SFR is 75% below the SFR at quenching onset
        drop75 = ionset + drop75_list[0]
        # drop75 = ionset + np.argmin(np.abs(SFH_after_peak - 0.25*onset_SFR))
        
        # drop_times = [times[i75], times[drop75]]
        # drop_labels = [r'$t_{\rm 0.75~through~episode}$',
        #                r'$t_{\rm 0.25~SFR_{\rm onset}}$']
        
        Re = ReHist[drop75]
        center = centerHist[drop75]
        
        string = 'save_skirt_input({}, {}, {}, {}*u.kpc, {})'.format(
            subIDfinal, drop75, subIDhist[drop75], Re, list(center))
        print(string)
        
        
        
        # seventyfives.append(times[drop75])
    # else :
    #     seventyfives.append(np.nan)
        
        # if (val < 0.025) and plot :
        #     plt.plot_simple_multi_with_times([times, times, times], [SFH, lo, hi],
        #     ['SFH', r'$\pm 2 \sigma$', ''], ['k', 'grey', 'grey'], ['', '', ''],
        #     ['-', '-.', '-.'], [1, 1, 1], np.nan, tonset, tterm, drop_times, drop_labels,
        #     scale='linear', xmin=-0.1, xmax=13.8,
        #     xlabel=r'$t$ (Gyr)', ylabel=r'SFR (${\rm M}_{\odot}$ yr$^{-1}$)',
        #     outfile=outDir + 'SFH_subID_{}.png'.format(subIDfinal), save=False)
    
    #     plt.plot_simple_multi_with_times([times, times, times], [SFH, lo, hi],
    #     ['SFH', r'$\pm 2 \sigma$', ''], ['k', 'grey', 'grey'], ['', '', ''],
    #     ['-', '-.', '-.'], [1, 1, 1], np.nan, tonset, tterm, [times[i75], np.nan], drop_labels,
    #     scale='linear', xmin=-0.1, xmax=13.8,
    #     xlabel=r'$t$ (Gyr)', ylabel=r'SFR (${\rm M}_{\odot}$ yr$^{-1}$)',
    #     outfile='D:/Desktop/SFH_q_subID_{}.png'.format(subID), save=False)

# plt.plot_scatter_dumb(times[i75s[quenched]], seventyfives, 'k', '', 'o',
#     xlabel=r'$t_{\rm 75}$ (Gyr)', ylabel=r'$t_{\rm 0.3~max}$',
#     xmin=0, xmax=14, ymin=0, ymax=14)
