
from os.path import exists
import numpy as np

import astropy.units as u
import h5py
from scipy.optimize import curve_fit

from core import add_dataset, bsPath, get, vertex
import plotting as plt

def determine_satellite_time_dynamical(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and file, and the file which contains the flags,
    # and the output file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    flags_file = inDir + '/{}_{}_primary_flags(t).hdf5'.format(simName, snapNum)
    outfile = inDir + '/{}_{}_satellite_times.hdf5'.format(simName, snapNum)
    
    # load necessary sample information
    with h5py.File(infile, 'r') as hf :
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]*u.Gyr
        quenched = hf['quenched'][:]
        tonsets = hf['onset_times'][:]
        logM = hf['logM'][:]
    
    # load all the flags
    with h5py.File(flags_file, 'r') as hf :
        flags = hf['flags'][:]
    
    # set the dynamical time at each snapshot
    t_dyn = (2*u.Gyr)*np.power(1 + redshifts, -1.5)
    
    # we require a satellite flag at three consecutive snapshots
    sss = [0, 0, 0]
    
    tsat_indices, tsats = [], []
    # loop over every galaxy in the sample
    for flag, use in zip(flags, quenched) :
        
        # only caclculate for the quenched galaxies
        if use :
            
            # limit the flags to times since redshift 2, as the early universe
            # was more chaotic
            flag = flag[33:]
            
            # determine if a galaxy has always been a primary, or has recently
            # gone back to being a primary -> 141 full primaries, 135 recent changes
            if np.all(flag == 1) or np.all(flag[-3:] == 1) :
                tsat_index, tsat = 99, times[99].value
            
            # if the galaxy hasn't always been a primary, use the dynamical
            # time at a given redshift/snapshot to determine when it was first
            # a satellite
            else :
                # find instances (ie. a subarray) of sss in the flag array
                length = len(flag) - len(sss) + 1
                indices = [x for x in range(length) if list(flag)[x:x+len(sss)] == sss]
                
                # indices returns the location of the start of the subarray
                indices = np.array(indices)
                
                # if there aren't three consecutive snapshots as a satellite,
                # do further checks to determine satellite times
                
                # these galaxies have also always been primaries
                if (len(indices) == 0) and (flag[-1] == 1) :
                    tsat_index, tsat = 99, times[99].value
                
                # these galaxies have recently become satellites
                if (len(indices) == 0) and (flag[-1] == 0) and (flag[-2] == 1):
                    tsat_index, tsat = 99, times[99].value
                
                # these galaxies have recently become satellites as well
                if (len(indices) == 0) and (flag[-1] == 0) and (flag[-2] == 0):
                    tsat_index, tsat = 98, times[98].value
                
                # if there are three consecutive satellite snapshots, compare
                # the dynamical time at the start of a given run of three to
                # the length of time that the galaxy has been a satellite
                # before becoming a primary again
                if len(indices) > 0 :
                    ratios = []
                    # loop over all the found indices and check each index
                    for index in indices :
                        
                        # get the dynamical time at the snapshot, but offset
                        t_dyn_at_snap = t_dyn[index + 33] # because we started
                        # at snapshot 33, corresponding to redshift 2
                        
                        # check if the subarray is followed by a primary flag
                        subflag = index + np.nonzero(flag[index:])[0] + 33
                        if len(subflag) > 0 :
                            end = subflag[0]
                        else :
                            end = 99
                        
                        # find the duration the galaxy was a satellite
                        time_diff = times[end] - times[index + 33] # see above
                        
                        # find the ratio of satellite duration to dynamical time
                        ratio = (time_diff/t_dyn_at_snap).value
                        ratios.append(ratio)
                    
                    # convert list to array
                    ratios = np.array(ratios)
                    
                    # we require the galaxy to be a satellite for at least one
                    # dynamical time in order for that snapshot to be considered
                    # when it actually became a satellite
                    sufficient = (ratios >= 1.0)
                    
                    # find the first location that fits the criteria
                    loc = np.where(sufficient > 0)[0]
                    if len(loc) > 0 :
                        loc = loc[0]
                        tsat_index = indices[loc] + 33
                    
                    # 27 galaxies don't have sufficiently long durations as
                    # satellites
                    else :
                        tsat_index = indices[0] + 33
                        
                        # a few galaxies have longer durations after the first
                        # episode of being a satellite
                        # if np.argmax(ratios) != 0 :
                            # print(flag)
                            # print(ratios)
                            # print()
                    
                    # set the time when the galaxy became a satellite
                    tsat = (times[tsat_index]).value
        else :
            tsat_index, tsat = np.nan, np.nan
        
        # append the found indices and times to the appropriate lists
        tsat_indices.append(tsat_index)
        tsats.append(tsat)
    
    # convert lists to arrays
    tsat_indices, tsats = np.array(tsat_indices), np.array(tsats)
    
    # investigate the situation for the 208 galaxies with very early satellite times
    # for flag, use, tsat_index, tsat in zip(flags, quenched, tsat_indices, tsats) :
        
        # only caclculate for the quenched galaxies
        # if use and (tsat < 3.3) :
        #     print(flag[int(tsat_index):])
        #     print()
    
    with h5py.File(outfile, 'w') as hf :
        add_dataset(hf, tsat_indices, 'tsat_indices')
        add_dataset(hf, tsats, 'tsats')
    
    return

def get_all_primary_flags(simName='TNG50-1', snapNum=99) :
    
    # define the input directory and file, and the output file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    outfile = inDir + '/{}_{}_primary_flags(t).hdf5'.format(simName, snapNum)
    
    if not exists(outfile) :
        with h5py.File(outfile, 'w') as hf :
            add_dataset(hf, np.full((8260, 100), np.nan), 'flags')
    
    with h5py.File(outfile, 'r') as hf :
        all_flags = hf['flags'][:]
    
    with h5py.File(infile, 'r') as hf :
        snaps = hf['snapshots'][:]
        subIDs = hf['subIDs'][:].astype(int)
        quenched = hf['quenched'][:]
    
    base = 'http://www.tng-project.org/api/{}/snapshots/'.format(simName)
    
    # loop over every galaxy in the sample, getting the subIDs along the mpb
    for i, (mpb_subIDs, use) in enumerate(zip(subIDs, quenched)) :
        
        # get the primary flags for the quenched sample if they don't exist
        if use and np.all(np.isnan(all_flags[i, :])) :
            flags = []
            
            # loop over every snapshot along the mpb
            for snap, subID in zip(snaps, mpb_subIDs) :
                
                # only get the primary flag if the subID isn't NaN
                if subID >= 0 :
                    url = base + '{}/subhalos/{}'.format(snap, subID)
                    sub = get(url)
                    flag = sub['primary_flag']
                else :
                    flag = np.nan
                flags.append(flag)
            
            # append the flags into the outfile
            with h5py.File(outfile, 'a') as hf :
                hf['flags'][i, :] = flags
    
    return

def save_time_comparison_plot() :
    
    textwidth = 7.10000594991006
    textheight = 9.095321710253218
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        # redshifts = hf['redshifts'][:]
        # times = hf['times'][:]*u.Gyr
        # subIDfinals = hf['SubhaloID'][:]
        quenched = hf['quenched'][:]
        tonsets = hf['onset_times'][:]
        logM = hf['logM'][:, -1]
        env = np.array([12*hf['cluster'][:], 10*hf['hm_group'][:],
                         8*hf['lm_group'][:], 6*hf['field'][:]]).T
        env = np.sum(env, axis=1)
    
    # with h5py.File('TNG50-1/TNG50-1_99_overdensity(t).hdf5', 'r') as hf :
    #     overdensity = hf['delta'][:, -1]
    
    with h5py.File('TNG50-1/TNG50-1_99_satellite_times.hdf5', 'r') as hf :
        tsats = hf['tsats'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_mechanism.hdf5', 'r') as hf :
        mech = np.array([1*hf['inside-out'][:], 3*hf['outside-in'][:],
                         5*hf['ambiguous'][:], 5*hf['uniform'][:]]).T
        mech = np.sum(mech, axis=1)
    
    # make a mask for the massive quenched sample
    mask = (logM >= 9.5) & quenched
    
    # apply the mask to the quantities of interest
    tonsets = tonsets[mask]
    tsats = tsats[mask]
    env = env[mask]
    mech = mech[mask]
    logM = logM[mask]
    
    # determine the size of the points for use in plotly
    mini, stretch = 10, 20 # define the minimum size and the maximum stretch
    diff = (np.max(logM) - np.min(logM))/2
    logM_fit_vals = np.array([np.min(logM), np.min(logM) + diff, np.max(logM)])
    size_fit_vals = np.array([1, np.sqrt(stretch), stretch])*mini
    # adapted from https://stackoverflow.com/questions/12208634
    popt, _ = curve_fit(lambda xx, aa: vertex(xx, aa, logM_fit_vals[0], mini),
                        logM_fit_vals, size_fit_vals) # fit the curve
    size = vertex(logM, popt[0], logM_fit_vals[0], mini) # get the size for the points
    
    # prepare values for plotting
    xs1 = [tsats[(mech == 1) & (env == 12)], tsats[(mech == 1) & (env == 10)],
           tsats[(mech == 1) & (env == 8)], tsats[(mech == 1) & (env == 6)]]
    ys1 = [tonsets[(mech == 1) & (env == 12)], tonsets[(mech == 1) & (env == 10)],
           tonsets[(mech == 1) & (env == 8)], tonsets[(mech == 1) & (env == 6)]]
    s1 = [size[(mech == 1) & (env == 12)], size[(mech == 1) & (env == 10)],
          size[(mech == 1) & (env == 8)], size[(mech == 1) & (env == 6)]]
    
    colors = ['r', 'gold', 'c', 'g']
    markers = ['o', 'o', 'o', 'o']
    alphas = [0.5, 0.5, 0.6, 0.3]
    xx = np.linspace(0, 14, 100)
    labels = ['cluster', 'high-mass group', 'low-mass group', 'field']
    xs2 = [tsats[(mech == 3) & (env == 12)], tsats[(mech == 3) & (env == 10)],
           tsats[(mech == 3) & (env == 8)], tsats[(mech == 3) & (env == 6)]]
    ys2 = [tonsets[(mech == 3) & (env == 12)], tonsets[(mech == 3) & (env == 10)],
           tonsets[(mech == 3) & (env == 8)], tonsets[(mech == 3) & (env == 6)]]
    s2 = [size[(mech == 3) & (env == 12)], size[(mech == 3) & (env == 10)],
          size[(mech == 3) & (env == 8)], size[(mech == 3) & (env == 6)]]
    
    plt.double_scatter_with_line(xs1, ys1, colors, markers, alphas, xx, xx,
        xs2, ys2, colors, markers, alphas, xx, xx, labels, s1=s1, s2=s2,
        xlabel1=r'$t_{\rm satellite}$ (Gyr)', ylabel1=r'$t_{\rm onset}$ (Gyr)',
        xlabel2=r'$t_{\rm satellite}$ (Gyr)', ylabel2='',
        titles=['inside-out', 'outside-in'],
        xmin1=1, xmax1=14, xmin2=1, xmax2=14, ymin=1, ymax=14,
        figsizewidth=textwidth, figsizeheight=textheight/2,
        save=False, outfile='time_comparison.pdf')
    
    return
