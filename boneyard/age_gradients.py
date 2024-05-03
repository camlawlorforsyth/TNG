
import numpy as np

from scipy.ndimage import gaussian_filter1d

from core import (get_ages_and_scalefactors, get_mpb_radii_and_centers,
                  get_particles)
import plotting as plt

def compute_age_gradient(ages, masses, rs, radius, snap) :
    
    if (len(rs) == 0) and (len(masses) == 0) and (len(ages) == 0) :
        gradient = np.nan
    else :
        
        # get scalefactors and age for interpolation later
        scalefactor, age = get_ages_and_scalefactors()
        
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
        
        # now determine the middle points in those shells, and the MWA gradient
        # within that shell
        centers, MWAs = [], []
        for first, second in zip(edges, edges[1:]) :
            
            # determine the middle points
            center = np.mean([first, second])
            centers.append(center)
            
            # limit the star particles to the given shell
            mask = (rs > first) & (rs <= second)
            
            # if there are star particles that are within that radius range
            if np.sum(mask) > 0 :
                
                # determine the mass-weighted average age in that shell
                ages_in_Gyr = np.interp(ages[mask], scalefactor, age)
                MWA = np.average(ages_in_Gyr, weights=masses[mask])
                MWAs.append(MWA)
            
            else :
                MWAs.append(np.nan)
        
        # mask out the nan values
        centers = np.array(centers)[~np.isnan(MWAs)]
        MWAs = np.array(MWAs)[~np.isnan(MWAs)]
        
        '''
        if snap % 10 == 0 :
            xlabel = r'$\log(r/R_e)$'
            ylabel = r'$\overline{\rm MWA}$ (Gyr)'
            plt.plot_scatter(centers, MWAs, 'k', '', 'o',
                             xmin=-1, xmax=0.5, xlabel=xlabel, ylabel=ylabel)
        '''
        
        # if we have sufficient data, then compute the gradient
        if (len(centers) > 1) and (len(MWAs) > 1) :
            gradient, intercept = np.polyfit(centers, MWAs, 1)
        else :
            gradient = np.nan
    
    return gradient

def determine_age_gradients(simName, snapNum, kernel=2, plot=False, save=False) :
    
    # define the output directory
    outDir = 'output/age_gradients(t)/'
    
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
        
        # mask the MPB to where the stellar halfmass radius exists
        mask = (radii > 0)
        snapNums = snapNums[mask]
        mpb_subIDs = mpb_subIDs[mask]
        radii = radii[mask]
        centers = centers[mask]
        
        # now get the star particle ages, masses, and distances at each
        # snapshot/time
        gradients = []
        for snap, subID, center, Re in zip(snapNums, mpb_subIDs, centers, radii) :
            
            # get all particles
            ages, masses, rs = get_particles(simName, snapNum, snap, subID,
                                             center)
            
            # only proceed if the ages, masses, and distances are intact
            if (ages is not None) and (masses is not None) and (rs is not None):
                
                # now compute the age gradient at each snapshot/time
                gradient = compute_age_gradient(ages, masses, rs, Re, snap)
                gradients.append(gradient)
            else :
                gradients.append(np.nan)
        
        if plot :
            # limit the time axis to valid snapshots
            ts = times[len(times)-len(gradients):]
            
            # mask out nan values so that savgol_filter works
            mask = ~np.isnan(gradients)
            gradients = np.array(gradients)[mask]
            ts = ts[mask]
            
            # smooth the function for plotting purposes
            smoothed = gaussian_filter1d(gradients, kernel)
            
            outfile = outDir + 'age_gradient_subID_{}.png'.format(subID)
            ylabel = r'$\nabla {\rm MWA}$'
            plt.plot_simple_multi_with_times([ts, ts], [gradients, smoothed],
                ['data', 'smoothed'], ['grey', 'k'], ['', ''], ['--', '-'], [0.5, 1],
                tsat, tonset, tterm, xlabel=r'$t$ (Gyr)', ylabel=ylabel,
                xmin=0, xmax=14, ymax=1.25*np.max(smoothed),
                scale='linear', save=save, outfile=outfile)
    
    return
