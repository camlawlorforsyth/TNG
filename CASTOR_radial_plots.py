
import numpy as np

import astropy.units as u
import h5py
from PIL import Image

from core import bsPath, get_mpb_radii_and_centers, get_particles, get_sf_particles
import plotting as plt

def combine() :
    
    inDir = 'output/CASTOR_proposal/'
    paths = [inDir + 'evolution_subID_14_snap_63.png',
             inDir + 'evolution_subID_14_snap_66.png',
             inDir + 'evolution_subID_14_snap_70.png',
             inDir + 'SFR-vs-r_subID_14_snap_63.png',
             inDir + 'SFR-vs-r_subID_14_snap_66.png',
             inDir + 'SFR-vs-r_subID_14_snap_70.png']
    
    # open the data from those images
    images = [Image.open(image) for image in paths]
    
    # create the top row and bottom row
    top = concat_horiz(images[:3], x_offset=7, extra=21)
    bottom = concat_horiz(images[3:])
    
    # save the final image
    final = concat_vert([top, bottom])
    final.save(inDir + 'CASTOR_example_subID_14.png')
    
    return

def concat_horiz(images, x_offset=0, extra=0) :
    
    # get the widths and heights to create an empty final image
    widths, heights = zip(*(i.size for i in images))
    final = Image.new('RGB', (np.sum(widths) + x_offset + extra*(len(images) - 1),
                              np.max(heights)),
                      color=(255, 255, 255))
    
    # populate the final image with the input images
    # x_offset = 0
    for im in images :
        final.paste(im, (x_offset, 0))
        x_offset += im.size[0] + extra
    
    return final

def concat_vert(images) :
    
    # get the widths and heights to create an empty final image
    widths, heights = zip(*(i.size for i in images))
    final = Image.new('RGB', (np.max(widths), np.sum(heights)),
                      color=(255, 255, 255))
    
    # populate the final image with the input images
    y_offset = 0
    for im in images :
        final.paste(im, (0, y_offset))
        y_offset += im.size[1]
    
    return final

def determine_mass_bin_indices(masses, mass, halfwidth=0.05, minNum=50) :
    
    mass_bin_mask = (masses >= mass - halfwidth) & (masses <= mass + halfwidth)
    
    if np.sum(mass_bin_mask) >= minNum :
        return mass_bin_mask
    else :
        return(determine_mass_bin_indices(masses, mass, halfwidth + 0.005))

def find_nearest(times, index_times) :
    
    indices = []
    for time in index_times :
        index = (np.abs(times - time)).argmin()
        indices.append(index)
    
    return indices

def show_outside_in_quenching_example(simName, snapNum, delta_t=100*u.Myr) :
    
    # define the input directory and the input file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    
    # testIDs satisfy:
    # 1) in old quenched sample so we have information about 
    #    a) tsat, b) tonset, c) tterm, and d) primary_confidence (all are confirmed satellites);
    # 2) in new quenched sample based on comparing SFHs of galaxies of similar mass,
    #    using a minimum of 50 comparison galaxies, in a mass bin halfwidth of initially 0.10 dex;
    # 3) have logM > 10^10 solMass, so that all necessary cutouts are already
    #    downloaded for comparison galaxies
    # THIS PRODUCES 6 POSSIBLE GALAXIES FOR THE CASTOR PROPOSAL
    testIDs = [5, 14, 96763,
               167398, 324126, 516101]
    testlogMs = [10.85793399810791, 10.441437721252441, 10.889863967895508,
                 10.367536544799805, 10.334203720092773, 10.68832778930664]
    testOnsetSnaps = [52, 63, 71, 57, 60, 59]
    testMidSnaps = [68, 66, 76, 65, 71, 71]
    testTermSnaps = [84, 70, 81, 73, 82, 82]
    
    # testOnsets = [6.188545890658511, 7.926577808275773, 9.21972994638489,
    #               6.992838764254372, 7.448463608541802, 7.30941773404718]
    # testTerms = [11.317308602268037, 9.05725531371842, 10.828399117909244,
    #              9.550536538544957, 11.010277545958493, 11.010277545958493]
    
    # transpose the snapshots of interest into one array for ease of use
    snapsets = np.transpose([testOnsetSnaps, testMidSnaps, testTermSnaps])
    
    # open basic infor for the qualifying satellite galaxies
    with h5py.File(infile, 'r') as hf :
        times = hf['times'][:]
        SFHs = hf['SFH'][:]
        subIDs = hf['SubhaloID'][:]
        masses = hf['SubhaloMassStars'][:]
        # tonsets = hf['onset_times'][:]
        # tterms = hf['termination_times'][:]
    
    # for tonset, tterm in zip(times[testOnsetSnaps], times[testTermSnaps]) :
    #     index_times = np.array([tonset, tonset + 0.5*(tterm - tonset), tterm])
    #     indices = find_nearest(times, index_times)
    
    # create a subsample of SFMS galaxies at z = 0 as a comparison for the quenched systems
    SFMS_at_z0_mask = (SFHs[:, -1] > 0.001) # 6337 galaxies
    SFMS_subIDs = subIDs[SFMS_at_z0_mask]
    SFMS_masses = masses[SFMS_at_z0_mask]
    
    # loop through the galaxies in the sample of quenched satellites
    for subID, mass, snapset in zip(testIDs[1:2], testlogMs[1:2], snapsets[1:2]) :
        for snap in snapset : # work through the onset, mid, and term snaps
            
            # define the center points of the radial bins
            edges = np.linspace(0, 5, 21)
            mids = []
            for start, end in zip(edges, edges[1:]) :
                mids.append(0.5*(start + end))
            
            # get the particles for the satellite galaxy of interest at the snapshot
            # of interest
            (snapNums_main_old, mpb_subIDs_main_old,
             radii_main_old, centers_main_old) = get_mpb_radii_and_centers(
                 simName, snapNum, subID)
            
            # pad the arrays to all have 100 entries
            snapNums_main = np.full(100, np.nan)
            snapNums_main[100-len(snapNums_main_old):] = snapNums_main_old
            
            mpb_subIDs_main = np.full(100, np.nan)
            mpb_subIDs_main[100-len(mpb_subIDs_main_old):] = mpb_subIDs_main_old
            
            radii_main = np.full(100, np.nan)
            radii_main[100-len(radii_main_old):] = radii_main_old
            
            centers_main = np.full((100, 3), np.nan)
            centers_main[100-len(centers_main_old):] = centers_main_old
            
            # define the effective radius at that snapshot
            Re_main = radii_main[snap]
            
            # open the corresponding cutouts and get their particles
            ages, masses, rs = get_particles(simName, snapNum, snap,
                                             int(mpb_subIDs_main[snap]),
                                             centers_main[snap])
            
            total_mass_in_bins = []
            # only proceed if the ages, masses, and distances are intact
            if (ages is not None) and (masses is not None) and (rs is not None) :
                # get the SF particles
                _, masses, rs = get_sf_particles(ages, masses, rs, times[snap],
                                                 delta_t=delta_t)
                
                # get SF particles within 5Re
                rs = rs/Re_main
                dist_mask = (rs <= 5)
                masses = masses[dist_mask]
                rs = rs[dist_mask]
                
                for start, end in zip(edges, edges[1:]) :
                    mass_in_bin = np.sum(masses[(rs > start) & (rs <= end)])
                    total_mass_in_bins.append(mass_in_bin)
            
            else :
                total_mass_in_bins.append(np.nan)
            
            SFRs_main = np.array(total_mass_in_bins)/1e8
            
            '''
            Now repeat the above for the comparison SFMS galaxies which are a similar mass
            '''
            
            # find galaxies of a similar mass in a small mass bin
            mass_bin = determine_mass_bin_indices(SFMS_masses, mass,
                                                  halfwidth=0.10, minNum=50)
            
            # create an empty array that will hold all of the SFRs at different radii
            SFR_vs_radius = np.full((len(SFMS_subIDs[mass_bin]), 20), np.nan)
            
            # for each galaxy in the mass bin, get basic information about the
            # MPB at the snapshots of interest
            for i, ID in enumerate(SFMS_subIDs[mass_bin]) :
                
                # get the mpb snapshot numbers, subIDs, stellar halfmassradii, and
                # galaxy centers
                (snapNums_old, mpb_subIDs_old,
                 radii_old, centers_old) = get_mpb_radii_and_centers(
                    simName, snapNum, ID)
                
                # pad the arrays to all have 100 entries
                snapNums = np.full(100, np.nan)
                snapNums[100-len(snapNums_old):] = snapNums_old
                
                mpb_subIDs = np.full(100, np.nan)
                mpb_subIDs[100-len(mpb_subIDs_old):] = mpb_subIDs_old
                
                radii = np.full(100, np.nan)
                radii[100-len(radii_old):] = radii_old
                
                centers = np.full((100, 3), np.nan)
                centers[100-len(centers_old):] = centers_old
                
                # define the effective radius at that snapshot
                Re = radii[snap]
                
                # open the corresponding cutouts and get their particles
                ages, masses, rs = get_particles(simName, snapNum, snap,
                                                 int(mpb_subIDs[snap]),
                                                 centers[snap])
                
                total_mass_in_bins = []
                # only proceed if the ages, masses, and distances are intact
                if (ages is not None) and (masses is not None) and (rs is not None) :
                    # get the SF particles
                    _, masses, rs = get_sf_particles(ages, masses, rs, times[snap],
                                                     delta_t=delta_t)
                    
                    # get SF particles within 2Re
                    rs = rs/Re
                    dist_mask = (rs <= 5)
                    masses = masses[dist_mask]
                    rs = rs[dist_mask]
                    
                    for start, end in zip(edges, edges[1:]) :
                        mass_in_bin = np.sum(masses[(rs > start) & (rs <= end)])
                        total_mass_in_bins.append(mass_in_bin)
                
                else :
                    total_mass_in_bins.append(np.nan)
                
                SFRs = np.array(total_mass_in_bins)/1e8
                SFR_vs_radius[i, :] = SFRs
            
            lo, med, hi = np.nanpercentile(SFR_vs_radius, [16, 50, 84], axis=0)
            
            if snap == snapset[-1] :
                legend = True
            else :
                legend = False
            
            outfile = 'output/CASTOR_proposal/SFR-vs-r_subID_{}_snap_{}.png'.format(
                subID, snap)
            plt.plot_simple_with_band(mids, SFRs_main, lo, med, hi, legend=legend,
                xlabel=r'$r/R_{\rm e}$', ylabel=r'SFR (M$_{\odot}$ yr$^{-1}$)',
                xmin=0, xmax=5, ymin=-0.05, ymax=1, outfile=outfile, save=True)
        
        # print('subID {} done'.format(subID))
    
    return
