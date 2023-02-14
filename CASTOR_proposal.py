
import numpy as np

from astropy.table import Table
import astropy.units as u
import h5py
from PIL import Image

from core import (bsPath, determine_mass_bin_indices, get_mpb_radii_and_centers,
                  get_particles, get_particle_positions, get_sf_particles,
                  get_sf_particle_positions)
import plotting as plt

def basics_for_subID_14(simName, snapNum, ) :
    
    # define the input directory and the input file
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample_SFHs(t).hdf5'.format(simName, snapNum)
    
    # open basi infor for the qualifying satellite galaxies
    with h5py.File(infile, 'r') as hf :
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        SFHs = hf['SFH'][:]
        subIDs = hf['SubhaloID'][:]
        logM = hf['SubhaloMassStars'][:]
    
    # create a subsample of SFMS galaxies at z = 0 as a comparison for the quenched system
    SFMS = (SFHs[:, -1] > 0.001) # 6337 galaxies
    
    # use subID 14 at z = 0 as the quenched example for the CASTOR proposal
    tonset, tterm = 7.926577808275773, 9.05725531371842
    
    # using tterm isn't very illustrative
    # index_times = np.array([tonset, tonset + 0.5*(tterm - tonset), tterm])
    
    # use alternative times based on the final snapshot being 75% of the
    # quenching mechanism duration
    index_times = np.array([tonset,
                            tonset + 0.375*(tterm - tonset),
                            tonset + 0.75*(tterm - tonset)])
    indices = find_nearest(times, index_times)
    
    subID, mass, snaps = 14, 10.441437721252441, indices
    
    return subID, mass, snaps, redshifts, times, SFHs, subIDs, logM, SFMS

def determine_sigmadot(simName, snapNum, times, subID, snap, edges,
                       delta_t=100*u.Myr, nelson2021version=False) :
    
    # get basic MPB info for the galaxy
    (snapNums, mpb_subIDs, radii, centers) = get_mpb_radii_and_centers(
        simName, snapNum, subID)
    
    # open the corresponding cutouts and get their particles
    ages, masses, rs = get_particles(simName, snapNum, snap,
                                     int(mpb_subIDs[snap]), centers[snap])
    
    # mask all particles to within 5Re
    Re = radii[snap]
    rs = rs/Re
    ages, masses, rs = ages[rs <= 5], masses[rs <= 5], rs[rs <= 5]
    
    # find the total mass and area (in kpc^2) in each annulus
    mass_in_annuli, areas = [], []
    for start, end in zip(edges, edges[1:]) :
        mass = np.sum(masses[(rs > start) & (rs <= end)])
        mass_in_annuli.append(mass)
        
        area = np.pi*(np.square(end*Re) - np.square(start*Re))
        areas.append(area)
    mass_in_annuli, areas = np.array(mass_in_annuli), np.array(areas)
    
    # get the SF particles
    _, masses, rs = get_sf_particles(ages, masses, rs, times[snap],
                                     delta_t=delta_t)
    
    # find the SF mass in each annulus
    SF_mass_in_annuli = []
    for start, end in zip(edges, edges[1:]) :
        total_mass = np.sum(masses[(rs > start) & (rs <= end)])
        SF_mass_in_annuli.append(total_mass)
    SF_mass_in_annuli = np.array(SF_mass_in_annuli)
    
    delta_t = (delta_t.to(u.yr).value)
    
    if nelson2021version :
        # sSFR in each annulus
        final = SF_mass_in_annuli/delta_t/mass_in_annuli # units of yr^-1
    else :
        # SFR surface density in each annulus
        final = SF_mass_in_annuli/delta_t/areas # units of Mdot/yr/kpc^2
    
    return final

def find_nearest(times, index_times) :
    
    indices = []
    for time in index_times :
        index = (np.abs(times - time)).argmin()
        indices.append(index)
    
    return indices

def radial_plot(simName, snapNum, delta_t=100*u.Myr, nelson2021version=False,
                save=False) :
    
    (subID, mass, snaps, redshifts, times,
     SFHs, subIDs, logM, SFMS) = basics_for_subID_14(simName, snapNum)
    
    # define the center points of the radial bins
    edges = np.linspace(0, 5, 21)
    mids = []
    for start, end in zip(edges, edges[1:]) :
        mids.append(0.5*(start + end))
    
    # work through the onset, mid, and term snaps
    for snap in snaps :
        
        # calculate the SFR surface density for the quenched galaxy
        SFR_density = determine_sigmadot(simName, snapNum, times, subID, snap,
                                         edges, delta_t=delta_t,
                                         nelson2021version=nelson2021version)
        
        # now find galaxies of a similar mass in a small mass bin
        mass_bin = determine_mass_bin_indices(logM[SFMS], mass,
                                              halfwidth=0.10, minNum=50)
        
        # create an empty array that will hold all of the SFR surface densities
        SFR_densities = np.full((len(subIDs[SFMS][mass_bin]), 20), np.nan)
        
        # loop over all the comparison galaxies and populate into the array
        for i, ID in enumerate(subIDs[SFMS][mass_bin]) :
            SFR_densities[i, :] = determine_sigmadot(simName, snapNum, times,
                ID, snap, edges, delta_t=delta_t,
                nelson2021version=nelson2021version)
        
        lo, med, hi = np.nanpercentile(SFR_densities, [16, 50, 84], axis=0)
        
        if nelson2021version :
            ylabel = r'sSFR (yr$^{-1}$)'
            ymin, ymax = 1e-12, 1e-9
        else :
            ylabel = r'$\log(\dot{\Sigma}_{\rm *, 100~Myr}/{\rm M}_{\odot}~{\rm yr}^{-1}~{\rm kpc}^{-2})$'
            ymin, ymax = 1e-5, 1
        
        outfile = 'output/CASTOR_proposal/SFR-vs-r_subID_{}_snap_{}.png'.format(
            subID, snap)
        plt.plot_simple_with_band(mids, SFR_density, lo, med, hi,
            xlabel=r'$r/R_{\rm e}$',
            ylabel=ylabel,
            xmin=0, xmax=5, ymin=ymin, ymax=ymax, outfile=outfile, save=save)
    
    return

def spatial_plot(simName, snapNum, delta_t=100*u.Myr, nelson2021version=False,
                 fast=True, save=False) :
    
    (subID, mass, snaps, redshifts, times,
     SFHs, subIDs, logM, SFMS) = basics_for_subID_14(simName, snapNum)
    
    # get the mpb snapshot numbers, subIDs, stellar halfmassradii, and
    # galaxy centers
    snapNums, mpb_subIDs, radii, centers = get_mpb_radii_and_centers(
        simName, snapNum, subID)
    
    edges = np.linspace(-5, 5, 61)
    XX, YY = np.meshgrid(edges, edges)
    
    # work through the onset, mid, and term snaps
    for time, snap, mpbsubID, center, Re in zip(times[snaps], snaps,
        mpb_subIDs[snaps], centers[snaps], radii[snaps]) :
        
        # get all particles
        ages, masses, dx, dy, dz = get_particle_positions(simName,
            snapNum, snap, mpbsubID, center)
        
        # get the SF particles
        _, sf_masses, sf_dx, sf_dy, sf_dz = get_sf_particle_positions(
            ages, masses, dx, dy, dz, time, delta_t=delta_t)
        
        
        # create 2D histograms of the particles and SF particles
        
        hh, _, _ = np.histogram2d(dx/Re, dz/Re, bins=(edges, edges))
        hh = hh.T
        
        hh_sf, _, _ = np.histogram2d(sf_dx/Re, sf_dz/Re, bins=(edges, edges))
        hh_sf = hh_sf.T
        
        area = np.square((edges[1] - edges[0])*Re)
        
        if nelson2021version :
            hist = hh_sf/(delta_t.to(u.yr).value)/hh
            cbar_label = r'sSFR (yr$^{-1}$)'
        else :
            hist = hh_sf/(delta_t.to(u.yr).value)/area
            cbar_label = r'$\log(\dot{\Sigma}_{\rm *, 100~Myr}/{\rm M}_{\odot}~{\rm yr}^{-1}~{\rm kpc}^{-2})$'
        
        # define an array of random numbers to select ~1000 particles
        np.random.seed(0)
        rand = np.random.random(len(dx))
        
        table = Table([rand, dx/Re, dz/Re, masses/1e8],
                      names=('rand', 'dx', 'dz', 'masses'))
        if fast :
            table = table[table['rand'] < 0.0024]
        df = table.to_pandas()
        # df = None
        
        if snap == snaps[-1] :
            legend = True
            if df is not None :
                figwidth = 8.7
        else :
            figwidth = 7
            legend = False
        
        title = (r'$z = {:.2f}$'.format(redshifts[snap]) + ', ' +
                 r'$\Delta t_{\rm since~onset} = $' +
                 '{:.2f} Gyr'.format(time - times[snaps][0]))
        
        outfile = 'output/CASTOR_proposal/evolution_subID_{}_snap_{}.png'.format(
            subID, snap)
        plt.plot_scatter_CASTOR(dx/Re, dz/Re, sf_dx/Re, sf_dz/Re,
                                sf_masses/1e8, [2, 4], hist, XX, YY,
                                df=df,
                                legend=legend, cbar_label=cbar_label,
                                bins=[edges, edges], title=title,
                                xlabel=r'$\Delta x$ ($R_{\rm e}$)',
                                ylabel=r'$\Delta z$ ($R_{\rm e}$)',
                                xmin=-5, xmax=5, ymin=-5, ymax=5,
                                figsizewidth=figwidth,
                                save=save, outfile=outfile)
    
    return

def combine() :
    
    inDir = 'output/CASTOR_proposal/'
    paths = [inDir + 'evolution_subID_14_snap_63.png',
             inDir + 'evolution_subID_14_snap_65.png',
             inDir + 'evolution_subID_14_snap_68.png',
             inDir + 'SFR-vs-r_subID_14_snap_63.png',
             inDir + 'SFR-vs-r_subID_14_snap_65.png',
             inDir + 'SFR-vs-r_subID_14_snap_68.png']
    
    # open the data from those images
    images = [Image.open(image) for image in paths]
    
    # create the top row and bottom row
    top = concat_horiz(images[:3], x_offset=26, extra=21)
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

# radial_plot('TNG50-1', 99, nelson2021version=True)
# radial_plot('TNG50-1', 99)

# spatial_plot('TNG50-1', 99, nelson2021version=True, fast=False, save=True)
# spatial_plot('TNG50-1', 99)
