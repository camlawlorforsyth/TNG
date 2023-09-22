
from os import makedirs
import numpy as np

import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import astropy.units as u
import h5py
from scipy.optimize import curve_fit
from scipy.stats import truncnorm
import xml.etree.ElementTree as ET

from core import find_nearest
import plotting as plt
from projection import (calculate_MoI_tensor, radial_distances,
                        rotation_matrix_from_MoI_tensor)

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

def check_Ngas_particles() :
    
    # open requisite information about the sample
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        subIDfinals = hf['SubhaloID'][:]
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        Res = hf['Re'][:]
        centers = hf['centers'][:]
        quenched = hf['quenched'][:]
        ionsets = hf['onset_indices'][:]
        tonsets = hf['onset_times'][:]
        iterms = hf['termination_indices'][:]
        tterms = hf['termination_times'][:]
    
    # define a mask to select the quenched galaxies with sufficient solar mass
    mask = quenched & (logM[:, -1] >= 9.5)
    
    # mask relevant properties
    subIDfinals = subIDfinals[mask]
    subIDs = subIDs[mask]
    logM = logM[mask]
    Res = Res[mask]
    centers = centers[mask]
    ionsets = ionsets[mask]
    tonsets = tonsets[mask]
    iterms = iterms[mask]
    tterms = tterms[mask]
    
    # find the snapshot corresponding to roughly 75% of the way through the
    # quenching episode, and the redshift at that snapshot
    iseventys = np.array(find_nearest(times, tonsets + 0.75*(tterms - tonsets)))
    z_75 = redshifts[iseventys]
    
    # get stellar masses and sizes at those snapshots
    firstDim = np.arange(278)
    subs = subIDs[firstDim, iseventys]
    masses = logM[firstDim, iseventys]
    rads = Res[firstDim, iseventys]*u.kpc # ckpc/h
    cents = centers[firstDim, iseventys]
    
    dim = (10*rads)*cosmo.arcsec_per_kpc_comoving(z_75)/(0.05*u.arcsec/u.pix)
    dim = np.ceil(dim).astype(int)
    
    '''
    for subIDfinal, iseventy, subID, Re, center in zip(subIDfinals,
        iseventys, subs, rads, cents) :
        params = [subIDfinal, iseventy, subID, Re.value, center.tolist()]
        print('save_skirt_input' + str(tuple(params)))
    '''
    
    Nstars, Ngass = [], []
    Nstar_maskeds, Ngas_maskeds = [], []
    for snap, subID, Re, center in zip(iseventys, subs, rads.value, cents) :
        inDir = 'F:/TNG50-1/mpb_cutouts_099/'
        cutout_file = inDir + 'cutout_{}_{}.hdf5'.format(snap, subID)
        
        with h5py.File(cutout_file, 'r') as hf :
            star_coords = hf['PartType4/Coordinates'][:]
            star_rs = radial_distances(center, star_coords)
            
            # only one galaxy have no gas particles at all
            if 'PartType0' not in hf.keys() :
                gas_rs = []
            else :
                gas_coords = hf['PartType0/Coordinates'][:]
                gas_rs = radial_distances(center, gas_coords)
        
        Nstar = len(star_rs)
        Ngas = len(gas_rs)
        
        Nstar_masked = len(star_rs[star_rs <= 5*Re])
        
        if Ngas == 0 :
            Ngas_masked = 0
        else :
            Ngas_masked = len(gas_rs[gas_rs <= 5*Re])
        
        Nstars.append(Nstar)
        Ngass.append(Ngas)
        Nstar_maskeds.append(Nstar_masked)
        Ngas_maskeds.append(Ngas_masked) # an additional galaxy has no gas <= 5Re

    tt = Table([subIDfinals, logM[:, -1],
                subs, masses, iseventys, z_75, rads,
                Nstars, Nstar_maskeds, Ngass, Ngas_maskeds, dim],
               names=('subID', 'logM',
                      'subID_75', 'logM_75', 'snap_75', 'z_75', 'Re_75',
                      'Nstar', 'Nstar_5Re', 'Ngas', 'Ngas_5Re', 'dim'))
    # tt.write('SKIRT/Ngas_particles_with_dim.fits')
    
    # from pypdf import PdfWriter
    # merger = PdfWriter()
    # inDir = 'TNG50-1/figures/comprehensive_plots/'
    # for subID in subIDfinals[np.argsort(z_75)] :
    #     merger.append(inDir + 'subID_{}.pdf'.format(subID))
    # outfile = 'SKIRT/comprehensive_plots_by_z75.pdf'
    # merger.write(outfile)
    # merger.close()
    
    return

def determine_runtime_with_photons() :
    
    xs = np.log10([1e6, 3162278, 1e7, 31622777, 1e8, 316227767, 1e9, 1e10])
    ys = np.log10([62, 70, 78, 124, 255, 696, 2177, 19333])
    
    popt_quad, _ = curve_fit(parabola, xs, ys, p0=[0.15, -1.8, 7.1])
    popt_exp, _ = curve_fit(exponential, xs, ys, p0=[0.040, 0.44, 1.1])
    
    xlin = np.linspace(6, 10, 100)
    ylin_para = parabola(xlin, *popt_quad)
    ylin_exp = exponential(xlin, *popt_exp)
    
    plt.plot_simple_multi([xlin, xlin, xs], [ylin_para, ylin_exp, ys],
        [r'$f(x) = ax^2 + bx + c$', r'$f(x) = Ae^{Bx} + C$', 'data'],
        ['r', 'b', 'k'], ['', '', 'o'], ['-', '--', ''], [1, 0.3, 1],
        xlabel=r'$\log(N_{\rm photons})$',
        ylabel=r'$\log({\rm runtime}/{\rm s})$', scale='linear')
    
    return

def exponential(xx, AA, BB, CC) :
    return AA*np.exp(BB*xx) + CC

def make_rgb() :
    
    import matplotlib.pyplot as plt
    
    from astropy.io import fits
    from astropy.visualization import make_lupton_rgb
    
    inDir = 'SKIRT/subID_513105_1e10/'
    infile = 'TNG_v0.5_fastTest_subID_513105_sed_cube_obs_total.fits'
    
    filters = ['HST_F218W',   'CASTOR_UV',   'HST_F225W',   'HST_F275W',
               'HST_F336W',   'CASTOR_U',    'HST_F390W',   'HST_F435W',
               'CASTOR_G',    'HST_F475W',   'HST_F555W',   'HST_F606W',
               'ROMAN_F062',  'HST_F625W',   'JWST_F070W',  'HST_F775W',
               'HST_F814W',   'ROMAN_F087',  'JWST_F090W',  'HST_F850LP',
               'HST_F105W',   'ROMAN_F106',  'HST_F110W',   'JWST_F115W',
               'HST_F125W',   'ROMAN_F129',  'HST_F140W',   'ROMAN_F146',
               'JWST_F150W',  'HST_F160W',   'ROMAN_F158',  'ROMAN_F184',
               'JWST_F200W',  'ROMAN_F213',  'JWST_F277W',  'JWST_F356W',
               'JWST_F410M',  'JWST_F444W'] # 'JWST_F560W',  'JWST_F770W',
               # 'JWST_F1000W', 'JWST_F1130W', 'JWST_F1280W', 'JWST_F1500W',
               # 'JWST_F1800W', 'JWST_F2100W', 'JWST_F2550W']
    
    # with h5py.File('SKIRT/SKIRT_cube_filters_and_waves_new.hdf5', 'w') as hf :
    #     add_dataset(hf, np.arange(47), 'index')
    #     add_dataset(hf, filters, 'filters', dtype=str)
    #     add_dataset(hf, waves, 'waves')
    
    with fits.open(inDir + infile) as hdu :
        # hdr = hdu[0].header
        data = hdu[0].data*u.MJy/u.sr # 38, 491, 491
        dim = data.shape
        # waves_hdr = hdu[1].header
        # waves = np.array(hdu[1].data.astype(float)) # effective wavelengths, in um
    
    # Re = 2.525036573410034*u.kpc
    # size = 10*Re*cosmo.arcsec_per_kpc_comoving(0.0485236299818059)/pixel_scale
    
    # define the area of a pixel, based on CASTOR resolution after dithering
    # pixel_scale = 0.05*u.arcsec/u.pix
    area = 0.0025*u.arcsec**2
    
    image_r = np.full(dim[1:], 0.0)
    for frame in data[20:] :
        image_r += (frame*area).to(u.Jy).value
    image_r = image_r/18
    
    image_g = np.full(dim[1:], 0.0)
    for frame in data[7:20] :
        image_g += (frame*area).to(u.Jy).value
    image_g = image_g/13
    
    image_b = np.full(dim[1:], 0.0)
    for frame in data[:7] :
        image_b += (frame*area).to(u.Jy).value
    image_b = image_b/7
    
    # image = make_lupton_rgb(image_r, image_g, image_b, Q=10, stretch=0.5)
    # plt.imshow(image)
    
    # for frame in data :
    #     m_AB = -2.5*np.log10(np.sum(frame*area).to(u.Jy)/(3631*u.Jy))
    
    return

def parabola(xx, aa, bb, cc) :
    return aa*np.square(xx) + bb*xx + cc

def save_skirt_input(subIDfinal, snap, subID, Re, center, gas_setup='voronoi',
                     star_setup='mappings', save_gas=True, save_stars=True,
                     save_ski=True, faceon_projection=False) :
    
    infile = 'F:/TNG50-1/mpb_cutouts_099/cutout_{}_{}.hdf5'.format(snap, subID)
    outDir = 'SKIRT/SKIRT_input_quenched/{}'.format(subIDfinal)
    outfile_gas = outDir + '/gas.txt'
    outfile_stars = outDir + '/stars.txt'
    outfile_oldstars = outDir + '/oldstars.txt'
    outfile_youngstars = outDir + '/youngstars.txt'
    outfile_ski = outDir + '/{}.ski'.format(subIDfinal)
    
    # create the output directory if it doesn't exist
    makedirs(outDir, exist_ok=True)
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        redshift = hf['redshifts'][snap]
    
    with h5py.File(infile, 'r') as hf :
        gas_coords = hf['PartType0/Coordinates'][:]
        Mgas = hf['PartType0/Masses'][:]*1e10/cosmo.h # in units of solMass
        Zgas = hf['PartType0/GFM_Metallicity'][:]
        gas_sfrs = hf['PartType0/StarFormationRate'][:]
        uu = hf['PartType0/InternalEnergy'][:]
        x_e = hf['PartType0/ElectronAbundance'][:]
        rho_gas = hf['PartType0/Density'][:]
        
        star_coords = hf['PartType4/Coordinates'][:]
        stellarHsml = hf['PartType4/StellarHsml'][:]
        Mstar = hf['PartType4/GFM_InitialMass'][:]*1e10/cosmo.h # solMass
        Zstar = hf['PartType4/GFM_Metallicity'][:]
        
        # formation times in units of scalefactor
        formation_scalefactors = hf['PartType4/GFM_StellarFormationTime'][:]
    
    # formation times in units of age of the universe (ie. cosmic time)
    formation_times = cosmo.age(1/formation_scalefactors - 1).value
    
    # calculate the rotation matrix to project the galaxy face on
    if faceon_projection :
        rot = rotation_matrix_from_MoI_tensor(calculate_MoI_tensor(
            Mgas, gas_sfrs, gas_coords, formation_times, Mstar, star_coords,
            Re, center))
        g_dx, g_dy, g_dz = np.matmul(np.asarray(rot['face-on']),
                                     (gas_coords-center).T)
        s_dx, s_dy, s_dz = np.matmul(np.asarray(rot['face-on']),
                                     (star_coords-center).T)
    else :
        # don't project the galaxy face-on
        g_dx, g_dy, g_dz = (gas_coords - center).T
        s_dx, s_dy, s_dz = (star_coords - center).T
    
    if save_gas : # save the input for the gas particles
        
        # adapted from https://www.tng-project.org/data/docs/faq/#gen6
        mu = 4/(1 + 3*0.76 + 4*0.76*x_e)*c.m_p.value # mean molecular weight
        k_B = c.k_B.to(u.erg/u.K).value # Boltzmann constant in cgs
        temp = (5/3 - 1)*uu/k_B*1e10*mu # temperature in Kelvin
        mask = (np.log10(temp) <= 6 + 0.25*np.log10(rho_gas)) # only cool gas
        
        if gas_setup == 'particle' :
            # find the distance to the 32nd other gas particle, for smoothing
            gasHsml = np.full(len(Mgas), np.nan)
            for i, coord in enumerate(gas_coords) :
                if i % 1000 == 0.0 :
                    print(i, len(gas_coords))
                gasHsml[i] = np.sort(np.sqrt(np.sum(
                    np.square(gas_coords - coord), axis=1)))[32]
            
            # https://skirt.ugent.be/skirt9/class_particle_medium.html
            g_hdr = ('subID {}\n'.format(subIDfinal) +
                     'Column 1: x-coordinate (kpc)\n' +
                     'Column 2: y-coordinate (kpc)\n' +
                     'Column 3: z-coordinate (kpc)\n' +
                     'Column 4: smoothing length (kpc)\n' +
                     'Column 5: gas mass (Msun)\n' +
                     'Column 6: metallicity (1)\n')
            gas = np.array([g_dx, g_dy, g_dz, gasHsml, Mgas, Zgas]).T
            
            # save the output to disk
            np.savetxt(outfile_gas, gas[mask], delimiter=' ', header=g_hdr)
        
        if gas_setup == 'voronoi' :
            # https://skirt.ugent.be/skirt9/class_voronoi_mesh_medium.html
            g_hdr = ('subID {}\n'.format(subIDfinal) +
                     'Column 1: x-coordinate (kpc)\n' +
                     'Column 2: y-coordinate (kpc)\n' +
                     'Column 3: z-coordinate (kpc)\n' +
                     'Column 4: gas mass (Msun)\n' +
                     'Column 5: metallicity (1)\n')
            gas = np.array([g_dx, g_dy, g_dz, Mgas, Zgas]).T
            
            # save the output to disk
            np.savetxt(outfile_gas, gas[mask], delimiter=' ', header=g_hdr)
    
    if save_stars : # save the input for the star particles
        
        # limit star particles to those that have positive formation times
        mask = (formation_scalefactors > 0)
        star_coords = star_coords[mask]
        stellarHsml = stellarHsml[mask]
        Mstar = Mstar[mask]
        Zstar = Zstar[mask]
        formation_times = formation_times[mask]
        s_dx = s_dx[mask]
        s_dy = s_dy[mask]
        s_dz = s_dz[mask]
        
        # convert the formation times to actual ages at the time of observation,
        # while also imposing a lower age limit of 1 Myr
        ages = cosmo.age(redshift).value - formation_times
        ages[ages < 0.001] = 0.001
        
        # https://skirt.ugent.be/skirt9/class_bruzual_charlot_s_e_d_family.html
        s_hdr = ('subID {}\n'.format(subIDfinal) +
                 'Column 1: x-coordinate (kpc)\n' +
                 'Column 2: y-coordinate (kpc)\n' +
                 'Column 3: z-coordinate (kpc)\n' +
                 'Column 4: smoothing length (kpc)\n' +
                 'Column 5: initial mass (Msun)\n' +
                 'Column 6: metallicity (1)\n' +
                 'Column 7: age (Gyr)\n')
        stars = np.array([s_dx, s_dy, s_dz, stellarHsml, Mstar, Zstar, ages]).T
        
        if star_setup == 'bc03' :
            # for all stellar populations, use the default Bruzual & Charlot
            # (2003) SEDs for simple stellar populations, with a Chabrier IMF,
            # and save the output to disk
            np.savetxt(outfile_stars, stars, delimiter=' ', header=s_hdr)
        
        if star_setup == 'mappings' :
            # make a mask for the young and old star particles
            oldmask = (ages > 0.01) # star particles older than 10 Myr
            youngmask = (ages <= 0.01) # star particles younger than 10 Myr
            length = np.sum(youngmask)
            
            # for old stellar populations, use the default Bruzual & Charlot
            # (2003) SEDs for simple stellar populations, with a Chabrier IMF,
            # and save the output to disk
            np.savetxt(outfile_oldstars, stars[oldmask], delimiter=' ',
                       header=s_hdr)
            
            # for young stellar populations, use the MAPPINGS-III library
            # https://skirt.ugent.be/skirt9/class_mappings_s_e_d_family.html
            ys_hdr = ('subID {}\n'.format(subIDfinal) +
                      'Column 1: x-coordinate (kpc)\n' +
                      'Column 2: y-coordinate (kpc)\n' +
                      'Column 3: z-coordinate (kpc)\n' +
                      'Column 4: smoothing length (kpc)\n' +
                      'Column 5: SFR (Msun/yr)\n' +
                      'Column 6: metallicity (1)\n' +
                      'Column 7: compactness (1)\n' +
                      'Column 8: pressure (Pa)\n' +
                      'Column 9: PDR fraction (1)\n')
            
            # define the SFR, compactness, ISM pressure, and PDR covering
            # factor, following a similar prescription as Trcka et al. (2022),
            # but use a truncated normal (ie. clipped Gaussian) for the age
            # distribution when calculating the PDR covering fraction, as we
            # want to maintain the lower age limit of 1 Myr from above, and an
            # upper limit of 10 Myr
            massrate = Mstar[youngmask]/1e7 # averaged over 10 Myr
            metallicity = Zstar[youngmask]
            compactness = np.random.normal(5*np.ones(length), 0.4)
            pressure = (1e5*c.k_B*u.K*np.power(u.cm, -3)).to(u.Pa)*np.ones(length)
            
            ages = ages[youngmask]
            aa, bb = (0.001 - ages)/0.0002, (0.01 - ages)/0.0002
            ages = truncnorm.rvs(aa, bb, loc=ages, scale=0.0002)
            fpdr = np.exp(-ages/0.003)
            if length == 1 :
                fpdr = [fpdr]
            
            youngstars = np.array([s_dx[youngmask], s_dy[youngmask],
                s_dz[youngmask], stellarHsml[youngmask], massrate,
                metallicity, compactness, pressure, fpdr]).T
            
            # save the output to disk
            np.savetxt(outfile_youngstars, youngstars, delimiter=' ',
                       header=ys_hdr)
    
    if save_ski : # save the SKIRT configuration file for processing
        
        # select the SKIRT configuration template file, based on the setup
        # for gas and star particles
        if (gas_setup == 'particle') and (star_setup == 'bc03') :
            template_file = 'SKIRT/TNG_v0.7_particle_BC03.ski'
        if (gas_setup == 'particle') and (star_setup == 'mappings') :
            template_file = 'SKIRT/TNG_v0.8_particle_MAPPINGS.ski'
        if (gas_setup == 'voronoi') and (star_setup == 'bc03') :
            template_file = 'SKIRT/TNG_v0.9_voronoi_BC03.ski'
        if (gas_setup == 'voronoi') and (star_setup == 'mappings') :
            template_file = 'SKIRT/TNG_v1.0_voronoi_MAPPINGS.ski'
        
        # define basic properties of the SKIRT run
        numPackets = int(1e8) # number of photon packets
        model_redshift = 0.5 # the redshift of the model run
        distance = 0*u.Mpc
        fdust = 0.2 # dust fraction
        minX, maxX = -10*Re.to(u.pc), 10*Re.to(u.pc) # extent of model space
        
        # define the FoV, and number of pixels for the redshift of interest
        plate_scale = 0.05*u.arcsec/u.pix
        fov = 20*Re
        nPix_raw = fov*cosmo.arcsec_per_kpc_proper(model_redshift)/plate_scale
        nPix = np.ceil(nPix_raw).astype(int).value
        
        # parse the template ski file
        ski = ET.parse(template_file)
        root = ski.getroot()
        
        # access attributes of the configuration and set basic properties
        sim = root.findall('MonteCarloSimulation')[0]
        sim.set('numPackets', str(numPackets))
        sim.findall('cosmology')[0].findall('FlatUniverseCosmology')[0].set(
            'redshift', str(model_redshift))
        
        # update medium attributes
        medium = sim.findall('mediumSystem')[0].findall('MediumSystem')[0]
        if gas_setup == 'particle' :
            dust = medium.findall('media')[0].findall('ParticleMedium')[0]
        if gas_setup == 'voronoi' :
            dust = medium.findall('media')[0].findall('VoronoiMeshMedium')[0]
            dust.set('minX', str(minX))
            dust.set('maxX', str(maxX))
            dust.set('minY', str(minX))
            dust.set('maxY', str(maxX))
            dust.set('minZ', str(minX))
            dust.set('maxZ', str(maxX))
        dust.set('massFraction', str(fdust))
        grid = medium.findall('grid')[0].findall('PolicyTreeSpatialGrid')[0]
        grid.set('minX', str(minX))
        grid.set('maxX', str(maxX))
        grid.set('minY', str(minX))
        grid.set('maxY', str(maxX))
        grid.set('minZ', str(minX))
        grid.set('maxZ', str(maxX))
        
        # update instrument attributes
        instruments = sim.findall('instrumentSystem')[0].findall(
            'InstrumentSystem')[0].findall('instruments')[0]
        for instrument in instruments :
            instrument.set('fieldOfViewX', str(fov))
            instrument.set('fieldOfViewY', str(fov))
            instrument.set('distance', str(distance))
            instrument.set('numPixelsX', str(nPix))
            instrument.set('numPixelsY', str(nPix))
        
        # write the configuration file to disk
        ski.write(outfile_ski, encoding='UTF-8', xml_declaration=True)
    
    print('{} done'.format(subIDfinal))
    
    return

# save_skirt_input(1, 38, 74114, 6.919701099395752*u.kpc, [5434.26318359375, 23616.751953125, 17887.64453125])
# save_skirt_input(2, 83, 351154, 4.704523086547852*u.kpc, [6739.55224609375, 23354.779296875, 19600.5859375])
# save_skirt_input(4, 55, 29285, 2.8483943939208984*u.kpc, [6491.39208984375, 25355.431640625, 22688.06640625])
# save_skirt_input(7, 50, 6, 3.7366392612457275*u.kpc, [8613.125, 23953.779296875, 21985.056640625])
# save_skirt_input(10, 84, 10, 4.965497970581055*u.kpc, [7026.8642578125, 24088.828125, 20683.41015625])
# save_skirt_input(13, 88, 14, 1.1463567018508911*u.kpc, [7166.50048828125, 24465.82421875, 21832.005859375])
# save_skirt_input(14, 69, 15, 2.4659676551818848*u.kpc, [8226.775390625, 24374.388671875, 21306.654296875])
# save_skirt_input(16, 63, 6, 2.527513265609741*u.kpc, [6941.310546875, 24125.052734375, 20470.5625])
# save_skirt_input(21, 52, 6, 5.278622150421143*u.kpc, [7838.95166015625, 24018.4453125, 21521.798828125])
# save_skirt_input(22, 24, 104053, 0.521422266960144*u.kpc, [10294.9052734375, 25877.40625, 25272.794921875])
# save_skirt_input(23, 68, 507406, 0.9742310643196106*u.kpc, [6907.7900390625, 25841.369140625, 24496.259765625])
# save_skirt_input(26, 62, 14, 2.598369598388672*u.kpc, [6863.08984375, 24267.4140625, 20918.943359375])
# save_skirt_input(28, 73, 561097, 1.2564334869384766*u.kpc, [8757.783203125, 23379.0703125, 20126.90234375])
# save_skirt_input(30, 81, 23, 2.138542413711548*u.kpc, [7258.22314453125, 24850.580078125, 22439.880859375])
# save_skirt_input(32, 50, 25246, 3.7992358207702637*u.kpc, [6240.41162109375, 25877.404296875, 22768.529296875])
# save_skirt_input(33, 67, 31, 3.721682548522949*u.kpc, [7062.49755859375, 24660.509765625, 21841.9375])
# save_skirt_input(34, 64, 17, 1.819235920906067*u.kpc, [6604.55712890625, 23643.322265625, 19719.943359375])
# save_skirt_input(35, 36, 8866, 3.591322660446167*u.kpc, [6849.68359375, 23852.560546875, 20356.9609375])
# save_skirt_input(36, 35, 8273, 1.5611287355422974*u.kpc, [7180.2900390625, 23525.1328125, 21057.548828125])
# save_skirt_input(38, 51, 25784, 3.0328338146209717*u.kpc, [6831.00341796875, 25378.078125, 23074.314453125])
# save_skirt_input(39, 72, 43, 1.818422555923462*u.kpc, [7175.0712890625, 24243.140625, 20891.255859375])
# save_skirt_input(40, 52, 14, 4.151195526123047*u.kpc, [6931.50341796875, 24411.431640625, 20427.673828125])
# save_skirt_input(41, 34, 5, 4.313936710357666*u.kpc, [6850.41748046875, 25669.892578125, 24534.396484375])
# save_skirt_input(42, 41, 15486, 1.2100837230682373*u.kpc, [7364.970703125, 24079.248046875, 20682.82421875])
# save_skirt_input(46, 73, 44, 2.460824728012085*u.kpc, [8163.775390625, 24273.59765625, 21427.751953125])
# save_skirt_input(47, 46, 6, 2.828500270843506*u.kpc, [6061.8525390625, 25990.482421875, 23215.443359375])
# save_skirt_input(50, 44, 15631, 5.784523963928223*u.kpc, [7215.57958984375, 23500.9765625, 20504.27734375])
# save_skirt_input(51, 68, 51, 3.93544340133667*u.kpc, [6761.806640625, 24407.7734375, 21684.45703125])
# save_skirt_input(58, 53, 26924, 4.255753993988037*u.kpc, [7108.6123046875, 25718.61328125, 23497.5])
# save_skirt_input(59, 45, 15424, 1.6405946016311646*u.kpc, [7222.03466796875, 24102.69921875, 20806.97265625])
# save_skirt_input(61, 65, 38316, 2.7552266120910645*u.kpc, [7358.517578125, 24734.5390625, 22527.283203125])
# save_skirt_input(64, 39, 97269, 3.1793203353881836*u.kpc, [6801.228515625, 23706.103515625, 19918.783203125])
# save_skirt_input(70, 38, 16, 1.700038194656372*u.kpc, [4727.572265625, 26248.708984375, 23313.3046875])
# save_skirt_input(74, 46, 13, 4.594496726989746*u.kpc, [6724.56982421875, 25512.4765625, 23598.291015625])
# save_skirt_input(79, 44, 83312, 6.019397735595703*u.kpc, [8540.1455078125, 24261.1171875, 22080.0859375])
# save_skirt_input(87, 39, 14, 4.017769813537598*u.kpc, [6113.62255859375, 25707.59765625, 24342.890625])
# save_skirt_input(92, 58, 33, 3.261753797531128*u.kpc, [7298.26806640625, 24560.673828125, 20954.353515625])
# save_skirt_input(94, 45, 15433, 2.5642805099487305*u.kpc, [6899.04052734375, 24458.458984375, 20841.50390625])
# save_skirt_input(98, 80, 55, 4.225803375244141*u.kpc, [7218.20654296875, 24063.216796875, 21064.794921875])
# save_skirt_input(124, 33, 10, 3.5367484092712402*u.kpc, [6466.31005859375, 25650.814453125, 24474.20703125])
# save_skirt_input(63871, 44, 327188, 1.007636547088623*u.kpc, [25952.037109375, 13654.326171875, 1629.3223876953125])
# save_skirt_input(63874, 68, 53740, 4.365604400634766*u.kpc, [23337.638671875, 15940.5439453125, 3432.876953125])
# save_skirt_input(63875, 34, 177281, 0.8749130964279175*u.kpc, [24296.3984375, 14042.9169921875, 4825.59375])
# save_skirt_input(63879, 80, 56446, 2.1986336708068848*u.kpc, [23664.947265625, 14457.259765625, 2870.424560546875])
# save_skirt_input(63880, 45, 31981, 2.105402708053589*u.kpc, [22917.166015625, 15839.3232421875, 3562.930908203125])
# save_skirt_input(63883, 68, 53751, 5.471149444580078*u.kpc, [23867.849609375, 14911.1142578125, 3029.632568359375])
# save_skirt_input(63885, 62, 50227, 3.190803050994873*u.kpc, [23526.720703125, 15256.330078125, 3269.130126953125])
# save_skirt_input(63886, 55, 46177, 6.470934867858887*u.kpc, [22915.181640625, 15977.6220703125, 3170.62744140625])
# save_skirt_input(63887, 93, 69078, 3.743358612060547*u.kpc, [23740.83203125, 15729.0703125, 3349.712158203125])
# save_skirt_input(63891, 50, 55638, 6.846566677093506*u.kpc, [22997.552734375, 15962.7880859375, 3272.555419921875])
# save_skirt_input(63893, 59, 48208, 3.8159537315368652*u.kpc, [22633.21484375, 15311.412109375, 3069.39892578125])
# save_skirt_input(63898, 96, 67381, 3.7795445919036865*u.kpc, [23545.0, 15016.5751953125, 3232.046142578125])
# save_skirt_input(63900, 60, 50355, 3.200077772140503*u.kpc, [23116.423828125, 15905.71875, 3192.859375])
# save_skirt_input(63901, 44, 31231, 1.7348716259002686*u.kpc, [22625.751953125, 15625.318359375, 3623.9384765625])
# save_skirt_input(63907, 47, 49839, 3.958181858062744*u.kpc, [22431.125, 15523.3115234375, 3161.415283203125])
# save_skirt_input(63910, 73, 52754, 3.3400063514709473*u.kpc, [23701.01171875, 14724.029296875, 3478.9404296875])
# save_skirt_input(63911, 67, 54068, 1.471694827079773*u.kpc, [23443.365234375, 14608.3271484375, 3058.72998046875])
# save_skirt_input(63917, 61, 49305, 3.9083659648895264*u.kpc, [23021.154296875, 15974.404296875, 3149.738037109375])
# save_skirt_input(63926, 38, 25073, 2.9337918758392334*u.kpc, [22486.287109375, 15150.759765625, 3092.587646484375])
# save_skirt_input(63928, 39, 25499, 4.598609447479248*u.kpc, [22284.310546875, 15495.310546875, 3662.435791015625])
# save_skirt_input(96763, 78, 423165, 3.690382242202759*u.kpc, [25902.939453125, 6506.41552734375, 2705.205078125])
# save_skirt_input(96764, 54, 273113, 3.7120327949523926*u.kpc, [27551.203125, 6635.87744140625, 1826.72900390625])
# save_skirt_input(96766, 69, 76902, 2.992084264755249*u.kpc, [27683.21875, 7614.2294921875, 4826.6171875])
# save_skirt_input(96767, 56, 66360, 2.0710253715515137*u.kpc, [28517.646484375, 7987.79150390625, 4003.575927734375])
# save_skirt_input(96769, 58, 67975, 1.8564605712890625*u.kpc, [27748.685546875, 6979.9765625, 3909.672119140625])
# save_skirt_input(96771, 44, 42759, 0.825415849685669*u.kpc, [28307.04296875, 7637.23876953125, 4297.02587890625])
# save_skirt_input(96772, 39, 273114, 1.3693691492080688*u.kpc, [27702.712890625, 6569.91015625, 5414.4169921875])
# save_skirt_input(96778, 85, 88214, 4.2367753982543945*u.kpc, [27741.09375, 6843.826171875, 3893.85986328125])
# save_skirt_input(96779, 61, 71742, 1.7090530395507812*u.kpc, [27482.357421875, 7386.04150390625, 4732.724609375])
# save_skirt_input(96780, 40, 36243, 1.159963846206665*u.kpc, [27918.42578125, 7139.97900390625, 4793.330078125])
# save_skirt_input(96781, 62, 72681, 2.995453119277954*u.kpc, [27930.240234375, 7844.5576171875, 3909.1494140625])
# save_skirt_input(96782, 62, 72683, 1.4666552543640137*u.kpc, [28157.080078125, 7649.35009765625, 3617.690185546875])
# save_skirt_input(96783, 62, 72675, 2.021986484527588*u.kpc, [27487.05859375, 7351.630859375, 4787.92626953125])
# save_skirt_input(96785, 52, 59781, 5.436735153198242*u.kpc, [27986.572265625, 7716.00927734375, 5105.67822265625])
# save_skirt_input(96791, 51, 41516, 7.395904541015625*u.kpc, [28195.703125, 7551.505859375, 3693.720947265625])
# save_skirt_input(96793, 97, 98726, 1.257693886756897*u.kpc, [27210.765625, 8053.125, 4197.70263671875])
# save_skirt_input(96795, 89, 99765, 2.336646556854248*u.kpc, [27366.3203125, 6995.21728515625, 3635.391357421875])
# save_skirt_input(96798, 52, 59786, 3.826808452606201*u.kpc, [28245.962890625, 7354.04931640625, 3528.866455078125])
# save_skirt_input(96800, 43, 42070, 2.931291341781616*u.kpc, [28411.380859375, 7912.8818359375, 4465.5458984375])
# save_skirt_input(96801, 61, 71741, 5.5939130783081055*u.kpc, [27578.640625, 7810.890625, 4677.9677734375])
# save_skirt_input(96804, 47, 37253, 1.1913484334945679*u.kpc, [27857.306640625, 7189.21630859375, 4691.7158203125])
# save_skirt_input(96805, 67, 77300, 4.619329452514648*u.kpc, [27595.748046875, 6975.54296875, 3679.213623046875])
# save_skirt_input(96806, 57, 67694, 3.8795738220214844*u.kpc, [28288.744140625, 7664.14697265625, 3744.3359375])
# save_skirt_input(96808, 44, 42768, 3.898806095123291*u.kpc, [27841.138671875, 7372.93359375, 4635.9697265625])
# save_skirt_input(117261, 63, 89870, 1.241865634918213*u.kpc, [16146.3427734375, 28838.658203125, 25621.716796875])
# save_skirt_input(117271, 79, 99400, 1.569605827331543*u.kpc, [16246.2607421875, 29218.20703125, 25858.818359375])
# save_skirt_input(117274, 62, 89105, 5.610778331756592*u.kpc, [16229.2763671875, 29225.40234375, 25847.056640625])
# save_skirt_input(117275, 69, 92026, 2.6896326541900635*u.kpc, [16409.80078125, 29328.6328125, 25754.064453125])
# save_skirt_input(117277, 80, 99814, 1.5543626546859741*u.kpc, [15826.2236328125, 29111.87890625, 25662.90625])
# save_skirt_input(117284, 66, 92481, 1.9755234718322754*u.kpc, [15998.5478515625, 29330.921875, 26064.099609375])
# save_skirt_input(117292, 90, 118204, 3.547374963760376*u.kpc, [15593.3232421875, 29337.71484375, 26313.587890625])
# save_skirt_input(117296, 57, 93037, 1.689876675605774*u.kpc, [16276.28125, 29084.431640625, 25652.70703125])
# save_skirt_input(117302, 52, 91282, 1.6807701587677002*u.kpc, [16128.8134765625, 29046.716796875, 25762.298828125])
# save_skirt_input(117306, 68, 92281, 3.4270215034484863*u.kpc, [15794.240234375, 29001.4296875, 25658.478515625])
# save_skirt_input(143895, 59, 241195, 2.1625211238861084*u.kpc, [23212.931640625, 5829.36083984375, 32287.716796875])
# save_skirt_input(143904, 59, 219804, 3.1096558570861816*u.kpc, [22059.375, 5862.70068359375, 30750.833984375])
# save_skirt_input(143907, 70, 118289, 2.386402130126953*u.kpc, [20631.26953125, 5513.98046875, 30191.2734375])
# save_skirt_input(167398, 64, 180374, 1.5275959968566895*u.kpc, [18064.6953125, 34308.98046875, 28819.6640625])
# save_skirt_input(167401, 61, 265446, 2.400195837020874*u.kpc, [16342.58203125, 33458.21484375, 28929.4296875])
# save_skirt_input(167409, 45, 114770, 1.6774942874908447*u.kpc, [17938.80078125, 34501.75390625, 27869.64453125])
# save_skirt_input(167412, 59, 133960, 1.4007476568222046*u.kpc, [17189.0703125, 340.875732421875, 30242.794921875])
# save_skirt_input(167418, 43, 87688, 2.1653599739074707*u.kpc, [17343.234375, 462.3441467285156, 30984.125])
# save_skirt_input(167420, 89, 132556, 1.9707319736480713*u.kpc, [17515.84375, 34551.45703125, 28939.12109375])
# save_skirt_input(167421, 64, 147191, 2.874843120574951*u.kpc, [17470.236328125, 808.1504516601562, 30396.40625])
# save_skirt_input(167422, 38, 60195, 1.365623950958252*u.kpc, [17789.162109375, 508.72576904296875, 31122.212890625])
# save_skirt_input(167434, 30, 41306, 0.8866535425186157*u.kpc, [18073.50390625, 82.76520538330078, 31237.720703125])
# save_skirt_input(184932, 55, 251127, 5.7491559982299805*u.kpc, [22664.27734375, 5809.5537109375, 7533.21923828125])
# save_skirt_input(184934, 67, 115153, 6.522535800933838*u.kpc, [23316.083984375, 4778.94091796875, 6079.46923828125])
# save_skirt_input(184943, 56, 91147, 1.3809112310409546*u.kpc, [23565.271484375, 4408.20849609375, 6112.8671875])
# save_skirt_input(184944, 75, 107514, 2.2174360752105713*u.kpc, [23274.404296875, 4606.64111328125, 5867.6474609375])
# save_skirt_input(184945, 44, 66840, 0.9806254506111145*u.kpc, [23049.83984375, 4271.59521484375, 6479.2353515625])
# save_skirt_input(184946, 54, 106129, 1.4133001565933228*u.kpc, [23513.736328125, 4718.4609375, 6318.044921875])
# save_skirt_input(184949, 91, 175621, 1.576864242553711*u.kpc, [24023.900390625, 4817.2509765625, 6129.0087890625])
# save_skirt_input(184952, 62, 109681, 1.2705119848251343*u.kpc, [23388.52734375, 4373.74853515625, 5784.23486328125])
# save_skirt_input(184954, 48, 87982, 2.1390504837036133*u.kpc, [23261.0078125, 4623.61181640625, 6270.169921875])
# save_skirt_input(184963, 46, 71435, 2.1442196369171143*u.kpc, [22951.408203125, 4317.70263671875, 6422.2744140625])
# save_skirt_input(198186, 53, 76760, 3.432344913482666*u.kpc, [32943.54296875, 30044.005859375, 12222.8515625])
# save_skirt_input(198188, 47, 63253, 0.8854263424873352*u.kpc, [32430.66796875, 30168.888671875, 12141.912109375])
# save_skirt_input(198190, 86, 169771, 2.069409132003784*u.kpc, [32689.90234375, 30230.966796875, 12009.375])
# save_skirt_input(198191, 98, 198574, 1.2872415781021118*u.kpc, [31909.6484375, 30052.583984375, 11763.1884765625])
# save_skirt_input(198192, 53, 537954, 3.7013840675354004*u.kpc, [31866.36328125, 29889.646484375, 12350.7568359375])
# save_skirt_input(198193, 56, 81473, 3.1956584453582764*u.kpc, [32388.373046875, 30758.25, 11488.6083984375])
# save_skirt_input(208821, 47, 72169, 1.2715272903442383*u.kpc, [4841.828125, 18173.693359375, 13254.49609375])
# save_skirt_input(220604, 73, 191069, 4.887711048126221*u.kpc, [9179.4326171875, 8770.296875, 2380.82421875])
# save_skirt_input(220606, 59, 122421, 1.4301583766937256*u.kpc, [9278.75, 8931.4189453125, 2554.07958984375])
# save_skirt_input(220609, 44, 79515, 2.054481029510498*u.kpc, [8546.142578125, 8962.072265625, 3025.50146484375])
# save_skirt_input(229934, 44, 246974, 0.6993401050567627*u.kpc, [441.8955993652344, 29535.720703125, 19032.15625])
# save_skirt_input(229938, 51, 139953, 0.9561034440994263*u.kpc, [34123.25390625, 31819.23046875, 18985.26171875])
# save_skirt_input(229944, 97, 218420, 2.7884552478790283*u.kpc, [532.0528564453125, 29835.04296875, 19402.9765625])
# save_skirt_input(229950, 53, 149557, 2.044487953186035*u.kpc, [34616.26953125, 30255.083984375, 19987.369140625])
# save_skirt_input(242789, 56, 180483, 4.562782287597656*u.kpc, [20917.033203125, 19690.8984375, 17111.205078125])
# save_skirt_input(242792, 46, 334957, 1.9745131731033325*u.kpc, [20753.87890625, 20613.111328125, 13708.2607421875])
# save_skirt_input(253865, 72, 166413, 4.1831769943237305*u.kpc, [1287.114990234375, 6028.1357421875, 26487.322265625])
# save_skirt_input(253871, 59, 155058, 2.835629940032959*u.kpc, [1245.82275390625, 5734.55908203125, 27642.740234375])
# save_skirt_input(253873, 81, 188295, 1.6667230129241943*u.kpc, [1306.615966796875, 5946.35595703125, 26629.623046875])
# save_skirt_input(253874, 56, 146205, 1.8737987279891968*u.kpc, [923.546630859375, 5860.53662109375, 27023.455078125])
# save_skirt_input(253884, 46, 98906, 2.8064401149749756*u.kpc, [1221.2706298828125, 5978.21875, 27790.755859375])
# save_skirt_input(264891, 81, 218716, 2.793318748474121*u.kpc, [21557.662109375, 2054.412841796875, 14768.23828125])
# save_skirt_input(264897, 88, 233049, 6.5292792320251465*u.kpc, [21894.140625, 2481.822265625, 14457.3369140625])
# save_skirt_input(275546, 51, 311495, 2.376377820968628*u.kpc, [2971.002197265625, 22772.833984375, 15073.3466796875])
# save_skirt_input(275550, 55, 126584, 2.006826639175415*u.kpc, [3970.59033203125, 22281.12890625, 16033.5146484375])
# save_skirt_input(275553, 52, 137093, 1.8947705030441284*u.kpc, [4409.40576171875, 22134.79296875, 15363.9736328125])
# save_skirt_input(275557, 63, 139489, 0.9585793614387512*u.kpc, [4189.87548828125, 21965.638671875, 15776.208984375])
# save_skirt_input(282780, 28, 117596, 0.5813378691673279*u.kpc, [3971.068359375, 21586.203125, 4909.3154296875])
# save_skirt_input(282790, 84, 226272, 2.696564197540283*u.kpc, [535.025146484375, 21271.201171875, 6125.66015625])
# save_skirt_input(289388, 45, 314678, 1.0405817031860352*u.kpc, [34943.0390625, 14269.501953125, 12258.4033203125])
# save_skirt_input(289389, 76, 216057, 0.8507285118103027*u.kpc, [607.306396484375, 15790.4052734375, 12028.19140625])
# save_skirt_input(289390, 71, 185742, 2.602957248687744*u.kpc, [793.7001342773438, 15254.9921875, 12209.6259765625])
# save_skirt_input(294867, 56, 418992, 1.2861616611480713*u.kpc, [30023.71875, 33106.765625, 7813.1162109375])
# save_skirt_input(294871, 70, 206401, 2.672353506088257*u.kpc, [31423.2734375, 34076.72265625, 7025.68212890625])
# save_skirt_input(294872, 67, 425465, 1.565247893333435*u.kpc, [31639.404296875, 34312.203125, 6507.662109375])
# save_skirt_input(294875, 72, 217916, 1.080196499824524*u.kpc, [31385.982421875, 33745.05078125, 6941.763671875])
# save_skirt_input(294879, 88, 263228, 1.5078620910644531*u.kpc, [31145.669921875, 33754.59765625, 7256.13671875])
# save_skirt_input(300912, 74, 217278, 3.476973056793213*u.kpc, [14647.009765625, 4974.1689453125, 33148.64453125])
# save_skirt_input(307485, 50, 189955, 5.004919052124023*u.kpc, [15832.6171875, 19699.033203125, 22148.666015625])
# save_skirt_input(307487, 49, 253392, 3.6569583415985107*u.kpc, [17992.498046875, 20927.701171875, 22983.119140625])
# save_skirt_input(313694, 90, 287325, 2.6654250621795654*u.kpc, [23885.90625, 25550.6484375, 5011.994140625])
# save_skirt_input(313698, 70, 210810, 2.7307660579681396*u.kpc, [23617.44921875, 26169.783203125, 5029.3720703125])
# save_skirt_input(319734, 48, 118495, 2.9442405700683594*u.kpc, [27362.283203125, 2479.93310546875, 7356.38330078125])
# save_skirt_input(324125, 76, 247426, 1.9658870697021484*u.kpc, [29505.654296875, 19900.181640625, 4887.1865234375])
# save_skirt_input(324126, 72, 227111, 3.995614528656006*u.kpc, [30141.76953125, 20090.212890625, 4894.41455078125])
# save_skirt_input(324129, 80, 263049, 1.069823145866394*u.kpc, [30392.91015625, 19979.162109375, 5153.845703125])
# save_skirt_input(324131, 46, 141121, 0.9843047857284546*u.kpc, [30521.306640625, 20117.3203125, 4721.1318359375])
# save_skirt_input(324132, 54, 164003, 2.43127703666687*u.kpc, [30602.798828125, 20019.61328125, 5136.88330078125])
# save_skirt_input(338447, 37, 140470, 1.1112414598464966*u.kpc, [31704.14453125, 25353.7734375, 30946.36328125])
# save_skirt_input(345873, 63, 428630, 1.4375026226043701*u.kpc, [8328.2373046875, 26634.998046875, 30696.939453125])
# save_skirt_input(355734, 70, 295779, 2.7876968383789062*u.kpc, [25157.583984375, 24984.271484375, 9704.9892578125])
# save_skirt_input(358609, 47, 240423, 4.075259208679199*u.kpc, [5637.240234375, 15550.5146484375, 11248.740234375])
# save_skirt_input(362994, 41, 152734, 2.3454973697662354*u.kpc, [10111.7333984375, 18734.53125, 4733.9736328125])
# save_skirt_input(366407, 26, 35261, 0.6124556064605713*u.kpc, [25529.533203125, 15449.326171875, 31379.794921875])
# save_skirt_input(377656, 74, 305307, 3.070284128189087*u.kpc, [19892.798828125, 20425.18359375, 19490.810546875])
# save_skirt_input(377658, 88, 355368, 1.5309075117111206*u.kpc, [19665.79296875, 20292.55859375, 18992.38671875])
# save_skirt_input(379803, 49, 204838, 4.43634557723999*u.kpc, [5507.28564453125, 7440.9794921875, 30852.828125])
# save_skirt_input(388545, 92, 366043, 1.0344150066375732*u.kpc, [8111.88037109375, 28445.078125, 18729.263671875])
# save_skirt_input(394623, 57, 376850, 0.9632445573806763*u.kpc, [11026.7763671875, 17260.828125, 2638.9716796875])
# save_skirt_input(404818, 62, 358211, 3.768686294555664*u.kpc, [1185.3515625, 10538.3291015625, 34949.78125])
# save_skirt_input(406941, 41, 190681, 2.9298694133758545*u.kpc, [33740.53125, 8565.4951171875, 25618.654296875])
# save_skirt_input(414918, 73, 454041, 1.6958820819854736*u.kpc, [8108.203125, 14466.990234375, 24470.166015625])
# save_skirt_input(416713, 57, 262272, 3.281362533569336*u.kpc, [31893.501953125, 17795.5, 10373.890625])
# save_skirt_input(418335, 54, 332435, 2.0327541828155518*u.kpc, [7831.1142578125, 3975.388916015625, 33039.1015625])
# save_skirt_input(421555, 84, 369589, 6.883451461791992*u.kpc, [33739.55859375, 10110.607421875, 11192.7734375])
# save_skirt_input(421556, 83, 359841, 2.184152841567993*u.kpc, [33833.79296875, 10178.794921875, 11289.9677734375])
# save_skirt_input(428177, 80, 396455, 3.730598211288452*u.kpc, [25678.8359375, 32901.51171875, 15656.0751953125])
# save_skirt_input(434356, 49, 278412, 2.151136875152588*u.kpc, [14733.322265625, 2557.226806640625, 19166.357421875])
# save_skirt_input(445626, 62, 295572, 3.969383955001831*u.kpc, [23633.310546875, 23665.021484375, 4890.91015625])
# save_skirt_input(446665, 53, 309497, 0.9388160109519958*u.kpc, [14660.34375, 19637.927734375, 21082.787109375])
# save_skirt_input(447914, 45, 179454, 1.921480655670166*u.kpc, [9991.69140625, 31006.05859375, 19240.74609375])
# save_skirt_input(450916, 56, 301528, 4.137515544891357*u.kpc, [12897.134765625, 16154.927734375, 29370.75])
# save_skirt_input(454172, 68, 398973, 2.5442001819610596*u.kpc, [33067.3125, 18051.40625, 5096.1123046875])
# save_skirt_input(457431, 57, 308972, 2.9916603565216064*u.kpc, [1892.830078125, 19613.013671875, 5747.8662109375])
# save_skirt_input(459558, 56, 360118, 0.9018786549568176*u.kpc, [8241.7099609375, 26125.974609375, 27421.431640625])
# save_skirt_input(466549, 46, 252055, 1.1439653635025024*u.kpc, [9726.927734375, 11637.38671875, 3618.622802734375])
# save_skirt_input(475619, 31, 100674, 1.3586828708648682*u.kpc, [14452.171875, 2939.702880859375, 32662.431640625])
# save_skirt_input(480803, 79, 405593, 2.296109199523926*u.kpc, [20308.201171875, 15898.251953125, 2483.97509765625])
# save_skirt_input(482155, 74, 370008, 5.354395866394043*u.kpc, [28428.658203125, 11384.9931640625, 29686.79296875])
# save_skirt_input(482891, 49, 317116, 0.6739818453788757*u.kpc, [8914.0654296875, 1.1891578435897827, 21530.509765625])
# save_skirt_input(483594, 63, 338236, 3.5494449138641357*u.kpc, [8922.822265625, 14293.255859375, 26284.55859375])
# save_skirt_input(484448, 47, 229841, 1.5786393880844116*u.kpc, [28339.68359375, 11355.0634765625, 3349.807861328125])
# save_skirt_input(486917, 64, 383587, 0.9261046648025513*u.kpc, [19670.880859375, 11504.57421875, 2759.96044921875])
# save_skirt_input(486919, 88, 447553, 0.8136436343193054*u.kpc, [20240.453125, 11896.216796875, 2531.94482421875])
# save_skirt_input(495451, 53, 296260, 0.8403759598731995*u.kpc, [34916.625, 31973.77734375, 16219.4853515625])
# save_skirt_input(496186, 51, 267448, 1.174773097038269*u.kpc, [11476.634765625, 29814.6875, 26593.642578125])
# save_skirt_input(503987, 40, 242135, 2.5162928104400635*u.kpc, [34925.921875, 21665.421875, 2332.586669921875])
# save_skirt_input(504559, 49, 279311, 1.0440152883529663*u.kpc, [13863.2880859375, 21591.70703125, 12750.03515625])
# save_skirt_input(507294, 79, 460075, 1.9364205598831177*u.kpc, [16731.833984375, 31390.927734375, 27040.513671875])
# save_skirt_input(507784, 54, 354136, 0.8554603457450867*u.kpc, [22225.37109375, 25135.744140625, 21645.037109375])
# save_skirt_input(508539, 72, 444540, 1.544507384300232*u.kpc, [27210.1640625, 12548.9814453125, 3522.6484375])
# save_skirt_input(513105, 94, 497904, 2.53871750831604*u.kpc, [34779.13671875, 26471.1796875, 26439.052734375])
# save_skirt_input(514272, 90, 487648, 2.5328171253204346*u.kpc, [15803.55859375, 25169.48828125, 32039.822265625])
# save_skirt_input(515296, 53, 295759, 1.4500712156295776*u.kpc, [7306.271484375, 2747.469482421875, 24515.1484375])
# save_skirt_input(516101, 79, 445410, 2.698007106781006*u.kpc, [21145.89453125, 18159.953125, 23016.107421875])
# save_skirt_input(516760, 40, 208817, 0.7851501703262329*u.kpc, [5418.78564453125, 9167.400390625, 21037.1875])
# save_skirt_input(524506, 45, 289349, 1.0616304874420166*u.kpc, [18507.498046875, 15147.849609375, 8127.251953125])
# save_skirt_input(526879, 60, 435509, 2.031116485595703*u.kpc, [594.8347778320312, 23681.908203125, 31970.3671875])
# save_skirt_input(529365, 78, 432059, 3.0339550971984863*u.kpc, [23975.82421875, 19487.3046875, 11709.255859375])
# save_skirt_input(531320, 72, 435217, 0.8108838200569153*u.kpc, [19841.197265625, 29619.501953125, 17910.669921875])
# save_skirt_input(536654, 76, 433088, 1.519272804260254*u.kpc, [2364.902587890625, 5387.15380859375, 1729.4039306640625])
# save_skirt_input(539667, 78, 466172, 4.427576065063477*u.kpc, [30962.96484375, 17877.24609375, 9677.2939453125])
# save_skirt_input(540082, 83, 469748, 3.325726270675659*u.kpc, [33698.82421875, 5794.43408203125, 28466.373046875])
# save_skirt_input(541218, 45, 303267, 1.368080496788025*u.kpc, [29840.169921875, 31575.064453125, 4590.26708984375])
# save_skirt_input(545003, 48, 395032, 0.8729383945465088*u.kpc, [5722.41796875, 1129.3597412109375, 24168.025390625])
# save_skirt_input(545703, 47, 325578, 1.0640292167663574*u.kpc, [22145.544921875, 24657.17578125, 22314.259765625])
# save_skirt_input(546870, 66, 411132, 2.682202100753784*u.kpc, [18950.51171875, 23546.283203125, 1906.41357421875])
# save_skirt_input(547545, 31, 204097, 1.1967297792434692*u.kpc, [9064.7802734375, 30419.783203125, 20829.953125])
# save_skirt_input(548151, 71, 422477, 0.8808767199516296*u.kpc, [18754.5078125, 22242.310546875, 24388.2421875])
# save_skirt_input(551541, 90, 530508, 0.9603878259658813*u.kpc, [676.5078125, 22394.4609375, 33638.57421875])
# save_skirt_input(555013, 56, 396417, 2.1972551345825195*u.kpc, [2932.37109375, 10982.5556640625, 1354.7694091796875])
# save_skirt_input(555815, 97, 552094, 0.6797944903373718*u.kpc, [21319.75390625, 26392.63671875, 26992.08984375])
# save_skirt_input(562029, 67, 444677, 2.0504655838012695*u.kpc, [7840.37841796875, 22433.306640625, 29135.87890625])
# save_skirt_input(564498, 58, 390982, 0.7486891150474548*u.kpc, [23029.587890625, 16468.25390625, 8772.826171875])
# save_skirt_input(567607, 91, 542669, 2.183142900466919*u.kpc, [24061.4296875, 5968.46875, 2243.404052734375])
# save_skirt_input(568646, 73, 471214, 1.8237674236297607*u.kpc, [18012.349609375, 19151.712890625, 23031.916015625])
# save_skirt_input(572328, 89, 527832, 2.9482383728027344*u.kpc, [7543.607421875, 26028.177734375, 27021.41796875])
# save_skirt_input(574037, 74, 500301, 2.0641438961029053*u.kpc, [27758.177734375, 1943.283447265625, 6960.0068359375])
# save_skirt_input(576516, 48, 337779, 1.0064384937286377*u.kpc, [9289.998046875, 12538.9248046875, 9136.7646484375])
# save_skirt_input(576705, 96, 583023, 0.6625633835792542*u.kpc, [10003.2705078125, 10086.1123046875, 28161.83984375])
# save_skirt_input(580907, 79, 501422, 4.120691299438477*u.kpc, [22916.15234375, 23667.650390625, 19517.50390625])
# save_skirt_input(584007, 74, 489691, 1.0839357376098633*u.kpc, [28515.037109375, 21472.6875, 4920.64453125])
# save_skirt_input(584724, 71, 485061, 1.0511302947998047*u.kpc, [2909.214111328125, 32453.072265625, 17644.837890625])
# save_skirt_input(588399, 73, 476962, 1.3440866470336914*u.kpc, [13380.865234375, 16539.662109375, 6907.58837890625])
# save_skirt_input(588831, 88, 543556, 2.90720272064209*u.kpc, [19624.541015625, 1594.2462158203125, 26156.87109375])
# save_skirt_input(591641, 89, 563245, 4.813155174255371*u.kpc, [22125.880859375, 26106.236328125, 23046.564453125])
# save_skirt_input(591796, 66, 463766, 0.7126250267028809*u.kpc, [15423.072265625, 21101.23828125, 16526.181640625])
# save_skirt_input(592021, 81, 515806, 1.0193214416503906*u.kpc, [21901.40234375, 4012.403564453125, 17790.822265625])
# save_skirt_input(593694, 57, 414095, 1.0779483318328857*u.kpc, [33979.21484375, 6133.1357421875, 31177.248046875])
# save_skirt_input(596401, 85, 607354, 1.0924177169799805*u.kpc, [5261.21044921875, 11739.267578125, 8969.2802734375])
# save_skirt_input(606223, 70, 484436, 0.6359179019927979*u.kpc, [19965.978515625, 10756.1572265625, 18495.34765625])
# save_skirt_input(607654, 55, 427014, 2.374330520629883*u.kpc, [3014.030517578125, 20786.876953125, 14766.6435546875])
# save_skirt_input(609710, 93, 593479, 1.1644617319107056*u.kpc, [1106.1356201171875, 27546.36328125, 16162.6728515625])
# save_skirt_input(610532, 87, 570004, 1.6736485958099365*u.kpc, [8859.7041015625, 21079.33984375, 12697.9716796875])
# save_skirt_input(613192, 60, 34598, 2.361635208129883*u.kpc, [7416.59765625, 25875.2890625, 22808.154296875])
# save_skirt_input(623367, 68, 493500, 1.0340344905853271*u.kpc, [7665.25732421875, 24403.033203125, 33188.5546875])
# save_skirt_input(625281, 83, 566307, 2.681144952774048*u.kpc, [16021.517578125, 7548.861328125, 24213.03515625])
# save_skirt_input(626287, 97, 621001, 4.127569675445557*u.kpc, [12597.595703125, 26697.8203125, 24813.966796875])
# save_skirt_input(626583, 92, 608571, 5.476966857910156*u.kpc, [30519.021484375, 9634.5693359375, 23674.69140625])
# save_skirt_input(634046, 90, 608951, 5.083594799041748*u.kpc, [12440.5322265625, 21563.009765625, 20298.564453125])
# save_skirt_input(634631, 95, 624747, 2.454105854034424*u.kpc, [8585.830078125, 26196.576171875, 29338.873046875])
# save_skirt_input(637199, 75, 553549, 0.8962335586547852*u.kpc, [2266.245849609375, 29348.4375, 10458.58984375])
# save_skirt_input(642671, 62, 485960, 1.3010234832763672*u.kpc, [16822.9921875, 34268.88671875, 29315.234375])
# save_skirt_input(651449, 87, 644801, 1.8590806722640991*u.kpc, [19370.79296875, 1428.2315673828125, 27299.810546875])
# save_skirt_input(657228, 84, 597037, 2.6778881549835205*u.kpc, [16712.91015625, 25100.27734375, 23662.9375])
# save_skirt_input(661599, 99, 661599, 1.5164704322814941*u.kpc, [30470.927734375, 15163.5615234375, 7637.79248046875])
# save_skirt_input(663512, 86, 617326, 4.313283443450928*u.kpc, [15500.2119140625, 21424.39453125, 15992.43359375])
# save_skirt_input(671746, 98, 669958, 1.966431736946106*u.kpc, [3420.68212890625, 3161.719970703125, 21596.59375])
# save_skirt_input(672526, 86, 630729, 1.0049163103103638*u.kpc, [22685.25390625, 14111.4794921875, 17170.87109375])
# save_skirt_input(678001, 94, 650019, 1.8563545942306519*u.kpc, [5228.0283203125, 31868.65625, 22707.541015625])
# save_skirt_input(692981, 62, 18, 1.379520297050476*u.kpc, [6690.6650390625, 23990.37890625, 21021.9296875])
# save_skirt_input(698432, 99, 698432, 2.4620347023010254*u.kpc, [24126.328125, 18035.5390625, 31814.82421875])
# save_skirt_input(708459, 84, 641448, 3.796165704727173*u.kpc, [4081.84521484375, 21873.060546875, 15929.71484375])
# save_skirt_input(719882, 80, 84054, 1.214719295501709*u.kpc, [27489.927734375, 8014.7880859375, 4025.9189453125])
# save_skirt_input(731524, 92, 714626, 0.8565922975540161*u.kpc, [8709.58984375, 8600.4013671875, 2310.703369140625])
# save_skirt_input(738736, 73, 92832, 0.8492041826248169*u.kpc, [15969.546875, 29337.400390625, 25776.44140625])
# save_skirt_input(799553, 73, 498318, 5.276242733001709*u.kpc, [34998.94921875, 17757.357421875, 23476.44140625])

# from os import listdir, mkdir
# subIDs = list(np.sort(np.int_(listdir('SKIRT/SKIRT_input_quenched'))))
