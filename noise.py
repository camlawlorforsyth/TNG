
import os
from os.path import exists
import numpy as np
import pickle

import astropy.constants as c
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import h5py
from scipy.ndimage import gaussian_filter

import filters
import plotting as plt
from CASTOR_proposal import spatial_plot_info

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def add_noise_and_psf(subID, telescope, version='', psf=0.15*u.arcsec,
                      display=False, save=True) :
    
    # telescope diameters from
    # https://www.castormission.org/mission
    # https://www.jwst.nasa.gov/content/forScientists/
    # faqScientists.html#collectingarea
    # https://jwst-docs.stsci.edu/jwst-observatory-hardware/jwst-telescope
    # https://roman.ipac.caltech.edu/sims/Param_db.html
    
    inDir = 'SKIRT/SKIRT_output_quenched/'
    outDir = 'SKIRT/SKIRT_processed_images_quenched/{}/'.format(subID)
    passbandsDir = 'passbands/'
    
    # open the SKIRT output file, and get the plate scale for the SKIRT images
    infile = '{}{}/{}_{}_total.fits'.format(inDir, subID, subID,
        telescope.split('_')[0].upper())
    with fits.open(infile) as hdu :
        plate_scale = (hdu[0].header)['CDELT1']*u.arcsec/u.pix
        data = hdu[0].data*u.MJy/u.sr
    
    # get all the parameters for every telescope
    dictionary = get_noise()
    
    if telescope == 'castor_wide' :
        exposures = np.array([1000, 1000, 2000])*u.s
        area = np.pi*np.square(100*u.cm/2)
    elif telescope == 'castor_deep' :
        exposures =  np.array([18000, 18000, 36000])*u.s
        area = np.pi*np.square(100*u.cm/2)
    elif telescope == 'castor_ultradeep' :
        exposures =  np.array([180000, 180000, 360000])*u.s
        area = np.pi*np.square(100*u.cm/2)
    elif telescope == 'hst_hff' :
        # get the maximum exposure time for each HFF filter
        with h5py.File('background/HFF_exposure_times.hdf5', 'r') as hf :
            exps = hf['exposures'][:]
        exps = np.nanmax(exps, axis=1)
        exposures = np.concatenate([exps[:1], exps[:5], exps[4:]])*u.s
        area = np.pi*np.square(240*u.cm/2)
    elif telescope == 'hst_deep' : # assume 30 hrs per filter for now
        exposures = 108000*np.ones(19)*u.s
        area = np.pi*np.square(240*u.cm/2)
    elif telescope == 'jwst_deep' : # assume 10 hrs per filter for now
        exposures = 36000*np.ones(18)*u.s
        area = np.pi*np.square(578.673*u.cm/2)
    elif telescope == 'roman_hlwas' :
        exposures = 146*np.ones(8)*u.s
        area = np.pi*np.square(236*u.cm/2)
    
    # get the filter names for the telescope
    filters = [key for key in dictionary.keys() if telescope.split('_')[0] in key]
    length = len(filters)
    
    # get certain attributes of the filters
    pivots = np.full(length, np.nan)*u.um
    widths = np.full(length, np.nan)*u.um
    backgrounds = np.full(length, np.nan)*u.mag/np.square(u.arcsec)
    dark_currents = np.full(length, np.nan)*u.electron/u.s/u.pix
    read_noises = np.full(length, np.nan)*u.electron/u.pix
    for i, filt in enumerate(filters) :
        pivots[i] = dictionary[filt]['pivot']
        widths[i] = dictionary[filt]['fwhm']
        backgrounds[i] = dictionary[filt]['background']
        dark_currents[i] = dictionary[filt]['dark_current']
        read_noises[i] = dictionary[filt]['read_noise']
    
    # get the sky background levels in Jy/arcsec^2
    bkg_Jy = mag_to_Jy(backgrounds)
    
    # get the throughputs at the pivot wavelengths
    throughputs = np.full(length, np.nan)
    for i, (filt, pivot) in enumerate(zip(filters, pivots)) :
        array = np.genfromtxt(passbandsDir + filt + '.txt')
        waves, response = array[:, 0]*u.um, array[:, 1]
        throughputs[i] = np.interp(pivot, waves, response)
    
    # get the area of a pixel on the sky, in arcsec^2
    pixel_area = np.square(plate_scale*u.pix)
    
    # calculate the conversion factor PHOTFNU to get janskys [per pixel]
    # from spatial electron flux electron/s/cm^2 [per pixel]
    photfnus = calculate_photfnu(1*u.electron/u.s/np.square(u.cm), pivots,
        widths, throughputs)
    
    # get the background electrons per second per pixel
    Bsky = bkg_Jy*area*pixel_area/photfnus
    
    # get the background electrons per pixel over the entire exposure
    background_electrons = Bsky*exposures
    
    # get the dark current electrons per second per pixel
    Bdet = dark_currents*u.pix
    
    # get the dark current electrons per pixel over the entire exposure
    detector_electrons = Bdet*exposures
    
    # get the number of reads, limiting a given exposure to 1000 s, as longer
    # exposures than 1000 s will be dominated by cosmic rays
    single_exposure = 1000*u.s
    Nreads = np.ceil(exposures/single_exposure)
    
    # get the read noise electrons per pixel
    read_electrons = read_noises*u.pix
    
    # get the total non-source noise per pixel over the entire exposure
    nonsource_level = background_electrons + detector_electrons
    
    # check the brightness of the galaxy in the given bands
    # mags = []
    # for frame in data :
    #     print(np.sum(frame*pixel_area).to(u.Jy))
    #     m_AB = -2.5*np.log10(np.sum(frame*pixel_area).to(u.Jy)/(3631*u.Jy))*u.mag
    #     mags.append(m_AB.value)
    # print(mags)
    
    # convert the PSF FWHM (that we'll use to convolve the images) into pixels
    sigma = psf/(2*np.sqrt(2*np.log(2))) # arcseconds
    sigma_pix = sigma/plate_scale # pixels
    
    for (filt, frame, pivot, width, throughput, exposure, level, Nread, RR,
         photfnu) in zip(filters, data, pivots, widths, throughputs, exposures,
                         nonsource_level, Nreads, read_electrons, photfnus) :
        
        # convert the noiseless synthetic SKIRT image to convenient units
        frame = frame.to(u.Jy/np.square(u.arcsec)) # Jy/arcsec^2 [per pixel]
        
        # get the noiseless synthetic SKIRT image
        image = frame*exposure*area*pixel_area/photfnu # electron [per pixel]
        # plt.display_image_simple(image.value, vmin=None, vmax=None)
        
        # define the convolution kernel and convolve the image
        kernel = Gaussian2DKernel(sigma_pix.value)
        convolved = convolve(image.value, kernel)*u.electron # electron [per pixel]
        # plt.display_image_simple(convolved.value, vmin=None, vmax=None)
        
        # add the non-source level to the convolved image
        noisey = convolved + level # electron [per pixel]
        # plt.display_image_simple(noisey.value, vmin=None, vmax=None)
        
        # sample from a Poisson distribution with the noisey data
        sampled = np.random.poisson(noisey.value)*u.electron # electron [per pixel]
        # plt.display_image_simple(sampled.value, vmin=None, vmax=None)
        
        # add the RMS noise value
        sampled = np.random.normal(sampled.value,
            scale=np.sqrt(Nread)*RR.value)*u.electron # electron [per pixel]
        # plt.display_image_simple(sampled.value, vmin=None, vmax=None)
        
        # subtract the background from the sampled image
        subtracted = sampled - level # electron [per pixel]
        # plt.display_image_simple(subtracted.value, vmin=None, vmax=None)
        
        # determine the final noise
        noise = np.sqrt(noisey.value +
            Nread*np.square(RR.value))*u.electron # electron [per pixel]
        
        # convert back to janskys [per pixel]
        # subtracted = subtracted/exposure/area*photfnu
        # noise = noise/exposure/area*photfnu
        # if display :
        #     plt.display_image_simple(final.value, vmin=1e-10, vmax=1e-6)
        #     plt.display_image_simple(noise.value, vmin=1e-9, vmax=1e-8)
        
        # save the output to file
        if save :
            os.makedirs(outDir, exist_ok=True)
            
            outfile = '{}_{}.fits'.format(telescope, filt.split('_')[1])
            save_cutout(subtracted.value, outDir + outfile,
                        exposure.value, photfnu.value, plate_scale.value, 0.5)
            
            noise_outfile = '{}_{}_noise.fits'.format(telescope, filt.split('_')[1])
            save_cutout(noise.value, outDir + noise_outfile,
                        exposure.value, photfnu.value, plate_scale.value, 0.5)
            
            snr_outfile = '{}_{}_snr.png'.format(telescope, filt.split('_')[1])
            plt.display_image_simple(subtracted.value/noise.value,
                lognorm=False, vmin=0.5, vmax=10, save=True,
                outfile=outDir + snr_outfile)
    
    return

def background_castor() :
    
    # adapted heavily from
    # https://github.com/CASTOR-telescope/ETC/blob/master/castor_etc/background.py
    
    # with additional discussion about renormalizing the zodiacal file from
    # https://github.com/CASTOR-telescope/ETC/tree/master/castor_etc/data/sky_background
    
    # from CASTOR ETC:
    # etc = np.array([27.72748, 24.24196, 22.58821])
    
    inDir = 'passbands/'
    filters = ['castor_uv', 'castor_u', 'castor_g']
    
    return determine_leo_background(inDir, filters)

def background_hst() :
    
    inDir = 'passbands/'
    filters = ['hst_f218w', 'hst_f225w', 'hst_f275w', 'hst_f336w',  'hst_f390w',
               'hst_f438w', 'hst_f435w', 'hst_f475w', 'hst_f555w',  'hst_f606w',
               'hst_f625w', 'hst_f775w', 'hst_f814w', 'hst_f850lp', 'hst_f105w',
               'hst_f110w', 'hst_f125w', 'hst_f140w', 'hst_f160w']
    
    return determine_leo_background(inDir, filters)

def background_jwst() :
    
    inDir = 'passbands/'
    filters = ['jwst_f070w',  'jwst_f090w',  'jwst_f115w',  'jwst_f150w',
               'jwst_f200w',  'jwst_f277w',  'jwst_f356w',  'jwst_f410m',
               'jwst_f444w',  'jwst_f560w',  'jwst_f770w',  'jwst_f1000w',
               'jwst_f1130w', 'jwst_f1280w', 'jwst_f1500w', 'jwst_f1800w',
               'jwst_f2100w', 'jwst_f2550w']
    
    return determine_l2_background(inDir, filters)

def background_roman() :
    
    inDir = 'passbands/'
    filters = ['roman_f062', 'roman_f087', 'roman_f106', 'roman_f129',
               'roman_f146', 'roman_f158', 'roman_f184', 'roman_f213']
    
    return determine_l2_background(inDir, filters)

def calculate_photfnu(electron_flux, lam_pivot, delta_lam,
                      throughput, gain=1*u.electron/u.photon) :
    
    lam_pivot = lam_pivot.to(u.m) # convert from um to m
    delta_lam = delta_lam.to(u.m) # convert from um to m
    
    # difference in wavelength to difference in frequency
    delta_nu = (c.c*delta_lam/np.square(lam_pivot)).to(u.Hz)
    
    # calculate the photon flux in photons/s/cm^2/Hz
    photnu = electron_flux/throughput/delta_nu/gain
    
    # calculate the flux density in janskys
    photfnu = photnu.to(u.Jy, equivalencies=u.spectral_density(lam_pivot))
    
    return photfnu*u.s/u.electron*np.square(u.cm)

def components_l2_background(waves) :
    # Sun-Earth L_2 Lagrange point
    
    # https://jwst-docs.stsci.edu/jwst-exposure-time-calculator-overview/
    # jwst-etc-calculations-page-overview/jwst-etc-backgrounds
    
    # https://jwst-docs.stsci.edu/jwst-general-support/jwst-background-model
    
    # https://jwst-docs.stsci.edu/jwst-other-tools/jwst-backgrounds-tool
    
    # https://github.com/spacetelescope/jwst_backgrounds/blob/master/
    # jwst_backgrounds/jbt.py
    
    # https://jwst-docs.stsci.edu/jwst-exposure-time-calculator-overview/
    # jwst-etc-pandeia-engine-tutorial/pandeia-backgrounds
    
    # https://jwst-docs.stsci.edu/jwst-exposure-time-calculator-overview/
    # jwst-etc-pandeia-engine-tutorial/pandeia-configuration-dictionaries
    
    if exists('background/minzodi_median.txt') :
        
        # total background, sum of in-field zodiacal light, in-field galactic
        # light, stray light, and thermal self-emission
        tab = Table.read('background/minzodi_median.txt', format='ascii')
        bkg_waves = tab['WAVELENGTH(micron)'].value*u.um
        bkg_fnu = tab['TOTALBACKGROUND(MJy/sr)'].value*u.MJy/u.sr
        
        # convert units in preparation of computing flam values
        bkg_waves = bkg_waves.to(u.AA)
        bkg_fnu = bkg_fnu.to(u.erg/u.s/u.Hz/np.square(u.cm*u.arcsec))
        
        # determine flam values
        bkg_flam = (c.c.to(u.AA/u.s))/np.square(bkg_waves)*bkg_fnu
        
        # interpolate those values at the specified wavelengths
        sky_background_flam = np.interp(waves, bkg_waves, bkg_flam)
    else :
        inDir = 'background/bathtubs/'
        
        means = np.full(2961, 0.0)   # in MJy/sr
        medians = np.full(2961, 0.0) # in MJy/sr
        
        waves = np.linspace(0.5, 30.1, 2961)
        
        # loop over every wavelength
        for i, wave in enumerate(waves) :
            # get the background values at that wavelength
            bkgs_at_wave = np.genfromtxt(
                inDir + 'bathtub_{:.2f}_micron.txt'.format(wave))[:, 1]
            
            # add the mean and median into the master arrays
            means[i] = np.mean(bkgs_at_wave)
            medians[i] = np.median(bkgs_at_wave)
        
        # jwst_backgrounds.jbt doesn't support wavelengths below 0.5 micron,
        # so we'll fit a linear function to the linear data, and sample at
        # the wavelengths we want, to account for Roman's F062 filter
        means_fit = np.polyfit(waves[:11], means[:11], 1)
        medians_fit = np.polyfit(waves[:11], medians[:11], 1)
        
        waves_front = np.linspace(0.4, 0.49, 10)
        means_front = means_fit[0]*waves_front + means_fit[1]
        medians_front = medians_fit[0]*waves_front + medians_fit[1]
        
        waves = np.concatenate([waves_front, waves])
        means = np.concatenate([means_front, means])
        medians = np.concatenate([medians_front, medians])
        
        means_array = np.array([waves, means]).T
        medians_array = np.array([waves, medians]).T
        
        np.savetxt('background/minzodi_mean.txt', means_array,
                   header='WAVELENGTH(micron) TOTALBACKGROUND(MJy/sr)')
        np.savetxt('background/minzodi_median.txt', medians_array,
                   header='WAVELENGTH(micron) TOTALBACKGROUND(MJy/sr)')
    
    return sky_background_flam

def components_leo_background(waves) :
    # low Earth orbit
    
    # https://etc.stsci.edu/etcstatic/users_guide/1_ref_9_background.html
    
    # https://hst-docs.stsci.edu/acsihb/chapter-9-exposure-time-calculations/
    # 9-4-detector-and-sky-backgrounds
    
    # https://hst-docs.stsci.edu/wfc3ihb/chapter-9-wfc3-exposure-time-calculation/
    # 9-7-sky-background
    
    # earthshine_model_001.fits from
    # https://ssb.stsci.edu/cdbs/work/etc/etc-cdbs/background/
    
    # earthshine
    es_tab = Table.read('background/earthshine_model_001.fits')
    es_waves = es_tab['Wavelength'].value*u.AA
    es_flam = es_tab['FLUX'].value*u.erg/u.s/np.square(u.cm*u.arcsec)/u.AA
    earthshine = np.interp(waves, es_waves, es_flam)
    
    # zodiacal_model_001.fits from
    # https://ssb.stsci.edu/cdbs/work/etc/etc-cdbs/background/
    
    # zodiacal renormalization? ->
    # https://github.com/gbrammer/wfc3/blob/master/etc_zodi.py#L61-L67
    
    # zodiacal
    zodi_tab = Table.read('background/zodiacal_model_001.fits')
    zodi_waves = zodi_tab['WAVELENGTH'].value*u.AA
    zodi_flam = zodi_tab['FLUX'].value*u.erg/u.s/np.square(u.cm*u.arcsec)/u.AA
    zodiacal = np.interp(waves, zodi_waves, zodi_flam)
    
    # geocoronal due to the [O II] 2471 A line
    geo_waves = np.linspace(2470, 2472, 201)*u.AA
    central = 2471*u.AA
    fwhm = 0.023*u.AA
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    flam = 1.5e-17*u.erg/u.s/np.square(u.cm*u.arcsec)/u.AA # low value
    gaussian = flam*np.exp(-0.5*np.square((geo_waves - central)/sigma))
    geocoronal = np.interp(waves, geo_waves, gaussian)
    
    # https://hst-docs.stsci.edu/wfc3ihb/chapter-7-ir-imaging-with-wfc3/
    # 7-9-other-considerations-for-ir-imaging
    
    # https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/
    # wfc3/documentation/instrument-science-reports-isrs/_documents/2014/WFC3-2014-03.pdf
    
    # airglow due to the He I 10830 A line
    airglow_waves = np.linspace(10795, 10865, 6001)*u.AA
    central = 10830*u.AA
    fwhm = 2*u.AA # from ETC User Manual, "Specifying the Appropriate Background"
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    flam = 0.1*1500*3.7e-14/10830*u.erg/u.s/np.square(u.cm*u.arcsec)/u.AA # avg value
    gaussian = flam*np.exp(-0.5*np.square((airglow_waves - central)/sigma))
    airglow = np.interp(waves, airglow_waves, gaussian)
    
    return earthshine, zodiacal, geocoronal, airglow

def determine_l2_background(inDir, filters) :
    
    bkg = []
    for filt in filters :
        # get the wavelengths and response curves for a given filter
        array = np.genfromtxt(inDir + filt + '.txt')
        waves, response = (array[:, 0]*u.um).to(u.AA), array[:, 1]
        
        # super sample the wavelengths and response at 1 angstrom intervals
        lam = np.arange(waves[0].value, waves[-1].value + 1, 1)*u.AA
        response = np.interp(lam.value, waves.value, response)
        
        # get the summed components of the background
        sky_background_flam = components_l2_background(lam)
        
        # ensure the units are correct
        sky_background_flam = sky_background_flam.to(
            u.erg/u.s/np.square(u.cm*u.arcsec)/u.AA)
        
        # integrate the background with the bandpass, and append the value
        mags_per_sq_arcsec = flam_to_mag(lam, sky_background_flam, response)
        bkg.append(mags_per_sq_arcsec)
    
    return np.array(bkg)

def determine_leo_background(inDir, filters) :
    
    bkg = []
    for filt in filters :
        # get the wavelengths and response curves for a given filter
        array = np.genfromtxt(inDir + filt + '.txt')
        waves, response = (array[:, 0]*u.um).to(u.AA), array[:, 1]
        
        # super sample the wavelengths and response at 1 angstrom intervals
        lam = np.arange(waves[0].value, waves[-1].value + 1, 1)*u.AA
        response = np.interp(lam.value, waves.value, response)
        
        # get the components of the background
        earthshine, zodiacal, geocoronal, airglow = components_leo_background(lam)
        
        # sum those components
        sky_background_flam = earthshine + zodiacal + geocoronal + airglow
        
        # integrate the background with the bandpass, and append the value
        mags_per_sq_arcsec = flam_to_mag(lam, sky_background_flam, response)
        bkg.append(mags_per_sq_arcsec)
    
    return np.array(bkg)

def determine_noise_components() :
    
    dictionary = filters.calculate_psfs()
    
    # CASTOR plate scales
    castor_filts = [key for key in dictionary.keys() if 'castor' in key]
    castor_scales = 0.1*np.ones(3)*u.arcsec/u.pix
    castor_bkg = background_castor()*u.mag/np.square(u.arcsec)
    castor_dark = np.array([0.00042, 0.00042, 0.002])*u.electron/u.s/u.pix
    castor_read = np.array([3.15, 3.15, 4.45])*u.electron/u.pix
    castor_scalings = np.square(castor_scales)/np.square(0.05*u.arcsec/u.pix)
    castor_dark = castor_dark/castor_scalings
    castor_read = castor_read/castor_scalings
    for filt, bkg, dark, read in zip(castor_filts, castor_bkg, castor_dark,
                                     castor_read) :
        dictionary[filt]['background'] = bkg
        dictionary[filt]['dark_current'] = dark
        dictionary[filt]['read_noise'] = read
    
    # HST plate scales
    hst_filts = [key for key in dictionary.keys() if 'hst' in key]
    hst_scales = np.array([0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395,
                           0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                           0.128, 0.128, 0.128, 0.128, 0.128])*u.arcsec/u.pix
    hst_bkg = background_hst()*u.mag/np.square(u.arcsec)
    hst_dark = np.array([0.00306, 0.00306, 0.00306, 0.00306, 0.00306, 0.00306,
                         0.0153, 0.0153, 0.0153, 0.0153, 0.0153, 0.0153, 0.0153,
                         0.0153, 0.048, 0.048, 0.048, 0.048,
                         0.048])*u.electron/u.s/u.pix
    hst_read = np.array([3.15, 3.15, 3.15, 3.15, 3.15, 3.15, 4.45, 4.45, 4.45,
                         4.45, 4.45, 4.45, 4.45, 4.45, 12.0, 12.0, 12.0, 12.0,
                         12.0])*u.electron/u.pix
    hst_scalings = np.square(hst_scales)/np.square(0.05*u.arcsec/u.pix)
    hst_dark = hst_dark/hst_scalings
    hst_read = hst_read/hst_scalings
    for filt, bkg, dark, read in zip(hst_filts, hst_bkg, hst_dark, hst_read) :
        dictionary[filt]['background'] = bkg
        dictionary[filt]['dark_current'] = dark
        dictionary[filt]['read_noise'] = read
    
    # JWST plate scales
    jwst_filts = [key for key in dictionary.keys() if 'jwst' in key]
    jwst_scales = np.array([0.031, 0.031, 0.031, 0.031, 0.031, 0.063, 0.063,
                            0.063, 0.063, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11,
                            0.11, 0.11, 0.11])*u.arcsec/u.pix
    jwst_bkg = background_jwst()*u.mag/np.square(u.arcsec)
    jwst_dark = np.array([0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0342,
                          0.0342, 0.0342, 0.0342, 0.2, 0.2, 0.2, 0.2, 0.2, 
                          0.2, 0.2, 0.2, 0.2])*u.electron/u.s/u.pix
    jwst_read = np.array([15.77, 15.77, 15.77, 15.77, 15.77, 13.25, 13.25,
                          13.25, 13.25, 14, 14, 14, 14, 14, 14, 14, 14,
                          14])*u.electron/u.pix
    jwst_scalings = np.square(jwst_scales)/np.square(0.05*u.arcsec/u.pix)
    jwst_dark = jwst_dark/jwst_scalings
    jwst_read = jwst_read/jwst_scalings
    for filt, bkg, dark, read in zip(jwst_filts, jwst_bkg, jwst_dark, jwst_read) :
        dictionary[filt]['background'] = bkg
        dictionary[filt]['dark_current'] = dark
        dictionary[filt]['read_noise'] = read
    
    # Roman plate scales
    roman_filts = [key for key in dictionary.keys() if 'roman' in key]
    roman_scales = 0.11*np.ones(8)*u.arcsec/u.pix
    roman_bkg = background_roman()*u.mag/np.square(u.arcsec)
    roman_dark = 0.005*np.ones(8)*u.electron/u.s/u.pix
    roman_read = 15.5*np.ones(8)*u.electron/u.pix
    roman_scalings = np.square(roman_scales)/np.square(0.05*u.arcsec/u.pix)
    roman_dark = roman_dark/roman_scalings
    roman_read = roman_read/roman_scalings
    for filt, bkg, dark, read in zip(roman_filts, roman_bkg, roman_dark,
                                     roman_read) :
        dictionary[filt]['background'] = bkg
        dictionary[filt]['dark_current'] = dark
        dictionary[filt]['read_noise'] = read
    
    return dictionary

def flam_to_mag(waves, flam, response) :
    
    # from Eq. 2 of Bessell & Murphy 2012
    numer = np.trapz(flam*response*waves*np.square(u.arcsec), x=waves)
    denom = np.trapz(response/waves, x=waves)
    const = c.c.to(u.AA/u.s)
    
    fnu = (numer/const/denom).value
    
    return -2.5*np.log10(fnu) - 48.6

def get_noise() :
    
    infile = 'noise/noise_components.txt'
    
    if exists(infile) :
        with open(infile, 'rb') as file :
            dictionary = pickle.load(file)
    else :
        with open(infile, 'wb') as file :
            pickle.dump(determine_noise_components(), file)
    
    return dictionary

def hff_exposure_time_vs_lam() :
    
    # get the average exposure time for each HFF filter
    with h5py.File('background/HFF_exposure_times.hdf5', 'r') as hf :
        exposures = hf['exposures'][:]
    # means = np.nanmean(exposures, axis=1)
    # medians = np.nanmedian(exposures, axis=1)
    exposures[np.isnan(exposures)] = 0 # ignore NaNs for plotting
    
    # means is larger than np.mean(exposures, axis=1)
    # medians is larger than np.median(exposures, axis=1)
    
    # get the pivot wavelengths
    dd = filters.calculate_psfs()
    pivs = [dd[key]['pivot'].value for key in dd.keys() if 'hst' in key]
    pivs = np.concatenate([pivs[1:5], pivs[6:]])
    
    # prepare values for plotting
    xs = [pivs, pivs, pivs, pivs, pivs, pivs]
    ys = np.array([exposures[:, 0], exposures[:, 1], exposures[:, 2],
                   exposures[:, 3], exposures[:, 4], exposures[:, 5]])/3600
    labels = ['a370', 'a1063', 'a2744', 'm416', 'm717', 'm1149']
    colors = ['k', 'r', 'b', 'm', 'g', 'c']
    markers = ['', '', '', '', '', '']
    styles = ['-', '--', ':', '-.', '-', '--']
    alphas = np.ones(6)
    
    # plot the exposure times for each HFF cluster
    plt.plot_simple_multi(xs, ys, labels, colors, markers, styles, alphas,
        xlabel='Wavelength (um)', ylabel='Average Exposure Time (hr)', loc=2,
        scale='linear')
    
    # create a table for visual inspection
    table = Table([column for column in exposures.T], names=labels)
    table.pprint(max_width=-1)
    
    times = np.nanmax(exposures, axis=1)
    
    return

def mag_to_Jy(mag) :
    # convert AB mag/arcsec^2 to Jy/arcsec^2
    mag = mag.to(u.mag/np.square(u.arcsec))
    return np.power(10, -0.4*(mag.value - 8.9))*u.Jy/np.square(u.arcsec)

def save_cutout(data, outfile, exposure, photfnu, scale, redshift) :
    
    hdu = fits.PrimaryHDU(data)
    
    hdr = hdu.header
    hdr['Z'] = redshift
    hdr.comments['Z'] = 'object spectroscopic redshift--by definition'
    hdr['EXPTIME'] = exposure
    hdr.comments['EXPTIME'] = 'exposure duration (seconds)--calculated'
    hdr['PHOTFNU'] = photfnu
    hdr.comments['PHOTFNU'] = 'inverse sensitivity, Jy*sec/electron'
    hdr['SCALE'] = scale
    hdr.comments['SCALE'] = 'Pixel size (arcsec) of output image'
    hdr['BUNIT'] = 'electron'
    hdr.comments['BUNIT'] = 'Physical unit of the array values'
    
    hdu.writeto(outfile)
    
    return

# import pprint
# pprint.pprint(get_noise())

# inDir = 'SKIRT/subID_198186/snap_51_variousRedshifts_allBC03/'
# inDir = 'SKIRT/subID_198186/snap_51_variousRedshifts_allBC03_revisedAges/'

# add_noise_and_psf(inDir, 'castor_wide', 'TNG_v0.7')
# add_noise_and_psf(inDir, 'castor_deep', 'TNG_v0.7')
# add_noise_and_psf(inDir, 'castor_ultradeep', 'TNG_v0.7')
# add_noise_and_psf(inDir, 'hst_hff', psf=0.14729817*u.arcsec) # PSF for F160W
# add_noise_and_psf(inDir, 'hst_deep', psf=0.14729817*u.arcsec) # PSF for F160W
# add_noise_and_psf(inDir, 'jwst_deep', psf=0.145026*u.arcsec) # PSF for NIRCam
# add_noise_and_psf(inDir, 'roman_hlwas', psf=0.151*u.arcsec) # PSF for F184 in HLWAS

# inDir = 'SKIRT/subID_198186/snap_51_variousRedshifts_MAPPINGSIII_revisedAges/'
# add_noise_and_psf(inDir, telescope='castor_wide', version='TNG_v0.8')

# inDir = 'SKIRT/subID_198186/snap_51_variousRedshifts_allBC03_revisedAges_onlyOldTest/'
# add_noise_and_psf(inDir, telescope='roman', version='v0.7_1e6photons')

# inDir = 'SKIRT/198186'
# add_noise_and_psf(inDir + '_particle_BC03/', 'castor_wide', version='198186')
# add_noise_and_psf(inDir + '_particle_MAPPINGS/', 'castor_wide', version='198186')
# add_noise_and_psf(inDir + '_voronoi_MAPPINGS/', 'castor_wide', version='198186')

# add_noise_and_psf(63871, 'castor_wide')
# add_noise_and_psf(63871, 'castor_deep')
# add_noise_and_psf(63871, 'castor_ultradeep')
# add_noise_and_psf(63871, 'roman_hlwas')

# add_noise_and_psf(198186, 'castor_ultradeep')
# add_noise_and_psf(198186, 'roman_hlwas')

# add_noise_and_psf(96771, 'castor_ultradeep')
# add_noise_and_psf(96771, 'roman_hlwas')

# subIDs = os.listdir('SKIRT/SKIRT_output_quenched')
# for subID in subIDs :
#     if subID != '14' :
#         add_noise_and_psf(subID, 'castor_ultradeep')
#         add_noise_and_psf(subID, 'roman_hlwas')

def compare_raw_to_SKIRT(subIDfinal, snap, subID, Re, center) :
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        times = hf['times'][:]
    
    # infile = 'F:/TNG50-1/mpb_cutouts_099/cutout_{}_{}.hdf5'.format(snap, subID)
    skirtDir = 'SKIRT/SKIRT_output_quenched/'
    processedDir = 'SKIRT/SKIRT_processed_images_quenched/'
    outDir = 'TNG50-1/figures/comparison_plots_SFR/'
    
    # test UV flux
    # minAge, maxAge = 0, 100
    # minAge, maxAge = 100, 250
    # minAge, maxAge = 250, 500
    # minAge, maxAge = 500, 1000
    # skirtDir = 'SKIRT/SKIRT_input_UV_flux_test/{}_{}_Myr/'.format(minAge, maxAge)
    # processedDir = 'SKIRT/SKIRT_input_UV_flux_test/{}_{}_Myr/'.format(minAge, maxAge)
    # outDir = 'SKIRT/SKIRT_input_UV_flux_test/'
    
    # use raw output from SKIRT, without adding noise
    uv_infile = '{}{}/{}_CASTOR_total.fits'.format(skirtDir, subIDfinal, subIDfinal)
    with fits.open(uv_infile) as hdu :
        skirt_pixel_area = np.square((hdu[0].header)['CDELT1']*u.arcsec)
        skirt_image = (hdu[0].data*u.MJy/u.sr*skirt_pixel_area).to(u.Jy).value
        skirt_image = skirt_image[0]
        skirt_image = np.rot90(skirt_image, k=3)
    
    nir_infile = '{}{}/{}_Roman_total.fits'.format(skirtDir, subIDfinal, subIDfinal)
    with fits.open(nir_infile) as hdu :
        skirt_pixel_area = np.square((hdu[0].header)['CDELT1']*u.arcsec)
        skirt_contour_image = (hdu[0].data*u.MJy/u.sr*skirt_pixel_area).to(u.Jy).value
        skirt_contour_image = skirt_contour_image[6]
        skirt_contour_image = np.rot90(skirt_contour_image, k=3)
    
    # use processed output, with adding noise
    uv_infile = '{}{}/castor_ultradeep_uv.fits'.format(processedDir, subIDfinal)
    if not exists(uv_infile) :
        add_noise_and_psf(subIDfinal, 'castor_ultradeep')
    with fits.open(uv_infile) as hdu :
        processed_image = hdu[0].data # janskys [per pixel]
        processed_image = np.rot90(processed_image, k=3)
    
    nir_infile = '{}{}/roman_hlwas_f184.fits'.format(processedDir, subIDfinal)
    if not exists(nir_infile) :
        add_noise_and_psf(subIDfinal, 'roman_hlwas')
    with fits.open(nir_infile) as hdu :
        processed_contour_image = hdu[0].data # janskys [per pixel]
        processed_contour_image = np.rot90(processed_contour_image, k=3)
    
    skirt_levels = np.power(10., np.array([-9.5, -8.5, -7.5]))
    # print(skirt_levels)
    # vals = np.sort(skirt_image.flatten())
    # vals = vals[vals > 0]
    # skirt_levels = np.percentile(vals, [50, 84, 95, 99])
    # print(skirt_levels)
    # print()
    
    processed_levels = np.power(10., np.array([-8.5, -7.5]))
    # print(processed_levels)
    # vals = np.sort(skirt_contour_image.flatten())
    # vals = vals[vals > 0]
    # processed_levels = np.percentile(vals, [50, 84, 95, 99])
    # print(processed_levels)
    
    spatial_edges = np.linspace(-10, 10, processed_image.shape[0] + 1)
    XX, YY = np.meshgrid(spatial_edges, spatial_edges)
    hist_centers = spatial_edges[:-1] + np.diff(spatial_edges)/2
    X_cent, Y_cent = np.meshgrid(hist_centers, hist_centers)
    
    # use TNG arrays, and create image similar to comprehensive plots
    tng_image, tng_contour_image, tng_levels = spatial_plot_info(times[snap],
        snap, subID, center, Re, spatial_edges, 100*u.Myr,
        nelson2021version=False, sfr_map=True)
    
    # specify and set new contour levels for the tng stellar mass image
    tng_levels = np.power(10., np.array([5.5, 6.5, 7.5]))
    
    # find vmin, vmax for tng_image
    tng_full = np.array(tng_image).flatten()
    tng_full[tng_full == 0.0] = np.nan
    tng_vmin, tng_vmax = np.nanpercentile(tng_full, [1, 99])
    
    # skirt_full = skirt_image.flatten()
    # skirt_full[skirt_full == 0.0] = np.nan
    # skirt_vmin, skirt_vmax = np.nanpercentile(skirt_full, [1, 99])
    skirt_vmin, skirt_vmax = 1e-10, 1e-8
    
    # pro_full = processed_image.flatten()
    # pro_full[pro_full <= 0.0] = np.nan
    # pro_vmin, pro_vmax = np.nanpercentile(pro_full, [1, 99])
    pro_vmin, pro_vmax = 1e-10, 1e-8
    
    # smooth contour images to match tng smoothed image
    tng_contour_image = gaussian_filter(tng_contour_image, 0.7)
    skirt_contour_image = gaussian_filter(skirt_contour_image, 0.6)
    processed_contour_image = gaussian_filter(processed_contour_image, 1.5)
    
    plt.plot_comparisons(tng_image, tng_contour_image, tng_levels,
        skirt_image, skirt_contour_image, skirt_levels, processed_image,
        processed_contour_image, processed_levels, XX, YY, X_cent, Y_cent,
        tng_vmin=tng_vmin, tng_vmax=tng_vmax, skirt_vmin=skirt_vmin,
        skirt_vmax=skirt_vmax, pro_vmin=pro_vmin, pro_vmax=pro_vmax,
        xlabel=r'$\Delta x$ ($R_{\rm e}$)', ylabel=r'$\Delta y$ ($R_{\rm e}$)',
        mtitle=r'subID$_{z = 0}$' + ' {}'.format(subIDfinal), xmin=-5, xmax=5,
        ymin=-5, ymax=5, save=True, outfile=outDir + '{}.png'.format(subIDfinal))
    
    return

compare_raw_to_SKIRT(96771, 44, 42759, 0.825415849685669,
                     [28307.04296875, 7637.23876953125, 4297.02587890625])

# compare_raw_to_SKIRT(1, 38, 74114, 6.919701099395752, [5434.26318359375, 23616.751953125, 17887.64453125])
# compare_raw_to_SKIRT(2, 83, 351154, 4.704523086547852, [6739.55224609375, 23354.779296875, 19600.5859375])
# compare_raw_to_SKIRT(4, 55, 29285, 2.8483943939208984, [6491.39208984375, 25355.431640625, 22688.06640625])
# compare_raw_to_SKIRT(7, 50, 6, 3.7366392612457275, [8613.125, 23953.779296875, 21985.056640625])
# compare_raw_to_SKIRT(10, 84, 10, 4.965497970581055, [7026.8642578125, 24088.828125, 20683.41015625])
# compare_raw_to_SKIRT(13, 88, 14, 1.1463567018508911, [7166.50048828125, 24465.82421875, 21832.005859375])
# compare_raw_to_SKIRT(14, 69, 15, 2.4659676551818848, [8226.775390625, 24374.388671875, 21306.654296875])
# compare_raw_to_SKIRT(16, 63, 6, 2.527513265609741, [6941.310546875, 24125.052734375, 20470.5625])
# compare_raw_to_SKIRT(21, 52, 6, 5.278622150421143, [7838.95166015625, 24018.4453125, 21521.798828125])
# compare_raw_to_SKIRT(22, 24, 104053, 0.521422266960144, [10294.9052734375, 25877.40625, 25272.794921875])
# compare_raw_to_SKIRT(23, 68, 507406, 0.9742310643196106, [6907.7900390625, 25841.369140625, 24496.259765625])
# compare_raw_to_SKIRT(26, 62, 14, 2.598369598388672, [6863.08984375, 24267.4140625, 20918.943359375])
# compare_raw_to_SKIRT(28, 73, 561097, 1.2564334869384766, [8757.783203125, 23379.0703125, 20126.90234375])
# compare_raw_to_SKIRT(30, 81, 23, 2.138542413711548, [7258.22314453125, 24850.580078125, 22439.880859375])
# compare_raw_to_SKIRT(32, 50, 25246, 3.7992358207702637, [6240.41162109375, 25877.404296875, 22768.529296875])
# compare_raw_to_SKIRT(33, 67, 31, 3.721682548522949, [7062.49755859375, 24660.509765625, 21841.9375])
# compare_raw_to_SKIRT(34, 64, 17, 1.819235920906067, [6604.55712890625, 23643.322265625, 19719.943359375])
# compare_raw_to_SKIRT(35, 36, 8866, 3.591322660446167, [6849.68359375, 23852.560546875, 20356.9609375])
# compare_raw_to_SKIRT(36, 35, 8273, 1.5611287355422974, [7180.2900390625, 23525.1328125, 21057.548828125])
# compare_raw_to_SKIRT(38, 51, 25784, 3.0328338146209717, [6831.00341796875, 25378.078125, 23074.314453125])
# compare_raw_to_SKIRT(39, 72, 43, 1.818422555923462, [7175.0712890625, 24243.140625, 20891.255859375])
# compare_raw_to_SKIRT(40, 52, 14, 4.151195526123047, [6931.50341796875, 24411.431640625, 20427.673828125])
# compare_raw_to_SKIRT(41, 34, 5, 4.313936710357666, [6850.41748046875, 25669.892578125, 24534.396484375])
# compare_raw_to_SKIRT(42, 41, 15486, 1.2100837230682373, [7364.970703125, 24079.248046875, 20682.82421875])
# compare_raw_to_SKIRT(46, 73, 44, 2.460824728012085, [8163.775390625, 24273.59765625, 21427.751953125])
# compare_raw_to_SKIRT(47, 46, 6, 2.828500270843506, [6061.8525390625, 25990.482421875, 23215.443359375])
# compare_raw_to_SKIRT(50, 44, 15631, 5.784523963928223, [7215.57958984375, 23500.9765625, 20504.27734375])
# compare_raw_to_SKIRT(51, 68, 51, 3.93544340133667, [6761.806640625, 24407.7734375, 21684.45703125])
# compare_raw_to_SKIRT(58, 53, 26924, 4.255753993988037, [7108.6123046875, 25718.61328125, 23497.5])
# compare_raw_to_SKIRT(59, 45, 15424, 1.6405946016311646, [7222.03466796875, 24102.69921875, 20806.97265625])
# compare_raw_to_SKIRT(61, 65, 38316, 2.7552266120910645, [7358.517578125, 24734.5390625, 22527.283203125])
# compare_raw_to_SKIRT(64, 39, 97269, 3.1793203353881836, [6801.228515625, 23706.103515625, 19918.783203125])
# compare_raw_to_SKIRT(70, 38, 16, 1.700038194656372, [4727.572265625, 26248.708984375, 23313.3046875])
# compare_raw_to_SKIRT(74, 46, 13, 4.594496726989746, [6724.56982421875, 25512.4765625, 23598.291015625])
# compare_raw_to_SKIRT(79, 44, 83312, 6.019397735595703, [8540.1455078125, 24261.1171875, 22080.0859375])
# compare_raw_to_SKIRT(87, 39, 14, 4.017769813537598, [6113.62255859375, 25707.59765625, 24342.890625])
# compare_raw_to_SKIRT(92, 58, 33, 3.261753797531128, [7298.26806640625, 24560.673828125, 20954.353515625])
# compare_raw_to_SKIRT(94, 45, 15433, 2.5642805099487305, [6899.04052734375, 24458.458984375, 20841.50390625])
# compare_raw_to_SKIRT(98, 80, 55, 4.225803375244141, [7218.20654296875, 24063.216796875, 21064.794921875])
# compare_raw_to_SKIRT(124, 33, 10, 3.5367484092712402, [6466.31005859375, 25650.814453125, 24474.20703125])
# compare_raw_to_SKIRT(63871, 44, 327188, 1.007636547088623, [25952.037109375, 13654.326171875, 1629.3223876953125])
# compare_raw_to_SKIRT(63874, 68, 53740, 4.365604400634766, [23337.638671875, 15940.5439453125, 3432.876953125])
# compare_raw_to_SKIRT(63875, 34, 177281, 0.8749130964279175, [24296.3984375, 14042.9169921875, 4825.59375])
# compare_raw_to_SKIRT(63879, 80, 56446, 2.1986336708068848, [23664.947265625, 14457.259765625, 2870.424560546875])
# compare_raw_to_SKIRT(63880, 45, 31981, 2.105402708053589, [22917.166015625, 15839.3232421875, 3562.930908203125])
# compare_raw_to_SKIRT(63883, 68, 53751, 5.471149444580078, [23867.849609375, 14911.1142578125, 3029.632568359375])
# compare_raw_to_SKIRT(63885, 62, 50227, 3.190803050994873, [23526.720703125, 15256.330078125, 3269.130126953125])
# compare_raw_to_SKIRT(63886, 55, 46177, 6.470934867858887, [22915.181640625, 15977.6220703125, 3170.62744140625])
# compare_raw_to_SKIRT(63887, 93, 69078, 3.743358612060547, [23740.83203125, 15729.0703125, 3349.712158203125])
# compare_raw_to_SKIRT(63891, 50, 55638, 6.846566677093506, [22997.552734375, 15962.7880859375, 3272.555419921875])
# compare_raw_to_SKIRT(63893, 59, 48208, 3.8159537315368652, [22633.21484375, 15311.412109375, 3069.39892578125])
# compare_raw_to_SKIRT(63898, 96, 67381, 3.7795445919036865, [23545.0, 15016.5751953125, 3232.046142578125])
# compare_raw_to_SKIRT(63900, 60, 50355, 3.200077772140503, [23116.423828125, 15905.71875, 3192.859375])
# compare_raw_to_SKIRT(63901, 44, 31231, 1.7348716259002686, [22625.751953125, 15625.318359375, 3623.9384765625])
# compare_raw_to_SKIRT(63907, 47, 49839, 3.958181858062744, [22431.125, 15523.3115234375, 3161.415283203125])
# compare_raw_to_SKIRT(63910, 73, 52754, 3.3400063514709473, [23701.01171875, 14724.029296875, 3478.9404296875])
# compare_raw_to_SKIRT(63911, 67, 54068, 1.471694827079773, [23443.365234375, 14608.3271484375, 3058.72998046875])
# compare_raw_to_SKIRT(63917, 61, 49305, 3.9083659648895264, [23021.154296875, 15974.404296875, 3149.738037109375])
# compare_raw_to_SKIRT(63926, 38, 25073, 2.9337918758392334, [22486.287109375, 15150.759765625, 3092.587646484375])
# compare_raw_to_SKIRT(63928, 39, 25499, 4.598609447479248, [22284.310546875, 15495.310546875, 3662.435791015625])
# compare_raw_to_SKIRT(96763, 78, 423165, 3.690382242202759, [25902.939453125, 6506.41552734375, 2705.205078125])
# compare_raw_to_SKIRT(96764, 54, 273113, 3.7120327949523926, [27551.203125, 6635.87744140625, 1826.72900390625])
# compare_raw_to_SKIRT(96766, 69, 76902, 2.992084264755249, [27683.21875, 7614.2294921875, 4826.6171875])
# compare_raw_to_SKIRT(96767, 56, 66360, 2.0710253715515137, [28517.646484375, 7987.79150390625, 4003.575927734375])
# compare_raw_to_SKIRT(96769, 58, 67975, 1.8564605712890625, [27748.685546875, 6979.9765625, 3909.672119140625])
# compare_raw_to_SKIRT(96771, 44, 42759, 0.825415849685669, [28307.04296875, 7637.23876953125, 4297.02587890625])
# compare_raw_to_SKIRT(96772, 39, 273114, 1.3693691492080688, [27702.712890625, 6569.91015625, 5414.4169921875])
# compare_raw_to_SKIRT(96778, 85, 88214, 4.2367753982543945, [27741.09375, 6843.826171875, 3893.85986328125])
# compare_raw_to_SKIRT(96779, 61, 71742, 1.7090530395507812, [27482.357421875, 7386.04150390625, 4732.724609375])
# compare_raw_to_SKIRT(96780, 40, 36243, 1.159963846206665, [27918.42578125, 7139.97900390625, 4793.330078125])
# compare_raw_to_SKIRT(96781, 62, 72681, 2.995453119277954, [27930.240234375, 7844.5576171875, 3909.1494140625])
# compare_raw_to_SKIRT(96782, 62, 72683, 1.4666552543640137, [28157.080078125, 7649.35009765625, 3617.690185546875])
# compare_raw_to_SKIRT(96783, 62, 72675, 2.021986484527588, [27487.05859375, 7351.630859375, 4787.92626953125])
# compare_raw_to_SKIRT(96785, 52, 59781, 5.436735153198242, [27986.572265625, 7716.00927734375, 5105.67822265625])
# compare_raw_to_SKIRT(96791, 51, 41516, 7.395904541015625, [28195.703125, 7551.505859375, 3693.720947265625])
# compare_raw_to_SKIRT(96793, 97, 98726, 1.257693886756897, [27210.765625, 8053.125, 4197.70263671875])
# compare_raw_to_SKIRT(96795, 89, 99765, 2.336646556854248, [27366.3203125, 6995.21728515625, 3635.391357421875])
# compare_raw_to_SKIRT(96798, 52, 59786, 3.826808452606201, [28245.962890625, 7354.04931640625, 3528.866455078125])
# compare_raw_to_SKIRT(96800, 43, 42070, 2.931291341781616, [28411.380859375, 7912.8818359375, 4465.5458984375])
# compare_raw_to_SKIRT(96801, 61, 71741, 5.5939130783081055, [27578.640625, 7810.890625, 4677.9677734375])
# compare_raw_to_SKIRT(96804, 47, 37253, 1.1913484334945679, [27857.306640625, 7189.21630859375, 4691.7158203125])
# compare_raw_to_SKIRT(96805, 67, 77300, 4.619329452514648, [27595.748046875, 6975.54296875, 3679.213623046875])
# compare_raw_to_SKIRT(96806, 57, 67694, 3.8795738220214844, [28288.744140625, 7664.14697265625, 3744.3359375])
# compare_raw_to_SKIRT(96808, 44, 42768, 3.898806095123291, [27841.138671875, 7372.93359375, 4635.9697265625])
# compare_raw_to_SKIRT(117261, 63, 89870, 1.241865634918213, [16146.3427734375, 28838.658203125, 25621.716796875])
# compare_raw_to_SKIRT(117271, 79, 99400, 1.569605827331543, [16246.2607421875, 29218.20703125, 25858.818359375])
# compare_raw_to_SKIRT(117274, 62, 89105, 5.610778331756592, [16229.2763671875, 29225.40234375, 25847.056640625])
# compare_raw_to_SKIRT(117275, 69, 92026, 2.6896326541900635, [16409.80078125, 29328.6328125, 25754.064453125])
# compare_raw_to_SKIRT(117277, 80, 99814, 1.5543626546859741, [15826.2236328125, 29111.87890625, 25662.90625])
# compare_raw_to_SKIRT(117284, 66, 92481, 1.9755234718322754, [15998.5478515625, 29330.921875, 26064.099609375])
# compare_raw_to_SKIRT(117292, 90, 118204, 3.547374963760376, [15593.3232421875, 29337.71484375, 26313.587890625])
# compare_raw_to_SKIRT(117296, 57, 93037, 1.689876675605774, [16276.28125, 29084.431640625, 25652.70703125])
# compare_raw_to_SKIRT(117302, 52, 91282, 1.6807701587677002, [16128.8134765625, 29046.716796875, 25762.298828125])
# compare_raw_to_SKIRT(117306, 68, 92281, 3.4270215034484863, [15794.240234375, 29001.4296875, 25658.478515625])
# compare_raw_to_SKIRT(143895, 59, 241195, 2.1625211238861084, [23212.931640625, 5829.36083984375, 32287.716796875])
# compare_raw_to_SKIRT(143904, 59, 219804, 3.1096558570861816, [22059.375, 5862.70068359375, 30750.833984375])
# compare_raw_to_SKIRT(143907, 70, 118289, 2.386402130126953, [20631.26953125, 5513.98046875, 30191.2734375])
# compare_raw_to_SKIRT(167398, 64, 180374, 1.5275959968566895, [18064.6953125, 34308.98046875, 28819.6640625])
# compare_raw_to_SKIRT(167401, 61, 265446, 2.400195837020874, [16342.58203125, 33458.21484375, 28929.4296875])
# compare_raw_to_SKIRT(167409, 45, 114770, 1.6774942874908447, [17938.80078125, 34501.75390625, 27869.64453125])
# compare_raw_to_SKIRT(167412, 59, 133960, 1.4007476568222046, [17189.0703125, 340.875732421875, 30242.794921875])
# compare_raw_to_SKIRT(167418, 43, 87688, 2.1653599739074707, [17343.234375, 462.3441467285156, 30984.125])
# compare_raw_to_SKIRT(167420, 89, 132556, 1.9707319736480713, [17515.84375, 34551.45703125, 28939.12109375])
# compare_raw_to_SKIRT(167421, 64, 147191, 2.874843120574951, [17470.236328125, 808.1504516601562, 30396.40625])
# compare_raw_to_SKIRT(167422, 38, 60195, 1.365623950958252, [17789.162109375, 508.72576904296875, 31122.212890625])
# compare_raw_to_SKIRT(167434, 30, 41306, 0.8866535425186157, [18073.50390625, 82.76520538330078, 31237.720703125])
# compare_raw_to_SKIRT(184932, 55, 251127, 5.7491559982299805, [22664.27734375, 5809.5537109375, 7533.21923828125])
# compare_raw_to_SKIRT(184934, 67, 115153, 6.522535800933838, [23316.083984375, 4778.94091796875, 6079.46923828125])
# compare_raw_to_SKIRT(184943, 56, 91147, 1.3809112310409546, [23565.271484375, 4408.20849609375, 6112.8671875])
# compare_raw_to_SKIRT(184944, 75, 107514, 2.2174360752105713, [23274.404296875, 4606.64111328125, 5867.6474609375])
# compare_raw_to_SKIRT(184945, 44, 66840, 0.9806254506111145, [23049.83984375, 4271.59521484375, 6479.2353515625])
# compare_raw_to_SKIRT(184946, 54, 106129, 1.4133001565933228, [23513.736328125, 4718.4609375, 6318.044921875])
# compare_raw_to_SKIRT(184949, 91, 175621, 1.576864242553711, [24023.900390625, 4817.2509765625, 6129.0087890625])
# compare_raw_to_SKIRT(184952, 62, 109681, 1.2705119848251343, [23388.52734375, 4373.74853515625, 5784.23486328125])
# compare_raw_to_SKIRT(184954, 48, 87982, 2.1390504837036133, [23261.0078125, 4623.61181640625, 6270.169921875])
# compare_raw_to_SKIRT(184963, 46, 71435, 2.1442196369171143, [22951.408203125, 4317.70263671875, 6422.2744140625])
# compare_raw_to_SKIRT(198186, 53, 76760, 3.432344913482666, [32943.54296875, 30044.005859375, 12222.8515625])
# compare_raw_to_SKIRT(198188, 47, 63253, 0.8854263424873352, [32430.66796875, 30168.888671875, 12141.912109375])
# compare_raw_to_SKIRT(198190, 86, 169771, 2.069409132003784, [32689.90234375, 30230.966796875, 12009.375])
# compare_raw_to_SKIRT(198191, 98, 198574, 1.2872415781021118, [31909.6484375, 30052.583984375, 11763.1884765625])
# compare_raw_to_SKIRT(198192, 53, 537954, 3.7013840675354004, [31866.36328125, 29889.646484375, 12350.7568359375])
# compare_raw_to_SKIRT(198193, 56, 81473, 3.1956584453582764, [32388.373046875, 30758.25, 11488.6083984375])
# compare_raw_to_SKIRT(208821, 47, 72169, 1.2715272903442383, [4841.828125, 18173.693359375, 13254.49609375])
# compare_raw_to_SKIRT(220604, 73, 191069, 4.887711048126221, [9179.4326171875, 8770.296875, 2380.82421875])
# compare_raw_to_SKIRT(220606, 59, 122421, 1.4301583766937256, [9278.75, 8931.4189453125, 2554.07958984375])
# compare_raw_to_SKIRT(220609, 44, 79515, 2.054481029510498, [8546.142578125, 8962.072265625, 3025.50146484375])
# compare_raw_to_SKIRT(229934, 44, 246974, 0.6993401050567627, [441.8955993652344, 29535.720703125, 19032.15625])
# compare_raw_to_SKIRT(229938, 51, 139953, 0.9561034440994263, [34123.25390625, 31819.23046875, 18985.26171875])
# compare_raw_to_SKIRT(229944, 97, 218420, 2.7884552478790283, [532.0528564453125, 29835.04296875, 19402.9765625])
# compare_raw_to_SKIRT(229950, 53, 149557, 2.044487953186035, [34616.26953125, 30255.083984375, 19987.369140625])
# compare_raw_to_SKIRT(242789, 56, 180483, 4.562782287597656, [20917.033203125, 19690.8984375, 17111.205078125])
# compare_raw_to_SKIRT(242792, 46, 334957, 1.9745131731033325, [20753.87890625, 20613.111328125, 13708.2607421875])
# compare_raw_to_SKIRT(253865, 72, 166413, 4.1831769943237305, [1287.114990234375, 6028.1357421875, 26487.322265625])
# compare_raw_to_SKIRT(253871, 59, 155058, 2.835629940032959, [1245.82275390625, 5734.55908203125, 27642.740234375])
# compare_raw_to_SKIRT(253873, 81, 188295, 1.6667230129241943, [1306.615966796875, 5946.35595703125, 26629.623046875])
# compare_raw_to_SKIRT(253874, 56, 146205, 1.8737987279891968, [923.546630859375, 5860.53662109375, 27023.455078125])
# compare_raw_to_SKIRT(253884, 46, 98906, 2.8064401149749756, [1221.2706298828125, 5978.21875, 27790.755859375])
# compare_raw_to_SKIRT(264891, 81, 218716, 2.793318748474121, [21557.662109375, 2054.412841796875, 14768.23828125])
# compare_raw_to_SKIRT(264897, 88, 233049, 6.5292792320251465, [21894.140625, 2481.822265625, 14457.3369140625])
# compare_raw_to_SKIRT(275546, 51, 311495, 2.376377820968628, [2971.002197265625, 22772.833984375, 15073.3466796875])
# compare_raw_to_SKIRT(275550, 55, 126584, 2.006826639175415, [3970.59033203125, 22281.12890625, 16033.5146484375])
# compare_raw_to_SKIRT(275553, 52, 137093, 1.8947705030441284, [4409.40576171875, 22134.79296875, 15363.9736328125])
# compare_raw_to_SKIRT(275557, 63, 139489, 0.9585793614387512, [4189.87548828125, 21965.638671875, 15776.208984375])
# compare_raw_to_SKIRT(282780, 28, 117596, 0.5813378691673279, [3971.068359375, 21586.203125, 4909.3154296875])
# compare_raw_to_SKIRT(282790, 84, 226272, 2.696564197540283, [535.025146484375, 21271.201171875, 6125.66015625])
# compare_raw_to_SKIRT(289388, 45, 314678, 1.0405817031860352, [34943.0390625, 14269.501953125, 12258.4033203125])
# compare_raw_to_SKIRT(289389, 76, 216057, 0.8507285118103027, [607.306396484375, 15790.4052734375, 12028.19140625])
# compare_raw_to_SKIRT(289390, 71, 185742, 2.602957248687744, [793.7001342773438, 15254.9921875, 12209.6259765625])
# compare_raw_to_SKIRT(294867, 56, 418992, 1.2861616611480713, [30023.71875, 33106.765625, 7813.1162109375])
# compare_raw_to_SKIRT(294871, 70, 206401, 2.672353506088257, [31423.2734375, 34076.72265625, 7025.68212890625])
# compare_raw_to_SKIRT(294872, 67, 425465, 1.565247893333435, [31639.404296875, 34312.203125, 6507.662109375])
# compare_raw_to_SKIRT(294875, 72, 217916, 1.080196499824524, [31385.982421875, 33745.05078125, 6941.763671875])
# compare_raw_to_SKIRT(294879, 88, 263228, 1.5078620910644531, [31145.669921875, 33754.59765625, 7256.13671875])
# compare_raw_to_SKIRT(300912, 74, 217278, 3.476973056793213, [14647.009765625, 4974.1689453125, 33148.64453125])
# compare_raw_to_SKIRT(307485, 50, 189955, 5.004919052124023, [15832.6171875, 19699.033203125, 22148.666015625])
# compare_raw_to_SKIRT(307487, 49, 253392, 3.6569583415985107, [17992.498046875, 20927.701171875, 22983.119140625])
# compare_raw_to_SKIRT(313694, 90, 287325, 2.6654250621795654, [23885.90625, 25550.6484375, 5011.994140625])
# compare_raw_to_SKIRT(313698, 70, 210810, 2.7307660579681396, [23617.44921875, 26169.783203125, 5029.3720703125])
# compare_raw_to_SKIRT(319734, 48, 118495, 2.9442405700683594, [27362.283203125, 2479.93310546875, 7356.38330078125])
# compare_raw_to_SKIRT(324125, 76, 247426, 1.9658870697021484, [29505.654296875, 19900.181640625, 4887.1865234375])
# compare_raw_to_SKIRT(324126, 72, 227111, 3.995614528656006, [30141.76953125, 20090.212890625, 4894.41455078125])
# compare_raw_to_SKIRT(324129, 80, 263049, 1.069823145866394, [30392.91015625, 19979.162109375, 5153.845703125])
# compare_raw_to_SKIRT(324131, 46, 141121, 0.9843047857284546, [30521.306640625, 20117.3203125, 4721.1318359375])
# compare_raw_to_SKIRT(324132, 54, 164003, 2.43127703666687, [30602.798828125, 20019.61328125, 5136.88330078125])
# compare_raw_to_SKIRT(338447, 37, 140470, 1.1112414598464966, [31704.14453125, 25353.7734375, 30946.36328125])
# compare_raw_to_SKIRT(345873, 63, 428630, 1.4375026226043701, [8328.2373046875, 26634.998046875, 30696.939453125])
# compare_raw_to_SKIRT(355734, 70, 295779, 2.7876968383789062, [25157.583984375, 24984.271484375, 9704.9892578125])
# compare_raw_to_SKIRT(358609, 47, 240423, 4.075259208679199, [5637.240234375, 15550.5146484375, 11248.740234375])
# compare_raw_to_SKIRT(362994, 41, 152734, 2.3454973697662354, [10111.7333984375, 18734.53125, 4733.9736328125])
# compare_raw_to_SKIRT(366407, 26, 35261, 0.6124556064605713, [25529.533203125, 15449.326171875, 31379.794921875])
# compare_raw_to_SKIRT(377656, 74, 305307, 3.070284128189087, [19892.798828125, 20425.18359375, 19490.810546875])
# compare_raw_to_SKIRT(377658, 88, 355368, 1.5309075117111206, [19665.79296875, 20292.55859375, 18992.38671875])
# compare_raw_to_SKIRT(379803, 49, 204838, 4.43634557723999, [5507.28564453125, 7440.9794921875, 30852.828125])
# compare_raw_to_SKIRT(388545, 92, 366043, 1.0344150066375732, [8111.88037109375, 28445.078125, 18729.263671875])
# compare_raw_to_SKIRT(394623, 57, 376850, 0.9632445573806763, [11026.7763671875, 17260.828125, 2638.9716796875])
# compare_raw_to_SKIRT(404818, 62, 358211, 3.768686294555664, [1185.3515625, 10538.3291015625, 34949.78125])
# compare_raw_to_SKIRT(406941, 41, 190681, 2.9298694133758545, [33740.53125, 8565.4951171875, 25618.654296875])
# compare_raw_to_SKIRT(414918, 73, 454041, 1.6958820819854736, [8108.203125, 14466.990234375, 24470.166015625])
# compare_raw_to_SKIRT(416713, 57, 262272, 3.281362533569336, [31893.501953125, 17795.5, 10373.890625])
# compare_raw_to_SKIRT(418335, 54, 332435, 2.0327541828155518, [7831.1142578125, 3975.388916015625, 33039.1015625])
# compare_raw_to_SKIRT(421555, 84, 369589, 6.883451461791992, [33739.55859375, 10110.607421875, 11192.7734375])
# compare_raw_to_SKIRT(421556, 83, 359841, 2.184152841567993, [33833.79296875, 10178.794921875, 11289.9677734375])
# compare_raw_to_SKIRT(428177, 80, 396455, 3.730598211288452, [25678.8359375, 32901.51171875, 15656.0751953125])
# compare_raw_to_SKIRT(434356, 49, 278412, 2.151136875152588, [14733.322265625, 2557.226806640625, 19166.357421875])
# compare_raw_to_SKIRT(445626, 62, 295572, 3.969383955001831, [23633.310546875, 23665.021484375, 4890.91015625])
# compare_raw_to_SKIRT(446665, 53, 309497, 0.9388160109519958, [14660.34375, 19637.927734375, 21082.787109375])
# compare_raw_to_SKIRT(447914, 45, 179454, 1.921480655670166, [9991.69140625, 31006.05859375, 19240.74609375])
# compare_raw_to_SKIRT(450916, 56, 301528, 4.137515544891357, [12897.134765625, 16154.927734375, 29370.75])
# compare_raw_to_SKIRT(454172, 68, 398973, 2.5442001819610596, [33067.3125, 18051.40625, 5096.1123046875])
# compare_raw_to_SKIRT(457431, 57, 308972, 2.9916603565216064, [1892.830078125, 19613.013671875, 5747.8662109375])
# compare_raw_to_SKIRT(459558, 56, 360118, 0.9018786549568176, [8241.7099609375, 26125.974609375, 27421.431640625])
# compare_raw_to_SKIRT(466549, 46, 252055, 1.1439653635025024, [9726.927734375, 11637.38671875, 3618.622802734375])
# compare_raw_to_SKIRT(475619, 31, 100674, 1.3586828708648682, [14452.171875, 2939.702880859375, 32662.431640625])
# compare_raw_to_SKIRT(480803, 79, 405593, 2.296109199523926, [20308.201171875, 15898.251953125, 2483.97509765625])
# compare_raw_to_SKIRT(482155, 74, 370008, 5.354395866394043, [28428.658203125, 11384.9931640625, 29686.79296875])
# compare_raw_to_SKIRT(482891, 49, 317116, 0.6739818453788757, [8914.0654296875, 1.1891578435897827, 21530.509765625])
# compare_raw_to_SKIRT(483594, 63, 338236, 3.5494449138641357, [8922.822265625, 14293.255859375, 26284.55859375])
# compare_raw_to_SKIRT(484448, 47, 229841, 1.5786393880844116, [28339.68359375, 11355.0634765625, 3349.807861328125])
# compare_raw_to_SKIRT(486917, 64, 383587, 0.9261046648025513, [19670.880859375, 11504.57421875, 2759.96044921875])
# compare_raw_to_SKIRT(486919, 88, 447553, 0.8136436343193054, [20240.453125, 11896.216796875, 2531.94482421875])
# compare_raw_to_SKIRT(495451, 53, 296260, 0.8403759598731995, [34916.625, 31973.77734375, 16219.4853515625])
# compare_raw_to_SKIRT(496186, 51, 267448, 1.174773097038269, [11476.634765625, 29814.6875, 26593.642578125])
# compare_raw_to_SKIRT(503987, 40, 242135, 2.5162928104400635, [34925.921875, 21665.421875, 2332.586669921875])
# compare_raw_to_SKIRT(504559, 49, 279311, 1.0440152883529663, [13863.2880859375, 21591.70703125, 12750.03515625])
# compare_raw_to_SKIRT(507294, 79, 460075, 1.9364205598831177, [16731.833984375, 31390.927734375, 27040.513671875])
# compare_raw_to_SKIRT(507784, 54, 354136, 0.8554603457450867, [22225.37109375, 25135.744140625, 21645.037109375])
# compare_raw_to_SKIRT(508539, 72, 444540, 1.544507384300232, [27210.1640625, 12548.9814453125, 3522.6484375])
# compare_raw_to_SKIRT(513105, 94, 497904, 2.53871750831604, [34779.13671875, 26471.1796875, 26439.052734375])
# compare_raw_to_SKIRT(514272, 90, 487648, 2.5328171253204346, [15803.55859375, 25169.48828125, 32039.822265625])
# compare_raw_to_SKIRT(515296, 53, 295759, 1.4500712156295776, [7306.271484375, 2747.469482421875, 24515.1484375])
# compare_raw_to_SKIRT(516101, 79, 445410, 2.698007106781006, [21145.89453125, 18159.953125, 23016.107421875])
# compare_raw_to_SKIRT(516760, 40, 208817, 0.7851501703262329, [5418.78564453125, 9167.400390625, 21037.1875])
# compare_raw_to_SKIRT(524506, 45, 289349, 1.0616304874420166, [18507.498046875, 15147.849609375, 8127.251953125])
# compare_raw_to_SKIRT(526879, 60, 435509, 2.031116485595703, [594.8347778320312, 23681.908203125, 31970.3671875])
# compare_raw_to_SKIRT(529365, 78, 432059, 3.0339550971984863, [23975.82421875, 19487.3046875, 11709.255859375])
# compare_raw_to_SKIRT(531320, 72, 435217, 0.8108838200569153, [19841.197265625, 29619.501953125, 17910.669921875])
# compare_raw_to_SKIRT(536654, 76, 433088, 1.519272804260254, [2364.902587890625, 5387.15380859375, 1729.4039306640625])
# compare_raw_to_SKIRT(539667, 78, 466172, 4.427576065063477, [30962.96484375, 17877.24609375, 9677.2939453125])
# compare_raw_to_SKIRT(540082, 83, 469748, 3.325726270675659, [33698.82421875, 5794.43408203125, 28466.373046875])
# compare_raw_to_SKIRT(541218, 45, 303267, 1.368080496788025, [29840.169921875, 31575.064453125, 4590.26708984375])
# compare_raw_to_SKIRT(545003, 48, 395032, 0.8729383945465088, [5722.41796875, 1129.3597412109375, 24168.025390625])
# compare_raw_to_SKIRT(545703, 47, 325578, 1.0640292167663574, [22145.544921875, 24657.17578125, 22314.259765625])
# compare_raw_to_SKIRT(546870, 66, 411132, 2.682202100753784, [18950.51171875, 23546.283203125, 1906.41357421875])
# compare_raw_to_SKIRT(547545, 31, 204097, 1.1967297792434692, [9064.7802734375, 30419.783203125, 20829.953125])
# compare_raw_to_SKIRT(548151, 71, 422477, 0.8808767199516296, [18754.5078125, 22242.310546875, 24388.2421875])
# compare_raw_to_SKIRT(551541, 90, 530508, 0.9603878259658813, [676.5078125, 22394.4609375, 33638.57421875])
# compare_raw_to_SKIRT(555013, 56, 396417, 2.1972551345825195, [2932.37109375, 10982.5556640625, 1354.7694091796875])
# compare_raw_to_SKIRT(555815, 97, 552094, 0.6797944903373718, [21319.75390625, 26392.63671875, 26992.08984375])
# compare_raw_to_SKIRT(562029, 67, 444677, 2.0504655838012695, [7840.37841796875, 22433.306640625, 29135.87890625])
# compare_raw_to_SKIRT(564498, 58, 390982, 0.7486891150474548, [23029.587890625, 16468.25390625, 8772.826171875])
# compare_raw_to_SKIRT(567607, 91, 542669, 2.183142900466919, [24061.4296875, 5968.46875, 2243.404052734375])
# compare_raw_to_SKIRT(568646, 73, 471214, 1.8237674236297607, [18012.349609375, 19151.712890625, 23031.916015625])
# compare_raw_to_SKIRT(572328, 89, 527832, 2.9482383728027344, [7543.607421875, 26028.177734375, 27021.41796875])
# compare_raw_to_SKIRT(574037, 74, 500301, 2.0641438961029053, [27758.177734375, 1943.283447265625, 6960.0068359375])
# compare_raw_to_SKIRT(576516, 48, 337779, 1.0064384937286377, [9289.998046875, 12538.9248046875, 9136.7646484375])
# compare_raw_to_SKIRT(576705, 96, 583023, 0.6625633835792542, [10003.2705078125, 10086.1123046875, 28161.83984375])
# compare_raw_to_SKIRT(580907, 79, 501422, 4.120691299438477, [22916.15234375, 23667.650390625, 19517.50390625])
# compare_raw_to_SKIRT(584007, 74, 489691, 1.0839357376098633, [28515.037109375, 21472.6875, 4920.64453125])
# compare_raw_to_SKIRT(584724, 71, 485061, 1.0511302947998047, [2909.214111328125, 32453.072265625, 17644.837890625])
# compare_raw_to_SKIRT(588399, 73, 476962, 1.3440866470336914, [13380.865234375, 16539.662109375, 6907.58837890625])
# compare_raw_to_SKIRT(588831, 88, 543556, 2.90720272064209, [19624.541015625, 1594.2462158203125, 26156.87109375])
# compare_raw_to_SKIRT(591641, 89, 563245, 4.813155174255371, [22125.880859375, 26106.236328125, 23046.564453125])
# compare_raw_to_SKIRT(591796, 66, 463766, 0.7126250267028809, [15423.072265625, 21101.23828125, 16526.181640625])
# compare_raw_to_SKIRT(592021, 81, 515806, 1.0193214416503906, [21901.40234375, 4012.403564453125, 17790.822265625])
# compare_raw_to_SKIRT(593694, 57, 414095, 1.0779483318328857, [33979.21484375, 6133.1357421875, 31177.248046875])
# compare_raw_to_SKIRT(596401, 85, 607354, 1.0924177169799805, [5261.21044921875, 11739.267578125, 8969.2802734375])
# compare_raw_to_SKIRT(606223, 70, 484436, 0.6359179019927979, [19965.978515625, 10756.1572265625, 18495.34765625])
# compare_raw_to_SKIRT(607654, 55, 427014, 2.374330520629883, [3014.030517578125, 20786.876953125, 14766.6435546875])
# compare_raw_to_SKIRT(609710, 93, 593479, 1.1644617319107056, [1106.1356201171875, 27546.36328125, 16162.6728515625])
# compare_raw_to_SKIRT(610532, 87, 570004, 1.6736485958099365, [8859.7041015625, 21079.33984375, 12697.9716796875])
# compare_raw_to_SKIRT(613192, 60, 34598, 2.361635208129883, [7416.59765625, 25875.2890625, 22808.154296875])
# compare_raw_to_SKIRT(623367, 68, 493500, 1.0340344905853271, [7665.25732421875, 24403.033203125, 33188.5546875])
# compare_raw_to_SKIRT(625281, 83, 566307, 2.681144952774048, [16021.517578125, 7548.861328125, 24213.03515625])
# compare_raw_to_SKIRT(626287, 97, 621001, 4.127569675445557, [12597.595703125, 26697.8203125, 24813.966796875])
# compare_raw_to_SKIRT(626583, 92, 608571, 5.476966857910156, [30519.021484375, 9634.5693359375, 23674.69140625])
# compare_raw_to_SKIRT(634046, 90, 608951, 5.083594799041748, [12440.5322265625, 21563.009765625, 20298.564453125])
# compare_raw_to_SKIRT(634631, 95, 624747, 2.454105854034424, [8585.830078125, 26196.576171875, 29338.873046875])
# compare_raw_to_SKIRT(637199, 75, 553549, 0.8962335586547852, [2266.245849609375, 29348.4375, 10458.58984375])
# compare_raw_to_SKIRT(642671, 62, 485960, 1.3010234832763672, [16822.9921875, 34268.88671875, 29315.234375])
# compare_raw_to_SKIRT(651449, 87, 644801, 1.8590806722640991, [19370.79296875, 1428.2315673828125, 27299.810546875])
# compare_raw_to_SKIRT(657228, 84, 597037, 2.6778881549835205, [16712.91015625, 25100.27734375, 23662.9375])
# compare_raw_to_SKIRT(661599, 99, 661599, 1.5164704322814941, [30470.927734375, 15163.5615234375, 7637.79248046875])
# compare_raw_to_SKIRT(663512, 86, 617326, 4.313283443450928, [15500.2119140625, 21424.39453125, 15992.43359375])
# compare_raw_to_SKIRT(671746, 98, 669958, 1.966431736946106, [3420.68212890625, 3161.719970703125, 21596.59375])
# compare_raw_to_SKIRT(672526, 86, 630729, 1.0049163103103638, [22685.25390625, 14111.4794921875, 17170.87109375])
# compare_raw_to_SKIRT(678001, 94, 650019, 1.8563545942306519, [5228.0283203125, 31868.65625, 22707.541015625])
# compare_raw_to_SKIRT(692981, 62, 18, 1.379520297050476, [6690.6650390625, 23990.37890625, 21021.9296875])
# compare_raw_to_SKIRT(698432, 99, 698432, 2.4620347023010254, [24126.328125, 18035.5390625, 31814.82421875])
# compare_raw_to_SKIRT(708459, 84, 641448, 3.796165704727173, [4081.84521484375, 21873.060546875, 15929.71484375])
# compare_raw_to_SKIRT(719882, 80, 84054, 1.214719295501709, [27489.927734375, 8014.7880859375, 4025.9189453125])
# compare_raw_to_SKIRT(731524, 92, 714626, 0.8565922975540161, [8709.58984375, 8600.4013671875, 2310.703369140625])
# compare_raw_to_SKIRT(738736, 73, 92832, 0.8492041826248169, [15969.546875, 29337.400390625, 25776.44140625])
# compare_raw_to_SKIRT(799553, 73, 498318, 5.276242733001709, [34998.94921875, 17757.357421875, 23476.44140625])
