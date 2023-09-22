
from os.path import exists
import numpy as np

import astropy.units as u
# import emcee
import h5py
from scipy.optimize import minimize
from scipy.special import gamma, gammainc

from core import add_dataset, find_nearest, get_particle_positions, get_sf_particles
import plotting as plt

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=RuntimeWarning)
warnings.simplefilter('ignore', category=AstropyWarning)

def bin_values_and_fit(edges, mids, masses, rs, Re, Ie, nn, nsteps=1000,
                       outfile=None, save_plot=True, plotfile=None) :
    
    # calculate the mass surface density values
    mass_in_bins, areas = [], []
    for start, end in zip(edges, edges[1:]) :
        mass_in_bin = np.sum(masses[(rs >= start) & (rs < end)])
        area = np.pi*(np.square(end) - np.square(start))
        mass_in_bins.append(mass_in_bin)
        areas.append(area)
    values = np.array(mass_in_bins)/np.array(areas)
    values = np.log10(values)
    
    # mask out infs and NaNs
    mask = np.isfinite(values)
    fit_mids = mids[mask]
    fit_values = values[mask]
    
    # fit the data using a Sersic profile, with input guesses for Re and Ie,
    # based on 2D values and the total 3D stellar mass
    res = minimize(log_likelihood, x0=[Re, Ie, nn], args=(fit_mids, fit_values),
        bounds=[(0.5*Re, 2*Re), (0.1*Ie, 10*Ie), (0.36, 10)],
        method='Nelder-Mead', options={'maxiter':10000})
    Re_fit, Ie_fit, n_fit = res.x
    '''
    if exists(outfile) :
        sampler = emcee.backends.HDFBackend(outfile)
    else :
        # run an MCMC sampler
        res = np.array([Re_fit, Ie_fit, n_fit])
        pos = res + 1e-4*np.random.rand(32, 3)
        nwalkers, ndim = pos.shape
        
        # set up the backend, to save chains to file
        backend = emcee.backends.HDFBackend(outfile)
        backend.reset(nwalkers, ndim)
        
        # initialize the sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                                        args=(fit_mids, fit_values, Re, Ie),
                                        backend=backend)
        
        # run the sampler
        sampler.run_mcmc(pos, nsteps, progress=True)
        
        # plot the chains to visually check convergence
        plt.plot_chains(sampler.get_chain(), ndim,
            [r'$R_{\rm e}$', r'$I_{\rm e}$', r'$n$'], save=save_plot,
            outfile=plotfile)
    '''
    return Re_fit, Ie_fit, n_fit, values, np.nan #sampler

def check_fit(mids, values, sf_values, Re_fit, Ie_fit, n_fit, sf_Re_fit,
              sf_Ie_fit, sf_n_fit, title, mini, maxi, scale, outfile, save) :
    
    # plot the fits for visual inspection
    xs = np.linspace(mini, maxi, 10000)
    fit = np.log10(sersic_fxn(xs, Re_fit, Ie_fit, n_fit))
    sf_fit = np.log10(sersic_fxn(xs, sf_Re_fit, sf_Ie_fit, sf_n_fit))
    
    labels = ['data', 'SF',
              (r'fit, $R_{\rm e}=$' + '{:.2f} kpc, '.format(Re_fit) + r'$n=$' +
               '{:.2f}'.format(n_fit)),
              (r'SF fit, $R_{\rm e}=$' + '{:.2f} kpc, '.format(sf_Re_fit) +
               r'$n=$' + '{:.2f}'.format(sf_n_fit))]
    
    try :
        ymin = min(0.8*np.min(values),
                   0.8*np.min(sf_values[np.isfinite(sf_values)]))
        ymax = max(1.25*np.max(values),
                   1.25*np.max(sf_values[np.isfinite(sf_values)]))
        
        plt.plot_simple_multi([mids, mids, xs, xs],
            [values, sf_values, fit, sf_fit], labels, ['k', 'b', 'r', 'grey'],
            ['o', 'o', '', ''], ['', '', '-', '-'], [1, 1, 1, 1],
            xlabel=r'$r$ (ckpc)', ylabel=r'$\log({\rm M}_{\odot}~{\rm kpc}^{-2})$',
            title=title, xmin=mini, xmax=maxi, ymin=ymin, ymax=ymax, scale=scale,
            outfile=outfile, save=save)
    except ValueError :
        pass
    
    return

def compare_input_and_fitted_values(sample='control', version='A', nBins=10,
                                    save_plots=False) :
    
    outDir = 'TNG50-1/Sersic_fits/TNG50-1_99_'
    if sample == 'control' :
        outfile = outDir + 'control_Sersic_fits_v{}_{}_bins.hdf5'.format(
            version, nBins)
    if sample == 'quenched' :
        outfile = outDir + 'quenched_Sersic_fits_v{}_{}_bins.hdf5'.format(
            version, nBins)
    
    if not exists(outfile) :
        if sample == 'control' :
            (logM, R50s, R90s, sf_R50s, sf_R90s, Iinits, fit_Res, fit_Ies, fit_ns,
             sf_fit_Res, sf_fit_Ies, sf_fit_ns) = get_control_sample_Sersic_fits(
                 version=version, nBins=nBins)
        if sample == 'quenched' :
            (logM, R50s, R90s, sf_R50s, sf_R90s, Iinits, fit_Res, fit_Ies, fit_ns,
             sf_fit_Res, sf_fit_Ies, sf_fit_ns) = get_quenched_sample_Sersic_fits(
                 version=version, nBins=nBins)
        
        with h5py.File(outfile, 'w') as hf :
            add_dataset(hf, logM, 'logM')
            add_dataset(hf, R50s, 'R50')
            add_dataset(hf, R90s, 'R90')
            add_dataset(hf, sf_R50s, 'sf_R50')
            add_dataset(hf, sf_R90s, 'sf_R90')
            add_dataset(hf, Iinits, 'Iinits')
            add_dataset(hf, fit_Res, 'fit_Res')
            add_dataset(hf, fit_Ies, 'fit_Ies')
            add_dataset(hf, fit_ns, 'fit_ns')
            add_dataset(hf, sf_fit_Res, 'sf_fit_Res')
            add_dataset(hf, sf_fit_Ies, 'sf_fit_Ies')
            add_dataset(hf, sf_fit_ns, 'sf_fit_ns')
    else :
        with h5py.File(outfile, 'r') as hf :
            logM = hf['logM'][:]
            R50s = hf['R50'][:]
            R90s = hf['R90'][:]
            sf_R50s = hf['sf_R50'][:]
            sf_R90s = hf['sf_R90'][:]
            Iinits = hf['Iinits'][:]
            fit_Res = hf['fit_Res'][:]
            fit_Ies = hf['fit_Ies'][:]
            fit_ns = hf['fit_ns'][:]
            sf_fit_Res = hf['sf_fit_Res'][:]
            sf_fit_Ies = hf['sf_fit_Ies'][:]
            sf_fit_ns = hf['sf_fit_ns'][:]
        
        # fit_logM, fit_R90s = get_integrated_quantities(fit_Res, fit_Ies, fit_ns)
        
        # plot_recoveries_with_nBins(logM, R50s, R90s, Iinits, fit_logM,
        #     fit_Res, fit_R90s, fit_Ies, fit_ns, sample=sample,
        #     version=version, nBins=nBins, save_plots=save_plots)
    
    return

def get_bn(nn) :
    bn = 1.9992*nn - 0.3271 # Capaccioli 1989 approximation, valid for
                            # 0.5 < n < 10
    
    # Ciotti & Bertin 1999 approximation, valid for n > 0.36
    bn = (2*nn - 1/3 + 4/(405*nn) + 46/(25515*np.square(nn)) +
          131/(1148175*np.power(nn, 3)) - 2194697/(30690717750*np.power(nn, 4)))
    
    return bn

def get_control_sample_Sersic_fits(version='A', nBins=10) :
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        times = hf['times'][:]
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        # Res = hf['Re'][:]
        centers = hf['centers'][:]
        # cluster = hf['cluster'][:]
        # hmGroup = hf['hm_group'][:]
        # lmGroup = hf['lm_group'][:]
        # field = hf['field'][:]
        comparison = hf['comparison'][:]
        i75s = hf['i75s'][:].astype(int)
    
    with h5py.File('TNG50-1/TNG50-1_99_diagnostics(t)_2d.hdf5', 'r') as hf :
        # sf_mass_within_1kpc = hf['sf_mass_within_1kpc'][:]
        # sf_mass_within_tenthRe = hf['sf_mass_within_tenthRe'][:]
        # sf_mass_tot = hf['sf_mass'][:]
        # R10s = hf['R10'][:]
        R50s = hf['R50'][:]
        R90s = hf['R90'][:]
        # sf_R10s = hf['sf_R10'][:]
        sf_R50s = hf['sf_R50'][:]
        sf_R90s = hf['sf_R90'][:]
    
    # get snapshots corresponding to 75% through the quenching episode
    i75s = i75s[comparison]
    
    # mask values to the control sample at the correct snapshot
    firstDim = np.arange(278)
    c_times = times[i75s]
    c_subIDs = (subIDs[comparison])[firstDim, i75s]
    c_logM = (logM[comparison])[firstDim, i75s]
    # c_Res = (Res[comparison])[firstDim, i75s]
    c_centers = (centers[comparison])[firstDim, i75s]
    
    # c_R10s = (R10s[comparison])[firstDim, i75s]
    c_R50s = (R50s[comparison])[firstDim, i75s]
    c_R90s = (R90s[comparison])[firstDim, i75s]
    
    # c_sf_R10s = (sf_R10s[comparison])[firstDim, i75s]
    c_sf_R50s = (sf_R50s[comparison])[firstDim, i75s]
    c_sf_R90s = (sf_R90s[comparison])[firstDim, i75s]
    
    # guess a value for Ie using n=1 and the total mass
    c_n = 1
    c_bn = get_bn(c_n)
    c_Iinits = (np.power(c_bn, 2*c_n)/(2*np.pi*c_n*np.exp(c_bn))*
                np.power(10, c_logM)/np.square(c_R50s)/gamma(2*c_n))
    
    # 2D values will be higher because R50_2D is generally smaller than Re_3D
    # c_Iinits_3D = (np.power(c_bn, 2*c_n)/(2*np.pi*c_n*np.exp(c_bn))*
    #                np.power(10, c_logM)/np.square(c_Res)/gamma(2*c_n))
    # plt.plot_simple_dumb(c_Iinits_3D, c_Iinits, label='data',
    #     xlabel=r'$\log(I_{\rm e, init}/{\rm M}_{\odot}~{\rm kpc}^{-2})_{\rm 3D}$',
    #     ylabel=r'$\log(I_{\rm e, init}/{\rm M}_{\odot}~{\rm kpc}^{-2})_{\rm 2D}$',
    #     xmin=2e6, xmax=4e10, ymin=2e6, ymax=4e10, scale='log')
    
    (fit_Res, fit_Ies, fit_ns, sf_fit_Res, sf_fit_Ies,
     sf_fit_ns) = get_Sersic_fits(c_times, c_subIDs, i75s, c_logM, c_R50s,
        c_sf_R50s, c_centers, c_Iinits, c_n, version=version, nBins=nBins)
    
    return (c_logM, c_R50s, c_R90s, c_sf_R50s, c_sf_R90s, c_Iinits, fit_Res,
            fit_Ies, fit_ns, sf_fit_Res, sf_fit_Ies, sf_fit_ns)

def get_integrated_quantities(fit_Res, fit_Ies, fit_ns, version='A') :
    
    fit_R90s = np.full(278, np.nan)
    
    if version == 'A' :
        mini = 0.01
    if version == 'B' :
        mini = 0
    
    # calculate the total mass using the analytic expression, from
    # Graham & Driver 2005
    bn = get_bn(fit_ns)
    Mstar_analytic = (2*np.pi*fit_ns*np.exp(bn)/np.power(bn, 2*fit_ns)*
                      fit_Ies*np.square(fit_Res)*gamma(2*fit_ns))
    
    # calculate the mass out to a given radius using the analytic expression
    # from Graham & Driver 2005
    for i, (Re_fit, Ie_fit, n_fit, Mstar) in enumerate(zip(fit_Res, fit_Ies,
        fit_ns, Mstar_analytic)) :
        bn = get_bn(n_fit)
        xs = np.linspace(mini, 100*Re_fit, 10000)
        xx = bn*np.power(xs/Re_fit, 1/n_fit)
        partial_analytic = (2*np.pi*n_fit*np.exp(bn)/np.power(bn, 2*n_fit)*
                            Ie_fit*np.square(Re_fit)*gamma(2*n_fit)*
                            gammainc(2*n_fit, xx))
        
        # check the radius that encloses 50% of the stellar mass -> these
        # differences are generally very small, <~1e-3
        # check_R50 = xs[(np.abs(partial_analytic/Mstar_analytic - 0.5)).argmin()]
        
        # find the radius that encloses 90% of the stellar mass
        fit_R90s[i] = xs[(np.abs(partial_analytic/Mstar - 0.9)).argmin()]
    
    return np.log10(Mstar_analytic), fit_R90s

def get_Sersic_fits(times, subIDs, indices, logM, R50s, sf_R50s, centers,
                    Ie_inits, n_init, version='A', nBins=10) :
    
    empty = np.full(278, np.nan)
    fit_Res = empty.copy()
    fit_Ies = empty.copy()
    fit_ns = empty.copy()
    sf_fit_Res = empty.copy()
    sf_fit_Ies = empty.copy()
    sf_fit_ns = empty.copy()
    
    for i, (time, subID, snap, mass, R50, sf_R50, center, Iinit) in enumerate(
            zip(times, subIDs, indices, logM, R50s, sf_R50s, centers, Ie_inits)) :
        
        # set the version and get relevant parameters, including bin edges
        if version == 'A' :
            edges = np.logspace(-2, np.log10(5*R50), nBins + 1)
            scale, mini, maxi = 'log', 0.01, 5*R50
        if version == 'B' :
            edges = np.linspace(0, 5*R50, nBins + 1)
            scale, mini, maxi = 'linear', 0, 5*R50
        
        # set the midpoints of the bins
        mids = edges[:-1] + np.diff(edges)/2
        
        # get all particles
        ages, masses, dx, dy, _ = get_particle_positions('TNG50-1', 99, snap,
                                                         subID, center)
        
        # get projected distances
        rs = np.sqrt(np.square(dx) + np.square(dy))
        
        # make distance log-friendly
        if version == 'A' :
            rs[rs < 0.01] = 0.01
        
        # bin values and fit with a Sersic profile
        nsteps = 10000
        outfile = 'TNG50-1/MCMC/subID_{}_stellar_{}_steps.h5'.format(
            subID, nsteps)
        plotfile = 'TNG50-1/figures/MCMC/subID_{}_stellar_{}_steps.png'.format(
            subID, nsteps)
        
        Re_fit, Ie_fit, n_fit, values, sampler = bin_values_and_fit(
            edges, mids, masses, rs, R50, Iinit, n_init, nsteps=nsteps,
            outfile=outfile, save_plot=True, plotfile=plotfile)
        
        '''
        tau = sampler.get_autocorr_time(quiet=True, tol=0)
        samples = sampler.get_chain(discard=int(2*np.max(tau)), flat=True,
                                    thin=int(0.5*np.min(tau)))
        
        nsamples = len(samples)
        xs = np.linspace(mini, maxi, 1000)
        ys = np.full((nsamples, 1000), np.nan)
        for j in range(nsamples) :
            fit = np.log10(sersic_fxn(xs, samples[j, 0], samples[j, 1],
                                      samples[j, 2]))
            ys[j] = fit
        plt.plot_simple_many(xs, ys, xlabel=r'$r$ (ckpc)',
            ylabel=r'$\log({\rm M}_{\odot}~{\rm kpc}^{-2})$',
            xmin=mini, xmax=maxi, ymin=2.3800309245872957,
            ymax=12.686962258307542, scale=scale,
            title='subID {}'.format(subID))
        '''
        
        # place fitted values into arrays
        fit_Res[i], fit_Ies[i], fit_ns[i] = Re_fit, Ie_fit, n_fit
        
        # get the SF particles
        _, sf_masses, sf_rs = get_sf_particles(ages, masses, rs, time,
                                               delta_t=100*u.Myr)
        
        # guess a value for Ie using n=1 and the total SF mass
        sf_mass = np.log10(np.sum(sf_masses))
        nn = 1
        bn = get_bn(nn)
        sf_Iinit = (np.power(bn, 2*nn)/(2*np.pi*nn*np.exp(bn))*
                     np.power(10, sf_mass)/np.square(sf_R50)/gamma(2*nn))
        
        # bin SF values and fit with a Sersic profile
        nsteps = 1000
        outfile = 'TNG50-1/MCMC/subID_{}_SF_{}_steps.h5'.format(subID, nsteps)
        plotfile = 'TNG50-1/figures/MCMC/subID_{}_SF_{}_steps.png'.format(
            subID, nsteps)
        
        sf_Re_fit, sf_Ie_fit, sf_n_fit, sf_values, sf_sampler = bin_values_and_fit(
            edges, mids, sf_masses, sf_rs, sf_R50, sf_Iinit, nn,
            nsteps=nsteps, outfile=outfile, save_plot=True, plotfile=plotfile)
        
        # place fitted values into arrays
        sf_fit_Res[i], sf_fit_Ies[i] = sf_Re_fit, sf_Ie_fit
        sf_fit_ns[i] = sf_n_fit
        
        # plot for visual inspection
        logM_label = r'logM$=$'
        Re_label = r'$R_{\rm e}=$'
        sf_Re_label = r'$R_{\rm e,SF}=$'
        title = 'subID {}, {}{:.2f}, {}{:.2f} kpc, {}{:.2f} kpc, ver {}'.format(
            subID, logM_label, mass, Re_label, R50, sf_Re_label, sf_R50, version)
        outfile = 'TNG50-1/figures/Sersic_binning/subID_{}_ver{}_{}bins.png'.format(
            subID, version, nBins)
        check_fit(mids, values, sf_values, Re_fit, Ie_fit, n_fit, sf_Re_fit,
                  sf_Ie_fit, sf_n_fit, title, mini, maxi, scale, outfile, True)
    
    return fit_Res, fit_Ies, fit_ns, sf_fit_Res, sf_fit_Ies, sf_fit_ns

def get_quenched_sample_Sersic_fits(version='A', nBins=10) :
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        times = hf['times'][:]
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        # Res = hf['Re'][:]
        centers = hf['centers'][:]
        tonsets = hf['onset_times'][:]
        tterms = hf['termination_times'][:]
        quenched = hf['quenched'][:]
        # cluster = hf['cluster'][:]
        # hmGroup = hf['hm_group'][:]
        # lmGroup = hf['lm_group'][:]
        # field = hf['field'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_diagnostics(t)_2d.hdf5', 'r') as hf :
        # sf_mass_within_1kpc = hf['sf_mass_within_1kpc'][:]
        # sf_mass_within_tenthRe = hf['sf_mass_within_tenthRe'][:]
        # sf_mass_tot = hf['sf_mass'][:]
        # R10s = hf['R10'][:]
        R50s = hf['R50'][:]
        R90s = hf['R90'][:]
        # sf_R10s = hf['sf_R10'][:]
        sf_R50s = hf['sf_R50'][:]
        sf_R90s = hf['sf_R90'][:]
    
    # with h5py.File('TNG50-1/TNG50-1_99_mechanism.hdf5', 'r') as hf :
    #     outside_in = hf['outside-in'][:] # 109
    #     inside_out = hf['inside-out'][:] # 103
    #     uniform = hf['uniform'][:]       # 8
    #     ambiguous = hf['ambiguous'][:]   # 58
    
    # select the quenched galaxies
    quenched = (logM[:, -1] >= 9.5) & quenched # 278 entries, but len(mask) = 8260
    
    # get snapshots corresponding to 75% through the quenching episode
    imajorities = np.array(find_nearest(times, tonsets + 0.75*(tterms - tonsets)))
    imajorities = imajorities[quenched]
    
    # get the quenching mechanisms
    # mech = np.array([4*outside_in, 3*inside_out, 2*uniform, ambiguous]).T
    # mech = np.sum(mech, axis=1)
    # mech = mech[quenched]
    
    # mask values to the quenched sample at the correct snapshot
    firstDim = np.arange(278)
    q_times = times[imajorities]
    q_subIDs = (subIDs[quenched])[firstDim, imajorities]
    q_logM = (logM[quenched])[firstDim, imajorities]
    # q_Res = (Res[quenched])[firstDim, imajorities]
    q_centers = (centers[quenched])[firstDim, imajorities]
    
    # q_R10s = (R10s[quenched])[firstDim, imajorities]
    q_R50s = (R50s[quenched])[firstDim, imajorities]
    q_R90s = (R90s[quenched])[firstDim, imajorities]
    
    # q_sf_R10s = (sf_R10s[quenched])[firstDim, imajorities]
    q_sf_R50s = (sf_R50s[quenched])[firstDim, imajorities]
    q_sf_R90s = (sf_R90s[quenched])[firstDim, imajorities]
    
    # guess a value for Ie using n=3 and the total mass
    q_n = 3
    q_bn = get_bn(q_n)
    q_Iinits = (np.power(q_bn, 2*q_n)/(2*np.pi*q_n*np.exp(q_bn))*
                np.power(10, q_logM)/np.square(q_R50s)/gamma(2*q_n))
    
    # 2D values will be higher because R50_2D is generally smaller than Re_3D
    # q_Iinits_3D = (np.power(q_bn, 2*q_n)/(2*np.pi*q_n*np.exp(q_bn))*
    #                np.power(10, q_logM)/np.square(q_Res)/gamma(2*q_n))
    # plt.plot_simple_dumb(q_Iinits_3D, q_Iinits, label='data',
    #     xlabel=r'$\log(I_{\rm e, init}/{\rm M}_{\odot}~{\rm kpc}^{-2})_{\rm 3D}$',
    #     ylabel=r'$\log(I_{\rm e, init}/{\rm M}_{\odot}~{\rm kpc}^{-2})_{\rm 2D}$',
    #     xmin=2e6, xmax=4e10, ymin=2e6, ymax=4e10, scale='log')
    
    (fit_Res, fit_Ies, fit_ns, sf_fit_Res, sf_fit_Ies,
     sf_fit_ns) = get_Sersic_fits(q_times, q_subIDs, imajorities, q_logM,
        q_R50s, q_sf_R50s, q_centers, q_Iinits, q_n, version=version,
        nBins=nBins)
    
    return (q_logM, q_R50s, q_R90s, q_sf_R50s, q_sf_R90s, q_Iinits, fit_Res,
            fit_Ies, fit_ns, sf_fit_Res, sf_fit_Ies, sf_fit_ns)

def log_likelihood(theta, RR, SB) :
    
    if np.any(np.array(theta) <= 0) :
        return np.inf
    
    model = sersic_fxn(RR, theta[0], theta[1], theta[2])
    
    return np.sqrt(np.mean(np.square(SB - np.log10(model))))

def log_prior(theta, Re, Ie) :
    
    if ((0.5*Re < theta[0] < 2*Re) and (0.1*Ie < theta[1] < 10*Ie) and
        (0.36 < theta[2] < 10)) :
        return 0.0
    
    return -np.inf

def log_prob(theta, RR, SB, Re, Ie) :
    
    lp = log_prior(theta, Re, Ie)
    
    if not np.isfinite(lp) :
        return -np.inf
    
    return lp + log_likelihood(theta, RR, SB)

def plot_recoveries_with_nBins(logM, R50s, R90s, Ie_inits, fit_logM, fit_Res,
                               fit_R90s, fit_Ies, fit_ns, sample='control',
                               version='A', nBins=10, save_plots=False) :
    
    # print(np.min(np.log10(R50s)), np.max(np.log10(R50s)))
    # print(np.min(np.log10(fit_Res)), np.max(np.log10(fit_Res)))
    # print(np.min(np.log10(R90s)), np.max(np.log10(R90s)))
    # print(np.min(np.log10(fit_R90s)), np.max(np.log10(fit_R90s)))
    # print(np.min(np.log10(Ie_inits)), np.max(np.log10(Ie_inits)))
    # print(np.min(np.log10(fit_Ies)), np.max(np.log10(fit_Ies)))
    # print(np.min(logM), np.max(logM))
    # print(np.min(fit_logM), np.max(fit_logM))
    # print()
    
    outDir = 'TNG50-1/figures/Sersic_binning/'
    title = '{} sample, version {}, using {} radial bins'.format(
        sample, version, nBins)
    file_info = 'recovery_v{}_{}_bins.png'.format(version, nBins)
    
    outfile = outDir + sample + '_R50_' + file_info
    plt.plot_scatter_dumb(np.log10(R50s), np.log10(fit_Res), fit_ns,
        sample, 'o', cbar_label=r'$n$', title=title,
        xlabel=r'$\log(R_{50}^{\rm 2D}/{\rm ckpc})$',
        ylabel=r'$\log(R_{\rm 50}^{\rm Sersic~fit}/{\rm ckpc})$',
        xmin=-0.3, xmax=1.2, ymin=-0.3, ymax=1.2, vmin=0.8, vmax=8,
        save=save_plots, outfile=outfile)
    
    outfile = outDir + sample + '_R90_' + file_info
    plt.plot_scatter_dumb(np.log10(R90s), np.log10(fit_R90s), fit_ns,
        sample, 'o', cbar_label=r'$n$', title=title,
        xlabel=r'$\log(R_{90}^{\rm 2D}/{\rm ckpc})$',
        ylabel=r'$\log(R_{\rm 90}^{\rm Sersic~fit}/{\rm ckpc})$',
        xmin=0.1, xmax=2.4, ymin=0.1, ymax=2.4, vmin=0.8, vmax=8,
        save=save_plots, outfile=outfile)
    
    outfile = outDir + sample + '_Ie_' + file_info
    plt.plot_scatter_dumb(np.log10(Ie_inits), np.log10(fit_Ies), fit_ns,
        sample, 'o', cbar_label=r'$n$', title=title,
        xlabel=r'$\log(I_{\rm e,~init}/{\rm M}_{\odot}~{\rm kpc}^{-2})$',
        ylabel=r'$\log(I_{\rm e,~fit}/{\rm M}_{\odot}~{\rm kpc}^{-2})$',
        xmin=6.5, xmax=10, ymin=6.5, ymax=10, vmin=0.8, vmax=8,
        save=save_plots, outfile=outfile)
    
    outfile = outDir + sample + '_logM_' + file_info
    plt.plot_scatter_dumb(logM, fit_logM, fit_ns, sample, 'o',
        cbar_label=r'$n$', title=title,
        xlabel=r'$\log(M_{*}/{\rm M}_{\odot})_{\rm 3D, catalog}$',
        ylabel=r'$\log(M_{*}/{\rm M}_{\odot})_{\rm 2D, Sersic~fit}$',
        xmin=9.4, xmax=11.9, ymin=9.4, ymax=11.9, vmin=0.8, vmax=8,
        save=save_plots, outfile=outfile)
    
    return

def sersic_fxn(RR, Re, Ie, nn) :
    
    bn = get_bn(nn)
    
    return Ie*np.exp(-bn*(np.power(RR/Re, 1/nn) - 1))

# compare_input_and_fitted_values(sample='control', version='A', nBins=5, save_plots=False)
# compare_input_and_fitted_values(sample='control', version='A', nBins=6, save_plots=False)
# compare_input_and_fitted_values(sample='control', version='A', nBins=7, save_plots=False)
# compare_input_and_fitted_values(sample='control', version='A', nBins=8, save_plots=False)
# compare_input_and_fitted_values(sample='control', version='A', nBins=9, save_plots=False)
# compare_input_and_fitted_values(sample='control', version='A', nBins=10, save_plots=False)
# compare_input_and_fitted_values(sample='control', version='B', nBins=5, save_plots=False)
# compare_input_and_fitted_values(sample='control', version='B', nBins=6, save_plots=False)
# compare_input_and_fitted_values(sample='control', version='B', nBins=7, save_plots=False)
# compare_input_and_fitted_values(sample='control', version='B', nBins=8, save_plots=False)
# compare_input_and_fitted_values(sample='control', version='B', nBins=9, save_plots=False)
# compare_input_and_fitted_values(sample='control', version='B', nBins=10, save_plots=False)

# compare_input_and_fitted_values(sample='quenched', version='A', nBins=5, save_plots=False)
# compare_input_and_fitted_values(sample='quenched', version='A', nBins=6, save_plots=False)
# compare_input_and_fitted_values(sample='quenched', version='A', nBins=7, save_plots=False)
# compare_input_and_fitted_values(sample='quenched', version='A', nBins=8, save_plots=False)
# compare_input_and_fitted_values(sample='quenched', version='A', nBins=9, save_plots=False)
# compare_input_and_fitted_values(sample='quenched', version='A', nBins=10, save_plots=False)
# compare_input_and_fitted_values(sample='quenched', version='B', nBins=5, save_plots=False)
# compare_input_and_fitted_values(sample='quenched', version='B', nBins=6, save_plots=False)
# compare_input_and_fitted_values(sample='quenched', version='B', nBins=7, save_plots=False)
# compare_input_and_fitted_values(sample='quenched', version='B', nBins=8, save_plots=False)
# compare_input_and_fitted_values(sample='quenched', version='B', nBins=9, save_plots=False)
# compare_input_and_fitted_values(sample='quenched', version='B', nBins=10, save_plots=False)

'''
# create various metrics of interest
xis = sf_mass_within_1kpc/sf_mass_tot
alt_xis = sf_mass_within_tenthRe/sf_mass_tot

R10_zetas = sf_R10s/R10s
zetas = sf_R50s/R50s
R90_zetas = sf_R90s/R90s

# get values for the comparison sample

i0s = i0s[comparison]
i25s = i25s[comparison]
i50s = i50s[comparison]

i100s = i100s[comparison]

c_xis = xis[comparison]
c_alt_xis = alt_xis[comparison]
c_R10_zetas = R10_zetas[comparison]
c_zetas = zetas[comparison]
c_R90_zetas = R90_zetas[comparison]

c_xis = np.array([c_xis[firstDim, i0s], c_xis[firstDim, i25s],
                  c_xis[firstDim, i50s], c_xis[firstDim, i75s],
                  c_xis[firstDim, i100s]])
c_alt_xis = np.array([c_alt_xis[firstDim, i0s], c_alt_xis[firstDim, i25s],
                      c_alt_xis[firstDim, i50s], c_alt_xis[firstDim, i75s],
                      c_alt_xis[firstDim, i100s]])
c_R10_zetas = np.array([c_R10_zetas[firstDim, i0s],
                        c_R10_zetas[firstDim, i25s],
                        c_R10_zetas[firstDim, i50s],
                        c_R10_zetas[firstDim, i75s],
                        c_R10_zetas[firstDim, i100s]])
c_zetas = np.array([c_zetas[firstDim, i0s], c_zetas[firstDim, i25s],
                    c_zetas[firstDim, i50s], c_zetas[firstDim, i75s],
                    c_zetas[firstDim, i100s]])
c_R90_zetas = np.array([c_R90_zetas[firstDim, i0s],
                        c_R90_zetas[firstDim, i25s],
                        c_R90_zetas[firstDim, i50s],
                        c_R90_zetas[firstDim, i75s],
                        c_R90_zetas[firstDim, i100s]])
comparison_dict = {'xis':c_xis, 'alt_xis':c_alt_xis,
                    'R10_zetas':c_R10_zetas, 'zetas':c_zetas,
                    'R90_zetas':c_R90_zetas}

ionsets = ionsets[mask]
iminorities = iminorities[mask]
ihalfways = ihalfways[mask]

iterms = iterms[mask]

xis = xis[mask]
alt_xis = alt_xis[mask]
R10_zetas = R10_zetas[mask]
zetas = zetas[mask]
R90_zetas = R90_zetas[mask]

# get values for the quenched sample
q_xis = np.array([xis[firstDim, ionsets], xis[firstDim, iminorities],
                  xis[firstDim, ihalfways], xis[firstDim, imajorities],
                  xis[firstDim, iterms]])
q_alt_xis = np.array([alt_xis[firstDim, ionsets],
                      alt_xis[firstDim, iminorities],
                      alt_xis[firstDim, ihalfways],
                      alt_xis[firstDim, imajorities],
                      alt_xis[firstDim, iterms]])
q_R10_zetas = np.array([R10_zetas[firstDim, ionsets],
                        R10_zetas[firstDim, iminorities],
                        R10_zetas[firstDim, ihalfways],
                        R10_zetas[firstDim, imajorities],
                        R10_zetas[firstDim, iterms]])
q_zetas = np.array([zetas[firstDim, ionsets],
                    zetas[firstDim, iminorities],
                    zetas[firstDim, ihalfways],
                    zetas[firstDim, imajorities],
                    zetas[firstDim, iterms]])
q_R90_zetas = np.array([R90_zetas[firstDim, ionsets],
                        R90_zetas[firstDim, iminorities],
                        R90_zetas[firstDim, ihalfways],
                        R90_zetas[firstDim, imajorities],
                        R90_zetas[firstDim, iterms]])
quenched_dict = {'xis':q_xis, 'alt_xis':q_alt_xis, 'R10_zetas':q_R10_zetas,
                  'zetas':q_zetas, 'R90_zetas':q_R90_zetas}
'''
