
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from scipy.optimize import curve_fit
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.svm import SVC
import plotting as plt

from core import surface, vertex

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # the TNG cosmology

def determine_oi_progress_boundaries() :
    
    textwidth = 7.10000594991006
    textheight = 9.095321710253218
    
    # get the masses, metrics, and episode progress values
    logM, metrics, episode_progress, episode_progress_class, subIDs, ages = get_oi_and_progress_data(
        return_extra=True)
    
    # populate an array of the elapsed time since the beginning of the quenching episode
    elapsed_delta_t = np.full(subIDs.shape[0], np.nan)
    for i, (subID, age) in enumerate(zip(subIDs, ages)) :
        first_instance = np.where(subIDs == subID)[0][0]
        elapsed_delta_t[i] = age - ages[first_instance]
    
    bins = np.arange(0, 1.01, 0.05)
    
    '''
    i = 0
    Niterations = 1000
    early_mid_coefficients = np.full((Niterations, 4), np.nan)
    mid_late_coefficients = np.full((Niterations, 4), np.nan)
    cms = np.full((Niterations, 3, 3), -1)
    early_histogram_values = np.full((Niterations, len(bins) - 1), -1)
    mid_histogram_values = np.full((Niterations, len(bins) - 1), -1)
    late_histogram_values = np.full((Niterations, len(bins) - 1), -1)
    while i < Niterations : # requires about 3225 total iterations for 1000 valid cases
        
        # create an array of zeros and an array of random numbers to sample from
        zeros = np.zeros(episode_progress_class.shape)
        randoms = np.random.rand(len(episode_progress_class))
        psuedo_randoms = np.full(episode_progress_class.shape, np.nan)
        
        # select the same number of mid-episode galaxies as early (372) and late (363)
        threshold = 368/644 # 644 galaxies are mid-episode
        psuedo_randoms[episode_progress_class == 7] = zeros[episode_progress_class == 7]
        psuedo_randoms[episode_progress_class == 8] = randoms[episode_progress_class == 8]
        psuedo_randoms[episode_progress_class == 9] = zeros[episode_progress_class == 9]
        select = (psuedo_randoms <= threshold)
        
        random_midpoint_oi_galaxies = select.sum() - 372 - 363
        
        if (random_midpoint_oi_galaxies >= 363) and (random_midpoint_oi_galaxies <= 372) :
            # separate the data using an SVM
            classifier = SVC(kernel='linear') # unweighted
            # classifier = SVC(kernel='linear', class_weight='balanced') # balanced
            
            fit = classifier.fit(metrics[select], episode_progress_class[select])
            
            # get the coefficients that separate early outside-in from mid
            a1, b1, c1, d1 = fit.coef_[0][0], fit.coef_[0][1], fit.coef_[0][2], fit.intercept_[0]
            # print(a1, b1, c1, d1)
            early_mid_coefficients[i] = [a1, b1, c1, d1]
            
            # get the coefficients that separate mid outside-in from late
            a2, b2, c2, d2 = fit.coef_[2][0], fit.coef_[2][1], fit.coef_[2][2], fit.intercept_[2]
            # print(a2, b2, c2, d2)
            mid_late_coefficients[i] = [a2, b2, c2, d2]
            
            # make predictions and populate a confusion matrix
            predictions = classifier.predict(metrics[select])
            cms[i] = confusion_matrix(episode_progress_class[select], predictions)
            
            # determine histogram values across all iterations
            early_histogram_values[i] = np.histogram(
                episode_progress[select][predictions == 7], bins=bins)[0] 
            mid_histogram_values[i] = np.histogram(
                episode_progress[select][predictions == 8], bins=bins)[0]
            late_histogram_values[i] = np.histogram(
                episode_progress[select][predictions == 9], bins=bins)[0]
            
            i += 1 # iterate the counter
    
    # print('early-mid')
    # print(np.percentile(early_mid_coefficients, 16, axis=0))
    # print(np.percentile(early_mid_coefficients, 50, axis=0))
    # print(np.percentile(early_mid_coefficients, 84, axis=0))
    # print()
    # print('mid-late')
    # print(np.percentile(mid_late_coefficients, 16, axis=0))
    # print(np.percentile(mid_late_coefficients, 50, axis=0))
    # print(np.percentile(mid_late_coefficients, 84, axis=0))
    # print()
    # print('cm')
    # print(np.percentile(cms, 16, axis=0))
    # print(np.percentile(cms, 50, axis=0))
    # print(np.percentile(cms, 84, axis=0))
    
    # disp = ConfusionMatrixDisplay(np.round(np.percentile(cms, 50, axis=0)).astype(int),
    #                               display_labels=['early', 'mid', 'late'])
    # disp.plot()
    # [[301.  31.  32.] # for 16th percentile
    # [166.  62. 105.]
    # [ 54.  40. 245.]]
    # [[302.  36.  35.] # for median
    # [173.  79. 115.]
    # [ 54.  54. 255.]]
    # [[302.    39.    39.  ] # for 84th percentile
    # [179.    93.   129.16]
    # [ 54.    64.   269.  ]]
    
    a1, b1, c1, d1 = -0.24363317, -0.16141724, 0.68043508, -2.30230267
    a2, b2, c2, d2 = -1.45916687, -0.05622891, 0.45285468, -0.37527552
    
    # now project the 3D plots into 2D for easier visualization
    xprimes1 = [-(a1*metrics[:, 0][episode_progress_class == 7] +
                  b1*metrics[:, 1][episode_progress_class == 7] + d1)/c1,
                -(a1*metrics[:, 0][episode_progress_class == 8] +
                  b1*metrics[:, 1][episode_progress_class == 8] + d1)/c1,
                -(a1*metrics[:, 0][episode_progress_class == 9] +
                  b1*metrics[:, 1][episode_progress_class == 9] + d1)/c1]
    ys1 = [np.random.normal(metrics[:, 2][episode_progress_class == 7], 0.03),
                np.random.normal(metrics[:, 2][episode_progress_class == 8], 0.03),
                np.random.normal(metrics[:, 2][episode_progress_class == 9], 0.03)]
    xlabel1 = r'$0.358 C_{\rm SF} + 0.237 R_{\rm SF} + 3.384$'
    ylabel1 = r'$R_{\rm outer}/R_{\rm e}$'
    
    xprimes2 = [-(a2*metrics[:, 0][episode_progress_class == 7] +
                  b2*metrics[:, 1][episode_progress_class == 7] + d2)/c2,
                -(a2*metrics[:, 0][episode_progress_class == 8] +
                  b2*metrics[:, 1][episode_progress_class == 8] + d2)/c2,
                -(a2*metrics[:, 0][episode_progress_class == 9] +
                  b2*metrics[:, 1][episode_progress_class == 9] + d2)/c2]
    ys2 = [np.random.normal(metrics[:, 2][episode_progress_class == 7], 0.03),
                np.random.normal(metrics[:, 2][episode_progress_class == 8], 0.03),
                np.random.normal(metrics[:, 2][episode_progress_class == 9], 0.03)]
    xlabel2 = r'$3.222 C_{\rm SF} + 0.124 R_{\rm SF} + 0.829$'
    ylabel2 = r'$R_{\rm outer}/R_{\rm e}$'
    
    xx = np.linspace(-0.1, 5.1, 1001)
    
    colors = ['darkgreen', 'gold', 'darkred']
    alphas = [0.2, 0.2, 0.2]
    labels = ['early outside-in', 'mid outside-in', 'late outside-in']
    markers = ['s', 's', 's']
    
    plt.double_scatter_with_line_other(xprimes1, ys1, colors, markers, alphas, xx, xx,
        xprimes2, ys2, colors, markers, alphas, xx, xx, labels,
        xlabel1=xlabel1, ylabel1=ylabel1, xlabel2=xlabel2, ylabel2=ylabel2,
        titles=['early/mid outside-in', 'mid/late outside-in'],
        xmin1=3.35, xmax1=3.66, xmin2=0.8, xmax2=4, ymin=-0.1, ymax=5.1,
        figsizewidth=textwidth, figsizeheight=textheight/3, loc=4,
        save=False, outfile='reclassification_boundaries.pdf')
    '''
    
    vals1 = np.array([np.histogram(episode_progress[episode_progress_class == 7],
                                   bins=bins)[0],
                      np.histogram(episode_progress[episode_progress_class == 8],
                                   bins=bins)[0],
                      np.histogram(episode_progress[episode_progress_class == 9],
                                   bins=bins)[0]])
    vals2 = np.array([[114, 50, 45, 50, 42, 24, 18, 26, 20, 15, 18, 13, 12, 14, 13, 10, 13, 8, 7, 16],
                      [10, 5, 7, 9, 5, 6, 8, 7, 7, 8, 8, 8, 9, 10, 7, 13, 17, 9, 7, 9],
                      [1, 3, 10, 10, 11, 10, 9, 7, 11, 10, 13, 13, 12, 15, 17, 36, 41, 44, 38, 96]])
    weights1 = vals1/(np.sum(vals1, axis=1))[:, None]
    weights2 = vals2/(np.sum(vals2, axis=1))[:, None]
    styles1 = ['-', '-', '-']
    styles2 =  ['--', '--', '--']
    labels1 = ['early', 'mid', 'late']
    labels2 = ['predicted early', 'predicted mid', 'predicted late']
    temp = [bins[:-1], bins[:-1], bins[:-1]]
    bins = [bins, bins, bins]
    colors = ['darkgreen', 'gold', 'darkred']
    
    plt.histogram_multi_double(temp, temp, weights1, weights2, styles1, styles2,
        labels1, labels2, colors, bins, ymin=1/700, ymax=0.35, loc=9,
        figsizewidth=3.35224200913242, figsizeheight=9.095321710253218/2,
        save=False, outfile='reclassified_quenching_duration.pdf')
    
    return

def get_oi_and_progress_data(return_extra=False) :

    # get the morphological metrics and the quenching episode progresses
    data = Table.read('TNG50-1/morphological_metrics_-10.5_+-1.fits')
    data = data[data['mechanism'].value == 3] # mask data to the outside-in population
    episode_progress = data['episode_progress'].value
    subIDs = data['quenched_subID'].value
    ages = cosmo.age(data['redshift'].value).value
    
    episode_progress_class = np.full(len(data), -1)
    episode_progress_class[(episode_progress >= 0.) & (episode_progress <= 0.25)] = 7
    episode_progress_class[(episode_progress > 0.25) & (episode_progress < 0.75)] = 8
    # episode_progress_class[(episode_progress >= 0.36) & (episode_progress <= 0.65)] = 8
    episode_progress_class[episode_progress >= 0.75] = 9
    
    # get parameters of interest
    logM = data['logM'].value
    metrics = np.array([data['C_SF'].value, np.log10(data['R_SF'].value),
                        data['Router'].value]).T
    
    # mask NaN data and -99 values for Rinner and Router
    good = ((np.sum(np.isnan(metrics), axis=1) == 0) & (metrics[:, 2] >= 0.0))
    data = data[good]
    logM = logM[good]
    metrics = metrics[good]
    episode_progress = episode_progress[good]
    episode_progress_class = episode_progress_class[good]
    subIDs = subIDs[good]
    ages = ages[good]
    
    if return_extra :
        return logM, metrics, episode_progress, episode_progress_class, subIDs, ages
    else :
        return logM, metrics, episode_progress_class

def show_oi_progress_boundaries(show=False) :
    
    # get the masses, metrics, and episode progress values
    logM, metrics, episode_progress_class = get_oi_and_progress_data()
    
    if show :
        # set up color information for plotting
        colors = np.full(len(metrics), '')
        colors[episode_progress_class == 7] = 'gold'
        colors[episode_progress_class == 8] = 'orange'
        colors[episode_progress_class == 9] = 'orangered'
        
        # setup a meshgrid to sample the dummy boundary surface, which is
        # simpler to include with a non-visible boundary, than to create a new
        # plotting function
        xx, yy = np.meshgrid(np.linspace(0, 1, 101), np.linspace(-1, 1, 201))
        zz = surface(1, 2, 3, 4, xx, yy)
        
        # mask surface features that are outside the plotting area
        zz[(zz < 0) | (zz > 5)] = np.nan
        
        # determine the size of the points for use in plotly
        mini, stretch = 1.5, 20 # define the minimum size and the maximum stretch
        logM_min, logM_max = np.min(logM), np.max(logM)
        diff = (logM_max - logM_min)/2
        logM_fit_vals = np.array([logM_min, logM_min + diff, logM_max])
        size_fit_vals = np.array([1, np.sqrt(stretch), stretch])*mini
        # adapted from https://stackoverflow.com/questions/12208634
        popt, _ = curve_fit(lambda xx, aa: vertex(xx, aa, logM_fit_vals[0], mini),
                            logM_fit_vals, size_fit_vals) # fit the curve
        size = vertex(logM, popt[0], logM_fit_vals[0], mini) # get the size for the points
        
        xs = [metrics[:, 0][episode_progress_class == 7],
              metrics[:, 0][episode_progress_class == 8],
              metrics[:, 0][episode_progress_class == 9]]
        ys = [metrics[:, 1][episode_progress_class == 7],
              metrics[:, 1][episode_progress_class == 8],
              metrics[:, 1][episode_progress_class == 9]]
        zs = [metrics[:, 2][episode_progress_class == 7],
              metrics[:, 2][episode_progress_class == 8],
              metrics[:, 2][episode_progress_class == 9],]
        colors = ['darkgreen', 'gold', 'darkred']
        markers = ['o', 'o', 'o']
        labels = ['early outside-in', 'mid outside-in', 'late outside-in']
        sizes = [size[episode_progress_class == 7],
                 size[episode_progress_class == 8],
                 size[episode_progress_class == 9]]
        
        plt.plot_scatter_3d(xs, ys, zs, colors, markers, labels, sizes, xx, yy, zz,
            xmin=0, xmax=1, ymin=-1, ymax=1, zmin=0, zmax=5,
            xlabel=r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
            ylabel=r'$R_{\rm SF} = \log{(R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}})}$',
            zlabel=r'$R_{\rm outer}/R_{\rm e}$')
    
    # separate the data using an SVM
    classifier = SVC(kernel='linear') # unweighted
    # classifier = SVC(kernel='linear', class_weight='balanced') # balanced
    
    fit = classifier.fit(metrics, episode_progress_class)
    
    # get the coefficients that separate early outside-in from mid
    a1, b1, c1, d1 = fit.coef_[0][0], fit.coef_[0][1], fit.coef_[0][2], fit.intercept_[0]
    
    # get the coefficients that separate mid outside-in from late
    a2, b2, c2, d2 = fit.coef_[2][0], fit.coef_[2][1], fit.coef_[2][2], fit.intercept_[2]
    
    if show :
        # set the boundary surface
        z1 = surface(a1, b1, c1, d1, xx, yy)
        z1[(z1 < 0) | (z1 > 5)] = np.nan
        
        plt.plot_scatter_3d(xs, ys, zs, colors, markers, labels, sizes, xx, yy, z1,
            xmin=0, xmax=1, ymin=-1, ymax=1, zmin=0, zmax=5,
            xlabel=r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
            ylabel=r'$R_{\rm SF} = \log{(R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}})}$',
            zlabel=r'$R_{\rm outer}/R_{\rm e}$')
        
        # set the boundary surface
        z2 = surface(a2, b2, c2, d2, xx, yy)
        z2[(z2 < 0) | (z2 > 5)] = np.nan
        
        plt.plot_scatter_3d(xs, ys, zs, colors, markers, labels, sizes, xx, yy, z2,
            xmin=0, xmax=1, ymin=-1, ymax=1, zmin=0, zmax=5,
            xlabel=r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$',
            ylabel=r'$R_{\rm SF} = \log{(R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}})}$',
            zlabel=r'$R_{\rm outer}/R_{\rm e}$')
    
    return
