
import numpy as np

from astropy.table import Table
import h5py
import imageio.v3 as iio
from scipy.ndimage import gaussian_filter # gaussian_filter1d
from scipy.stats import gaussian_kde
import sklearn.metrics as met
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.inspection import DecisionBoundaryDisplay

import plotting as plt

def classify(simple=False, single=False, mask=False, multi=False) :
    
    # get the data for the classification
    # data has columns [redshift, SFR_onset_frac, logM, C_SF, R_SF, Rinner, Router]
    data, mech = get_data()
    
    if mask : # mask for only the io and oi populations
        data = data[mech < 3]
        mech = mech[mech < 3]
    
    if simple :
        # perform the simplest fit, including all available data
        fit(data, mech) # accuracies around 67% (90% if discarding uni and amb)
    
    if single :
        # delete single columns of the data and retest for changes
        for i in range(data.shape[1]) : # distributions look ~Gaussian, +/-5%
            # accuracies without and (with) masking to the two populations
            # 65% (86%) without redshift
            # 68% (90%) without SFR_onset_frac
            # 60% (80%) without logM           -> largest influence
            # 65% (85%) without C_SF
            # 64% (85%) without R_SF
            # 68% (90%) without Rinner
            # 68% (88%) without Router
            fit(np.delete(data.copy(), i, axis=1), mech)
    
    if multi :
        # now we'll delete multiple columns at a time and retest for changes
        cols = [[0, 1], [0, 1, 2], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6],
                [5, 6], [3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
        for i, col in enumerate(cols) : # distributions look ~Gaussian, +/-5%
            # accuracies without and (with) masking to the two populations
            # 65% (86%) without redshift and SFR_onset_frac
            # 59% (78%) without redshift, SFR_onset_frac, and logM
            # 64% (85%) without C_SF and R_SF
            # 64% (85%) without C_SF and Rinner
            # 64% (83%) without C_SF and Router
            # 63% (83%) without R_SF and Rinner
            # 62% (80%) without R_SF and Router
            # 68% (89%) without Rinner and Router
            # 61% (80%) C_SFR, R_SF, Rinner, and Router
            # 35% (45%) using only redshift
            fit(np.delete(data.copy(), col, axis=1), mech)
    
    return

def confusion(cm=False, io_test=False, oi_test=False) :
    
    # get the data for the classification
    # data has columns [redshift, SFR_onset_frac, logM, C_SF, R_SF, Rinner, Router]
    data, mech = get_data()
    
    # make predictions using SVC with a linear kernel
    predict = SVC(kernel='linear').fit(data, mech).predict(data)
    
    if cm :
        # use sklearn's confusion matrix for multi classification
        # https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
        
        # !!! update to use .from_estimator or .from_predictions method, as opposed to class
        # cm = met.confusion_matrix(mech, predict)
        # met.ConfusionMatrixDisplay(cm,
        #     display_labels=['io', 'oi', 'uni', 'amb']).plot() # sum of 267
        
        Niterations = 1000
        cms = np.full((Niterations, 4, 4), 0)
        for i in range(Niterations) :
            X_train, X_test, y_train, y_test = train_test_split(data, mech, test_size=0.2)
            y_predict = SVC(kernel='linear').fit(X_train, y_train).predict(X_test)
            cm = met.confusion_matrix(y_test, y_predict)
            
            # somtimes the confusion matrix does not predict the uniform class,
            # so we need to reshape the arrays to conform to our 4x4 expectations
            if cm.shape == (3, 3) :
                cm = cm.ravel()
                cm = np.array([[cm[0], cm[1], 0, cm[2]],
                               [cm[3], cm[4], 0, cm[5]],
                               [0, 0, 0, 0],
                               [cm[6], cm[7], 0, cm[8]]])
                cm = cm.reshape(4, 4)
            
            cms[i] = cm
        
        lo, med, hi = np.percentile(cms, [16, 50, 84], axis=0)
        # print((med-lo)/54)
        # print()
        # print(med/54)
        # print()
        # print((hi-med)/54)
        
        # print(np.sum(med*267/54))
        # !!! update to use .from_estimator or .from_predictions method, as opposed to class
        met.ConfusionMatrixDisplay(med*267/54,
            display_labels=['io', 'oi', 'uni', 'amb']).plot()
    
    if io_test :
        # https://scikit-learn.org/stable/modules/model_evaluation.html#multi-label-confusion-matrix
        
        # !!! update to use .from_estimator or .from_predictions method, as opposed to class
        met.ConfusionMatrixDisplay(np.rot90(met.multilabel_confusion_matrix(
            mech, predict)[0], k=2), display_labels=['io', 'other']).plot()
        
        # non-io set to dummy class to compute confusion matrix
        true = mech.copy()
        pred = predict.copy()
        true[mech != 1] = 5
        pred[pred != 1] = 5
        print('io')
        print(met.confusion_matrix(true, pred)) # accuracy 78%, precision 67%
        print(met.accuracy_score(true, pred)) # accuracy 78%
        print(met.precision_score(true, pred)) # precision 67%
    
    if oi_test :
        
        # !!! update to use .from_estimator or .from_predictions method, as opposed to class
        met.ConfusionMatrixDisplay(np.rot90(met.multilabel_confusion_matrix(
            mech, predict)[1], k=2), display_labels=['oi', 'other']).plot()
        
        # non-oi set to dummy class to compute confusion matrix
        true = mech.copy()
        pred = predict.copy()
        true[mech != 2] = 5
        pred[pred != 2] = 5
        print('oi')
        print(met.confusion_matrix(true, pred)) # accuracy 87%, precision 79%
        print(met.accuracy_score(true, pred)) # accuracy 87%
        print(met.precision_score(true, pred, pos_label=2)) # precision 79%
    
    return

def fit(data, mech, Niterations=10, show=False) :  
    
    # setup the SVC classifier using a linear kernel
    classifier = SVC(kernel='linear')
    
    
    # classify the data using an 80/20 split for training/testing data, and
    # check the accuracy of the classification
    scores = np.full(Niterations, -1.0)
    for i in range(Niterations) :
        X_train, X_test, y_train, y_test = train_test_split(data, mech, test_size=0.2)
        
        # get the scores and save for future use
        scores[i] = classifier.fit(X_train, y_train).score(X_test, y_test)
    
    if show :
        if Niterations == 1000 :
            bins = 20
        else :
            bins = 10
        plt.histogram(scores, 'scores', bins=bins)
        print(np.percentile(scores, [16, 50, 84]))
    
    # for comparison, use the entire dataset
    # score = classifier.fit(data, mech).score(data, mech) # 0.70
    
    # print(classifier.classes_)
    # print(classifier.coef_)
    # print(classifier.dual_coef_)
    # print(classifier.intercept_)
    
    # params = np.delete(data.copy(), [0, 1, 2, 5, 6], axis=1) # use C_SF and R_SF
    # X_train, X_test, y_train, y_test = train_test_split(data, mech, test_size=0.2)
    # classifier.fit(X_train, y_train)
    # display = DecisionBoundaryDisplay.from_estimator(classifier, data)
    # display.ax_.scatter(data[:, 0], data[:, 1])
    # display.plot()
    
    return

def get_contours(sf_x, sf_y, quenched_x, quenched_y, xmin, xmax, ymin, ymax) :
    
    # define the bin edges in the x- and y-directions
    xbins = np.linspace(xmin, xmax, 101)
    ybins = np.linspace(ymin, ymax, 101)
    XX, YY = np.meshgrid(xbins[:-1] + np.diff(xbins)/2, ybins[:-1] + np.diff(ybins)/2)
    
    # get the 2D histograms
    sf_hist, _, _ = np.histogram2d(sf_x, sf_y, bins=(xbins, ybins))
    quenched_hist , _, _ = np.histogram2d(quenched_x, quenched_y,
                                          bins=(xbins, ybins))
    
    # normalize the histograms to recover probabilities
    sf_hist = sf_hist.T/sf_hist.sum()
    quenched_hist = quenched_hist.T/quenched_hist.sum()
    
    # smooth the histograms for visualization purposes
    xsigma = 300*(xbins[1] - xbins[0]) # 3
    ysigma = 300*(ybins[1] - ybins[0]) # 6
    sf_hist_smoothed = gaussian_filter(sf_hist, (xsigma, ysigma))
    quenched_hist_smoothed = gaussian_filter(quenched_hist, (xsigma, ysigma))
    
    # plt.display_image_simple(sf_hist, lognorm=False)
    # plt.display_image_simple(sf_hist_smoothed, lognorm=False)
    # plt.display_image_simple(quenched_hist, lognorm=False)
    # plt.display_image_simple(quenched_hist_smoothed, lognorm=False)
    
    # determine contour levels
    sf_levels = np.array([0.16, 0.5, 0.84])*sf_hist_smoothed.max()
    quenched_levels = np.array([0.16, 0.5, 0.84])*quenched_hist_smoothed.max()
    
    return XX, YY, [sf_hist_smoothed, quenched_hist_smoothed], [sf_levels, quenched_levels]

def get_data() :
    
    # get manual classifications
    with h5py.File('TNG50-1/TNG50-1_99_mechanism.hdf5', 'r') as hf :
        io = hf['inside-out'][:]
        oi = hf['outside-in'][:]
        uni = hf['uniform'][:]
        amb = hf['ambiguous'][:]
    
    # get basic information to mask to quenched populations
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        logM = hf['logM'][:, -1]
        quenched = hf['quenched'][:]
    
    # get the manually identified quenching mechanism
    mask = (logM >= 9.5) & quenched
    io = io[mask]
    oi = oi[mask]
    uni = uni[mask]
    amb = amb[mask]
    mech = np.sum(np.array([1*io, 2*oi, 3*uni, 4*amb]).T, axis=1) # mechanism
    
    # get the morphological parameters
    data = np.genfromtxt('TNG50-1/morphological_parameters_quenched.txt')
    data = data[:, 1:] # we don't need the subID information
    
    mask = (data[:, -1] >= 0.0) # mask the one bad row and the 10 rows that
    data = data[mask]           # need to be updated
    mech = mech[mask] # 101 io, 104 oi, 8 uni, 54 amb, compared to
                      # 103 io, 109 oi, 8 uni, 58 amb if not masking bad rows
    
    return data, mech

def get_kde(sf_x, sf_y, quenched_x, quenched_y, xmin, xmax, ymin, ymax) :
    
    # define the meshgrid and sampling positions
    XX, YY = np.mgrid[xmin:xmax:101j, ymin:ymax:101j]
    positions = np.vstack((XX.ravel(), YY.ravel()))
    
    # get the SF and quenched gaussian density estimate kernels
    sf_kernel = gaussian_kde(np.vstack((sf_x, sf_y)))
    quenched_kernel = gaussian_kde(np.vstack((quenched_x, quenched_y)))
    
    # get the densities
    sf_ZZ = np.reshape(sf_kernel(positions).T, XX.shape)
    quenched_ZZ = np.reshape(quenched_kernel(positions).T, XX.shape)
    
    # map the densities to the range [0, 1]
    sf_ZZ = (sf_ZZ - sf_ZZ.min())/(sf_ZZ.max() - sf_ZZ.min())
    quenched_ZZ = (quenched_ZZ - quenched_ZZ.min())/(quenched_ZZ.max() - quenched_ZZ.min())
    
    # set the contour levels based on 2D gaussians
    levels = 1 - np.exp(-0.5*np.square([0.5, 1, 1.5, 2]))
    
    return XX, YY, [sf_ZZ, quenched_ZZ], [levels, levels]

def permutation() :
    
    # get the data for the classification
    # data has columns [redshift, SFR_onset_frac, logM, C_SF, R_SF, Rinner, Router]
    data, mech = get_data()
    
    # use permutation score to understand "how likely an observed performance
    # of the classifier would be obtained by chance"
    from sklearn.model_selection import permutation_test_score
    score, permutation_scores, pvalue = permutation_test_score(
        SVC(kernel='linear'), data, mech, n_permutations=1000)
    print(score)
    # print(permutation_scores.mean(), permutation_scores.std())
    print(np.percentile(permutation_scores, [16, 50, 84]))
    # print(pvalue)
    # plt.histogram(permutation_scores, 'random scores', bins=20)
    
    return

def save_oi_misclassified_gifs() :
    
    frames = [iio.imread(f'TNG50-1/figures/misclassified_OI_galaxies_in_metric_space/RSF_vs_CSF_{i}.png') for i in range(100)]
    iio.imwrite('TNG50-1/figures/misclassified_OI_galaxies_RSF_vs_CSF.gif',
                np.stack(frames, axis=0), fps=4)
    
    frames = [iio.imread(f'TNG50-1/figures/misclassified_OI_galaxies_in_metric_space/Router_vs_CSF_{i}.png') for i in range(100)]
    iio.imwrite('TNG50-1/figures/misclassified_OI_galaxies_Router_vs_CSF.gif',
                np.stack(frames, axis=0), fps=4)
    
    frames = [iio.imread(f'TNG50-1/figures/misclassified_OI_galaxies_in_metric_space/Router_vs_RSF_{i}.png') for i in range(100)]
    iio.imwrite('TNG50-1/figures/misclassified_OI_galaxies_Router_vs_RSF.gif',
                np.stack(frames, axis=0), fps=4)
    
    return

def split_and_crossvalidate() :
    
    # get the data for the classification
    # data has columns [redshift, SFR_onset_frac, logM, C_SF, R_SF, Rinner, Router]
    data, mech = get_data()
    
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, mech, test_size=0.2)
    # note that this is very similar in some respects to using ShuffleSplit,
    # but not sure what the difference is between ShuffleSplit vs
    # StratifiedShuffleSplit or others, and which is better to use?
    # also why use this as opposed to train_test_split, especially if using
    # to run ~1000 times?
    from sklearn.model_selection import ShuffleSplit
    ss = ShuffleSplit(test_size=0.2, n_splits=1)
    # for train_index, test_index in ss.split(data, mech) :
    #     print(train_index, test_index)
    
    from sklearn.model_selection import cross_val_score
    # https://scikit-learn.org/stable/modules/cross_validation.html
    scores = cross_val_score(SVC(kernel='linear'), data, mech) # scoring='accuracy' by default
    # print()
    # print(scores)
    print(scores.mean(), scores.std())
    print(np.percentile(scores, [16, 50, 84]))
    
    return

def test_classifier(thresh=-10.5, slope=1, include_sf=False, Niterations=1000) :
    
    # get the morphological metrics and the quenching episode progresses
    infile = 'TNG50-1/morphological_metrics_{}_+-{}.fits'.format(thresh, slope)
    data = Table.read(infile)
    episode_progress = data['episode_progress'].value
    
    # create masks for the quenching progress epochs of interest
    early_mask = (episode_progress < 0.25)
    early_mid_mask = (episode_progress >= 0.25) & (episode_progress < 0.5)
    mid_late_mask = (episode_progress >= 0.5) & (episode_progress < 0.75)
    late_mask = (episode_progress >= 0.75)
    
    masks = [early_mask, early_mid_mask, mid_late_mask, late_mask]
    epochs = ['early', 'early-mid', 'mid-late', 'late']
    for mask, epoch in zip(masks, epochs) :
        
        print(epoch)
        
        # mask data to the epoch of interest
        table = data.copy()
        table = table[mask]
        
        # get parameters of interest
        epoch_quenched = table['quenched_status'].value
        epoch_sf = table['sf_status'].value
        epoch_mech = table['mechanism'].value
        
        # exclude redshift and stellar mass information, and mask
        # to epoch of interest
        epoch_data = np.array([table['C_SF'].value,
                               np.log10(table['R_SF'].value),
                               # table['Rinner'].value,
                               table['Router'].value]).T
        
        # mask NaN data and -99 values for Rinner and Router
        good = ((np.sum(np.isnan(epoch_data), axis=1) == 0) &
                (epoch_data[:, 2] >= 0.0) )#& (epoch_data[:, 3] >= 0.0))
        epoch_quenched = epoch_quenched[good]
        epoch_sf = epoch_sf[good]
        epoch_data = epoch_data[good]
        epoch_mech = epoch_mech[good]
        
        '''
        # setup the SVC classifier using a linear kernel and balanced weighting
        classifier = SVC(kernel='linear', class_weight='balanced')
        
        # classify the data using an 80/20 split for training/testing data, and
        # check the accuracy of the classification
        X_train, X_test, y_train, y_test = train_test_split(epoch_data,
            epoch_mech, test_size=0.2)
        y_predict = SVC(kernel='linear').fit(X_train, y_train).predict(X_test)
        score = classifier.fit(X_train, y_train).score(X_test, y_test)
        print(score)
        
        met.ConfusionMatrixDisplay(met.confusion_matrix(y_test, y_predict),
            display_labels=['sf', 'io', 'oi', 'amb']).plot()
        '''
        
        # setup the SVC classifier using a linear kernel
        # classifier = make_pipeline(StandardScaler(), SVC(kernel='linear'))
        # classifier = make_pipeline(RobustScaler(), SVC(kernel='linear'))
        classifier = SVC(kernel='linear')#, probability=True)
        
        if include_sf :
            # select the same number of SF galaxies as quenching galaxies
            threshold = np.sum(epoch_quenched)/np.sum(epoch_sf)
            
            epoch_quenched_mech = epoch_mech[epoch_mech > 0.0]
            epoch_sf_mech = epoch_mech[epoch_mech == 0.0]
            
            epoch_quenched_data = epoch_data[epoch_mech > 0.0]
            epoch_sf_data = epoch_data[epoch_mech == 0.0]
            
            np.random.seed(0) # use the same SF galaxies each time
            select = (np.random.rand(np.sum(epoch_sf)) <= threshold)
            epoch_sf_data = epoch_sf_data[select]
            epoch_sf_mech = epoch_sf_mech[select]
            
            epoch_data_final = np.concatenate([epoch_quenched_data,
                                               epoch_sf_data])
            epoch_mech_final = np.concatenate([epoch_quenched_mech,
                                               epoch_sf_mech])
            
            # cms = np.full((Niterations, 4, 4), 0)
        else :
            epoch_data_final = epoch_data[epoch_mech > 0.0]
            epoch_mech_final = epoch_mech[epoch_mech > 0.0]
            
            # cms = np.full((Niterations, 3, 3), 0)
        
        # classify the data using an 80/20 split for
        # training/testing data, and check the accuracy of the 
        # classification
        scores = np.full(Niterations, np.nan)
        
        # linearSVC = np.full(Niterations, np.nan)
        # rbf_SVC = np.full(Niterations, np.nan)
        # poly_SVC = np.full(Niterations, np.nan)
        
        # random_forest = np.full(Niterations, np.nan)
        # random_forest_opt = np.full(Niterations, np.nan)
        
        # neural_net = np.full(Niterations, np.nan)
        # decision_tree = np.full(Niterations, np.nan)
        # nearest_neighbors = np.full(Niterations, np.nan)
        
        # gaussian_process = np.full(Niterations, np.nan)
        # adaboost = np.full(Niterations, np.nan)
        # qda = np.full(Niterations, np.nan)
        # naive_bayes = np.full(Niterations, np.nan)
        
        # ridge = np.full(Niterations, np.nan)
        for i in range(Niterations) :
            X_train, X_test, y_train, y_test = train_test_split(
                epoch_data_final, epoch_mech_final, test_size=0.2)
            # y_predict = classifier.fit(X_train, y_train).predict(X_test)
            scores[i] = classifier.fit(X_train, y_train).score(
                X_test, y_test)
            # cms[i] = met.confusion_matrix(y_test, y_predict)
            
            # fit = classifier.fit(X_train, y_train)
            # probs = fit.predict_proba(X_test)
            # highest_prob = np.argmax(probs, axis=1)
            # predict = fit.classes_[highest_prob]
            # scores[i] = met.accuracy_score(y_test, predict)
            
            # linearSVC[i] = LinearSVC(dual=False).fit(
            #     X_train, y_train).score(X_test, y_test)
            # rbf_SVC[i] = SVC(kernel='rbf', gamma='scale').fit(
            #     X_train, y_train).score(X_test, y_test)
            # poly_SVC[i] = SVC(kernel='poly', degree=3, gamma='auto').fit(
            #     X_train, y_train).score(X_test, y_test)
            
            # random_forest[i] = RandomForestClassifier(
            #     max_depth=5, n_estimators=10, max_features=1).fit(
            #     X_train, y_train).score(X_test, y_test)
            # random_forest_opt[i] = RandomForestClassifier(
            #     max_depth=7, n_estimators=100, max_features=None).fit(
            #     X_train, y_train).score(X_test, y_test)
            
            # neural_net[i] = MLPClassifier(alpha=1, max_iter=1000).fit(
            #     X_train, y_train).score(X_test, y_test)
            # decision_tree[i] = DecisionTreeClassifier(max_depth=5).fit(
            #     X_train, y_train).score(X_test, y_test)
            # nearest_neighbors[i] = KNeighborsClassifier(3).fit(
            #     X_train, y_train).score(X_test, y_test)
            
            # gaussian_process[i] = GaussianProcessClassifier(1.0*RBF(1.0)).fit(
            #     X_train, y_train).score(X_test, y_test)
            # adaboost[i] = AdaBoostClassifier(algorithm='SAMME').fit(
            #     X_train, y_train).score(X_test, y_test)
            # qda[i] = QuadraticDiscriminantAnalysis().fit(
            #     X_train, y_train).score(X_test, y_test)
            # naive_bayes[i] = GaussianNB().fit(
            #     X_train, y_train).score(X_test, y_test)
            
            # ridge[i] = RidgeClassifier().fit(
            #     X_train, y_train).score(X_test, y_test)
        
        print('SVM linear kernel       {}'.format(
            np.percentile(scores, [16, 50, 84])))
        
        # print('LinearSVC         {}'.format(
        #     np.percentile(linearSVC, [16, 50, 84])))
        # print('RBF SVC           {}'.format(
        #     np.percentile(rbf_SVC, [16, 50, 84])))
        # print('poly SVC          {}'.format(
        #     np.percentile(poly_SVC, [16, 50, 84])))
        
        # print('random forest     {}'.format(
        #     np.percentile(random_forest, [16, 50, 84])))
        # print('random forest optimized {}'.format(
        #     np.percentile(random_forest_opt, [16, 50, 84])))
        
        # print('neural net        {}'.format(
        #     np.percentile(neural_net, [16, 50, 84])))
        # print('decision tree     {}'.format(
        #     np.percentile(decision_tree, [16, 50, 84])))
        # print('nearest neighbors {}'.format(
        #     np.percentile(nearest_neighbors, [16, 50, 84])))
        
        # print('gaussian process  {}'.format(
        #     np.percentile(gaussian_process, [16, 50, 84])))
        # print('adaboost          {}'.format(
        #     np.percentile(adaboost, [16, 50, 84])))
        # print('qda               {}'.format(
        #     np.percentile(qda, [16, 50, 84])))
        # print('naive bayes       {}'.format(
        #     np.percentile(naive_bayes, [16, 50, 84])))
        
        # print('ridge classifer         {}'.format(
        #     np.percentile(ridge, [16, 50, 84])))
        
        # print(np.percentile(cms, 50, axis=0))
        print()
        # met.ConfusionMatrixDisplay(med,
        #     display_labels=['sf', 'io', 'oi', 'amb']).plot()
        # met.ConfusionMatrixDisplay(
        #     met.confusion_matrix(y_test, y_predict),
        #     display_labels=['sf', 'io', 'oi', 'amb']).plot()
    
    return

def tests() :
    
    # use random classes to test recovery -> recovery should be 1/Nclasses
    scores = [0.495,  0.498,  0.5035, 0.499,  0.493,  0.4865, 0.506,  0.494,
              0.5105, 0.5045, 0.475,  0.4915, 0.498,  0.5085, 0.511,  0.5065,
              0.515,  0.4735, 0.478,  0.5095, 0.5015, 0.488,  0.482,  0.4765,
              0.5115, 0.507,  0.4965, 0.515,  0.489,  0.504,  0.498,  0.497,
              0.505,  0.4915, 0.4905, 0.49,   0.4895, 0.482,  0.488,  0.4905,
              0.4915, 0.5125, 0.501,  0.501,  0.498,  0.499,  0.496,  0.4975,
              0.4805, 0.5055, 0.497,  0.5055, 0.497,  0.4985, 0.51,   0.4695,
              0.5,    0.499,  0.4925, 0.492,  0.501,  0.4855, 0.5145, 0.493,
              0.5105, 0.498,  0.5,    0.505,  0.498,  0.5035, 0.497,  0.493,
              0.5115, 0.5085, 0.5075, 0.4985, 0.4965, 0.508,  0.5085, 0.501,
              0.5205, 0.5135, 0.5265, 0.5125, 0.484,  0.494,  0.504,  0.5025,
              0.486,  0.4775, 0.47,   0.483,  0.5135, 0.4885, 0.499,  0.489,
              0.487,  0.4995, 0.496,  0.5145]
    if not scores :
        classifier = SVC(kernel='linear')
        Nclasses, length = 2, 10000 # use a large data set for random data
        train = int(0.8*length)
        scores = np.full(100, -1.0)
        for i in range(100) :
            values = np.reshape(np.random.rand(length), (length, 1))
            classes = np.random.randint(0, Nclasses, size=length)
            scores[i] = classifier.fit(values[:train], classes[:train]).score(
                values[train:], classes[train:])
    plt.histogram(scores, 'scores')
    print(np.percentile(scores, [16, 50, 84]), np.mean(scores))
    
    # use correlated classes to test recovery -> recovery should be 100%
    scores = [0.9995, 0.999,  0.9985, 0.9975, 0.999,  0.9995, 1.0,    0.999,
              0.9995, 0.9995, 0.9995, 1.0,    0.9975, 0.9995, 0.9995, 0.999,
              0.9995, 0.994,  0.9985, 0.9985, 1.0,    0.998,  0.999,  0.999,
              0.9965, 0.9995, 0.9975, 0.9985, 1.0,    0.999,  1.0,    0.9965,
              1.0,    1.0,    0.998,  1.0,    0.999,  0.9985, 0.999,  0.998,
              0.998,  0.9975, 0.9985, 0.996,  0.9985, 0.9995, 0.9955, 1.0,
              1.0,    0.999,  0.9975, 0.9985, 0.9975, 1.0,    1.0,    1.0,
              0.9965, 0.997,  1.0,    1.0,    0.9995, 0.9995, 1.0,    1.0,
              1.0,    0.998,  0.998,  0.998,  1.0,    0.9985, 0.9995, 1.0,
              0.9995, 0.997,  0.9975, 1.0,    1.0,    0.998,  0.996,  1.0,
              0.997,  0.999,  0.997,  0.996,  1.0,    0.9995, 0.9995, 0.9995,
              0.999,  1.0,    1.0,    0.999,  0.9985, 0.9985, 1.0,    0.9995,
              1.0,    0.9995, 0.9995, 0.9985]
    if not scores :
        classifier = SVC(kernel='linear')
        Nclasses, length = 2, 10000 # use a large data set for random data
        train = int(0.8*length)
        scores = np.full(100, -1.0)
        for i in range(100) :
            values = np.random.rand(length)
            classes = np.full(length, 1)
            classes[values <= 0.5] = 0  # for 2 classes
            # classes[values <= 0.33] = 0 # for 3 classes
            # classes[values > 0.66] = 2  # for 3 classes
            values = np.reshape(values, (length, 1))
            scores[i] = classifier.fit(values[:train], classes[:train]).score(
                values[train:], classes[train:])
    plt.histogram(scores, 'scores')
    print(np.percentile(scores, [16, 50, 84]), np.mean(scores))
    
    return

# tests()
# classify(simple=1) # use all data, but with 80/20 train/test split
# classify(simple=1, mask=1) # only use io and oi classes
# classify(single=1) # delete single columns to check accuracy
# classify(single=1, mask=1) # delete single columns, but only use io and oi
# classify(multi=1) # delete multiple columns to check accuracy
# classify(multi=1, mask=1) # delete multiple columns, but only use io and oi
# confusion(cm=1) # find the confusion matrix for all classes
# confusion(io_test=1) # find the confusion matrix for the io class
# confusion(oi_test=1) # find the confusion matrix for the oi class

# split_and_crossvalidate() # different ways to split data, cross validate?
# permutation() # check how often classifier performs at given level by chance

# xs = [[0.448, 0.476, 0.504], [0.53125, 0.5625, 0.59375],
#       [0.61085973, 0.64253394, 0.66968326], [0.69008264, 0.71487603, 0.73966942]]
# ys = [[0.125, 0.125, 0.125], [0.375, 0.375, 0.375],
#       [0.625, 0.625, 0.625], [0.875, 0.875, 0.875]]
# labels = ['early', 'early-mid', 'mid-late', 'late']
# colors = ['b', 'm', 'r', 'k']
# markers = ['', '', '', '']
# styles = ['-', '-', '-', '-']
# alphas = [1, 1, 1, 1]
# plt.plot_simple_multi(xs, ys, labels, colors, markers, styles, alphas,
#     xlabel='SVM classifer accuracy', ylabel='quenching episode progress',
#     xmin=0.4, xmax=0.8, ymin=0, ymax=1)

# test_classifier(Niterations=10, include_sf=False)

# save_oi_misclassified_gifs()

'''
import matplotlib
# from matplotlib.mlab import bivariate_normal
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

X, Y = np.mgrid[-3:3:100j, -3:3:100j]
# z1 = bivariate_normal(X, Y, .5, .5, 0., 0.)
# z2 = bivariate_normal(X, Y, .4, .4, .5, .5)
# z3 = bivariate_normal(X, Y, .6, .2, -1.5, 0.)
z1 = st.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]])
z2 = st.multivariate_normal([0.5, 0.5], [[0.4, 0], [0, 0.4]])
z3 = st.multivariate_normal([-1.5, 0], [[0.6, 0], [0, 0.2]])
z = z1.pdf(np.dstack((X, Y))) + z2.pdf(np.dstack((X, Y))) + z3.pdf(np.dstack((X, Y)))
z = z / z.sum()

n = 1000
t = np.linspace(0, z.max(), n)
integral = ((z >= t[:, None, None]) * z).sum(axis=(1,2))

from scipy import interpolate
f = interpolate.interp1d(integral, t)
t_contours = f(np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))
plt.imshow(z.T, origin='lower', extent=[-3,3,-3,3], cmap="gray")
plt.contour(z.T, t_contours, extent=[-3,3,-3,3])
plt.show()
'''

# met.ConfusionMatrixDisplay(np.rot90(met.multilabel_confusion_matrix(
#     epoch_mech, y_predict)[0], k=2), display_labels=['io', 'other']).plot()

# X_train, X_test, y_train, y_test = train_test_split(data, mech, test_size=0.2)
# y_predict = SVC(kernel='linear').fit(X_train, y_train).predict(X_test)
# cm = met.confusion_matrix(y_test, y_predict)
# met.ConfusionMatrixDisplay(cm, display_labels=['io', 'oi', 'amb', 'sf']).plot()
