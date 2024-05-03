
import numpy as np

from sklearn.svm import SVC

from core import get_late_data
import plotting as plt

def calculate_purity_completeness(balance_populations_manually=True, loc=0,
                                  Niterations=10, SF_and_IO=False, save=False,
                                  fix_coefficients=False) :
    
    y_quenched, y_sf, X_quenched, X_sf, X_io, X_oi, X_amb = get_late_data()
    
    # setup the SVC classifier using a linear kernel
    if balance_populations_manually :
        classifier = SVC(kernel='linear')
        balance_mode = 'Manual'
        # select the same number of SF galaxies as quenching galaxies
        threshold = len(y_quenched)/len(y_sf)
    else :
        classifier = SVC(kernel='linear', class_weight='balanced')
        balance_mode = 'SVM'
        Niterations = 1
        # select all the SF galaxies
        threshold = 1
    
    if SF_and_IO :
        colors = ['k', 'grey', 'm', 'indigo']
        labels = ['SF purity', 'SF completeness', 'IO purity', 'IO completeness']
        col = 2
        pop = 'IO'
    else :
        colors = ['k', 'grey', 'r', 'darkred']
        labels = ['SF purity', 'SF completeness', 'OI purity', 'OI completeness']
        col = 3
        pop = 'OI'
    
    Noffsets = 41
    sf_purities = np.full((Niterations, Noffsets), -1.0)
    sf_completenesses = np.full((Niterations, Noffsets), -1.0)
    quenched_purities = np.full((Niterations, Noffsets), -1.0)
    quenched_completenesses = np.full((Niterations, Noffsets), -1.0)
    coefficients = np.full((Niterations, 4), -1.0)
    offsets = np.linspace(-2, 2, 41)
    for i in range(Niterations) :
        select = (np.random.rand(len(y_sf)) <= threshold)
        X_sf_it = X_sf[select]
        y_sf_it = y_sf[select]
        
        X_final = np.concatenate([X_quenched, X_sf_it])
        y_final = np.concatenate([y_quenched, y_sf_it])
        
        if fix_coefficients :
            if SF_and_IO :
                aa = -2.90513387
                bb = -2.45755192
                cc = -1.26427034
                dd = 2.26105976
            else :
                aa = -0.52796386
                bb = 2.07145622
                cc = 0.39705317
                dd = -0.11663121
        else :
            # classify the data
            fit = classifier.fit(X_final, y_final)
            
            if SF_and_IO :
                # define the coefficients to separate SF from IO quenchers,
                # using C_SF, R_SF, and Rinner (median, -1sigma, +1sigma values
                # listed for 1000 iterations)
                aa = fit.coef_[0][0]   # -2.90513387 -0.34961385 +0.36058785
                bb = fit.coef_[0][1]   # -2.45755192 -0.2669109  +0.29921949
                cc = fit.coef_[0][2]   # -1.26427034 -0.07732092 +0.08821039
                dd = fit.intercept_[0] #  2.26105976 -0.19093801 +0.15973421
            else :
                # define the coefficients to separate SF from OI quenchers,
                # using C_SF, R_SF, and Router (median, -1sigma, +1sigma values
                # listed for 1000 iterations)
                aa = fit.coef_[1][0]   # -0.52796386 -0.2211186  +0.19816578
                bb = fit.coef_[1][1]   #  2.07145622 -0.3309247  +0.38390364
                cc = fit.coef_[1][3]   #  0.39705317 -0.03814128 +0.03362633
                dd = fit.intercept_[1] # -0.11663121 -0.13552855 +0.145316
        
        # keep track of the coefficients across all iterations
        coefficients[i] = [aa, bb, cc, dd]
        
        # determine the orthogonal distances between planes with different
        # intercepts, here determined using an additional offset
        # distance_max = np.sqrt(np.sum(np.square([aa, bb, cc])))
        # offsets[i] = np.linspace(-distance_max, distance_max, Noffsets)
        
        # calculate purity and completeness for various offsets from intercept
        for j, offset in enumerate(offsets) :
            
            sf_above_boundary_check = (X_sf_it[:, col] >= surface(aa, bb,
                cc, dd + offset, X_sf_it[:, 0], X_sf_it[:, 1]))
            sf_above_boundary = np.sum(sf_above_boundary_check)
            sf_below_boundary = np.sum(~sf_above_boundary_check)
            
            oi_above_boundary_check = (X_oi[:, col] >= surface(aa, bb,
                cc, dd + offset, X_oi[:, 0], X_oi[:, 1]))
            oi_above_boundary = np.sum(oi_above_boundary_check)
            oi_below_boundary = np.sum(~oi_above_boundary_check)
            
            io_above_boundary_check = (X_io[:, col] >= surface(aa, bb,
                cc, dd + offset, X_io[:, 0], X_io[:, 1]))
            io_above_boundary = np.sum(io_above_boundary_check)
            io_below_boundary = np.sum(~io_above_boundary_check)
            
            am_above_boundary_check = (X_amb[:, col] >= surface(aa, bb,
                cc, dd + offset, X_amb[:, 0], X_amb[:, 1]))
            am_above_boundary = np.sum(am_above_boundary_check)
            am_below_boundary = np.sum(~am_above_boundary_check)
            
            # print('{:5.2f} {:4} {:4} {:4} | {:4} {:4} {:4}'.format(
            #     offset/distance_max, sf_total, sf_above_boundary,
            #     sf_below_boundary, oi_total, oi_above_boundary,
            #     oi_below_boundary))
            
            if SF_and_IO :
                sf_purity_denom = (sf_below_boundary + io_below_boundary +
                                   oi_below_boundary + am_below_boundary)
                if sf_purity_denom >= 100 :
                    sf_purity = sf_below_boundary/sf_purity_denom
                else :
                    sf_purity = np.nan
                
                quenched_purity = io_above_boundary/(sf_above_boundary +
                    io_above_boundary + oi_above_boundary + am_above_boundary)
                
                sf_completeness = sf_below_boundary/(sf_above_boundary +
                                                     sf_below_boundary)
                quenched_completeness = io_above_boundary/(io_above_boundary +
                                                           io_below_boundary)
                
            else :
                sf_purity = sf_above_boundary/(sf_above_boundary +
                    io_above_boundary + oi_above_boundary + am_above_boundary)
                quenched_purity = oi_below_boundary/(sf_below_boundary +
                    io_below_boundary + oi_below_boundary + am_below_boundary)
                
                sf_completeness = sf_above_boundary/(sf_above_boundary +
                                                     sf_below_boundary)
                quenched_completeness = oi_below_boundary/(oi_above_boundary +
                                                           oi_below_boundary)
            
            sf_purities[i, j] = sf_purity
            sf_completenesses[i, j] = sf_completeness
            quenched_purities[i, j] = quenched_purity
            quenched_completenesses[i, j] = quenched_completeness
    
    coeff_16, coeff_50, coeff_84 = np.percentile(coefficients, [16, 50, 84], axis=0)
    # print(coeff_50)
    # print(coeff_16 - coeff_50)
    # print(coeff_84 - coeff_50)
    
    sf_pur_16, sf_pur_50, sf_pur_84 = np.nanpercentile(
        sf_purities, [16, 50, 84], axis=0)
    sf_com_16, sf_com_50, sf_com_84 = np.percentile(
        sf_completenesses, [16, 50, 84], axis=0)
    quenched_pur_16, quenched_pur_50, quenched_pur_84 = np.percentile(
        quenched_purities, [16, 50, 84], axis=0)
    quenched_com_16, quenched_com_50, quenched_com_84 = np.percentile(
        quenched_completenesses, [16, 50, 84], axis=0)
    
    los = [sf_pur_16, sf_com_16, quenched_pur_16, quenched_com_16]
    meds = [sf_pur_50, sf_com_50, quenched_pur_50, quenched_com_50]
    his = [sf_pur_84, sf_com_84, quenched_pur_84, quenched_com_84]
    styles = ['-', '--', '-', '--']
    xlabel = r'orthogonal distance from boundary surface ($ax + by + cz + d = 0$)'
    ylabel = 'fraction'
    outfile = 'purity_completeness_{}_{}iterations_balance{}.png'.format(
        pop, Niterations, balance_mode)
    plt.plot_simple_multi_with_bands(offsets,
        los, meds, his, colors, styles, labels, xlabel=xlabel, ylabel=ylabel,
        xmin=-2, xmax=2,
        ymin=0, ymax=1,
        figsizeheight=8, #figsizeheight=9,
        figsizewidth=10.5, #figsizewidth=12,
        loc=loc, save=save, outfile=outfile)
    
    return

def locations_of_misclassified(balance_populations_manually=True) :
    
    CSF_label = r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$'
    RSF_label = r'$\log{(R_{\rm SF} = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}})}$'
    # Rinner_label = r'$R_{\rm inner}/R_{\rm e}$'
    Router_label = r'$R_{\rm outer}/R_{\rm e}$'
    outDir = 'TNG50-1/figures/misclassified_OI_galaxies_in_metric_space/'
    
    y_quenched, y_sf, X_quenched, X_sf, X_io, X_oi, X_amb = get_late_data()
    
    # setup the SVC classifier using a linear kernel
    if balance_populations_manually :
        classifier = SVC(kernel='linear')
        # select the same number of SF galaxies as quenching galaxies
        threshold = len(y_quenched)/len(y_sf)
    else :
        classifier = SVC(kernel='linear', class_weight='balanced')
        # select all the SF galaxies
        threshold = 1
    
    select = (np.random.rand(len(y_sf)) <= threshold)
    X_sf_it = X_sf[select]
    y_sf_it = y_sf[select]
    
    X_final = np.concatenate([X_quenched, X_sf_it])
    y_final = np.concatenate([y_quenched, y_sf_it])
    
    # classify the data
    fit = classifier.fit(X_final, y_final)
    y_predict = fit.predict(X_final)
    
    # find the galaxies that are OI but misclassified as SF
    oi_misclassified_as_sf = (y_final == 3) & (y_predict == 0)
    locations = np.where(oi_misclassified_as_sf == True)[0]
    
    # get the morphological metrics for those galaxies
    oi_misclassified_as_sf_metrics = X_final[locations]
    
    # 1) R_SF vs C_SF
    outfile = outDir + 'RSF_vs_CSF.png'
    xs = oi_misclassified_as_sf_metrics[:, 0]
    ys = oi_misclassified_as_sf_metrics[:, 1]
    contour_xs = [X_sf[:, 0], X_oi[:, 0]]
    contour_ys = [X_sf[:, 1], X_oi[:, 1]]
    plt.plot_scatter_with_contours(xs, ys, contour_xs, contour_ys,
        ['k', 'r'], xlabel=CSF_label, ylabel=RSF_label, xmin=0, xmax=1,
        ymin=-1, ymax=1, save=False, outfile=outfile)
    
    # 3) Router vs C_SF
    outfile = outDir + 'Router_vs_CSF.png'
    xs = oi_misclassified_as_sf_metrics[:, 0]
    ys = oi_misclassified_as_sf_metrics[:, 3]
    contour_xs = [X_sf[:, 0], X_oi[:, 0]]
    contour_ys = [X_sf[:, 3], X_oi[:, 3]]
    plt.plot_scatter_with_contours(xs, ys, contour_xs, contour_ys,
        ['k', 'r'], xlabel=CSF_label, ylabel=Router_label, xmin=0, xmax=1,
        ymin=0, ymax=5, loc=3, save=False, outfile=outfile)
    
    # 5) Router vs R_SF
    outfile = outDir + 'Router_vs_RSF.png'
    xs = oi_misclassified_as_sf_metrics[:, 1]
    ys = oi_misclassified_as_sf_metrics[:, 3]
    contour_xs = [X_sf[:, 1], X_oi[:, 1]]
    contour_ys = [X_sf[:, 3], X_oi[:, 3]]
    plt.plot_scatter_with_contours(xs, ys, contour_xs, contour_ys,
        ['k', 'r'], xlabel=RSF_label, ylabel=Router_label, xmin=-1, xmax=1,
        ymin=0, ymax=5, loc=4, save=False, outfile=outfile)
    
    '''
    # 2) Rinner vs C_SF
    xs = oi_misclassified_as_sf_metrics[:, 0]
    ys = oi_misclassified_as_sf_metrics[:, 2]
    contour_xs = [X_sf[:, 0], X_oi[:, 0]]
    contour_ys = [X_sf[:, 2], X_oi[:, 2]]
    plt.plot_scatter_with_contours(xs, ys, contour_xs, contour_ys,
        ['k', 'r'], xlabel=CSF_label, ylabel=Rinner_label, xmin=0, xmax=1,
        ymin=0, ymax=5)
    
    # 4) Rinner vs R_SF
    xs = oi_misclassified_as_sf_metrics[:, 1]
    ys = oi_misclassified_as_sf_metrics[:, 2]
    contour_xs = [X_sf[:, 1], X_oi[:, 1]]
    contour_ys = [X_sf[:, 2], X_oi[:, 2]]
    plt.plot_scatter_with_contours(xs, ys, contour_xs, contour_ys,
        ['k', 'r'], xlabel=RSF_label, ylabel=Rinner_label, xmin=-1,
        xmax=1, ymin=0, ymax=5)
    
    # 6) Router vs Rinner
    xs = oi_misclassified_as_sf_metrics[:, 2]
    ys = oi_misclassified_as_sf_metrics[:, 3]
    contour_xs = [X_sf[:, 2], X_oi[:, 2]]
    contour_ys = [X_sf[:, 3], X_oi[:, 3]]
    plt.plot_scatter_with_contours(xs, ys, contour_xs, contour_ys,
        ['k', 'r'], xlabel=Rinner_label, ylabel=Router_label, xmin=0,
        xmax=5, ymin=0, ymax=5)
    '''
    
    return

def find_boundaries(balance_populations_manually=True, save=False) :
    
    y_quenched, y_sf, X_quenched, X_sf, X_io, X_oi, X_amb = get_late_data()
    
    if balance_populations_manually :
        threshold = len(y_quenched)/len(y_sf) # select the same number of SF
    else :                                    # galaxies as quenching galaxies
        threshold = 1 # select all the SF galaxies
    
    select = (np.random.rand(len(y_sf)) <= threshold)
    X_sf_it = X_sf[select]
        
    # coefficients that separate SF from IO, using CSF+RSF+Rinner
    # (median, -1sigma, +1sigma values listed for 1000 iterations)
    # fit.coef_[0][0]   # -2.90513387 -0.34961385 +0.36058785
    # fit.coef_[0][1]   # -2.45755192 -0.2669109  +0.29921949
    # fit.coef_[0][2]   # -1.26427034 -0.07732092 +0.08821039
    # fit.intercept_[0] #  2.26105976 -0.19093801 +0.15973421
    create_boundary_plot(-2.90513387, -2.45755192, -1.26427034, 2.26105976,
        X_sf_it, X_io, X_oi, X_amb, 'Rinner', save=save)
    
    # coefficients that separate SF from OI, using CSF+RSF+Router
    # (median, -1sigma, +1sigma values listed for 1000 iterations)
    # fit.coef_[1][0]   # -0.52796386 -0.2211186  +0.19816578
    # fit.coef_[1][1]   #  2.07145622 -0.3309247  +0.38390364
    # fit.coef_[1][3]   #  0.39705317 -0.03814128 +0.03362633
    # fit.intercept_[1] # -0.11663121 -0.13552855 +0.145316
    create_boundary_plot(-0.52796386, 2.07145622, 0.39705317, -0.11663121,
        X_sf_it, X_io, X_oi, X_amb, 'Router', save=save)
    
    # create a single figure with two panels
    create_side_by_side_plot(-2.90513387, -2.45755192, -1.26427034, 2.26105976,
        -0.52796386, 2.07145622, 0.39705317, -0.11663121,
        X_sf_it, X_io, X_oi, X_amb)
    
    # classify the data
    # classifier = SVC(kernel='linear') # class_weight='balanced'
    # X_final = np.concatenate([X_quenched, X_sf_it])
    # y_final = np.concatenate([y_quenched, y_sf[select]])
    # fit = classifier.fit(X_final, y_final)
    
    return

def create_side_by_side_plot(a1, b1, c1, d1, a2, b2, c2, d2, X_sf, X_io, X_oi, X_am) :
    
    textwidth = 7.10000594991006
    textheight = 9.095321710253218
    
    xprimes1 = [-(a1*X_sf[:, 0] + b1*X_sf[:, 1] + d1)/c1,
                -(a1*X_io[:, 0] + b1*X_io[:, 1] + d1)/c1,
                -(a1*X_oi[:, 0] + b1*X_oi[:, 1] + d1)/c1,
                -(a1*X_am[:, 0] + b1*X_am[:, 1] + d1)/c1]
    # ys1 = [X_sf[:, 2], X_io[:, 2], X_oi[:, 2], X_am[:, 2]]
    ys1 = [np.random.normal(X_sf[:, 2], 0.03), np.random.normal(X_io[:, 2], 0.03),
           np.random.normal(X_oi[:, 2], 0.03), np.random.normal(X_am[:, 2], 0.03)]
    xlabel1 = r'$-2.298 C_{\rm SF} - 1.944 R_{\rm SF} + 1.788$'
    ylabel1 = r'$R_{\rm inner}/R_{\rm e}$'
    
    xprimes2 = [-(a2*X_sf[:, 0] + b2*X_sf[:, 1] + d2)/c2,
                -(a2*X_io[:, 0] + b2*X_io[:, 1] + d2)/c2,
                -(a2*X_oi[:, 0] + b2*X_oi[:, 1] + d2)/c2,
                -(a2*X_am[:, 0] + b2*X_am[:, 1] + d2)/c2]
    # ys2 = [X_sf[:, 3], X_io[:, 3], X_oi[:, 3], X_am[:, 3]]
    ys2 = [np.random.normal(X_sf[:, 3], 0.03), np.random.normal(X_io[:, 3], 0.03),
           np.random.normal(X_oi[:, 3], 0.03), np.random.normal(X_am[:, 3], 0.03)]
    xlabel2 = r'$1.330 C_{\rm SF} - 5.217 R_{\rm SF} + 0.294$'
    ylabel2 = r'$R_{\rm outer}/R_{\rm e}$'
    
    xx = np.linspace(-0.1, 5.1, 1001)
    
    colors = ['k', 'm', 'r', 'orange']
    alphas = [0.2, 0.2, 0.2, 0.2]
    labels = ['SF', 'inside-out', 'outside-in', 'ambiguous']
    markers = ['o', 's', 's', 's']
    
    plt.double_scatter_with_line(xprimes1, ys1, colors, markers, alphas, xx, xx,
        xprimes2, ys2, colors, markers, alphas, xx, xx, labels,
        xlabel1=xlabel1, ylabel1=ylabel1, xlabel2=xlabel2, ylabel2=ylabel2,
        xmin1=-1.3, xmax1=2.3, xmin2=-8, xmax2=9, ymin=-0.1, ymax=5.1,
        figsizewidth=textwidth, figsizeheight=textheight/3, loc=2,
        save=False, outfile='classification_boundaries.pdf')
    
    return

def create_boundary_plot(aa, bb, cc, dd, X_sf, X_io, X_oi, X_am, variety, save=False) :
    
    CSF_label = r'$C_{\rm SF} = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$'
    RSF_label = r'$R_{\rm SF} = \log{(R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}})}$'
    Rinner_label = r'$R_{\rm inner}/R_{\rm e}$'
    Router_label = r'$R_{\rm outer}/R_{\rm e}$'
    
    if variety == 'Rinner' :
        zlabel, col, loc = Rinner_label, 2, 2
        xprime_label = r'$-2.298 C_{\rm SF} - 1.944 R_{\rm SF} + 1.788$'
        xprime_min, xprime_max = -1.3, 2.3
    else :
        zlabel, col, loc = Router_label, 3, 3
        xprime_label = r'$1.330 C_{\rm SF} - 5.217 R_{\rm SF} + 0.294$'
        xprime_min, xprime_max = -8, 9
    
    # setup a meshgrid to sample the boundary surface
    xx, yy = np.meshgrid(np.linspace(0, 1, 101), np.linspace(-1, 1, 201))
    
    # find the surface which separates the SF galaxies from the quenched class
    # of interest (IO or OI)
    zz = surface(aa, bb, cc, dd, xx, yy)
    
    # mask surface features that are outside the plotting area
    zz[(zz < 0) | (zz > 5)] = np.nan
    
    # C_SF vs R_SF vs Rinner/Router
    xs = [X_sf[:, 0], X_io[:, 0], X_oi[:, 0], X_am[:, 0]]
    ys = [X_sf[:, 1], X_io[:, 1], X_oi[:, 1], X_am[:, 1]]
    zs = [X_sf[:, col], X_io[:, col], X_oi[:, col], X_am[:, col]]
    outfile = 'parameter_space_3d_{}.png'.format(variety)
    # plt.plot_scatter_3d(xs, ys, zs, ['k', 'm', 'r', 'orange'],
    #     ['o', 's', 's', 's'], ['late SF', 'late IO', 'late OI', 'late AMB'],
    #     [20, 20, 20, 20], xx, yy, zz, xlabel=CSF_label, ylabel=RSF_label,
    #     zlabel=zlabel, xmin=0, xmax=1, ymin=-1, ymax=1, zmin=0, zmax=5,
    #     figsizewidth=9.5, figsizeheight=9, save=save, outfile=outfile)
        # azim=187.25, elev=0
    
    # now project the 3D plot into 2D
    xprimes = [-(aa*X_sf[:, 0] + bb*X_sf[:, 1] + dd)/cc,
               -(aa*X_io[:, 0] + bb*X_io[:, 1] + dd)/cc,
               -(aa*X_oi[:, 0] + bb*X_oi[:, 1] + dd)/cc,
               -(aa*X_am[:, 0] + bb*X_am[:, 1] + dd)/cc]
    outfile = 'parameter_space_3d-into-2d_{}.pdf'.format(variety)
    equality = np.linspace(-0.1, 5.1, 1001)
    plt.plot_scatter_multi_with_line(xprimes, zs, ['k', 'm', 'r', 'orange'],
        ['SF', 'inside-out', 'outside-in', 'ambiguous'], ['o', 's', 's', 's'],
        [0.3, 0.3, 0.3, 0.3], equality, equality, xlabel=xprime_label,
        ylabel=zlabel, xmin=xprime_min, xmax=xprime_max, ymin=-0.1, ymax=5.1,
        figsizewidth=7.10000594991006/2, figsizeheight=9.095321710253218/3,
        save=False, outfile=outfile, loc=loc)
    
    return

def surface(aa, bb, cc, dd, xs, ys) :
    # given a plane of the form aa*xx + bb*yy + cc*zz + dd = 0, solve for zz
    surf = lambda xx, yy : (-dd - aa*xx - bb*yy)/cc
    return surf(xs, ys)

'''
y_quenched, y_sf, X_quenched, X_sf, X_io, X_oi, X_amb = get_late_data()

# setup the SVC classifier using a linear kernel
if True :
    classifier = SVC(kernel='linear')
    # select the same number of SF galaxies as quenching galaxies
    threshold = len(y_quenched)/len(y_sf)
else :
    classifier = SVC(kernel='linear', class_weight='balanced')
    # select all the SF galaxies
    threshold = 1

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

np.random.seed(0)
select = (np.random.rand(len(y_sf)) <= threshold)
X_sf_it = X_sf[select]
y_sf_it = y_sf[select]

X_final = np.concatenate([X_quenched, X_sf_it])
y_final = np.concatenate([y_quenched, y_sf_it])

# determine the principal components using PCA
# pca = PCA(n_components=3, svd_solver='full')
# fit = pca.fit(X_final)

# print(fit.components_)
# print(fit.explained_variance_ratio_)
# print(np.cumsum(fit.explained_variance_ratio_))

# X_r = fit.transform(X_final)

# plt.plot_scatter_3d([X_r[:, 0]], [X_r[:, 1]], [X_r[:, 2]], ['k'], ['o'], ['pca'],
#                     [20], None, None, None)


# lda = LinearDiscriminantAnalysis(n_components=3)
# fit = lda.fit(X_final, y_final)
# print(fit.intercept_)
# print(fit.coef_)
# print(fit.explained_variance_ratio_)
# print(np.cumsum(fit.explained_variance_ratio_))

# X_r = fit.transform(X_final)

# xs = X_r[:, 0]

# plt.plot_scatter_3d([X_r[:, 0]], [X_r[:, 1]], [X_r[:, 2]], ['k'], ['o'], ['lda'],
#                     [20], None, None, None)

'''
