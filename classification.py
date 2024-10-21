
import numpy as np

from sklearn.svm import SVC

from core import get_late_data, surface
import plotting as plt

def calculate_purity_completeness(Niterations, offsets, X_sf, X_io, X_oi, X_am,
                                  col, aa, bb, cc, dd) :
    
    sf_purities = np.full(len(offsets), -1.0)
    sf_completenesses = np.full(len(offsets), -1.0)
    quenched_purities = np.full(len(offsets), -1.0)
    quenched_completenesses = np.full(len(offsets), -1.0)
    quenched_amb_purs = np.full(len(offsets), -1.0)
    quenched_amb_coms = np.full(len(offsets), -1.0)
    
    # calculate purity and completeness for various offsets from intercept
    for i, offset in enumerate(offsets) :
        
        sf_above_boundary_check = (X_sf[:, col] >= surface(aa, bb,
            cc, dd + offset, X_sf[:, 0], X_sf[:, 1]))
        sf_abv = np.sum(sf_above_boundary_check)
        sf_bel = np.sum(~sf_above_boundary_check)
        
        oi_above_boundary_check = (X_oi[:, col] >= surface(aa, bb,
            cc, dd + offset, X_oi[:, 0], X_oi[:, 1]))
        oi_abv = np.sum(oi_above_boundary_check)
        oi_bel = np.sum(~oi_above_boundary_check)
        
        io_above_boundary_check = (X_io[:, col] >= surface(aa, bb,
            cc, dd + offset, X_io[:, 0], X_io[:, 1]))
        io_abv = np.sum(io_above_boundary_check)
        io_bel = np.sum(~io_above_boundary_check)
        
        am_above_boundary_check = (X_am[:, col] >= surface(aa, bb,
            cc, dd + offset, X_am[:, 0], X_am[:, 1]))
        am_abv = np.sum(am_above_boundary_check)
        am_bel = np.sum(~am_above_boundary_check)
        
        # print('{:5.2f} {:4} {:4} {:4} | {:4} {:4} {:4}'.format(
        #     offset/distance_max, sf_total, sf_above_boundary,
        #     sf_below_boundary, oi_total, oi_above_boundary,
        #     oi_below_boundary))
        
        if (col == 2) :
            sf_purity_denom = (sf_bel + io_bel + oi_bel + am_bel)
            if sf_purity_denom >= 100 :
                sf_purity = sf_bel/sf_purity_denom
            else :
                sf_purity = np.nan
            quenched_purity = io_abv/(sf_abv + io_abv + oi_abv + am_abv)
            quenched_amb_pur = (io_abv + am_abv)/(sf_abv + io_abv + oi_abv + am_abv)
            
            sf_completeness = sf_bel/(sf_abv + sf_bel)
            quenched_completeness = io_abv/(io_abv + io_bel)
            quenched_amb_com = (io_abv + am_abv)/(io_abv + io_bel + am_abv + am_bel)
            
        if (col == 3) :
            sf_purity = sf_abv/(sf_abv + io_abv + oi_abv + am_abv)
            quenched_purity = oi_bel/(sf_bel + io_bel + oi_bel + am_bel)
            quenched_amb_pur = (oi_bel + am_bel)/(sf_bel + io_bel + oi_bel + am_bel)
            
            sf_completeness = sf_abv/(sf_abv + sf_bel)
            quenched_completeness = oi_bel/(oi_abv + oi_bel)
            quenched_amb_com = (oi_bel + am_bel)/(oi_abv + oi_bel + am_abv + am_bel)
    
        sf_purities[i] = sf_purity
        sf_completenesses[i] = sf_completeness
        quenched_purities[i] = quenched_purity
        quenched_completenesses[i] = quenched_completeness
        quenched_amb_purs[i] = quenched_amb_pur
        quenched_amb_coms[i] = quenched_amb_com
    
    return (sf_purities, sf_completenesses, quenched_purities,
            quenched_completenesses, quenched_amb_purs, quenched_amb_coms)

def determine_purity_completeness(y_quenched, y_sf, X_quenched, X_sf, X_io,
                                  X_oi, X_am, col, offsets, threshold,
                                  Niterations=10) :
    
    if Niterations == 1 :
        # use all SF galaxies and fix the coefficients
        X_final = np.concatenate([X_quenched, X_sf])
        y_final = np.concatenate([y_quenched, y_sf])
        
        if (col == 2) :
            aa = -2.90513387
            bb = -2.45755192
            cc = -1.26427034
            dd = 2.26105976
        else :
            aa = -0.52796386
            bb = 2.07145622
            cc = 0.39705317
            dd = -0.11663121
        
        (sf_purities, sf_completenesses, quenched_purities,
         quenched_completenesses, quenched_amb_purs,
         quenched_amb_coms) = calculate_purity_completeness(
             Niterations, offsets, X_sf, X_io, X_oi, X_am, col, aa, bb, cc, dd)
    else :
        classifier = SVC(kernel='linear')
        
        sf_purities = np.full((Niterations, len(offsets)), -1.0)
        sf_completenesses = np.full((Niterations, len(offsets)), -1.0)
        quenched_purities = np.full((Niterations, len(offsets)), -1.0)
        quenched_completenesses = np.full((Niterations, len(offsets)), -1.0)
        quenched_amb_purs = np.full((Niterations, len(offsets)), -1.0)
        quenched_amb_coms = np.full((Niterations, len(offsets)), -1.0)
        
        coefficients = np.full((Niterations, 4), -1.0)
        for i in range(Niterations) :
            select = (np.random.rand(len(y_sf)) <= threshold)
            X_sf_it = X_sf[select]
            y_sf_it = y_sf[select]
            
            X_final = np.concatenate([X_quenched, X_sf_it])
            y_final = np.concatenate([y_quenched, y_sf_it])
            
            # classify the data
            fit = classifier.fit(X_final, y_final)
            
            if (col == 2) :
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
            
            (sf_purities[i], sf_completenesses[i], quenched_purities[i],
             quenched_completenesses[i], quenched_amb_purs[i],
             quenched_amb_coms[i]) = calculate_purity_completeness(
                  Niterations, offsets, X_sf_it, X_io, X_oi, X_am, col, aa, bb, cc, dd)
    
    # coeff_16, coeff_50, coeff_84 = np.percentile(coefficients, [16, 50, 84], axis=0)
    
    if Niterations > 1 :
        sf_pur_16, sf_pur_50, sf_pur_84 = np.nanpercentile(
            sf_purities, [16, 50, 84], axis=0)
        sf_com_16, sf_com_50, sf_com_84 = np.percentile(
            sf_completenesses, [16, 50, 84], axis=0)
        quenched_pur_16, quenched_pur_50, quenched_pur_84 = np.percentile(
            quenched_purities, [16, 50, 84], axis=0)
        quenched_com_16, quenched_com_50, quenched_com_84 = np.percentile(
            quenched_completenesses, [16, 50, 84], axis=0)
        quenched_amb_pur_16, quenched_amb_pur_50, quenched_amb_pur_84 = np.percentile(
            quenched_amb_purs, [16, 50, 84], axis=0)
        quenched_amb_com_16, quenched_amb_com_50, quenched_amb_com_84 = np.percentile(
            quenched_amb_coms, [16, 50, 84], axis=0)
        
        los = [sf_pur_16, sf_com_16, quenched_pur_16, quenched_com_16,
               quenched_amb_pur_16, quenched_amb_com_16]
        meds = [sf_pur_50, sf_com_50, quenched_pur_50, quenched_com_50,
                quenched_amb_pur_50, quenched_amb_com_50]
        his = [sf_pur_84, sf_com_84, quenched_pur_84, quenched_com_84,
               quenched_amb_pur_84, quenched_amb_com_84]
        
        return los, meds, his
    else :
        return [sf_purities, sf_completenesses, quenched_purities,
                quenched_completenesses, quenched_amb_purs, quenched_amb_coms]

def display_boundaries(balance_populations_manually=True) :
    
    y_quenched, y_sf, X_quenched, X_sf, X_io, X_oi, X_am = get_late_data()
    
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
    display_boundaries_helper(-2.90513387, -2.45755192, -1.26427034, 2.26105976,
        X_sf_it, X_io, X_oi, X_am, 'Rinner')
    
    # coefficients that separate SF from OI, using CSF+RSF+Router
    # (median, -1sigma, +1sigma values listed for 1000 iterations)
    # fit.coef_[1][0]   # -0.52796386 -0.2211186  +0.19816578
    # fit.coef_[1][1]   #  2.07145622 -0.3309247  +0.38390364
    # fit.coef_[1][3]   #  0.39705317 -0.03814128 +0.03362633
    # fit.intercept_[1] # -0.11663121 -0.13552855 +0.145316
    display_boundaries_helper(-0.52796386, 2.07145622, 0.39705317, -0.11663121,
        X_sf_it, X_io, X_oi, X_am, 'Router')
    
    # classify the data
    # classifier = SVC(kernel='linear') # class_weight='balanced'
    # X_final = np.concatenate([X_quenched, X_sf_it])
    # y_final = np.concatenate([y_quenched, y_sf[select]])
    # fit = classifier.fit(X_final, y_final)
    
    return

def display_boundaries_helper(aa, bb, cc, dd, X_sf, X_io, X_oi, X_am, variety) :
    
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
    plt.plot_scatter_3d(xs, ys, zs, ['k', 'm', 'r', 'orange'],
        ['o', 's', 's', 's'], ['SF', 'inside-out', 'outside-in', 'ambiguous'],
        [20, 20, 20, 20], xx, yy, zz, xlabel=CSF_label, ylabel=RSF_label,
        zlabel=zlabel, xmin=0, xmax=1, ymin=-1, ymax=1, zmin=0, zmax=5,
        figsizewidth=9.5, figsizeheight=9)
        # azim=187.25, elev=0
    
    # now project the 3D plot into 2D
    xprimes = [-(aa*X_sf[:, 0] + bb*X_sf[:, 1] + dd)/cc,
               -(aa*X_io[:, 0] + bb*X_io[:, 1] + dd)/cc,
               -(aa*X_oi[:, 0] + bb*X_oi[:, 1] + dd)/cc,
               -(aa*X_am[:, 0] + bb*X_am[:, 1] + dd)/cc]
    equality = np.linspace(-0.1, 5.1, 1001)
    plt.plot_scatter_multi_with_line(xprimes, zs, ['k', 'm', 'r', 'orange'],
        ['SF', 'inside-out', 'outside-in', 'ambiguous'], ['o', 's', 's', 's'],
        [0.3, 0.3, 0.3, 0.3], equality, equality, xlabel=xprime_label,
        ylabel=zlabel, xmin=xprime_min, xmax=xprime_max, ymin=-0.1, ymax=5.1,
        figsizewidth=7.10000594991006/2, figsizeheight=9.095321710253218/3,
        loc=loc)
    
    return

def save_classification_boundaries_plot(balance_populations_manually=True) :
    
    textwidth = 7.10000594991006
    textheight = 9.095321710253218
    
    y_quenched, y_sf, X_quenched, X_sf, X_io, X_oi, X_am = get_late_data()
    
    if balance_populations_manually :
        threshold = len(y_quenched)/len(y_sf) # select the same number of SF
    else :                                    # galaxies as quenching galaxies
        threshold = 1 # select all the SF galaxies
    
    select = (np.random.rand(len(y_sf)) <= threshold)
    X_sf = X_sf[select]
    
    a1, b1, c1, d1 = -2.90513387, -2.45755192, -1.26427034,  2.26105976
    a2, b2, c2, d2 = -0.52796386,  2.07145622,  0.39705317, -0.11663121
    
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
        titles=['inside-out', 'outside-in'],
        xmin1=-1.3, xmax1=2.3, xmin2=-8, xmax2=9, ymin=-0.1, ymax=5.1,
        figsizewidth=textwidth, figsizeheight=textheight/3, loc=2,
        save=False, outfile='classification_boundaries.pdf')
    
    return

def save_purity_completness_plot(Noffsets=41) :
    
    textwidth = 7.10000594991006
    textheight = 9.095321710253218
    
    # get the data for late times
    (y_quenched, y_sf, X_quenched, X_sf, X_io, X_oi, X_am, _, _, y_below,
     X_below, _) = get_late_data(basic=False)
    
    # setup the orthogonal offsets from the classification boundary planes
    offsets = np.linspace(-2, 2, Noffsets)
    
    # np.random.seed(0)
    
    '''
    # fix_coefficients=True, threshold=1, and Niterations=1 should all employ
    # the same behaviour, with the other behaviour being
    # fix_coefficients=False, threshold=len/len, Niterations=1000
    '''
    
    # select the same number of SF galaxies as quenching galaxies, for IO
    los1, meds1, his1 = determine_purity_completeness(y_quenched, y_sf,
        X_quenched, X_sf, X_io, X_oi, X_am, 2, offsets, len(y_quenched)/len(y_sf),
        Niterations=1000)
    
    # select the same number of SF galaxies as quenching galaxies, for OI
    los2, meds2, his2 = determine_purity_completeness(y_quenched, y_sf,
        X_quenched, X_sf, X_io, X_oi, X_am, 3, offsets, len(y_quenched)/len(y_sf),
        Niterations=1000)
    
    # select all the SF galaxies, for IO
    # meds3 = determine_purity_completeness(y_quenched, y_sf, X_quenched, X_sf,
    #     X_io, X_oi, X_am, 2, offsets, 1, Niterations=1)
    
    # select all the SF galaxies, for OI
    # meds4 = determine_purity_completeness(y_quenched, y_sf, X_quenched, X_sf,
    #     X_io, X_oi, X_am, 3, offsets, 1, Niterations=1)
    
    # colors1 = ['k', 'grey', 'm', 'indigo']
    # labels1 = ['SF purity', 'SF completeness', 'inside-out purity', 'inside-out completeness']
    # colors2 = ['k', 'grey', 'r', 'darkred']
    # labels2 = ['SF purity', 'SF completeness', 'outside-in purity', 'outside-in completeness']
    # colors3 = colors1 + ['gold', 'darkorange']
    # labels3 = labels1 + ['+ambiguous purity', '+ambiguous completeness']
    # colors4 = colors2 + ['gold', 'darkorange']
    # labels4 = labels2 + ['+ambiguous purity', '+ambiguous completeness']
    # styles = ['-', '--', '-', '--', '-', '--']
    
    # plt.quad_grid_plot(offsets, meds1, los1, his1, colors1, labels1,
    #     meds2, los2, his2, colors2, labels2, meds3, colors3, labels3,
    #     meds4, colors4, labels4, styles, titles=['inside-out', 'outside-in'],
    #     xlabel=r'orthogonal distance from boundary surface',
    #     ylabel='fraction', save=False, outfile='purity_completeness.pdf',
    #     figsizeheight=textheight/2, figsizewidth=textwidth)
    
    colors1 = ['k', 'grey', 'm', 'indigo', 'gold', 'darkorange']
    labels1 = ['SF purity', '', 'inside-out/+ambig. purity',
               'inside-out/+ambig. compl.', 'inside-out/+ambig. purity',
               'inside-out/+ambig. compl.']
    colors2 = ['k', 'grey', 'r', 'darkred', 'gold', 'darkorange']
    labels2 = ['', 'SF completeness', 'outside-in/+ambig. purity',
               'outside-in/+ambig. compl.', 'outside-in/+ambig. purity',
               'outside-in/+ambig. compl.']
    styles = ['-', '--', '-', '--', '-', '--']
    zorders = np.array([4, 1, 5, 2, 6, 3])/6
    
    plt.double_grid_plot(offsets, meds1, los1, his1, colors1, labels1,
        meds2, los2, his2, colors2, labels2, styles, zorders=zorders,
        titles=['inside-out', 'outside-in'],
        xlabel=r'orthogonal distance from boundary surface',
        ylabel='fraction', save=False, outfile='purity_completeness.pdf',
        figsizeheight=textheight/3, figsizewidth=textwidth)
    
    return
