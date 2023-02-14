
import copy
import matplotlib.pyplot as plt

from matplotlib import cm
import matplotlib.colors as mcol
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import seaborn as sns

currentFig = 1

def histogram(data, label, title=None, bins=None, log=False, histtype='bar',
              vlines=[], colors=[], labels=[], loc='upper left') :
    
    global currentFig
    fig = plt.figure(currentFig)
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    if bins and not log :
        ax.hist(data, bins=bins, color='k', histtype=histtype)
    elif bins and log :
        ax.hist(data, bins=bins, log=log, color='k', histtype=histtype)
    elif log and not bins :
        ax.hist(data, log=log, color='k', histtype=histtype)
    else :
        ax.hist(data, histtype=histtype)
    
    if len(vlines) > 0 :
        for i in range(len(vlines)) :
            ax.axvline(vlines[i], ls='--', color=colors[i], lw=1, alpha=0.5,
                       label=labels[i])
    
    ax.set_xlabel('{}'.format(label), fontsize = 15)    
    
    if len(vlines) > 0 :
        ax.legend(loc=loc, facecolor='whitesmoke', framealpha=1,
                  fontsize=15)
    
    plt.tight_layout()
    plt.show()
    
    return

def histogram_2d(xhist, yhist, label=None, #xscatter, yscatter, xs, ys, fitx, fity, labels,
                 # styles,
                 bad='white', bins=[20,20], cmap=cm.Blues, title=None,
                 norm=LogNorm(), outfile=None, xlabel=None, ylabel=None,
                 xmin=None, xmax=None, ymin=None, ymax=None, save=False,
                 figsizewidth=9.5, figsizeheight=7, loc=0) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    cmap = copy.copy(cmap)
    cmap.set_bad(bad, 1)
    
    ax.hist2d(xhist, yhist, bins=bins, #range=[[xmin, xmax], [ymin, ymax]],
               cmap=cmap, norm=norm, alpha=0.7, label=label)
    
    # for i in range(len(ys)) :
    #     ax.plot(xs[i], ys[i], styles[i], color='k')
    
    # for i in range(len(fity)) :
    #     ax.plot(fitx[i], fity[i], 'r-', label=labels[i])
    
    # ax.plot(xscatter, yscatter, 'ro', label=labels[-1])
    
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # ax.legend(loc=loc, facecolor='whitesmoke', framealpha=1, fontsize=15)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter(xs, ys, color, label, marker, cbar_label='',
                 xlabel=None, ylabel=None, title=None, cmap=cm.rainbow,
                 xmin=None, xmax=None, ymin=None, ymax=None, loc=0,
                 figsizewidth=9.5, figsizeheight=7, scale='linear',
                 vmin=None, vmax=None, save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    cmap = copy.copy(cmap)
    # norm = Normalize(vmin=vmin, vmax=vmax)
        
    frame = ax.scatter(xs, ys, c=color, marker=marker, label=label, cmap=cmap,
                       edgecolors='grey')
    # cbar = plt.colorbar(frame)
    # cbar.set_label(cbar_label, fontsize=15)
    
    ax.set_yscale(scale)
    ax.set_xscale(scale)
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter_3d(xs, ys, zs, colors, markers, scale='linear',
                    xlabel=None, ylabel=None, zlabel=None, xmin=None, xmax=None,
                    ymin=None, ymax=None, zmin=None, zmax=None,
                    figsizewidth=9.5, figsizeheight=7, save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
        
    for i in range(len(xs)) :
        ax.scatter(xs[i], ys[i], zs[i], c=colors[i], marker=markers[i])
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_zscale(scale)
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_zlabel(zlabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter_CASTOR(dx, dy, sf_dx, sf_dy, sf_masses, radii, hist, XX, YY,
                        df=None, legend=True, title=None, cbar_label=None,
                        bad='black', bins=[20,20], cmap=cm.inferno, #cmap=cm.Blues,
                        norm=LogNorm(vmin=1e-11, vmax=2e-09), #(vmin=0.001, vmax=0.7),
                        xlabel=None, ylabel=None,
                        xmin=None, xmax=None, ymin=None, ymax=None,
                        figsizewidth=7, figsizeheight=7, save=False,
                        outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    cmap = copy.copy(cmap)
    cmap.set_bad(bad, 1)
    
    if df is not None :
        sns.kdeplot(data=df, x='dx', y='dz', color='limegreen', weights='masses',
                    levels=[0.1, 0.2, 0.3, 0.6, 0.9], ax=ax, linewidths=2)
        ax.plot([2*xmax], [2*xmax], color='limegreen', ls='-',
                label='non-SF stellar particles', lw=2)
        # hist, _, _, image = ax.hist2d(sf_dx, sf_dy, bins=bins, cmap=cmap,
        #                               norm=norm, alpha=0.9, weights=sf_masses)
        
        image = ax.pcolormesh(XX, YY, hist, cmap=cmap, norm=norm, alpha=0.9)
        
        if legend :
            cbar = plt.colorbar(image)
            cbar.set_label(cbar_label, fontsize=20)
            
            ax.text(-1.25, -2.25, r'$2 R_{\rm e}$', fontsize=18, color='w')
            ax.text(2.8, -3.1, r'$4 R_{\rm e}$', fontsize=18, color='w')
        
    else :
        ax.scatter(dx, dy, color='r', alpha=0.05,
                   label='non-SF stellar particles')
        ax.scatter(sf_dx, sf_dy, color='b', alpha=0.1, label='SF particles')
    
    for i in range(len(radii)) :
        circle = Circle((0, 0), radius=radii[i], facecolor='none', ls=':',
                        edgecolor='w', linewidth=1.5, alpha=1, zorder=3)
        ax.add_patch(circle)
    
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    ax.tick_params(axis='both', which='major', labelsize=15)
    # ax.tick_params(axis='both', which='minor', labelsize=8)
    
    if legend :
        ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=18)
    
    ax.axes.set_aspect('equal')
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter_multi(dx, dy, dz, sf_dx, sf_dy, sf_dz, xlabel=None, ylabel=None,
                       zlabel=None, figsizewidth=18, figsizeheight=6, save=False,
                       outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    ax1.scatter(dx, dy, color='r', alpha=0.05)
    ax1.scatter(sf_dx, sf_dy, color='b', alpha=0.1)
    ax1.set_xlabel(xlabel, fontsize=15)
    ax1.set_ylabel(ylabel, fontsize=15)
    ax1.set_xlim(-30, 30)
    ax1.set_ylim(-30, 30)
    
    ax2.scatter(dx, dz, color='r', alpha=0.05)
    ax2.scatter(sf_dx, sf_dz, color='b', alpha=0.1)
    ax2.set_xlabel(xlabel, fontsize=15)
    ax2.set_ylabel(zlabel, fontsize=15)
    ax2.set_xlim(-30, 30)
    ax2.set_ylim(-30, 30)
    
    ax3.scatter(dy, dz, color='r', alpha=0.05)
    ax3.scatter(sf_dy, sf_dz, color='b', alpha=0.1)
    ax3.set_xlabel(ylabel, fontsize=15)
    ax3.set_ylabel(zlabel, fontsize=15)
    ax3.set_xlim(-30, 30)
    ax3.set_ylim(-30, 30)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_simple_dumb(xs, ys, label='',
                     xlabel=None, ylabel=None, title=None,
                     xmin=None, xmax=None, ymin=None, ymax=None,
                     figsizewidth=9.5, figsizeheight=7) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    ax.plot(xs, ys, 'k-', label=label)
    
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if label != '' :
        ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15)
    
    plt.tight_layout()
    plt.show()
    
    return

def plot_simple_many(xs, ys, xlabel=None, ylabel=None, xmin=None, xmax=None,
                     ymin=None, ymax=None, save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(10, 7))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    for yy in ys :
        ax.plot(xs, yy, 'C1', alpha=0.2)
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_simple_multi(xs, ys, labels, colors, markers, styles, alphas,
                      xlabel=None, ylabel=None, title=None,
                      xmin=None, xmax=None, ymin=None, ymax=None,
                      figsizewidth=9.5, figsizeheight=7, scale='log', loc=0,
                      outfile=None, save=False) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    for i in range(len(xs)) :
        ax.plot(xs[i], ys[i], marker=markers[i], linestyle=styles[i],
                color=colors[i], label=labels[i], alpha=alphas[i])
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # if labels[0] != '' :
    ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_simple_with_band(xs, ys, lo, med, hi, xlabel=None, ylabel=None,
                          xmin=None, xmax=None, ymin=None, ymax=None,
                          figsizewidth=7, figsizeheight=7, scale='linear', loc=0,
                          outfile=None, save=False, legend=False) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    ax.plot(xs, med, 'k:')
    ax.plot(xs, ys, 'k-')
    ax.fill_between(xs, lo, hi, color='grey', edgecolor='darkgrey', alpha=0.2)
    
    ax.set_xscale(scale)
    ax.set_yscale('log')
    
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if legend :
        ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=18, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_simple_multi_with_times(xs, ys, labels, colors, markers, styles,
                                 alphas, tsat, tonset, ttermination,
                                 xlabel=None, ylabel=None, title=None,
                                 xmin=None, xmax=None, ymin=None, ymax=None,
                                 figsizewidth=9.5, figsizeheight=7, scale='log',
                                 loc=0, outfile=None, save=False) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    for i in range(len(xs)) :
        ax.plot(xs[i], ys[i], marker=markers[i], linestyle=styles[i],
                color=colors[i], label=labels[i], alpha=alphas[i])
    
    if (ymin == None) :
        _, _, ymin, _ = plt.axis()
    if (ymax == None) :
        _, _, _, ymax = plt.axis()
    cmap = mcol.LinearSegmentedColormap.from_list('BlRd',['b','r'])
    ax.imshow([[0.,1.], [0.,1.]], extent=(tonset, ttermination, ymin, ymax),
               cmap=cmap, interpolation='bicubic', alpha=0.15, aspect='auto')
    
    ax.axvline(tsat, color='k', ls='--', label=r'$t_{\rm sat}$')
    ax.axvline(tonset, color='b', ls=':', label=r'$t_{\rm onset}$', alpha=0.15)
    ax.axvline(ttermination, color='r', ls=':', label=r'$t_{\rm termination}$',
               alpha=0.15)
    
    delta_t_label = (r'$\Delta t_{\rm quench} = $' +
                     '{:.1f} Gyr'.format(ttermination-tonset))
    ax.plot(xmin-1, ymin-1, '-', color='whitesmoke', label=delta_t_label)
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return
