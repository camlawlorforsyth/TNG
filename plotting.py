
import copy
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import LogNorm

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
