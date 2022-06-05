
import numpy as np
import matplotlib.pyplot as plt

import astropy.constants as c
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
import astropy.units as u

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
    if title :
        if title[0] == 'a' :
            title = 'Abell ' + title[1:]
        if title[0] == 'm' :
            title = 'MACS J' + title[1:]
    ax.set_title(title, fontsize=18)
    
    if len(vlines) > 0 :
        ax.legend(loc=loc, facecolor='whitesmoke', framealpha=1,
                  fontsize=15)
    
    plt.tight_layout()
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
    
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if label != '' :
        ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15)
    
    plt.tight_layout()
    plt.show()
    
    return

def plot_simple_multi(xs, ys, labels, colors, markers, styles, alphas=None,
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
    
    ax.set_yscale(scale)
    ax.set_xscale(scale)
    
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
