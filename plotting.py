
import copy
import matplotlib.pyplot as plt
import numpy as np

import corner
from matplotlib import cm
import matplotlib.colors as mcol
from matplotlib.colors import LogNorm, Normalize
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

currentFig = 1

import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['text.usetex'] = True

def display_image_simple_with_contour(image, contour_image, levels, bad='black',
                                      cbar_label='', cmap=cm.viridis, vmin=None,
                                      vmax=None, figsizewidth=9, figsizeheight=9) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    cmap = copy.copy(cmap)
    cmap.set_bad(bad, 1)
    
    norm = LogNorm(vmin=vmin, vmax=vmax)
    frame = ax.imshow(image, origin='lower', cmap=cmap, norm=norm)
    cbar = plt.colorbar(frame)
    cbar.set_label(cbar_label, fontsize=15)
    
    ax.contour(contour_image, levels=levels, colors='white', alpha=0.9)
    
    plt.tight_layout()
    plt.show()
    
    return

def display_image_simple(data, bad='black', cbar_label='', cmap=cm.gray, 
                         vmin=None, vmax=None, figsizewidth=9, figsizeheight=9,
                         lognorm=True, save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    cmap = copy.copy(cmap)
    cmap.set_bad(bad, 1)
    
    if lognorm :
        norm = LogNorm(vmin=vmin, vmax=vmax)
        frame = ax.imshow(data, origin='lower', cmap=cmap, norm=norm)
    else :
        frame = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(frame)
    cbar.set_label(cbar_label, fontsize=15)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def histogram(data, label, title=None, bins=None, log=False, histtype='step',
              vlines=[], colors=[], labels=[], loc='upper left',
              figsizewidth=9.5, figsizeheight=7) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    # if bins and not log :
    ax.hist(data, bins=bins, color='k', histtype=histtype)
    # elif bins and log :
    #     ax.hist(data, bins=bins, log=log, color='k', histtype=histtype)
    # elif log and not bins :
    #     ax.hist(data, log=log, color='k', histtype=histtype)
    # else :
    #     ax.hist(data, color='k', histtype=histtype)
    
    # if len(vlines) > 0 :
    #     for i in range(len(vlines)) :
    #         ax.axvline(vlines[i], ls='--', color=colors[i], lw=1, alpha=0.5,
    #                    label=labels[i])
    
    ax.set_xlabel('{}'.format(label), fontsize = 15)    
    
    # if len(vlines) > 0 :
    #     ax.legend(loc=loc, facecolor='whitesmoke', framealpha=1,
    #               fontsize=15)
    
    plt.tight_layout()
    plt.show()
    
    return

def histogram_large(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13,
                    h14, h15, h16, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10,
                    w11, w12, w13, w14, w15, w16, colors, labels, titles,
                    bins=12, ylabel1=None, ylabel2=None, ylabel3=None,
                    ylabel4=None, figsizewidth=9.5, figsizeheight=7, loc=0,
                    save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    
    gs = fig.add_gridspec(4, 4, hspace=0, wspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    ax4 = fig.add_subplot(gs[0, 3], sharey=ax1)
    
    ax5 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax6 = fig.add_subplot(gs[1, 1], sharex=ax2, sharey=ax5)
    ax7 = fig.add_subplot(gs[1, 2], sharex=ax3, sharey=ax5)
    ax8 = fig.add_subplot(gs[1, 3], sharex=ax4, sharey=ax5)
    
    ax9 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax10 = fig.add_subplot(gs[2, 1], sharex=ax2, sharey=ax9)
    ax11 = fig.add_subplot(gs[2, 2], sharex=ax3, sharey=ax9)
    ax12 = fig.add_subplot(gs[2, 3], sharex=ax4, sharey=ax9)
    
    ax13 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax14 = fig.add_subplot(gs[3, 1], sharex=ax2, sharey=ax13)
    ax15 = fig.add_subplot(gs[3, 2], sharex=ax3, sharey=ax13)
    ax16 = fig.add_subplot(gs[3, 3], sharex=ax4, sharey=ax13)
    
    for i in range(len(h1)) :
        ax1.hist(h1[i], color=colors[i], linestyle='-', label=labels[i],
                 histtype='step', bins=bins, weights=w1[i],
                 orientation='horizontal', log=True)
    
    for i in range(len(h2)) :
        ax2.hist(h2[i], color=colors[i], linestyle='-', label=labels[i],
                 histtype='step', bins=bins, weights=w2[i],
                 orientation='horizontal', log=True)
    
    for i in range(len(h3)) :
        ax3.hist(h3[i], color=colors[i], linestyle='-', label=labels[i],
                 histtype='step', bins=bins, weights=w3[i],
                 orientation='horizontal', log=True)
    
    for i in range(len(h4)) :
        ax4.hist(h4[i], color=colors[i], linestyle='-', label=labels[i],
                 histtype='step', bins=bins, weights=w4[i],
                 orientation='horizontal', log=True)
    
    for i in range(len(h5)) :
        ax5.hist(h5[i], color=colors[i], linestyle='-', label=labels[i],
                 histtype='step', bins=bins, weights=w5[i],
                 orientation='horizontal', log=True)
    
    for i in range(len(h6)) :
        ax6.hist(h6[i], color=colors[i], linestyle='-', label=labels[i],
                 histtype='step', bins=bins, weights=w6[i],
                 orientation='horizontal', log=True)
    
    for i in range(len(h7)) :
        ax7.hist(h7[i], color=colors[i], linestyle='-', label=labels[i],
                 histtype='step', bins=bins, weights=w7[i],
                 orientation='horizontal', log=True)
    
    for i in range(len(h8)) :
        ax8.hist(h8[i], color=colors[i], linestyle='-', label=labels[i],
                 histtype='step', bins=bins, weights=w8[i],
                 orientation='horizontal', log=True)
    
    for i in range(len(h9)) :
        ax9.hist(h9[i], color=colors[i], linestyle='-', label=labels[i],
                 histtype='step', bins=bins, weights=w9[i],
                 orientation='horizontal', log=True)
    
    for i in range(len(h10)) :
        ax10.hist(h10[i], color=colors[i], linestyle='-', label=labels[i],
                  histtype='step', bins=bins, weights=w10[i],
                  orientation='horizontal', log=True)
    
    for i in range(len(h11)) :
        ax11.hist(h11[i], color=colors[i], linestyle='-', label=labels[i],
                  histtype='step', bins=bins, weights=w11[i],
                  orientation='horizontal', log=True)
    
    for i in range(len(h12)) :
        ax12.hist(h12[i], color=colors[i], linestyle='-', label=labels[i],
                  histtype='step', bins=bins, weights=w12[i],
                  orientation='horizontal', log=True)
    
    for i in range(len(h13)) :
        ax13.hist(h13[i], color=colors[i], linestyle='-', label=labels[i],
                  histtype='step', bins=bins, weights=w13[i],
                  orientation='horizontal', log=True)
    
    for i in range(len(h14)) :
        ax14.hist(h14[i], color=colors[i], linestyle='-', label=labels[i],
                  histtype='step', bins=bins, weights=w14[i],
                  orientation='horizontal', log=True)
    
    for i in range(len(h15)) :
        ax15.hist(h15[i], color=colors[i], linestyle='-', label=labels[i],
                  histtype='step', bins=bins, weights=w15[i],
                  orientation='horizontal', log=True)
    
    for i in range(len(h16)) :
        ax16.hist(h16[i], color=colors[i], linestyle='-', label=labels[i],
                  histtype='step', bins=bins, weights=w16[i],
                  orientation='horizontal', log=True)
    
    ax1.tick_params(bottom=False, labelbottom=False)
    ax2.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax3.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax4.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    
    ax5.tick_params(bottom=False, labelbottom=False)
    ax6.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax7.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax8.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    
    ax9.tick_params(bottom=False, labelbottom=False)
    ax10.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax11.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax12.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    
    ax14.tick_params(left=False, labelleft=False)
    ax15.tick_params(left=False, labelleft=False)
    ax16.tick_params(left=False, labelleft=False)
    
    # ax.xaxis.set_label_position('top')
    # ax.xaxis.tick_top()
    
    ax1.set_ylabel(ylabel1)
    ax5.set_ylabel(ylabel2)
    ax9.set_ylabel(ylabel3)
    ax13.set_ylabel(ylabel4)
    
    ax1.set_xlim(0.003, 1)
    ax2.set_xlim(0.003, 1)
    ax3.set_xlim(0.003, 1)
    ax4.set_xlim(0.003, 1)
    
    ax1.set_ylim(0, 1)
    ax5.set_ylim(-1, 1)
    ax9.set_ylim(0, 5)
    ax13.set_ylim(0, 5)
    
    ax13.set_xticks([0.01, 0.1, 1]) # [0, 0.5]
    ax14.set_xticks([0.01, 0.1, 1]) # [0, 0.5]
    ax15.set_xticks([0.01, 0.1, 1]) # [0, 0.5]
    ax16.set_xticks([0.01, 0.1, 1]) # [0, 0.5, 1]
    
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax5.set_yticks([-1, -0.5, 0, 0.5])
    ax9.set_yticks([0, 1, 2, 3, 4])
    ax13.set_yticks([0, 1, 2, 3, 4])
    
    ax1.title.set_text(titles[0])
    ax2.title.set_text(titles[1])
    ax3.title.set_text(titles[2])
    ax4.title.set_text(titles[3])
    
    ax9.legend(loc=loc, facecolor='whitesmoke', framealpha=1)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def histogram_multi(data, hlabel, colors, styles, labels, bins, weights,
                    xmin=None, xmax=None, ymin=None, ymax=None, title=None,
                    figsizewidth=9.5, figsizeheight=7, loc=0, histtype='step',
                    save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    for i in range(len(data)) :
        ax.hist(data[i], color=colors[i], linestyle=styles[i], label=labels[i],
                histtype=histtype, bins=bins[i], weights=weights[i])
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(hlabel, fontsize=15)
    # ax.set_ylabel('Fractional Frequency', fontsize=15)
    
    # ax.set_xscale('log')
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc=loc, facecolor='whitesmoke', framealpha=1,
              fontsize=15)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
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

def plot_CASTOR_proposal(title1, title2, title3, df1, df2, df3,
                         hist1, hist2, hist3, fwhm1, fwhm2, fwhm3,
                         xs, main1, main2, main3,
                         lo1, lo2, lo3, med1, med2, med3, hi1, hi2, hi3, XX, YY,
                         xlabel_t=None, ylabel_t=None, xlabel_b=None, ylabel_b=None,
                         xmin_t=None, xmax_t=None, ymin_t=None, ymax_t=None,
                         xmin_b=None, xmax_b=None, ymin_b=None, ymax_b=None,
                         save=False, outfile=None, label=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(13, 9))
    currentFig += 1
    plt.clf()
    
    gs = fig.add_gridspec(2, 3, wspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    norm=LogNorm(vmin=1e-11, vmax=2e-09)
    
    cmap = copy.copy(cm.inferno)
    cmap.set_bad('black', 1)
    
    ax1.pcolormesh(XX, YY, hist1, cmap=cmap, norm=norm, alpha=0.9)
    sns.kdeplot(data=df1, x='dx', y='dz', color='lime', weights='masses',
                levels=[0.1, 0.2, 0.3, 0.6, 0.9], ax=ax1, linewidths=3)
    circle1_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle1_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax1.add_patch(circle1_in)
    ax1.add_patch(circle1_out)
    beam1 = Circle((4, -4), radius=fwhm1, facecolor='none', ls='-',
                        edgecolor='w', linewidth=1.5, alpha=1, zorder=3)
    ax1.add_patch(beam1)
    ax1.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    ax1.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax1.set_title(title1, fontsize=15)
    ax1.set_xlabel(xlabel_t, fontsize=15)
    ax1.set_ylabel(ylabel_t, fontsize=15)
    ax1.set_xlim(xmin_t, xmax_t)
    ax1.set_ylim(ymin_t, ymax_t)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    # ax1.axes.set_aspect('equal')
    
    ax2.pcolormesh(XX, YY, hist2, cmap=cmap, norm=norm, alpha=0.9)
    sns.kdeplot(data=df2, x='dx', y='dz', color='lime', weights='masses',
                levels=[0.1, 0.2, 0.3, 0.6, 0.9], ax=ax2, linewidths=3)
    circle2_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle2_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax2.add_patch(circle2_in)
    ax2.add_patch(circle2_out)
    beam2 = Circle((4, -4), radius=fwhm2, facecolor='none', ls='-',
                        edgecolor='w', linewidth=1.5, alpha=1, zorder=3)
    ax2.add_patch(beam2)
    ax2.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    ax2.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax2.set_title(title2, fontsize=15)
    ax2.set_xlabel(xlabel_t, fontsize=15)
    ax2.set(ylabel=None)
    ax2.set_xlim(xmin_t, xmax_t)
    ax2.set_ylim(ymin_t, ymax_t)
    ax2.tick_params(axis='x', which='major', labelsize=11)
    ax2.yaxis.set_ticks([])
    ax2.yaxis.set_ticklabels([])
    # ax2.axes.set_aspect('equal')
    
    image3 = ax3.pcolormesh(XX, YY, hist3, cmap=cmap, norm=norm, alpha=0.9)
    sns.kdeplot(data=df3, x='dx', y='dz', color='lime', weights='masses',
                levels=[0.1, 0.2, 0.3, 0.6, 0.9], ax=ax3, linewidths=3)
    circle3_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle3_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax3.add_patch(circle3_in)
    ax3.add_patch(circle3_out)
    beam3 = Circle((4, -4), radius=fwhm3, facecolor='none', ls='-',
                        edgecolor='w', linewidth=1.5, alpha=1, zorder=3)
    ax3.add_patch(beam3)
    ax3.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    ax3.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax3.set_title(title3, fontsize=15)
    ax3.set_xlabel(xlabel_t, fontsize=15)
    ax3.set(ylabel=None)
    ax3.set_xlim(xmin_t, xmax_t)
    ax3.set_ylim(ymin_t, ymax_t)
    ax3.tick_params(axis='x', which='major', labelsize=11)
    ax3.yaxis.set_ticks([])
    ax3.yaxis.set_ticklabels([])
    # ax3.axes.set_aspect('equal')
    ax3.plot([2*xmax_t], [2*xmax_t], color='limegreen', ls='-',
             label='non-SF stellar particles', lw=2)
    ax3.legend(facecolor='whitesmoke', framealpha=1, fontsize=13)
    axins = inset_axes(ax3, width='5%', height='100%', loc='lower left',
                        bbox_to_anchor=(1.05, 0., 1, 1),
                        bbox_transform=ax3.transAxes, borderpad=0)
    cbar = plt.colorbar(image3, cax=axins)
    cbar.set_label(label, fontsize=15)
    
    ax4.plot(xs, med1, 'k:')
    ax4.plot(xs, main1, 'k-')
    ax4.fill_between(xs, lo1, hi1, color='grey', edgecolor='darkgrey', alpha=0.2)
    ax4.set_yscale('log')
    ax4.set_xlabel(xlabel_b, fontsize=15)
    ax4.set_ylabel(ylabel_b, fontsize=15)
    ax4.set_xlim(xmin_b, xmax_b)
    ax4.set_ylim(ymin_b, ymax_b)
    ax4.tick_params(axis='both', which='major', labelsize=11)
    ax4.xaxis.set_ticks([0, 1, 2, 3, 4])
    ax4.xaxis.set_ticklabels(['0', '1', '2', '3', '4'])
    # ax4.axes.set_aspect('equal')
    
    ax5.plot(xs, med2, 'k:')
    ax5.plot(xs, main2, 'k-')
    ax5.fill_between(xs, lo2, hi2, color='grey', edgecolor='darkgrey', alpha=0.2)
    ax5.set_yscale('log')
    ax5.set_xlabel(xlabel_b, fontsize=15)
    ax5.set(ylabel=None)
    ax5.set_xlim(xmin_b, xmax_b)
    ax5.set_ylim(ymin_b, ymax_b)
    ax5.tick_params(axis='x', which='major', labelsize=11)
    ax5.tick_params(axis='y', which='minor', left=False)
    ax5.xaxis.set_ticks([0, 1, 2, 3, 4])
    ax5.xaxis.set_ticklabels(['0', '1', '2', '3', '4'])
    ax5.yaxis.set_ticks([])
    ax5.yaxis.set_ticklabels([])
    # ax5.axes.set_aspect('equal')
    
    ax6.plot(xs, med3, 'k:')
    ax6.plot(xs, main3, 'k-')
    ax6.fill_between(xs, lo3, hi3, color='grey', edgecolor='darkgrey', alpha=0.2)
    ax6.set_yscale('log')
    ax6.set_xlabel(xlabel_b, fontsize=15)
    ax6.set(ylabel=None)
    ax6.set_xlim(xmin_b, xmax_b)
    ax6.set_ylim(ymin_b, ymax_b)
    ax6.tick_params(axis='x', which='major', labelsize=11)
    ax6.tick_params(axis='y', which='minor', left=False)
    ax6.xaxis.set_ticks([0, 1, 2, 3, 4, 5])
    ax6.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5'])
    ax6.yaxis.set_ticks([])
    ax6.yaxis.set_ticklabels([])
    # ax6.axes.set_aspect('equal')
    
    # gs.tight_layout(fig)
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_chains(samples, ndim, labels, save=False, outfile=None) :
    
    global currentFig
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(14, 10), sharex=True,
                                        clear=True)
    currentFig += 1
    
    ax1.plot(samples[:, :, 0], 'k', alpha=0.3)
    ax1.set_xlim(0, len(samples))
    ax1.set_ylabel(labels[0], fontsize=15)
    ax1.yaxis.set_label_coords(-0.1, 0.5)
    
    ax2.plot(samples[:, :, 1], 'k', alpha=0.3)
    ax2.set_xlim(0, len(samples))
    ax2.set_ylabel(labels[1], fontsize=15)
    ax2.yaxis.set_label_coords(-0.1, 0.5)
    
    ax3.plot(samples[:, :, 2], 'k', alpha=0.3)
    ax3.set_xlim(0, len(samples))
    ax3.set_ylabel(labels[2], fontsize=15)
    ax3.yaxis.set_label_coords(-0.1, 0.5)
    ax3.set_xlabel('step number', fontsize=15)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_comparisons(tng_image, tng_contour_image, tng_levels,
                     skirt_image, skirt_contour_image, skirt_levels,
                     processed_image, processed_contour_image, processed_levels,
                     XX, YY, X_cent, Y_cent,
                     tng_vmin=None, tng_vmax=None,
                     skirt_vmin=None, skirt_vmax=None,
                     pro_vmin=None, pro_vmax=None,
                     xlabel=None, ylabel=None, mtitle=None,
                     xmin=None, xmax=None, ymin=None, ymax=None,
                     save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(30, 10))
    currentFig += 1
    plt.clf()
    
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0)
    
    # gs00 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0], wspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    cmap = copy.copy(cm.inferno)
    cmap.set_bad('black', 1)
    
    ax1.pcolormesh(XX, YY, tng_image, cmap=cmap,
                   norm=LogNorm(vmin=tng_vmin, vmax=tng_vmax))
    ax1.contour(X_cent, Y_cent, tng_contour_image, colors='lime',
                levels=tng_levels, linewidths=3)
    circle1_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle1_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax1.add_patch(circle1_in)
    ax1.add_patch(circle1_out)
    # ax1.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    # ax1.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax1.set_title('raw from TNG', fontsize=13)
    ax1.set_xlabel(xlabel, fontsize=15)
    ax1.set_ylabel(ylabel, fontsize=15)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    
    ax2.pcolormesh(XX, YY, skirt_image, cmap=cmap,
                   norm=LogNorm(vmin=skirt_vmin, vmax=skirt_vmax))
    ax2.contour(X_cent, Y_cent, skirt_contour_image, colors='lime',
                levels=skirt_levels, linewidths=3)
    circle2_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle2_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax2.add_patch(circle2_in)
    ax2.add_patch(circle2_out)
    # ax2.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    # ax2.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax2.set_title('raw from SKIRT: CASTOR UV + Roman F184', fontsize=13)
    ax2.set_xlabel(xlabel, fontsize=15)
    ax2.set(ylabel=None)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.tick_params(axis='x', which='major', labelsize=11)
    ax2.yaxis.set_ticks([])
    ax2.yaxis.set_ticklabels([])
    
    image3 = ax3.pcolormesh(XX, YY, processed_image, cmap=cmap,
                            norm=LogNorm(vmin=pro_vmin, vmax=pro_vmax))
    ax3.contour(X_cent, Y_cent, processed_contour_image, colors='lime',
                levels=processed_levels, linewidths=3)
    circle3_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle3_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax3.add_patch(circle3_in)
    ax3.add_patch(circle3_out)
    # ax3.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    # ax3.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax3.set_title('processed: CASTOR UV + Roman F184', fontsize=13)
    ax3.set_xlabel(xlabel, fontsize=15)
    ax3.set(ylabel=None)
    ax3.set_xlim(xmin, xmax)
    ax3.set_ylim(ymin, ymax)
    ax3.tick_params(axis='x', which='major', labelsize=11)
    ax3.yaxis.set_ticks([])
    ax3.yaxis.set_ticklabels([])
    # ax3.plot([2*xmax_t], [2*xmax_t], color='limegreen', ls='-',
    #           label='non-SF stellar particles', lw=2)
    # ax3.legend(facecolor='whitesmoke', framealpha=1, fontsize=13)
    # axins = inset_axes(ax3, width='5%', height='100%', loc='lower left',
    #                     bbox_to_anchor=(1.05, 0., 1, 1),
    #                     bbox_transform=ax3.transAxes, borderpad=0)
    # cbar = plt.colorbar(image3, cax=axins)
    # cbar.set_label(vlabel, fontsize=15)
    
    plt.suptitle(mtitle, fontsize=20)
    
    # gs.tight_layout(fig)
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_comprehensive_plot(title1, title2, title3, hist1, hist2, hist3,
                            contour1, contour2, contour3, level1, level2, level3,
                            xs, main1, main2, main3,
                            lo1, lo2, lo3, med1, med2, med3, hi1, hi2, hi3, XX, YY, X_cent, Y_cent,
                            times, sm, lo_sm, hi_sm, tonset, tterm, thirtySeven, seventyFive,
                            SMH, UVK_X_cent, UVK_Y_cent, UVK_contour, UVK_levels,
                            UVK_snaps_xs, UVK_snaps_ys, mtitle=None,
                            xlabel_t=None, ylabel_t=None, xlabel_b=None, ylabel_b=None,
                            xlabel_SFH=None, ylabel_SFH=None, xlabel_SMH=None, ylabel_SMH=None,
                            xlabel_UVK=None, ylabel_UVK=None, vmin=None, vmax=None,
                            xmin_t=None, xmax_t=None, ymin_t=None, ymax_t=None,
                            xmin_b=None, xmax_b=None, ymin_b=None, ymax_b=None,
                            xmin_SFH=None, xmax_SFH=None, ymin_SFH=None, ymax_SFH=None,
                            xmin_SMH=None, xmax_SMH=None, ymin_SMH=None, ymax_SMH=None,
                            xmin_UVK=None, xmax_UVK=None, ymin_UVK=None, ymax_UVK=None,
                            save=False, outfile=None, vlabel=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(25, 10))
    currentFig += 1
    plt.clf()
    
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.6, 0.4], wspace=0.25)
    
    gs00 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0], wspace=0)
    ax1 = fig.add_subplot(gs00[0, 0])
    ax2 = fig.add_subplot(gs00[0, 1])
    ax3 = fig.add_subplot(gs00[0, 2])
    ax4 = fig.add_subplot(gs00[1, 0])
    ax5 = fig.add_subplot(gs00[1, 1])
    ax6 = fig.add_subplot(gs00[1, 2])
    
    gs01 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1],
                                            width_ratios=[0.4, 0.6], wspace=0.4)
    ax7 = fig.add_subplot(gs01[0, :])
    ax8 = fig.add_subplot(gs01[1, 0])
    ax9 = fig.add_subplot(gs01[1, 1])
    
    norm=LogNorm(vmin=vmin, vmax=vmax)
    
    cmap = copy.copy(cm.inferno)
    cmap.set_bad('black', 1)
    
    ax1.pcolormesh(XX, YY, hist1, cmap=cmap, norm=norm)
    ax1.contour(X_cent, Y_cent, contour1, colors='lime', levels=level1,
               linewidths=3)
    circle1_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle1_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax1.add_patch(circle1_in)
    ax1.add_patch(circle1_out)
    ax1.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    ax1.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax1.set_title(title1, fontsize=13)
    ax1.set_xlabel(xlabel_t, fontsize=15)
    ax1.set_ylabel(ylabel_t, fontsize=15)
    ax1.set_xlim(xmin_t, xmax_t)
    ax1.set_ylim(ymin_t, ymax_t)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    # ax1.axes.set_aspect('equal')
    
    ax2.pcolormesh(XX, YY, hist2, cmap=cmap, norm=norm)
    ax2.contour(X_cent, Y_cent, contour2, colors='lime', levels=level2,
               linewidths=3)
    circle2_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle2_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax2.add_patch(circle2_in)
    ax2.add_patch(circle2_out)
    ax2.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    ax2.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax2.set_title(title2, fontsize=13)
    ax2.set_xlabel(xlabel_t, fontsize=15)
    ax2.set(ylabel=None)
    ax2.set_xlim(xmin_t, xmax_t)
    ax2.set_ylim(ymin_t, ymax_t)
    ax2.tick_params(axis='x', which='major', labelsize=11)
    ax2.yaxis.set_ticks([])
    ax2.yaxis.set_ticklabels([])
    # ax2.axes.set_aspect('equal')
    
    image3 = ax3.pcolormesh(XX, YY, hist3, cmap=cmap, norm=norm)
    ax3.contour(X_cent, Y_cent, contour3, colors='lime', levels=level3,
               linewidths=3)
    circle3_in = Circle((0, 0), radius=2, facecolor='none', ls=':',
                        edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    circle3_out = Circle((0, 0), radius=4, facecolor='none', ls=':',
                         edgecolor='w', linewidth=2.5, alpha=1, zorder=3)
    ax3.add_patch(circle3_in)
    ax3.add_patch(circle3_out)
    ax3.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    ax3.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax3.set_title(title3, fontsize=13)
    ax3.set_xlabel(xlabel_t, fontsize=15)
    ax3.set(ylabel=None)
    ax3.set_xlim(xmin_t, xmax_t)
    ax3.set_ylim(ymin_t, ymax_t)
    ax3.tick_params(axis='x', which='major', labelsize=11)
    ax3.yaxis.set_ticks([])
    ax3.yaxis.set_ticklabels([])
    # ax3.axes.set_aspect('equal')
    ax3.plot([2*xmax_t], [2*xmax_t], color='limegreen', ls='-',
             label='non-SF stellar particles', lw=2)
    ax3.legend(facecolor='whitesmoke', framealpha=1, fontsize=13)
    axins = inset_axes(ax3, width='5%', height='100%', loc='lower left',
                        bbox_to_anchor=(1.05, 0., 1, 1),
                        bbox_transform=ax3.transAxes, borderpad=0)
    cbar = plt.colorbar(image3, cax=axins)
    cbar.set_label(vlabel, fontsize=15)
    
    ax4.plot(xs, med1, 'k:')
    ax4.plot(xs, main1, 'ro')
    ax4.fill_between(xs, lo1, hi1, color='grey', edgecolor='darkgrey', alpha=0.2)
    ax4.set_yscale('log')
    ax4.set_xlabel(xlabel_b, fontsize=15)
    ax4.set_ylabel(ylabel_b, fontsize=15)
    ax4.set_xlim(xmin_b, xmax_b)
    ax4.set_ylim(ymin_b, ymax_b)
    ax4.tick_params(axis='both', which='major', labelsize=11)
    ax4.xaxis.set_ticks([0, 1, 2, 3, 4])
    ax4.xaxis.set_ticklabels(['0', '1', '2', '3', '4'])
    # ax4.axes.set_aspect('equal')
    
    ax5.plot(xs, med2, 'k:')
    ax5.plot(xs, main2, 'ro')
    ax5.fill_between(xs, lo2, hi2, color='grey', edgecolor='darkgrey', alpha=0.2)
    ax5.set_yscale('log')
    ax5.set_xlabel(xlabel_b, fontsize=15)
    ax5.set(ylabel=None)
    ax5.set_xlim(xmin_b, xmax_b)
    ax5.set_ylim(ymin_b, ymax_b)
    ax5.tick_params(axis='x', which='major', labelsize=11)
    ax5.tick_params(axis='y', which='minor', left=False)
    ax5.xaxis.set_ticks([0, 1, 2, 3, 4])
    ax5.xaxis.set_ticklabels(['0', '1', '2', '3', '4'])
    ax5.yaxis.set_ticks([])
    ax5.yaxis.set_ticklabels([])
    # ax5.axes.set_aspect('equal')
    
    ax6.plot(xs, med3, 'k:')
    ax6.plot(xs, main3, 'ro')
    ax6.fill_between(xs, lo3, hi3, color='grey', edgecolor='darkgrey', alpha=0.2)
    ax6.set_yscale('log')
    ax6.set_xlabel(xlabel_b, fontsize=15)
    ax6.set(ylabel=None)
    ax6.set_xlim(xmin_b, xmax_b)
    ax6.set_ylim(ymin_b, ymax_b)
    ax6.tick_params(axis='x', which='major', labelsize=11)
    ax6.tick_params(axis='y', which='minor', left=False)
    ax6.xaxis.set_ticks([0, 1, 2, 3, 4, 5])
    ax6.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5'])
    ax6.yaxis.set_ticks([])
    ax6.yaxis.set_ticklabels([])
    # ax6.axes.set_aspect('equal')
    
    ax7.plot(times, sm, color='k', label='SFH', marker='', linestyle='-',
             alpha=1)
    ax7.plot(times, lo_sm, color='grey', label='lo, hi', marker='', linestyle='-.',
             alpha=0.8)
    ax7.plot(times, hi_sm, color='grey', label='', marker='', linestyle='-.',
             alpha=0.8)
    if (ymin_SFH == None) :
        _, _, ymin_SFH, _ = ax7.axis()
    if (ymax_SFH == None) :
        _, _, _, ymax_SFH = ax7.axis()
    cmap_SFH = mcol.LinearSegmentedColormap.from_list('BlRd',['b','r'])
    ax7.imshow([[0.,1.], [0.,1.]], extent=(tonset, tterm, ymin_SFH, ymax_SFH),
                cmap=cmap_SFH, interpolation='bicubic', alpha=0.15, aspect='auto')
    ax7.axvline(thirtySeven, color='k', ls=':')
    ax7.axvline(seventyFive, color='k', ls='--')
    ax7.axvline(tonset, color='b', ls=':', alpha=0.15) # label=r'$t_{\rm onset}$'
    ax7.axvline(tterm, color='r', ls=':', alpha=0.15) # label=r'$t_{\rm termination}$'
    ax7.set_title(r'$\Delta t_{\rm quench} = $' + '{:.1f} Gyr'.format(tterm-tonset),
                  fontsize=13)
    ax7.set_xlabel(xlabel_SFH, fontsize=15)
    ax7.set_ylabel(ylabel_SFH, fontsize=15)
    ax7.set_xlim(xmin_SFH, xmax_SFH)
    ax7.set_ylim(ymin_SFH, ymax_SFH)
    ax7.legend(facecolor='whitesmoke', framealpha=1, fontsize=13, loc=0)
    
    ax8.plot(times, SMH, color='k', label='SMH', marker='', linestyle='-',
             alpha=1)
    if (ymin_SMH == None) :
        _, _, ymin_SMH, _ = ax8.axis()
    if (ymax_SMH == None) :
        _, _, _, ymax_SFH = ax8.axis()
    cmap_SFH = mcol.LinearSegmentedColormap.from_list('BlRd',['b','r'])
    ax8.imshow([[0.,1.], [0.,1.]], extent=(tonset, tterm, ymin_SFH, ymax_SFH),
                cmap=cmap_SFH, interpolation='bicubic', alpha=0.15, aspect='auto')
    ax8.axvline(thirtySeven, color='k', ls=':')
    ax8.axvline(seventyFive, color='k', ls='--')
    ax8.axvline(tonset, color='b', ls=':', alpha=0.15) # label=r'$t_{\rm onset}$'
    ax8.axvline(tterm, color='r', ls=':', alpha=0.15) # label=r'$t_{\rm termination}$'
    ax8.set_xlabel(xlabel_SMH, fontsize=15)
    ax8.set_ylabel(ylabel_SMH, fontsize=15)
    ax8.set_xlim(xmin_SMH, xmax_SMH)
    ax8.set_ylim(ymin_SMH, ymax_SMH)
    ax8.legend(facecolor='whitesmoke', framealpha=1, fontsize=13, loc=0)
    
    ax9.contour(UVK_X_cent, UVK_Y_cent, UVK_contour, colors='grey',
                levels=UVK_levels, linewidths=1) # levels=[0.1, 0.3, 0.5, 0.7, 0.9]
    ax9.scatter(UVK_snaps_xs, UVK_snaps_ys, c=['blue', 'purple', 'red', 'k'],
                marker='o', edgecolors='grey', s=40, zorder=3)
    ax9.set_xlabel(xlabel_UVK, fontsize=15)
    ax9.set_ylabel(ylabel_UVK, fontsize=15)
    ax9.set_xlim(xmin_UVK, xmax_UVK)
    ax9.set_ylim(ymin_UVK, ymax_UVK)
    
    plt.suptitle(mtitle, fontsize=20)
    
    # gs.tight_layout(fig)
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_comprehensive_mini(hist1, hist2, hist3, contour1, contour2, contour3,
                            level1, level2, level3, xs, main1, main2, main3,
                            lo1, lo2, lo3, med1, med2, med3, hi1, hi2, hi3,
                            XX, YY, X_cent, Y_cent, vmin=None, vmax=None,
                            ymin=None, ymax=None, save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(7.10000594991006, 9.095321710253218/2))
    currentFig += 1
    plt.clf()
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0, wspace=0)
    # gs = fig.add_gridspec(2, 3, hspace=0, wspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1], sharey=ax4)
    ax6 = fig.add_subplot(gs[1, 2], sharey=ax4)
    
    # norm = LogNorm(vmin=vmin, vmax=vmax)
    norm = Normalize(vmin=-10.5, vmax=-8.5)
    
    cmap = copy.copy(cm.inferno)
    cmap.set_bad('black', 1)
    
    small1 = Circle((0, 0), radius=2, facecolor='none', ls=':',
                    edgecolor='w', alpha=1, zorder=3)
    large1 = Circle((0, 0), radius=4, facecolor='none', ls=':',
                    edgecolor='w', alpha=1, zorder=3)
    small2 = Circle((0, 0), radius=2, facecolor='none', ls=':',
                    edgecolor='w', alpha=1, zorder=3)
    large2 = Circle((0, 0), radius=4, facecolor='none', ls=':',
                    edgecolor='w', alpha=1, zorder=3)
    small3 = Circle((0, 0), radius=2, facecolor='none', ls=':',
                    edgecolor='w', alpha=1, zorder=3)
    large3 = Circle((0, 0), radius=4, facecolor='none', ls=':',
                    edgecolor='w', alpha=1, zorder=3)
    
    ax1.pcolormesh(XX, YY, hist1, cmap=cmap, norm=norm)
    ax1.contour(X_cent, Y_cent, contour1, colors='lime', levels=level1)
    ax1.add_patch(small1)
    ax1.add_patch(large1)
    ax1.text(-1.25, -2.7, r'$2 R_{\rm e}$', color='w')
    ax1.text(1.4, -4.6, r'$4 R_{\rm e}$', color='w')
    
    ax2.pcolormesh(XX, YY, hist2, cmap=cmap, norm=norm)
    ax2.contour(X_cent, Y_cent, contour2, colors='lime', levels=level2)
    ax2.add_patch(small2)
    ax2.add_patch(large2)
    ax2.text(-1.25, -2.7, r'$2 R_{\rm e}$', color='w')
    ax2.text(1.4, -4.6, r'$4 R_{\rm e}$', color='w')
    
    image3 = ax3.pcolormesh(XX, YY, hist3, cmap=cmap, norm=norm)
    ax3.contour(X_cent, Y_cent, contour3, colors='lime', levels=level3)
    ax3.add_patch(small3)
    ax3.add_patch(large3)
    ax3.text(-1.25, -2.7, r'$2 R_{\rm e}$', color='w')
    ax3.text(1.4, -4.6, r'$4 R_{\rm e}$', color='w')
    ax3.plot([np.nan], [np.nan], color='limegreen', ls='-', lw=2,
             label='old stellar particles')
    # ax3.legend(facecolor='whitesmoke', framealpha=1)
    axins = inset_axes(ax3, width='5%', height='100%', loc='lower left',
                       bbox_to_anchor=(1.01, 0., 1, 1),
                       bbox_transform=ax3.transAxes, borderpad=0)
    cbar = plt.colorbar(image3, cax=axins)
    cbar.set_label(r'$\log{({\rm sSFR}/{\rm yr}^{-1})}$')
    
    ax4.plot(xs, med1, 'k:', label='median')
    ax4.fill_between(xs, lo1, hi1, color='grey', edgecolor='darkgrey', alpha=0.2,
                     label=r'$\pm 2 \sigma$')
    ax4.scatter(xs, main1, c='r', marker='o', s=15, label='quenched')
    
    ax5.plot(xs, med2, 'k:')
    ax5.fill_between(xs, lo2, hi2, color='grey', edgecolor='darkgrey', alpha=0.2)
    ax5.scatter(xs, main2, c='r', marker='o', s=15)
    
    ax6.plot(xs, med3, 'k:')
    ax6.fill_between(xs, lo3, hi3, color='grey', edgecolor='darkgrey', alpha=0.2)
    ax6.scatter(xs, main3, c='r', marker='o', s=15)
    
    ax1.tick_params(left=True, labelleft=True, top=True, labeltop=True)
    ax2.tick_params(left=False, labelleft=False, top=True, labeltop=True)
    ax3.tick_params(left=False, labelleft=False, top=True, labeltop=True)
    ax4.tick_params(left=True, labelleft=True, bottom=True, labelbottom=True)
    ax5.tick_params(left=False, labelleft=False, bottom=True, labelbottom=True)
    ax6.tick_params(left=False, labelleft=False, bottom=True, labelbottom=True)
    
    ax1.set_xlabel(r'$\Delta x$ ($R_{\rm e}$)')
    ax2.set_xlabel(r'$\Delta x$ ($R_{\rm e}$)')
    ax3.set_xlabel(r'$\Delta x$ ($R_{\rm e}$)')
    ax4.set_xlabel(r'$r/R_{\rm e}$')
    ax5.set_xlabel(r'$r/R_{\rm e}$')
    ax6.set_xlabel(r'$r/R_{\rm e}$')
    
    ax1.set_ylabel(r'$\Delta y$ ($R_{\rm e}$)')
    ax4.set_ylabel(r'$\log{({\rm sSFR}/{\rm yr}^{-1})}$')
    
    ax1.set_xlim(-5, 5)
    ax2.set_xlim(-5, 5)
    ax3.set_xlim(-5, 5)
    ax4.set_xlim(0, 5)
    ax5.set_xlim(0, 5)
    ax6.set_xlim(0, 5)
    ax1.set_ylim(-5, 5)
    ax4.set_ylim(ymin, ymax)
    
    ax1.set_xticks([-4, -2, 0, 2, 4])
    ax2.set_xticks([-4, -2, 0, 2, 4])
    ax3.set_xticks([-4, -2, 0, 2, 4])
    ax4.set_xticks([0, 1, 2, 3, 4])
    ax5.set_xticks([0, 1, 2, 3, 4])
    ax6.set_xticks([0, 1, 2, 3, 4, 5])
    ax1.set_yticks([-4, -2, 0, 2, 4])
    # ax4.set_yticks([0, 1, 2, 3, 4])
    
    ax1.xaxis.set_label_position('top')
    ax2.xaxis.set_label_position('top')
    ax3.xaxis.set_label_position('top')
    # ax.xaxis.tick_top()
    
    ax3.legend(facecolor='whitesmoke', framealpha=1, loc=1)
    ax4.legend(facecolor='whitesmoke', framealpha=1, loc=3)
    
    # gs.tight_layout(fig)
    # plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_contour_with_hists(xs, ys, colors, markers, alphas,
                            X_cent, Y_cent, contours, levels,
                            xlabel=None, ylabel=None, title=None, loc=0,
                            xmin=None, xmax=None, ymin=None, ymax=None,
                            figsizewidth=9.5, figsizeheight=7, save=False,
                            outfile=None, xbins=None, ybins=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histy.tick_params(axis='y', labelleft=False)
    
    for xx, yy, color, marker, alpha, contour, level in zip(xs, ys, colors,
        markers, alphas, contours, levels) :
        
        ax.contour(X_cent, Y_cent, contour, colors=color,
                   levels=level, linewidths=1)
        if alpha > 0.5 : # alpha is capped at 1
            alpha = 0.5
        if marker != '' :
            ax_histx.hist(xx, bins=xbins, color=color, histtype='step',
                          alpha=1, weights=np.ones(len(xx))/len(xx))
            ax_histy.hist(yy, bins=ybins, color=color, histtype='step',
                          alpha=1, weights=np.ones(len(yy))/len(yy),
                          orientation='horizontal')
    
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    
    ax_histx.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax_histx.set_ylim(0, 0.5)
    ax_histy.set_xlim(0, 0.25)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_lines(xx, yy, xs, ys, label=None,
               xlabel=None, ylabel=None, xmin=None, xmax=None,
               ymin=None, ymax=None, save=False, outfile=None,
               figsizewidth=9.5, figsizeheight=7, loc=0) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    ax.plot(xx, yy, '-', color='grey')
    ax.scatter(xx, yy, s=100, linestyles='-', color=['b', 'purple', 'r'],
               edgecolors='k', zorder=5, label=label)
    
    for i, (x_indv, y_indv) in enumerate(zip(xs, ys)) :
        if i == 0 :
            slabel = 'control sample'
        else :
            slabel = ''
        # ax.plot(x_indv, y_indv, '-', color='grey', alpha=0.2)
        ax.scatter(x_indv[1], y_indv[1], linestyles='-', label=slabel,
                   # color=['b', 'purple', 'r'],
                   color='purple',
                   alpha=0.3, zorder=5)
    
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
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

def plot_scatter(xs, ys, color, label, marker, cbar_label='', size=30,
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
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    temp = np.linspace(xmin, xmax, 1000)
    ax.plot(temp, temp, 'k-')
    # ax.plot(temp, temp+0.5, 'k--')
    # ax.plot(temp, temp+1, 'k:')
    
    frame = ax.scatter(xs, ys, c=color, marker=marker, label=label, cmap=cmap,
                        # edgecolors='grey',
                       norm=norm, s=size, alpha=0.3)
    # cbar = plt.colorbar(frame)
    # cbar.set_label(cbar_label, fontsize=15)
    ax.plot(-5, -5, 'o', c=cmap(327), alpha=0.3, label='cluster')
    ax.plot(-5, -5, 'o', c=cmap(170), alpha=0.3, label='high mass group')
    ax.plot(-5, -5, 'o', c=cmap(100), alpha=0.3, label='low mass group')
    ax.plot(-5, -5, 'o', c=cmap(0), alpha=0.3, label='field')
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    
    ax.set_title(title, fontsize=18)
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

def plot_scatter_dumb(xs, ys, color, label, marker, cbar_label='', size=30,
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
    
    if (xmin is None) and (xmax is None) :
        xmin, xmax = np.min(xs), np.max(xs)
    xx = np.linspace(xmin, xmax, 1000)
    ax.plot(xx, xx, 'k-', label='equality')
    frame = ax.scatter(xs, ys, c=color, marker=marker, label=label, cmap=cmap,
                        # edgecolors='grey', norm=norm,
                       vmin=vmin, vmax=vmax, s=size, alpha=1, zorder=3)
    # cbar = plt.colorbar(frame)
    # cbar.set_label(cbar_label, fontsize=15)
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    
    ax.set_title(title, fontsize=18)
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

def plot_scatter_err(xs, ys, lo, hi, xlabel=None, ylabel=None,
                     xmin=None, xmax=None, ymin=None, ymax=None,
                     figsizewidth=9.5, figsizeheight=7, save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    ax.errorbar(xs, ys, xerr=[lo, hi], fmt='ko', ecolor='k', elinewidth=0.5,
                capsize=2)
    
    ax.axhline(50, c='grey', ls='--')
    
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

def plot_scatter_3d(xs, ys, zs, colors, markers, labels, sizes, xx, yy, zz,
                    xlabel=None, ylabel=None, zlabel=None, xmin=None, xmax=None,
                    ymin=None, ymax=None, zmin=None, zmax=None, scale='linear',
                    figsizewidth=9.5, figsizeheight=7, save=False, outfile=None,
                    azim=230, elev=30) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    
    # adapted from
    # https://matplotlib.org/stable/gallery/mplot3d/projections.html
    ax.set_proj_type('ortho')
    
    # adapted from
    # https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
    # ax.view_init(elev=20, azim=45)
    ax.view_init(elev=elev, azim=azim)
    
    for i in range(len(xs)) :
        ax.scatter(xs[i], ys[i], zs[i], c=colors[i], marker=markers[i],
                   label=labels[i], s=sizes[i])
    
    ax.plot_surface(xx, yy, zz, color='k', alpha=0.5)
    
    # ax.contour(xx, yy, zz, zdir='x', offset=0, levels=0)
    # ax.contour(xx, yy, zz, zdir='y', offset=1, levels=0)
    # ax.contour(xx, yy, zz, zdir='z', offset=0, levels=0)
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_zscale(scale)
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_zlabel(zlabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=0)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter_multi(xs, ys, colors, labels, markers, alphas,
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
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    for i in range(len(xs)) :
        ax.scatter(xs[i], ys[i], c=colors[i], marker=markers[i], label=labels[i],
                   cmap=cmap, norm=norm, alpha=alphas[i])
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    
    ax.set_title(title, fontsize=18)
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

def plot_scatter_multi_with_bands(SF_xs, SF_ys, q_xs, q_ys, other_xs, other_ys,
                                  centers, lo, hi, xlabel=None, ylabel=None,
                                  xmin=None, xmax=None, ymin=None, ymax=None, loc=0,
                                  figsizewidth=9.5, figsizeheight=7,
                                  save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    ax.fill_between(centers, lo, hi, color='grey', alpha=0.2)
    ax.scatter(other_xs, other_ys, c='k', marker='o', label='', alpha=0.2)
    ax.scatter(SF_xs, SF_ys, c='b', marker='o', label='SFMS', alpha=0.1)
    ax.scatter(q_xs, q_ys, c='r', marker='o', label='quenched', alpha=0.3)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(facecolor='whitesmoke', framealpha=1, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter_multi_with_line(xs, ys, colors, labels, markers, alphas,
                                 xline, yline,
                                 xlabel=None, ylabel=None, title=None,
                                 xmin=None, xmax=None, ymin=None, ymax=None, loc=0,
                                 figsizewidth=9.5, figsizeheight=7, scale='linear',
                                 vmin=None, vmax=None, save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    for i in range(len(xs)) :
        ax.scatter(xs[i], ys[i], c=colors[i], marker=markers[i], label=labels[i],
                   alpha=alphas[i], s=10)
    
    ax.plot(xline, yline, 'k-')
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=10, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def double_scatter_with_line(xs1, ys1, colors1, markers1, alphas1, x1, y1,
                             xs2, ys2, colors2, markers2, alphas2, x2, y2, labels,
                             s1=None, s2=None, titles=None, xlabel1=None,
                             ylabel1=None, xlabel2=None, ylabel2=None,
                             xmin1=None, xmax1=None, xmin2=None, xmax2=None,
                             ymin=None, ymax=None, figsizewidth=9.5,
                             figsizeheight=7, loc=0, save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    
    gs = fig.add_gridspec(1, 2, wspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    
    # ax2.set_xticks([-5, 0, 5])
    # ax2.yaxis.set_label_position('right')
    # ax2.yaxis.tick_right()
    ax2.tick_params(left=False, labelleft=False)
    ax1.set_xticks([2, 4, 6, 8, 10, 12])
    
    if (s1 and s2) :
        legendSize = [50]
    else :
        s1, s2, legendSize = [10]*len(xs1), [10]*len(xs2), [10]
    
    for i in range(len(xs1)) :
        ax1.scatter(xs1[i], ys1[i], c=colors1[i], marker=markers1[i],
                    label=labels[i], alpha=alphas1[i], s=s1[i])
    
    ax1.plot(x1, y1, 'k-', lw=1)
    
    for i in range(len(xs2)) :
        ax2.scatter(xs2[i], ys2[i], c=colors2[i], marker=markers2[i],
                    alpha=alphas2[i], s=s2[i])
    
    ax2.plot(x2, y2, 'k-', lw=1)
    
    ax1.set_xlabel(xlabel1)
    ax1.set_ylabel(ylabel1)
    
    ax2.set_xlabel(xlabel2)
    ax2.set_ylabel(ylabel2)
    
    ax1.set_xlim(xmin1, xmax1)
    ax2.set_xlim(xmin2, xmax2)
    
    ax1.set_ylim(ymin, ymax)
    
    if titles :
        ax1.title.set_text(titles[0])
        ax2.title.set_text(titles[1])
    
    legend = ax1.legend(facecolor='whitesmoke', framealpha=1, loc=loc)
    
    # adapted from https://stackoverflow.com/questions/24706125
    for i in range(len(xs1)) :
        legend.legendHandles[i]._sizes = legendSize
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def quad_scatter(xs1, ys1, xs2, ys2, xs3, ys3, xs4, ys4, colors,
                 markers, alphas, labels, titles, xlabel=None,
                 ylabel=None, xmin=None, xmax=None, ymin=None,
                 ymax=None, figsizewidth=9.5, figsizeheight=7,
                 loc=0, save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    
    gs = fig.add_gridspec(1, 4, wspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    ax4 = fig.add_subplot(gs[0, 3], sharey=ax1)
    # ax2.yaxis.set_label_position('right')
    # ax2.yaxis.tick_right()
    
    for i in range(len(xs1)) :
        ax1.scatter(xs1[i], ys1[i], c=colors[i], marker=markers[i],
                    alpha=alphas[i], s=10)
    
    for i in range(len(xs2)) :
        ax2.scatter(xs2[i], ys2[i], c=colors[i], marker=markers[i],
                    alpha=alphas[i], s=10)
    
    for i in range(len(xs3)) :
        ax3.scatter(xs3[i], ys3[i], c=colors[i], marker=markers[i],
                    alpha=alphas[i], s=10)
    
    for i in range(len(xs4)) :
        ax4.scatter(xs4[i], ys4[i], c=colors[i], marker=markers[i],
                    alpha=alphas[i], s=10, label=labels[i])
    
    ax1.set_xlabel(xlabel)
    ax2.set_xlabel(xlabel)
    ax3.set_xlabel(xlabel)
    ax4.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    
    ax1.set_xlim(xmin, xmax)
    ax2.set_xlim(xmin, xmax)
    ax3.set_xlim(xmin, xmax)
    ax4.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    
    # ax1.tick_params(bottom=False, labelbottom=False)
    ax2.tick_params(left=False, labelleft=False)
    ax3.tick_params(left=False, labelleft=False)
    ax4.tick_params(left=False, labelleft=False)
    
    ax1.set_xticks([0, 0.5])
    ax2.set_xticks([0, 0.5])
    ax3.set_xticks([0, 0.5])
    ax4.set_xticks([0, 0.5, 1])
    
    # ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    
    ax1.title.set_text(titles[0])
    ax2.title.set_text(titles[1])
    ax3.title.set_text(titles[2])
    ax4.title.set_text(titles[3])
    
    ax4.legend(facecolor='whitesmoke', framealpha=1, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_scatter_particles(dx, dy, dz, sf_dx, sf_dy, sf_dz, xlabel=None,
                           ylabel=None, zlabel=None, figsizewidth=18,
                           figsizeheight=6, save=False, outfile=None) :
    
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

def plot_scatter_with_bands(xs, ys, centers, lo, hi,
                            xlabel=None, ylabel=None,
                            xmin=None, xmax=None, ymin=None, ymax=None, loc=0,
                            figsizewidth=9.5, figsizeheight=7,
                            save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    ax.fill_between(centers, lo, hi, color='grey', alpha=0.2,
                    edgecolor='darkgrey')
    ax.scatter(xs, ys, c='b', marker='o', label='', edgecolors='grey', alpha=0.01)
    
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

def plot_scatter_with_contours(xs, ys, contour_xs, contour_ys, colors, bins=100,
                               smooth=4, xlabel=None, ylabel=None, xmin=None,
                               xmax=None, ymin=None, ymax=None, loc=0,
                               save=False, outfile=None) : # smooth=(3,6)
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(9.5, 7))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    levels = 1 - np.exp(-0.5*np.square([0.5, 1, 1.5, 2]))
    
    for i in range(len(contour_xs)) :
        corner.hist2d(contour_xs[i], contour_ys[i], bins=bins, #levels=levels,
            levels=[0.3934693402873666, 0.8646647167633873],
            smooth=smooth, ax=ax, color=colors[i], quiet=True,
            plot_datapoints=False, plot_density=False,
            plot_contours=True, no_fill_contours=True,
            fill_contours=False, new_fig=False)
    
    ax.scatter(xs, ys, c='grey', edgecolors='k', marker='o', alpha=1, s=50,
               label='OI misclassified as SF', zorder=2)
    
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    
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

def plot_scatter_with_hists(xs, ys, colors, labels, markers, alphas,
                            xlabel=None, ylabel=None, title=None, loc=0,
                            xmin=None, xmax=None, ymin=None, ymax=None,
                            figsizewidth=9.5, figsizeheight=7, save=False,
                            outfile=None, xbins=None, ybins=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histy.tick_params(axis='y', labelleft=False)
    
    for xx, yy, color, label, marker, alpha in zip(xs, ys, colors, labels,
                                                   markers, alphas) :
        
        ax.scatter(xx, yy, c=color, marker=marker, label=label, alpha=alpha)
        if alpha > 0.5 : # alpha is capped at 1
            alpha = 0.5
        if marker != '' :
            ax_histx.hist(xx, bins=xbins, color=color, histtype='step',
                          alpha=1, weights=np.ones(len(xx))/len(xx)) # alpha=2*alpha
            ax_histy.hist(yy, bins=ybins, color=color, histtype='step',
                          alpha=1, weights=np.ones(len(yy))/len(yy),
                          orientation='horizontal') # alpha=2*alpha
    
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    
    ax_histx.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax_histx.set_ylim(0, 0.5)
    ax_histy.set_xlim(0, 0.25)
    ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_simple_dumb(xs, ys, label='', save=False,
                     xlabel=None, ylabel=None, title=None, outfile=None,
                     xmin=None, xmax=None, ymin=None, ymax=None, loc=0,
                     figsizewidth=9.5, figsizeheight=7, scale='linear') :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    # xx = np.linspace(xmin, xmax, 1000)
    # ax.plot(xx, xx, 'r-', label='equality')
    # ax.plot(xx, xx+0.18, 'b-', label='y = x + 0.18')
    ax.plot(xs, ys, 'ko', label=label) #alpha=0.2)
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if label != '' :
        ax.legend(facecolor='whitesmoke', framealpha=1, fontsize=15, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_simple_many(xs, ys, xlabel=None, ylabel=None, xmin=None, xmax=None,
                     ymin=None, ymax=None, save=False, outfile=None,
                     title=None, scale='linear') :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(9.5, 7))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    for yy in ys :
        ax.plot(xs, yy, 'C1', alpha=0.01)
    
    ax.set_xscale(scale)
    ax.set_yscale('linear')
    
    ax.set_title(title, fontsize=18)
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
                      figsizewidth=9.5, figsizeheight=7, scale='linear', loc=0,
                      outfile=None, save=False) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    # lws = [2, 4]
    for i in range(len(xs)) :
        ax.plot(xs[i], ys[i], marker=markers[i], linestyle=styles[i],
                color=colors[i], label=labels[i], alpha=alphas[i])
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # if labels[0] != '' :
    ax.legend(facecolor='whitesmoke', framealpha=1, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_simple_multi_secax(xs, ys, labels, colors, markers, styles,
                            alphas, reg, xlabel=None, ylabel=None,
                            seclabel=None, title=None, xmin=None,
                            xmax=None, ymin=None, ymax=None,
                            secmin=None, secmax=None, loc=0,
                            figsizewidth=9.5, figsizeheight=7,
                            xscale='linear', yscale='linear',
                            secscale='linear', outfile=None,
                            save=False) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    secax = ax.twinx()
    secax.format_coord = make_format(secax, ax)
    
    for i in range(reg) :
        ax.plot(xs[i], ys[i], marker=markers[i], linestyle=styles[i],
                color=colors[i], label=labels[i], alpha=alphas[i])
    
    for i in range(reg, len(xs)) :
        secax.plot(xs[i], ys[i], marker=markers[i], linestyle=styles[i],
                color=colors[i], label=labels[i], alpha=alphas[i])
    
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    secax.set_yscale(secscale)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    secax.set_ylabel(seclabel)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    secax.set_ylim(secmin, secmax)
    
    lines, labels = ax.get_legend_handles_labels()
    seclines, seclabels = secax.get_legend_handles_labels()
    
    ax.legend(lines + seclines, labels + seclabels,
              facecolor='whitesmoke', framealpha=1, loc=loc,
               bbox_to_anchor=(0.045, 0.15, 0.5, 0.5))
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def make_format(current, other) :
    # adapted from https://stackoverflow.com/questions/21583965
    
    # current and other are axes
    def format_coord(x, y) :
        # x, y are data coordinates
        # convert to display coords
        display_coord = current.transData.transform((x, y))
        inv = other.transData.inverted()
        # convert back to data coords with respect to ax
        ax_coord = inv.transform(display_coord)
        coords = [ax_coord, (x, y)]
        return ('Left: {:<40}    Right: {:<}'
                .format(*['({:.3g}, {:.3g})'.format(x, y) for x, y in coords]))
    return format_coord

def plot_simple_multi_with_bands(xs, los, meds, his, colors, styles, labels,
                                 xlabel=None, ylabel=None, xmin=None, xmax=None,
                                 ymin=None, ymax=None, figsizewidth=9.5,
                                 figsizeheight=7, scale='linear', loc=0,
                                 save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    for i in range(len(los)) :
        ax.fill_between(xs, los[i], his[i], color=colors[i], alpha=0.2)
        ax.plot(xs, meds[i], marker='', linestyle=styles[i], color=colors[i],
                label=labels[i], lw=2)
    
    ax.axvline(0, c='k', ls=':', alpha=0.3)
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    ax.tick_params(axis='both', which='major', labelsize=15)
    
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

def double_simple_with_times(xs1, ys1, colors1, styles1, alphas1, labels1,
                             xs2, ys2, colors2, styles2, alphas2, labels2,
                             tonset, tterm, xlabel=None, ylabel=None,
                             xmin=None, xmax=None, ymin=None, ymax=None,
                             figsizewidth=9.5, figsizeheight=7, loc=0,
                             save=False, outfile=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(figsizewidth, figsizeheight))
    currentFig += 1
    plt.clf()
    
    gs = fig.add_gridspec(2, 1, hspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    # ax2.yaxis.set_label_position('right')
    # ax2.yaxis.tick_right()
    
    for i in range(len(xs1)) :
        ax1.plot(xs1[i], ys1[i], c=colors1[i], ls=styles1[i], marker='',
                 label=labels1[i], alpha=alphas1[i], zorder=len(xs1)-i)
    
    for i in range(len(xs2)) :
        ax2.plot(xs2[i], ys2[i], c=colors2[i], ls=styles2[i], marker='',
                 label=labels2[i], alpha=alphas2[i], zorder=len(xs2)-i)
    
    if (ymin == None) :
        _, _, ymin, _ = plt.axis()
    if (ymax == None) :
        _, _, _, ymax = plt.axis()
    cmap = mcol.LinearSegmentedColormap.from_list('BlRd',['b','r'])
    ax2.imshow([[0.,1.], [0.,1.]], extent=(tonset, tterm, ymin, ymax),
               cmap=cmap, interpolation='bicubic', alpha=0.15, aspect='auto')
    
    ax1.tick_params(bottom=False, labelbottom=False)
    
    ax1.set_ylabel(ylabel)
    
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    
    ax1.set_xlim(xmin, xmax)
    
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)
    
    # ax1.set_yticks([0, 2, 4, 6, 8, 10])
    # ax2.set_yticks([0, 2, 4, 6, 8])
    
    # ax2.set_xticks([0, 2, 4, 6, 8, 10, 12])
    
    ax1.legend(facecolor='whitesmoke', framealpha=1, loc=loc)
    ax2.legend(facecolor='whitesmoke', framealpha=1, loc=loc)
    
    plt.tight_layout()
    
    if save :
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else :
        plt.show()
    
    return

def plot_simple_multi_with_times(xs, ys, labels, colors, markers, styles,
                                 alphas, tsat, tonset, ttermination, drop_times, drop_labels,
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
    
    # ax.axvline(tsat, color='k', ls=':', label=r'$t_{\rm sat}$', alpha=0.5)
    # ax.axvline(tonset, color='b', ls=':', alpha=0.15) # label=r'$t_{\rm onset}$', )
    # ax.axvline(ttermination, color='r', ls=':', alpha=0.15) # label=r'$t_{\rm termination}$'
    
    # new as of July 17th, 2023 -> to be removed in future
    # ax.axvline(drop_times[0], color='k', ls=':', label=drop_labels[0], alpha=0.5)
    # ax.axvline(drop_times[1], color='m', ls=':', label=drop_labels[1], alpha=0.8)
    
    # delta_t_label = (r'$\Delta t_{\rm quench} = $' +
    #                   '{:.1f} Gyr'.format(ttermination-tonset))
    # ax.plot(xmin-1, ymin-1, '-', color='whitesmoke', label=delta_t_label)
    
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    
    ax.set_title(title, fontsize=18)
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

def test_contour(XX, YY, hist, X_cent, Y_cent, contours, levels,
                 xlabel=None, ylabel=None, save=False, outfile=None,
                 xmin=None, xmax=None, ymin=None, ymax=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(8, 8))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    norm = LogNorm(vmin=1e-11, vmax=2e-09)
    
    cmap = copy.copy(cm.inferno)
    cmap.set_bad('black', 1)
    
    ax.pcolormesh(XX, YY, hist, cmap=cmap, norm=norm, alpha=0.9)
    ax.contour(X_cent, Y_cent, contours, colors='grey', levels=levels,
               linewidths=1)
    
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
