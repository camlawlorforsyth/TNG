
import copy
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib import cm
import matplotlib.colors as mcol
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
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

def plot_comprehensive_plot(title1, title2, title3, df1, df2, df3,
                            hist1, hist2, hist3,
                            xs, main1, main2, main3,
                            lo1, lo2, lo3, med1, med2, med3, hi1, hi2, hi3, XX, YY,
                            times, sm, lo_sm, hi_sm, tonset, tterm, thirtySeven, seventyFive,
                            SMH, UVK_df, UVK_snaps_xs, UVK_snaps_ys, mtitle=None,
                            xlabel_t=None, ylabel_t=None, xlabel_b=None, ylabel_b=None,
                            xlabel_SFH=None, ylabel_SFH=None, xlabel_SMH=None, ylabel_SMH=None,
                            xlabel_UVK=None, ylabel_UVK=None,
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
    ax1.text(-1.25, -2.7, r'$2 R_{\rm e}$', fontsize=18, color='w')
    ax1.text(1.4, -4.6, r'$4 R_{\rm e}$', fontsize=18, color='w')
    ax1.set_title(title1, fontsize=13)
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
    
    image3 = ax3.pcolormesh(XX, YY, hist3, cmap=cmap, norm=norm, alpha=0.9)
    sns.kdeplot(data=df3, x='dx', y='dz', color='lime', weights='masses',
                levels=[0.1, 0.2, 0.3, 0.6, 0.9], ax=ax3, linewidths=3)
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
    
    sns.kdeplot(data=UVK_df, x='V-K', y='U-V', color='grey',
                levels=[0.1, 0.3, 0.5, 0.7, 0.9], ax=ax9, linewidths=1)
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
    
    ax.fill_between(centers, lo, hi, color='grey', alpha=0.2,
                    edgecolor='darkgrey')
    ax.scatter(SF_xs, SF_ys, c='b', marker='o', label='SFMS', edgecolors='grey',
               alpha=0.1)
    ax.scatter(q_xs, q_ys, c='r', marker='o', label='', edgecolors='grey',
               alpha=0.1)
    ax.scatter(other_xs, other_ys, c='k', marker='o', label='', edgecolors='grey',
               alpha=0.2)
    
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
    lws = [2, 4]
    for i in range(len(xs)) :
        ax.plot(xs[i], ys[i], marker=markers[i], linestyle=styles[i],
                color=colors[i], label=labels[i], alpha=alphas[i], lw=lws[i])
    
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
    
    # ax.axvline(tsat, color='k', ls='--', label=r'$t_{\rm sat}$')
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

def test_contour(XX, YY, hist, X_cent, Y_cent, non_sf, levels,
                 xlabel=None, ylabel=None, save=False, outfile=None,
                 xmin=None, xmax=None, ymin=None, ymax=None) :
    
    global currentFig
    fig = plt.figure(currentFig, figsize=(8, 8))
    currentFig += 1
    plt.clf()
    ax = fig.add_subplot(111)
    
    norm=LogNorm(vmin=1e-11, vmax=2e-09)
    
    cmap = copy.copy(cm.inferno)
    cmap.set_bad('black', 1)
    
    ax.pcolormesh(XX, YY, hist, cmap=cmap, norm=norm, alpha=0.9)
    ax.contour(X_cent, Y_cent, non_sf, colors='lime', levels=levels,
               linewidths=3)
    
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
