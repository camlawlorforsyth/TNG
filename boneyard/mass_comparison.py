
import numpy as np

import plotting as plt

def plot_comparison(lookbacktimes, masses, SFH, smoothed, window_length, polyorder) :

    # calculate the area under the original and also smoothed curves
    area_original = np.trapz(SFH, lookbacktimes)
    area_smoothed = np.trapz(smoothed, lookbacktimes)
    
    # convert these areas into solar masses
    mass_diff = np.log10(area_original*1e9) - np.log10(area_smoothed*1e9)
    
    # plot the difference of the two mass estimates
    title = 'Window Length = {}, Polyorder={}'.format(window_length, polyorder)
    plt.plot_scatter(masses, mass_diff, 'k', 'data', 'o',
                     xlabel=r'$\log(M_{*}/M_{\odot})$',
                     ylabel=r'$\Delta M_{*}$',
                     title=title)
    
    return
