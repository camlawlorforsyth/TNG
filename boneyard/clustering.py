
import numpy as np

# from astropy.table import Table
import h5py
from sklearn.cluster import KMeans # DBSCAN

from core import find_nearest
import plotting as plt

def main() :
    
    (data, oi_xis, oi_zetas, oi_Rtruncs, io_xis, io_zetas, io_Rtruncs,
     c_xis, c_zetas, c_Rtruncs) = get_params()
    
    # perfrom K-means clustering
    (clus1_x, clus1_y, clus1_z, clus2_x, clus2_y, clus2_z,
     clus3_x, clus3_y, clus3_z) = perform_kmeans(data)
    
    xlabel = r'$\xi = {\rm SFR}_{<1~{\rm kpc}}/{\rm SFR}_{\rm total}$'
    ylabel = r'$\log{(\zeta = R_{{\rm e}_{*, {\rm SF}}}/R_{{\rm e}_{*, {\rm total}}})}$'
    zlabel = r'$R_{\rm trunc}/R_{\rm e}$'
    xmin, xmax, ymin, ymax, zmin, zmax = -0.03, 1.03, -1.2, 1.5, 0, 5
    width, height = 20, 14
    
    # create 3D plots
    plt.plot_scatter_3d([c_xis, oi_xis, io_xis],
        [c_zetas, oi_zetas, io_zetas], [c_Rtruncs, oi_Rtruncs, io_Rtruncs],
        ['k', 'r', 'm'], ['o', 's', 's'], xlabel=xlabel, ylabel=ylabel,
        zlabel=zlabel, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin,
        zmax=zmax, figsizewidth=width, figsizeheight=height,
        outfile='D:/Desktop/3D_metrics.png', save=False)
    
    # plt.plot_scatter_3d([clus1_x, clus2_x, clus3_x],
    #     [clus1_y, clus2_y, clus3_y], [clus1_z, clus2_z, clus3_z],
    #     ['orange', 'g', 'b'], ['s', 's', 's'], xlabel=xlabel, ylabel=ylabel,
    #     zlabel=zlabel, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin,
    #     zmax=zmax, figsizewidth=width, figsizeheight=height)
    
    '''
    # create 2D plots
    title = '75% through the quenching event'
    outfile = ''
    xbins = np.arange(np.around(xmin, 1), np.around(xmax, 1), 0.1)
    ybins = np.arange(np.around(ymin, 1), np.around(ymax, 1), 0.1)
    
    plt.plot_scatter_with_hists([c_xis, oi_xis, io_xis],
        [c_zetas, oi_zetas, io_zetas], ['k', 'r', 'm'],
        ['control', 'outside-in', 'inside-out'],
        ['o', 's', 's'], [0.3, 0.5, 0.5],
        xlabel=xlabel, ylabel=ylabel, title=title,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        outfile=outfile, save=False, loc=1, xbins=xbins, ybins=ybins)
    
    plt.plot_scatter_multi([c_xis, oi_xis, io_xis],
        [c_zetas, oi_zetas, io_zetas], ['k', 'r', 'm'],
        ['control', 'outside-in', 'inside-out'],
        ['o', 's', 's'], [0.3, 0.5, 0.5],
        xlabel=xlabel, ylabel=ylabel, title=title,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        outfile=outfile, save=False, loc=1)
    
    plt.plot_scatter_multi([clus1_x, clus2_x, clus3_x],
        [clus1_y, clus2_y, clus3_y], ['b', 'g', 'y'],
        ['cluster 1', 'cluster 2', 'cluster 3'],
        ['o', 'o', 'o'], [0.3, 0.5, 0.5],
        xlabel=xlabel, ylabel=ylabel, title=title,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        outfile=outfile, save=False, loc=1)
    '''
    
    return

def get_params() :
    
    # open requisite information about the sample
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        times = hf['times'][:]
        # subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        # Res = hf['Re'][:]
        quenched = hf['quenched'][:]
        tonsets = hf['onset_times'][:]
        tterms = hf['termination_times'][:]
        comparison = hf['comparison'][:]
        i75s = hf['i75s'][:].astype(int)
    
    # determine the snapshots that correspond to 75% through the quenching event
    imajorities = np.array(find_nearest(times, tonsets + 0.75*(tterms - tonsets)))
    
    with h5py.File('TNG50-1/TNG50-1_99_diagnostics(t)_2d.hdf5', 'r') as hf :
        sf_mass_within_1kpc = hf['sf_mass_within_1kpc'][:]
        # sf_mass_within_tenthRe = hf['sf_mass_within_tenthRe'][:]
        sf_mass_tot = hf['sf_mass'][:]
        # R10s = hf['R10'][:]
        R50s = hf['R50'][:]
        # R90s = hf['R90'][:]
        # sf_R10s = hf['sf_R10'][:]
        sf_R50s = hf['sf_R50'][:]
        # sf_R90s = hf['sf_R90'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_mechanism.hdf5', 'r') as hf :
        outside_in = hf['outside-in'][:] # 109
        inside_out = hf['inside-out'][:] # 103
        uniform = hf['uniform'][:]       # 8
        ambiguous = hf['ambiguous'][:]   # 58
    
    # get the truncation radii for the quenched sample
    Rtruncs = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, 2.375, 5.0, 2.625, 2.125, 1.375, 4.625,
        5.0, 5.0, 5.0, 4.375, 1.875, 5.0, np.nan, 5.0, np.nan, 2.375, 4.875,
        0.875, 5.0, 5.0, 1.625, np.nan, np.nan, 1.875, 5.0, 5.0, 3.875, 5.0,
        0.375, 5.0, 5.0, np.nan, np.nan, 2.625, 2.875, np.nan, 1.625, 0.875,
        1.125, 1.875, 1.625, 2.875, 3.125, 1.625, 4.375, 0.875, 1.375, 1.125,
        np.nan, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        0.875, 1.875, np.nan, 4.875, 5.0, 0.875, 4.125, 4.875, 0.875, 1.375,
        5.0, 2.375, 5.0, 4.625, 1.125, 3.125, 4.625, np.nan, np.nan, 1.125,
        2.125, 0.875, 5.0, 1.125, 5.0, 3.625, np.nan, np.nan, 2.375, 4.125,
        np.nan, 4.875, np.nan, np.nan, 5.0, 2.375, 2.875, 4.875, 2.125, np.nan,
        np.nan, np.nan, 3.375, np.nan, 3.875, 1.125, 5.0, np.nan, 5.0, 3.125,
        np.nan, 1.875, 1.875, 4.875, 5.0, 5.0, np.nan, np.nan, 5.0, np.nan,
        5.0, 1.375, 4.125, np.nan, np.nan, np.nan, 3.125, 5.0, np.nan, np.nan,
        1.375, 0.875, np.nan, np.nan, np.nan, 4.375, np.nan, 1.375, np.nan,
        np.nan, 1.875, np.nan, np.nan, np.nan, np.nan, 2.375, np.nan, np.nan,
        np.nan, np.nan, 2.125, 5.0, np.nan, 1.875, np.nan, 4.125, 1.625, np.nan,
        np.nan, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4.125,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3.875, np.nan, np.nan,
        np.nan, 3.625, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, 3.375, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        3.875, np.nan, np.nan, 0.625])
    
    # get the truncation radii for the comparison sample
    # FIXME
    np.random.seed(0)
    c_Rtruncs = np.random.normal(loc=4.8, scale=0.1, size=278)
    c_Rtruncs[c_Rtruncs >= 5] = 5
    
    # get the quenching mechanisms
    mech = np.array([4*outside_in, 3*inside_out, 2*uniform, ambiguous]).T
    mech = np.sum(mech, axis=1)
    
    # create various metrics of interest
    xis = sf_mass_within_1kpc/sf_mass_tot
    # alt_xis = sf_mass_within_tenthRe/sf_mass_tot
    
    # R10_zetas = sf_R10s/R10s
    zetas = sf_R50s/R50s
    # R90_zetas = sf_R90s/R90s
    
    # get values for the comparison sample
    firstDim = np.arange(278)
    i75s = i75s[comparison]
    c_xis = (xis[comparison])[firstDim, i75s]
    c_zetas = np.log10((zetas[comparison])[firstDim, i75s])
    # c_R90_zetas = (R90_zetas[comparison])[firstDim, i75s]
    
    # select the quenched galaxies
    mask = (logM[:, -1] >= 9.5) & quenched # 278 entries, but len(mask) = 8260
    imajorities = imajorities[mask]
    mech = mech[mask]
    q_xis = (xis[mask])[firstDim, imajorities]
    q_zetas = np.log10((zetas[mask])[firstDim, imajorities])
    # q_R90_zetas = (R90_zetas[mask])[firstDim, imajorities]
    
    # categorize data based on quenching mechanism
    oi_xis, oi_zetas = q_xis[mech == 4], q_zetas[mech == 4]
    io_xis, io_zetas = q_xis[mech == 3], q_zetas[mech == 3]
    oi_Rtruncs = Rtruncs[mech == 4]
    io_Rtruncs = Rtruncs[mech == 3]
    
    # reshape data into a 2D array
    data = np.array([np.concatenate([c_xis, oi_xis, io_xis]),
                     np.concatenate([c_zetas, oi_zetas, io_zetas]),
                     np.concatenate([c_Rtruncs, oi_Rtruncs, io_Rtruncs])]).T
    
    # mask various values
    data = data[~np.isnan(data).any(axis=1), :] # mask any row containing a NaN
    data = data[data[:, 1] > -3] # mask one very low value
    
    return (data, oi_xis, oi_zetas, oi_Rtruncs, io_xis, io_zetas, io_Rtruncs,
            c_xis, c_zetas, c_Rtruncs)

def perform_kmeans(data) :
    
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
    labels = kmeans.labels_
    
    # get the found clusters using labels
    clus1, clus2, clus3 = data[labels == 0], data[labels == 1], data[labels == 2]
    
    # get x, y, z values for plotting in 3D
    clus1_x, clus1_y, clus1_z = clus1[:, 0], clus1[:, 1], clus1[:, 2]
    clus2_x, clus2_y, clus2_z = clus2[:, 0], clus2[:, 1], clus2[:, 2]
    clus3_x, clus3_y, clus3_z = clus3[:, 0], clus3[:, 1], clus3[:, 2]
    
    return (clus1_x, clus1_y, clus1_z, clus2_x, clus2_y, clus2_z,
            clus3_x, clus3_y, clus3_z)
