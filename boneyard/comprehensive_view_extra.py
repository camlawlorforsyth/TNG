
import numpy as np

from astropy.table import Table
import astropy.units as u
import h5py
from pypdf import PdfWriter

from core import add_dataset, bsPath, cosmo
import plotting as plt

def comp_prop_plots_for_starbursts(simName='TNG50-1', snapNum=99, hw=0.1,
                                   minNum=50, nelson2021version=True,
                                   save=False) :
    
    # define the input and output directories, and the input and helper files
    inDir = bsPath(simName)
    infile = inDir + '/{}_{}_sample(t).hdf5'.format(simName, snapNum)
    helper_file = inDir + '/{}_{}_profiles(t).hdf5'.format(simName, snapNum)
    outDir = inDir + '/figures/comprehensive_plots_starbursts/'
    
    # get relevant information for the general sample, and quenched systems
    with h5py.File(infile, 'r') as hf :
        redshifts = hf['redshifts'][:]
        times = hf['times'][:]
        subIDfinals = hf['SubhaloID'][:].astype(int)
        subIDs = hf['subIDs'][:].astype(int)
        logM = hf['logM'][:]
        Re = hf['Re'][:]
        centers = hf['centers'][:]
        UVK = hf['UVK'][:]
        # primary = hf['primary_flag'][:]
        # cluster = hf['cluster'][:]
        # hm_group = hf['hm_group'][:]
        # lm_group = hf['lm_group'][:]
        # field = hf['field'][:]
        SFHs = hf['SFH'][:]
        SFMS = hf['SFMS'][:].astype(bool) # (boolean) SFMS at each snapshot
        lo_SFH = hf['lo_SFH'][:]
        hi_SFH = hf['hi_SFH'][:]
        # quenched = hf['quenched'][:]
        # ionsets = hf['onset_indices'][:].astype(int)
        # tonsets = hf['onset_times'][:]
        # iterms = hf['termination_indices'][:].astype(int)
        # tterms = hf['termination_times'][:]
    
    with h5py.File('TNG50-1/TNG50-1_99_potential_starburst_locations.hdf5', 'r') as hf :
        starburst = hf['starburst'][:]
        starburst_snaps = hf['starburst_snaps'][:].astype(int)
    
    # get basic information for the UVK and radial profile plots
    UVK_X_cent, UVK_Y_cent, UVK_contour, UVK_levels = determine_UVK_contours(UVK)
    (radial_edges, mids,
     spatial_edges, XX, YY, X_cent, Y_cent) = set_basic_info_for_plots()
    
    # set which version to use
    if nelson2021version :
        label = r'sSFR (yr$^{-1}$)'
    else :
        label = (r'$\log(\dot{\Sigma}_{\rm *, 100~Myr}/' +
                 r'{\rm M}_{\odot}~{\rm yr}^{-1}~{\rm kpc}^{-2})$')
    
    s_subIDfinals = subIDfinals[starburst]
    s_subIDs = subIDs[starburst]
    s_logM = logM[starburst]
    s_SFHs = SFHs[starburst]
    s_Re = Re[starburst]
    s_centers = centers[starburst]
    s_UVK = UVK[starburst]
    s_lo_SFH = lo_SFH[starburst]
    s_hi_SFH = hi_SFH[starburst]
    s_starburst_snaps = starburst_snaps[starburst]
    
    for i, (s_subfin, s_subID, s_logM, s_SFH, s_rad, s_cent, s_UVK_ind, s_lo,
        s_hi, s_snap) in enumerate(zip(s_subIDfinals, s_subIDs, s_logM, s_SFHs,
        s_Re, s_centers, s_UVK, s_lo_SFH, s_hi_SFH, s_starburst_snaps)) :
        
        outfile = outDir + 'subID_{}.pdf'.format(s_subfin)
        comprehensive_plot(s_subfin, s_subID, s_logM, s_SFH, s_rad,
                            s_cent, s_UVK_ind, s_lo, s_hi, redshifts,
                            times, subIDs, logM, Re, centers, UVK, SFHs,
                            SFMS, times[s_snap-3], s_snap+6, times[s_snap+3],
                            [s_snap-3, s_snap, s_snap+3],
                            radial_edges, mids, spatial_edges, XX, YY,
                            X_cent, Y_cent, UVK_X_cent, UVK_Y_cent,
                            UVK_contour, UVK_levels, label, helper_file,
                            outfile, nelson2021version=nelson2021version,
                            save=True)
    
    return

def file_prep_for_CRM() :
    
    infile = 'TNG50-1/TNG50-1_99_sample(t).hdf5'
    helper_file = 'TNG50-1/TNG50-1_99_profiles(t).hdf5'
    
    with h5py.File(infile, 'r') as hf :
        logM = hf['logM'][:, -1]
        Re = hf['Re'][:, -1]
        SFMS = hf['SFMS'][:, -1].astype(bool) # (boolean) SFMS at each snapshot
    
    with h5py.File(helper_file, 'r') as hf :
        edges = hf['edges'][:]
        mids = hf['mids'][:]
        SFR_profiles = hf['SFR_profiles'][:, -1, :]
        area_profiles = hf['area_profiles'][:, -1, :]
        nSFparticles = hf['nSFparticles'][:, -1, :].astype(int)
    
    # limit galaxies to those on the SFMS at redshift 0
    masses = logM[SFMS]
    Re = Re[SFMS]
    nSFparticles = nSFparticles[SFMS]
    
    # SFR surface density in each annulus, units of Mdot/yr/kpc^2
    profiles = SFR_profiles/area_profiles
    profiles = profiles[SFMS]
    
    with h5py.File('TNG50_SFMS_z0_SFRsurfaceArea_profiles.hdf5', 'w') as hf :
        add_dataset(hf, edges, 'edges')
        add_dataset(hf, mids, 'mids')
        add_dataset(hf, masses, 'logM')
        add_dataset(hf, Re, 'Re')
        add_dataset(hf, profiles, 'profiles')
        add_dataset(hf, nSFparticles, 'nSFparticles')
    
    return






def merge_comp_plots(simName='TNG50-1', snapNum=99) :
    
    # define the input and output directories and files
    base = bsPath(simName)
    infile = base + '/TNG50-1_99_sample(t).hdf5'
    inDir = base + '/figures/comprehensive_plots/'
    outDir = base + '/figures/comprehensive_plots_merged/'
    
    # get information about the quenched sample
    with h5py.File(infile, 'r') as hf :
        subIDfinals = hf['SubhaloID'][:]
        logM = hf['logM'][:]
        quenched = hf['quenched'][:]
    
    # get the subIDfinals
    quenched_subIDs = subIDfinals[quenched]
    quenched_logM_z0 = logM[:, -1][quenched]
    
    # now sort the subIDs and stellar masses
    sort = np.argsort(quenched_logM_z0)
    quenched_subIDs_sorted = quenched_subIDs[sort]
    quenched_logM_z0_sorted = quenched_logM_z0[sort]
    
    # set the mass bin edges to group quenched galaxies of a simiarl mass
    edges = [8.00, 8.25, 8.50, 8.75, 9.00, 9.50, 10.00, 11.00, 13.00]
    # 356, 276, 255, 186, 226, 111, 144, 23 quenched galaxies per bin
    
    # loop over every bin
    for start, end in zip(edges, edges[1:]) :
        mass_bin = ((quenched_logM_z0_sorted >= start) &
                    (quenched_logM_z0_sorted < end))
        
        # create a merged object
        merger = PdfWriter()
        
        # loop over every quenched galaxy in the mass bin
        for subID in quenched_subIDs_sorted[mass_bin] :
            merger.append(inDir + 'subID_{}.pdf'.format(subID))
        
        # write the merged file and close it
        outfile = outDir + 'merged_{:.2f}_{:.2f}.pdf'.format(start, end)
        merger.write(outfile)
        merger.close()
    
    table = Table([quenched_subIDs_sorted, quenched_logM_z0_sorted],
                  names=('subID', 'logM'))
    table.write('comprehensive_plots_visual_inspection.csv', overwrite=False)
    
    return

def merge_comprehensive_plots_based_on_mechanism() :
    
    with h5py.File('TNG50-1/TNG50-1_99_sample(t).hdf5', 'r') as hf :
        subIDfinals = hf['SubhaloID'][:]
        logM = hf['logM'][:, -1]
        quenched = hf['quenched'][:]
    
    # 278 total galaxies with logM >= 9.5
    with h5py.File('TNG50-1/TNG50-1_99_mechanism.hdf5', 'r') as hf :
        outside_in = hf['outside-in'][:] # 109
        inside_out = hf['inside-out'][:] # 103
        uniform = hf['uniform'][:]       # 8
        ambiguous = hf['ambiguous'][:]   # 58
    
    inDir = 'TNG50-1/figures/comprehensive_plots/'
    
    mask = (logM >= 9.5) & quenched
    subIDfinals = subIDfinals[mask]
    outside_in = outside_in[mask]
    inside_out = inside_out[mask]
    uniform = uniform[mask]
    ambiguous = ambiguous[mask]
    logM = logM[mask]
    
    sort = np.argsort(logM)
    subIDfinals = subIDfinals[sort]
    outside_in = outside_in[sort]
    inside_out = inside_out[sort]
    uniform = uniform[sort]
    ambiguous = ambiguous[sort]
    logM = logM[sort]
    
    merger = PdfWriter()
    for subID in subIDfinals[outside_in] :
        merger.append(inDir + 'subID_{}.pdf'.format(subID))
    outfile = 'TNG50-1/figures/comprehensive_plots_merged_mechanism/outside_in.pdf'
    merger.write(outfile)
    merger.close()
    
    merger = PdfWriter()
    for subID in subIDfinals[inside_out] :
        merger.append(inDir + 'subID_{}.pdf'.format(subID))
    outfile = 'TNG50-1/figures/comprehensive_plots_merged_mechanism/inside_out.pdf'
    merger.write(outfile)
    merger.close()
    
    merger = PdfWriter()
    for subID in subIDfinals[uniform] :
        merger.append(inDir + 'subID_{}.pdf'.format(subID))
    outfile = 'TNG50-1/figures/comprehensive_plots_merged_mechanism/uniform.pdf'
    merger.write(outfile)
    merger.close()
    
    merger = PdfWriter()
    for subID in subIDfinals[ambiguous] :
        merger.append(inDir + 'subID_{}.pdf'.format(subID))
    outfile = 'TNG50-1/figures/comprehensive_plots_merged_mechanism/ambiguous.pdf'
    merger.write(outfile)
    merger.close()
    
    return

def merge_starburst_comprehensive_plots() :
    
    inDir = 'TNG50-1/figures/comprehensive_plots_starbursts/'
    outfile = inDir + 'merged.pdf'
    
    merger = PdfWriter()
    for subID in [167404, 673933, 592984, 622298, 220605, 184939, 568923,
                  167395, 408534] :
        merger.append(inDir + 'subID_{}.pdf'.format(subID))
    merger.write(outfile)
    merger.close()
    
    return

def proposal_plot(q_subfin, q_subID, q_mass, q_radii, q_center, snaps,
                  redshifts, times, subIDs, logM, Re, centers, SFHs, SFMS,
                  outfile, nelson2021version=True) :
    
    spatial_edges = np.linspace(-5, 5, 61)
    XX, YY = np.meshgrid(spatial_edges, spatial_edges)
    
    # define the center points of the radial bins
    radial_edges = np.linspace(0, 5, 21)
    mids = []
    for start, end in zip(radial_edges, radial_edges[1:]) :
        mids.append(0.5*(start + end))
    
    if nelson2021version :
        label = r'sSFR (yr$^{-1}$)'
        ymin_b, ymax_b = 1e-12, 1e-9
    else :
        label = (r'$\log(\dot{\Sigma}_{\rm *, 100~Myr}/' +
                 r'{\rm M}_{\odot}~{\rm yr}^{-1}~{\rm kpc}^{-2})$')
        ymin_b, ymax_b = 1e-5, 1
    
    dfs, hists, fwhms, titles = [], [], [], []
    mains, los, meds, his = [], [], [], []
    
    for snap in snaps :
        
        resolution = (0.15*u.arcsec)/cosmo().arcsec_per_kpc_proper(
            redshifts[snap])
        res = resolution/(q_radii[snap]*u.kpc)
        fwhms.append(res)
        
        title = (r'$z = {:.2f}$'.format(redshifts[snap]) + ', ' +
                  r'$\Delta t_{\rm since~onset} = $' +
                  '{:.2f} Gyr'.format(times[snap] - times[snaps][0]))
        titles.append(title)
        
        df, hist = spatial_plot(times[snap], snap, q_subID[snap],
            q_center[snap], q_radii[snap], spatial_edges, 100*u.Myr,
            nelson2021version=nelson2021version)
        dfs.append(df)
        hists.append(hist)
        
        main, lo, med, hi = radial_plot(times, subIDs, logM, Re, centers, SFMS,
            snap, q_subID, q_mass, q_radii, q_center, radial_edges, 100*u.Myr,
            nelson2021version=nelson2021version)
        mains.append(main)
        los.append(lo)
        meds.append(med)
        his.append(hi)
    
    plt.plot_CASTOR_proposal(titles[0], titles[1], titles[2],
                             dfs[0], dfs[1], dfs[2],
                             hists[0], hists[1], hists[2],
                             fwhms[0], fwhms[1], fwhms[2],
                             mids,
                             mains[0], mains[1], mains[2],
                             los[0], los[1], los[2],
                             meds[0], meds[1], meds[2],
                             his[0], his[1], his[2],
                             XX, YY,
                             label=label,
                             xlabel_t=r'$\Delta x$ ($R_{\rm e}$)',
                             ylabel_t=r'$\Delta z$ ($R_{\rm e}$)',
                             xlabel_b=r'$r/R_{\rm e}$',
                             ylabel_b=label,
                             xmin_t=-5, xmax_t=5, ymin_t=-5, ymax_t=5,
                             xmin_b=0, xmax_b=5, ymin_b=ymin_b, ymax_b=ymax_b,
                             outfile=outfile, save=True)
    
    return

def radial_plot_info_old(snap, q_subID, helper_file, nelson2021version=True) :
    
    # define the label for the snapshot and quenched subID
    keyBase = 'Snapshot_{}/subID_{}_'.format(snap, q_subID)
    
    # get the radial profiles for the quenched galaxy and the
    # comparison control sample
    with h5py.File(helper_file, 'r') as hf :
        main_SFR_profile = hf[keyBase + 'main_SFR_profile'][:]
        main_mass_profile = hf[keyBase + 'main_mass_profile'][:]
        main_area_profile = hf[keyBase + 'main_area_profile'][:]
        
        control_SFR_profiles = hf[keyBase + 'control_SFR_profiles'][:]
        control_mass_profiles = hf[keyBase + 'control_mass_profiles'][:]
        control_area_profiles = hf[keyBase + 'control_area_profiles'][:]
    
    return

# count how many radial profiles were computed before
# total = 0
# with h5py.File('TNG50-1/TNG50-1_99_comprehensive_plots_helper_incorrect.hdf5', 'r') as hf :
#     keys = list(hf.keys())
#     for key in keys :
#         subkeys = list(hf[key].keys())
#         for subkey in subkeys :
#             if 'control' in subkey :
#                 entries = hf[key][subkey].shape[0]
#                 total += entries
# print(total) # 1729137

# but there are 465661 total SFMS galaxies across all snapshots, and 15070 quenched
# galaxies from the onset indices to the termination indices => 480731 total
