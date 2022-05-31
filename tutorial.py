
# import numpy as np

import h5py
import requests

# import astropy.constants as c
# from astropy.cosmology import Planck15 as cosmo
# from astropy.table import Table
# import astropy.units as u

baseURL = 'http://www.tng-project.org/api/'

from core import get
'''
rr = get(baseURL)
names = [sim['name'] for sim in rr['simulations']]
index = names.index('TNG50-1')

sim = get(rr['simulations'][index]['url'])

snaps = get(sim['snapshots'])

snap = get(snaps[-1]['url'])

subs = get(snap['subhalos'], {'limit':20, 'order_by':'-mass_stars'})

sub = get(subs['results'][0]['url'])

url = sub['related']['parent_halo'] + 'info.json'
parent_fof = get(url)
print(parent_fof.keys())

# mpb1 = get(sub['trees']['sublink_mpb'])

'''
