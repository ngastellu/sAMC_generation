#!/usr/bin/env python

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import histogram, multiple_histograms 

datadir = Path("~/Desktop/simulation_outputs/all_ones_MAP_expt/").expanduser()

suffix = '_init'
ifilters = np.arange(40)

hdata = np.load(datadir / 'pre_activation' / f'hdata{suffix}2.npy')
vdata = np.load(datadir / 'pre_activation' / f'vdata{suffix}.npy')

hdata_sum = hdata.sum(axis=1)
vdata_sum = vdata.sum(axis=1)

fig, axs = plt.subplots(1,2,sharex=True,sharey=True)
im = axs[0].imshow(hdata_sum[0,:20,:20])
axs[0].set_title(r'$\Sigma$ hdata (with v_to_h added)')
axs[1].imshow(vdata_sum[0,:20,:20],
              vmin=im.get_clim()[0],
              vmax=im.get_clim()[1])
axs[1].set_title(r'$\Sigma$ vdata')

cbar = fig.colorbar(im, ax=axs, orientation='vertical',
                    fraction=0.046, pad=0.04)

plt.show()

# v_nnz = (vdata != 0).nonzero()[1:]
# print(f'Found {len(v_nnz[0])} nonzero indices in vdata{suffix}.npy')

# ifilter_nnz = set(v_nnz[1])

# for ifilter in ifilters:
#     fig, axs = plt.subplots(1,2,sharex=True, sharey=True)

#     hd = hdata[0,ifilter,:20,:20]
#     hck = hd == hd[0,0]
#     print(np.all(hck))

#     print((~hck).nonzero())

#     # if not np.all(hck):
#     #     inds = np.vstack((~hck).nonzero())
#     #     for ij in inds.T:
#     #         i,j =ij
#     #         if i > 20: break
#     #         print(f'{(i,j)} -->  {np.abs(hd[i,j] - hd[0,0])}')

#     vd = vdata[0,ifilter,:20,:20]
#     # vck = vd == vd[0,0]
#     # print(np.all(vck))

#     # if not np.all(vck):
#     #     inds = np.vstack((~vck).nonzero())
#     #     for ij in inds.T:
#     #         i,j =ij
#     #         if i > 20: break
#     #         print(f'{(i,j)} -->  {np.abs(vd[i,j]- vd[0,0])}')

#     axs[0].imshow(hd)
#     axs[1].imshow(vd)

#     plt.show()

#     # multiple_histograms([hdata[0,ifilter,:,:],vdata[0,ifilter,:,:]], labels=['hdata', 'vdata'], bins=30, log_counts=True)
    
#     # fig2, axs = plt.subplots(1,2,sharex=True, sharey=True)
#     # fig2, axs[0] = histogram(hdata[0,ifilter,:,:], bins=30, plt_objs=(fig,axs[0]),show=False, log_counts=True)
#     # fig2, axs[1] = histogram(vdata[0,ifilter,:,:], bins=30, plt_objs=(fig,axs[1]),show=False, log_counts=True)

#     # plt.show()