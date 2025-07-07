#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from ase import build
from utils import _pixelize_data


def bitmap_graphene(N,edge_type, npad, pxl2angstrom=0.2):
    """Generates a masked pixel image of a graphene nanoribbon, with size (N,N)."""
    # Unit cell dimensions for GNR type
    DX_A = 1.2297560733739028
    DY_A = 2.84
    DX_Z = 0.71
    DY_Z = 1.2297560733739028

    if edge_type == 'armchair':
        dx = DX_A
        dy = DY_A
    elif edge_type == 'zigzag':
        dx = DX_Z
        dy = DY_Z
    else:
        print(f'!!! Invalid edge type {edge_type}. Assuming zigzag edge for what follows. !!!')
        dx = DX_Z
        dy = DY_Z


    Nx = N + 2*npad # pad both left and right sides of image
    Lx = Nx * pxl2angstrom
    Ny = N + npad # pad only top part of image
    Ly = Ny * pxl2angstrom

    n = 1 + int(Lx // dx) # number of unit cells along x-direction
    m = 1 + int(Ly // dy) # number of unit cells along y-direction

    print(m)
    print(n)

    gnr = build.graphene_nanoribbon(n,m,edge_type).positions
    natoms = [gnr.shape[0]]
    gnr = gnr[None,:,:] # add axis along first dim to make it compatible with _pixelize_data and the model
    icoords = (2,0)
    img = _pixelize_data(gnr, 1, natoms, icoords, img_shape=(Ny,Nx))
    print('# of nonzero pixels in img = ', img.sum())
    
    #apply mask; zero all pixels expect the padding
    masked_img = np.zeros((1,Ny,Nx))
    masked_img[0,:npad, :] = img[0,:npad, :] #pad top
    masked_img[0, :, :npad] = img[0,:, :npad] #pad left
    masked_img[0, :, -npad:] = img[0,:, -npad:] #pad right
    
    print('# of nonzero pixels in masked_img = ',  masked_img.sum())

    return img, masked_img




N = 300
npad = 66

for edge_type in ['armchair', 'zigzag']:
    img, masked_img = bitmap_graphene(N,edge_type,npad)
    
    # fig, axs = plt.subplots(1,2,sharex=True,sharey=True)
    # axs[0].imshow(img[0,:,:])
    plt.imshow(masked_img[0,:,:])
    plt.show()