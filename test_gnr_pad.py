#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from ase import build

width=1
length=1


def bitmap_graphene(N,edge_type):
    DX_A = 0
    pass

for edge_type in ['armchair', 'zigzag']:
    print(f'\n*** {edge_type} ***')

    gnr = build.graphene_nanoribbon(width,length,edge_type).positions
    print(gnr.shape)
    gnr = gnr[:,[0,2]]
    print(f'Lx = {np.max(gnr[:,0] - np.min(gnr[:,0]))}')
    print(f'Ly = {np.max(gnr[:,1] - np.min(gnr[:,1]))}')
