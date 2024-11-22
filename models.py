import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.special as special
from itertools import combinations
import sys

class MaskedConv2d_h(nn.Conv2d):  # add a mask to the regular Conv2D function, so that it cannot learn about the future
    def __init__(self, mask_type, channels, *args, **kwargs):
        super(MaskedConv2d_h, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}  # mask A is for the first convolutional layer only, all deeper layers use mask B
        _, _, kH, kW = self.weight.size()  # get the size of the convolutional filter
        self.register_buffer('mask', self.weight.data.clone())  # initialize mask
        self.mask.fill_(1)  # start building the masks

        # spatial masking - prevent information from neighbours
        if mask_type=='A':
            self.mask[:, :, :, -1] = 0  # mask type B allows access to the 'present' pixel, mask A does not

        if channels > 1:
            # channel masking - block information from nearby color channels - ONLY 2 CHANNELS
            ''' 
            filters will be stacked as x1,x2,x3,x1,x2,x3,... , therefore, we will mask such that 
            e.g. filter 2 serving x2 can see previous outputs from x3, but not x1
            we will achieve this by building a connections graph, which will zero-out all elements from given channels 
            '''
            # mask A only allows information from lower channels
            Cin = self.mask.shape[1] # input filters
            Cout = self.mask.shape[0] # output filters
            def channel_mask(i_out, i_in): # a map which defines who is allowed to see what
                cout_idx = np.expand_dims(np.arange(Cout) % 2 == i_out, 1)
                cin_idx = np.expand_dims(np.arange(Cin) % 2 == i_in, 0)
                a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
                return a1 * a2

            mask = np.array(self.mask)
            for c in range(2): # mask B allows information from current and lower channels
                mask[channel_mask(c, c), kH // 2, kW // 2] = 0.0 if mask_type == 'A' else 1.0

            mask[channel_mask(0, 1), kH // 2, kW // 2] = 0.0
            self.mask = torch.from_numpy(mask)
    def forward(self, x):
        self.weight.data *= self.mask  # at each forward pass, apply the mask to all the filters (zero-out information about the future)
        return super(MaskedConv2d_h, self).forward(x)


def gated_activation(input):
    # implement gated activation from Conditional Generation with PixelCNN Encoders
    assert (input.shape[1] % 2) == 0
    a, b = torch.chunk(input, 2, 1) # split input into two equal parts - only works for even number of filters
    a = torch.tanh(a)
    b = torch.sigmoid(b)

    return torch.mul(a,b) # return element-wise (sigmoid-gated) product

class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return gated_activation(input)


class Activation(nn.Module):
    def __init__(self, activation_func, filters, *args, **kwargs):
        super().__init__()
        if activation_func == 'gated':
            self.activation = gated_activation
        elif activation_func == 'relu':
            self.activation = F.relu

    def forward(self, input):
        return self.activation(input)


class StackedConvolution(nn.Module):
    def __init__(self, f_in, f_out, kernel_size, padding, dilation, activation, *args, **kwargs):
        super(StackedConvolution, self).__init__(*args, **kwargs)

        self.padding = padding
        self.act_func = activation
        self.pad = dilation * (kernel_size // 2)
        self.v_activation = Activation(self.act_func, f_out) # for ReLU, must change number of filters as gated approach halves filters on each application
        self.h_activation = Activation(self.act_func, f_out)
        if activation == 'gated': # filter ratio - need to double a bunch of filters for gated activation
            f_rat = 2
        else:
            f_rat = 1
        self.v_Conv2d = nn.Conv2d(f_in, f_rat * f_out, (kernel_size//2 + 1, kernel_size), 1, (padding * (self.pad), padding * self.pad), dilation, bias=False, padding_mode='zeros')
        self.v_to_h_fc = nn.Conv2d(f_rat * f_out, f_rat * f_out, 1, bias=False)
        #self.h_Conv2d = nn.Conv2d(f_in, f_rat * f_out, (1, kernel_size // 2 + 1), 1, (0, padding * self.pad), dilation, bias=True, padding_mode='zeros')
        self.h_Conv2d = nn.Conv2d(f_in, f_rat * f_out, (1, kernel_size // 2 + 1), 1, (0, padding * self.pad), dilation, bias=False, padding_mode='zeros')
        #self.h_to_skip = nn.Conv2d(f_out, f_out, 1, bias=False)
        self.h_to_h = nn.Conv2d(f_out, f_out, 1, bias=False)

    def forward(self, v_in, h_in):
        residue = h_in.clone() # residual track

        if self.padding == 0:
            v_in = self.v_Conv2d(v_in) # remove extra padding
            v_to_h = self.v_to_h_fc(v_in)#[:,:,:-1,:] # align v stack to h
            h_in = self.h_Conv2d(h_in)[:, :, (self.pad):, :-self.pad]  # unpad by 1 on rhs
            residue = residue[:,:,self.pad:,self.pad:-self.pad]
        else:
            v_in = self.v_Conv2d(v_in)[:, :, :-(self.pad), :]  # remove extra padding
            v_to_h = self.v_to_h_fc(v_in)#[:,:,:-1,:] # align v stack to h
            h_in = self.h_Conv2d(h_in)[:, :, :, :-self.pad]  # unpad by 1 on rhs
        h_out = self.h_activation(torch.add(h_in, v_to_h))
        v_out = self.v_activation(v_in)

        #skip = self.h_to_skip(h_out)
        h_out = self.h_to_h(h_out)
        h_out = torch.add(h_out, residue) # add the residue if the sizes are the same

        return v_out, h_out#, skip

class GatedPixelCNN(nn.Module):  # Dense or residual, gated, blocked, dilated PixelCNN with batchnorm
    def __init__(self, configs, dataDims):
        super(GatedPixelCNN, self).__init__()

        ### initialize constants
        self.act_func = configs.activation_function
        if self.act_func == 'gated': # filter ratio - need to double a bunch of filters for gated activation
            f_rat = 2
        else:
            f_rat = 1
        kernel_size = configs.conv_size
        initial_convolution_size = configs.init_conv_size
        self.initial_pad = (initial_convolution_size - 1) // 2
        padding = 1 # DO NOT CHANGE THIS
        channels = dataDims['channels']
        self.layers = configs.conv_layers
        self.filters = configs.conv_filters
        initial_filters = configs.init_conv_filters

        f_in = (np.ones(configs.conv_layers + 1) * configs.conv_filters).astype(int)
        f_in[0        ] = initial_filters
        f_out = (np.ones(configs.conv_layers + 1) * configs.conv_filters).astype(int)
        self.h_init_activation = Activation(self.act_func, initial_filters)
        self.v_init_activation = Activation(self.act_func, initial_filters)
        out_maps = dataDims['classes'] + 1


        # initial layer
        self.v_initial_convolution = nn.Conv2d(channels, f_rat * initial_filters, (self.initial_pad + 1, initial_convolution_size), 1, padding * (self.initial_pad + 1, self.initial_pad), padding_mode='zeros', bias=False)
        self.v_to_h_initial = nn.Conv2d(f_rat * initial_filters, f_rat * initial_filters, 1, bias=False)
        self.h_initial_convolution = MaskedConv2d_h('A', channels, channels, f_rat * initial_filters, (1, self.initial_pad + 1), 1, padding * (0, self.initial_pad), padding_mode='zeros', bias=False)
        self.h_to_skip_initial = nn.Conv2d(initial_filters, initial_filters, 1, bias=False)
        self.h_to_h_initial = nn.Conv2d(initial_filters, initial_filters, 1, bias=False)

        # stack hidden layers in blocks
        self.conv_layer = [StackedConvolution(f_in[i], f_out[i], kernel_size, padding, 1, self.act_func) for i in range(configs.conv_layers)] # stacked convolution (no blind spot)
        self.conv_layer = nn.ModuleList(self.conv_layer)

        #output layers
        fc_filters = configs.fc_depth
        self.fc_activation = Activation('relu', fc_filters)
        self.fc_dropout = nn.Dropout(configs.fc_dropout_probability)

        if configs.fc_norm is None:
            self.fc_norm = nn.Identity()
        elif configs.fc_norm == 'batch':
            self.fc_norm = nn.BatchNorm2d(fc_filters//2)
        else:
            print(configs.fc_norm + ' is not an implemented norm')
            sys.exit()

        self.fc1 = nn.Conv2d(f_out[-1], fc_filters, kernel_size=(1,1), bias=True)  # add skip connections
        self.fc2 = nn.Conv2d(fc_filters, fc_filters//2, kernel_size=(1,1), bias=True) # gated activation cuts filters by 2
        
        self.fc3 = nn.Conv2d(fc_filters // 2, out_maps * channels, 1) # gated activation cuts filters by 2
       
        # output fully connected (1x1 convolution) layers
    #    self.fc1 = nn.Conv2d(f_out[-1], 256, (1,1)) # final filters are 256, but can really be any even number >= 4
        

    def forward(self, input):
        # clean input

        # initial convolution
        v_data = self.v_initial_convolution(input)[:, :, :-(self.initial_pad + 2), :]  # remove extra
        v_to_h_data = self.v_to_h_initial(v_data)#[:,:,:-1,:] # align with h-stack
        h_data = self.h_initial_convolution(input)[:,:,:,:-self.initial_pad] # unpad rhs of image
        v_data = self.v_init_activation(v_data)
        h_data = self.h_init_activation(torch.add(v_to_h_data, h_data))

        h_data = self.h_to_h_initial(h_data)

        # hidden layers
        for i in range(self.layers):
            v_data, h_data = self.conv_layer[i](v_data, h_data) # stacked convolutions fix blind spot

        # output convolutions
        x = self.fc1(h_data)
        x = self.fc2(x)

        x = self.fc_norm(x)
        x = self.fc_activation(x)
        x = self.fc_dropout(x)
        x = self.fc3(x)
        

        return x

