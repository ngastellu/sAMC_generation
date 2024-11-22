import torch
import pickle
import time
import tqdm
import json
from torch import nn, optim
from models import *
a_file = open("datadims.pkl","rb")

dataDims = pickle. load(a_file)

import argparse
import json

parser = argparse.ArgumentParser()
def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action = 'store_true')
    group.add_argument('--no-' + name, dest=name, action = 'store_false')
    parser.set_defaults(**{name:default})

parser.add_argument('--run_num', type = int, default = 0)
parser.add_argument('--experiment_name', type = str, default = 'testing')
# model architecture
parser.add_argument('--model', type = str, default = 'gated1') # model architecture -- 'gated1'
parser.add_argument('--fc_depth', type = int, default = 512) # number of neurons for final fully connected layers
parser.add_argument('--init_conv_size', type=int, default= 3) # size of the initial convolutional window # ODD NUMBER
parser.add_argument('--conv_filters', type = int, default = 20) # number of filters per gated convolutional layer
parser.add_argument('--init_conv_filters', type=int, default = 20) # number of filters for the first convolutional layer  # MUST BE THE SAME AS 'conv_filters'
parser.add_argument('--conv_size', type = int, default = 3) # ODD NUMBER
parser.add_argument('--conv_layers', type = int, default = 70) # number of layers in the convnet - should be larger than the correlation length
parser.add_argument('--dilation', type = int, default = 1) # must be 1 - greater than 1 is deprecated
parser.add_argument('--activation_function', type = str, default = 'relu') # 'gated' is only working option
parser.add_argument('--fc_dropout_probability', type = float, default = 0.2) # dropout probability on hidden FC layer(s) [0,1)
parser.add_argument('--fc_norm', type = str, default = 'batch') # None or 'batch'
parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='gloo', type=str, help='')

# add_bool_arg(parser, 'subsample_images', default = True) # cut training images in transverse direction by a custom amount at runtime
#
# add_bool_arg(parser,'do_conditioning', default = True) # incorporate conditioning variables in model training
# # parser.add_argument('--init_conditioning_filters', type=int, default=64) # number of filters for optional conditioning layers
# parser.add_argument('-l', '--generation_conditions', nargs='+', default=[0.23, 0.22]) # conditions used to generate samples at runtime

# training parameters
parser.add_argument('--training_dataset', type = str, default = 'amorphous') # name of training dataset - 'fake welds', 'welds 1'
parser.add_argument('--training_batch_size', type = int, default = 4) # maximum training batch size
add_bool_arg(parser, 'auto_training_batch', default = True) # whether to automatically set training batch size to largest value < the max
parser.add_argument('--max_epochs', type = int, default =500) # number of epochs over which to train
parser.add_argument('--convergence_moving_average_window', type = int, default = 500) # BROKEN - moving average window used to compute convergence criteria
parser.add_argument('--max_dataset_size', type = int, default = 10000) # maximum dataset size (limited by size of actual dataset) - not sure the dataset is properly shuffled prior to this being applied
parser.add_argument('--convergence_margin', type = float, default = 1e-4) # cutoff which determines when the model has converged
parser.add_argument('--dataset_seed', type = int, default = 0)
parser.add_argument('--model_seed', type = int, default = 0)

# sample generation parameters
parser.add_argument('--bound_type', type = str, default = 'empty') # what is outside the image during training and generation 'empty'
parser.add_argument('--boundary_layers', type = int, default = 0) # number of layers of conv_field between sample and actual image boundary
parser.add_argument('--sample_outpaint_ratio', type = int, default = 4) # size of sample images, relative to the input images
parser.add_argument('--sample_generation_mode', type = str, default = 'serial') # 'parallel' or 'serial' - serial is currently untested
parser.add_argument('--sample_batch_size', type = int, default = 1) # maximum sample batch size - no automated test but can generally be rather large (1e3),
parser.add_argument('--generation_period', type = int, default = 100) # how often to run (expensive) generation during training
# utility of higher batch sizes for parallel generation is only realized with extremely large samples
parser.add_argument('--n_samples', type = int, default = 20) # number of samples to generate

add_bool_arg(parser, 'CUDA', default=True)
add_bool_arg(parser, 'comet', default=False)

configs,unknown= parser.parse_known_args()


model = GatedPixelCNN(configs,dataDims)
device = torch.device('cuda:0')
model.eval()
model.to(torch.device("cuda:0"))

optimizer = optim.SGD(model.parameters(),lr=1e-1, momentum=0.9, nesterov=True)#optim.AdamW(ddp_model.parameters(),lr=0.05, amsgrad=True)# optim.SGD(ddp_model.parameters(),lr=1e-1, momentum=0.9, nesterov=True)#optim.SGD(net.parameters(),lr=1e-4, momentum=0.9, nesterov=True)#optim.AdamW(ddp_model.parameters(),lr=0.01, amsgrad=True)

checkpoint = torch.load('model-128.pt', map_location=device)
bc_old=checkpoint['model_state_dict']
bc_new=bc_old.copy()
for items in bc_old.items():
    s1 = (items[0])
    s2 = s1[7:]
   
    bc_new[s2] = bc_new.pop(s1)
model.load_state_dict(bc_new)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if configs.sample_generation_mode == 'serial':
 

        sample_x_padded = dataDims['sample x dim'] + 2 * dataDims['conv field'] * configs.boundary_layers
        sample_y_padded = dataDims['sample y dim'] + dataDims['conv field'] * configs.boundary_layers  # don't need to pad the bottom

        batches = int(np.ceil(configs.n_samples/configs.sample_batch_size))
        #n_samples = sample_batch_size * batches
        sample = torch.zeros(configs.n_samples, dataDims['channels'], dataDims['sample y dim'], dataDims['sample x dim'])  # sample placeholder
        print('Generating {} Samples'.format(configs.n_samples))

        for batch in range(batches):  # can't do these all at once so we do it in batches
            print('Batch {} of {} batches'.format(batch + 1, batches))
            sample_batch = torch.FloatTensor(configs.sample_batch_size, dataDims['channels'] , sample_y_padded + 2 * dataDims['conv field'] + 1 - dataDims['conv field'] * 1, sample_x_padded + 2 * dataDims['conv field'])  # needs to be explicitly padded by the convolutional field
            sample_batch.fill_(0)  # initialize with minimum value

            # if configs.do_conditioning: # assign conditions so the model knows what we want
            #     for i in range(len(configs.generation_conditions)):
            #         sample_batch[:,1+i,:,:] = (configs.generation_conditions[i] - dataDims['conditional mean']) / dataDims['conditional std']

            if configs.CUDA:
                sample_batch = sample_batch.cuda()

            #generator.train(False)
            model.eval()
            with torch.no_grad():  # we will not be updating weights
                for i in tqdm.tqdm(range(dataDims['conv field'] + 1, sample_y_padded + dataDims['conv field'] + 1)):  # for each pixel
                    for j in range(dataDims['conv field'], sample_x_padded + dataDims['conv field']):
                        for k in range(dataDims['channels']): # should only ever be 1
                            #out = generator(sample_batch.float())
                            out = model(sample_batch[:, :, i - dataDims['conv field'] - 1:i  + 1, j - dataDims['conv field']:j + dataDims['conv field'] + 1].float())
                            out = torch.reshape(out, (out.shape[0], dataDims['classes'] + 1, dataDims['channels'], out.shape[-2], out.shape[-1]))
                            probs = F.softmax(out[:, 1:, k, -1, dataDims['conv field']]/1, dim=1).data # the remove the lowest element (boundary)
                            sample_batch[:, k, i, j] = (torch.multinomial(probs, 1).float() + 1).squeeze(1) / dataDims['classes']  # convert output back to training space

                            del out, probs

            for k in range(dataDims['channels']):
                sample[batch * configs.sample_batch_size:(batch + 1) * configs.sample_batch_size, k, :, :] = sample_batch[:, k, (configs.boundary_layers + 1) * dataDims['conv field'] + 1:, (configs.boundary_layers + 1) * dataDims['conv field']:-((configs.boundary_layers + 1) * dataDims['conv field'])] * dataDims['classes'] - 1  # convert back to input space

    
        np.save('samples/run_{}_amorphoussamples_loadedmodel'.format(configs.run_num), sample.cpu())

