#!/usr/bin/env python
from pathlib import Path
import json
from glob import glob 
import numpy as np

def max_epoch(run_name):
    saved_epochs = np.array([int(chk.split('_')[-1].split('.')[0]) for chk in glob(f'{run_name}/model-epoch*.pt')])
    imax = np.max(saved_epochs)
    return imax

def best_epoch(run_name):
    logfile = f'{run_name}/{run_name}.log'
    epochs, _, te_loss = parse_losses(logfile)
    saved_epochs = np.array([int(chk.split('_')[-1].split('.')[0]) for chk in glob(f'{run_name}/model-epoch*.pt')])
    ii = (epochs == saved_epochs[:,None]).nonzero()[1]
    saved_te_losses = te_loss[ii]
    ibest = saved_epochs[np.argmin(saved_te_losses)]
    return ibest

def load_train_args(run_name):
    run_name = Path(run_name)
    json_file = run_name / 'train_configs.json'
    with open(json_file, 'r') as fo:
        args_dict = json.load(fo)
    return args_dict

def add_training_args(parser, overwrite=False):
    """Merges training args saved to JSON with the generation args defined in this script."""
    args_dict = load_train_args(parser.training_run_name)
    for k, v in args_dict.items():
        if overwrite or not hasattr(parser, k):
            setattr(parser, k, v)
    return parser


def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action = 'store_true')
    group.add_argument('--no-' + name, dest=name, action = 'store_false')
    parser.set_defaults(**{name:default})


def save_args(configs, run_type):
    """Saves run arguments to JSON file to keep track of experimens/ensure reproducibility."""
    if run_type == 'train':
        outdir = Path(configs.experiment_name)
        outdir.mkdir(exist_ok=True)
    elif run_type == 'generate':
        outdir = Path(configs.training_run_name)
    json_file = outdir / f'{run_type}_configs.json'
    print(f'Saving args to {json_file}...',end = ' ', flush=True)
    args_dict = vars(configs)

    with open(json_file, 'w') as fo:
        json.dump(args_dict, fo, indent=4)
    print('Done!', flush=True)

def parse_losses(logfile,nepochs=1000, log_frequency=2):
    npts = nepochs // log_frequency # nb of logged loss entries
    epochs = np.zeros(npts,dtype=int)
    tr_loss = np.zeros(npts)
    te_loss = np.zeros(npts)
    with open(logfile) as fo:
        k = 0
        for line in fo:
            line = line.strip()
            if len(line) == 0:
                continue # skip empty lines
            split_line = line.split()
            epochs[k] = int(split_line[0])
            tr_loss[k] = float(split_line[1].split('(')[1][:-1]) # get rid of comma at end of number
            te_loss[k] = float(split_line[4].split('(')[1][:-1]) # get rid of comma at end of number
            k+=1
    
    return keep_nonzero(epochs), keep_nonzero(tr_loss), keep_nonzero(te_loss)

def keep_nonzero(arr):
    return arr[arr != 0]