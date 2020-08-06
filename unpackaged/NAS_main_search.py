from argparse import Namespace as APNamespace, _SubParsersAction,ArgumentParser
from pathlib import Path
import os
# import logging


import math
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import torch
import yaml
import shutil

from early_stop import EarlyStop
from train_help import *
from optim import get_optimizer_scheduler

from utils import parse_config
from metrics import Metrics
from models import get_net
from data import get_data
from AdaS import AdaS
import  global_vars as GLOBALS

if __name__ == '__main__':

    parser = ArgumentParser(description=__doc__)
    args(parser)
    args_true = parser.parse_args()
    
    train_loader,test_loader,device,optimizer,scheduler,output_path,starting_conv_sizes = initialize(args_true,0)
    output_path = output_path / f"conv_{GLOBALS.CONFIG['init_conv_setting']}_deltaThresh={GLOBALS.CONFIG['delta_threshold']}_minScaleLimit={GLOBALS.CONFIG['min_scale_limit']}_beta={GLOBALS.CONFIG['beta']}_epochpert={GLOBALS.CONFIG['epochs_per_trial']}_adaptnum={GLOBALS.CONFIG['adapt_trials']}"
    GLOBALS.OUTPUT_PATH_STRING = str(output_path)

    if not os.path.exists(GLOBALS.OUTPUT_PATH_STRING):
        os.mkdir(GLOBALS.OUTPUT_PATH_STRING)

    print('~~~Initialization Complete. Beginning first training~~~')

    epochs = range(0, GLOBALS.CONFIG['epochs_per_trial'])
    full_train_epochs = range(0, GLOBALS.CONFIG['max_epoch'])

    output_path_string_trials = GLOBALS.OUTPUT_PATH_STRING +'\\'+ 'Trials'
    output_path_string_modelweights = GLOBALS.OUTPUT_PATH_STRING +'\\'+ 'model_weights'
    output_path_string_graph_files = GLOBALS.OUTPUT_PATH_STRING +'\\'+ 'graph_files'
    output_path_string_full_train = GLOBALS.OUTPUT_PATH_STRING +'\\'+ 'full_train'
    output_path_train = output_path / f"Trials"
    output_path_fulltrain = output_path / f"full_train"

    if not os.path.exists(output_path_string_trials):
        os.mkdir(output_path_string_trials)

    if not os.path.exists(output_path_string_modelweights):
        os.mkdir(output_path_string_modelweights)

    if not os.path.exists(output_path_string_graph_files):
        os.mkdir(output_path_string_graph_files)

    if not os.path.exists(output_path_string_full_train):
        os.mkdir(output_path_string_full_train)

    conv_data,rank_final_data,rank_stable_data,output_sizes,delta_info=run_trials(train_loader,test_loader,device,optimizer,scheduler,epochs,output_path_train)
    create_trial_data_file(conv_data,delta_info,rank_final_data,rank_stable_data,output_path_string_trials,output_path_string_graph_files,output_path_string_modelweights)
    print('Done Trials.')

    #run_saved_weights_full_train(train_loader,test_loader,device,output_sizes,range(0,250),output_path_fulltrain)
    #Note Last Iter not used
    #output_sizes=[[40,40,40,58,40,42,40],[252,90,142,90,78,90],[388,174,198,174,116,174],[334,116,176, 116, 82, 116],[388,30,40,30,44,30]]

    run_fresh_full_train(train_loader,test_loader,device,output_sizes,full_train_epochs,output_path_fulltrain)
    create_full_data_file(GLOBALS.NET,output_path_string_full_train+'\\'+f"AdaS_last_iter_fulltrain_trial=0_net={GLOBALS.CONFIG['network']}_dataset={GLOBALS.CONFIG['dataset']}.xlsx",
                                 output_path_string_full_train+'\\'+f"AdaS_fresh_fulltrain_trial=0_net={GLOBALS.CONFIG['network']}_dataset={GLOBALS.CONFIG['dataset']}.xlsx",
                                 output_path_string_full_train)

    print('Done Full Train and Trials.')
