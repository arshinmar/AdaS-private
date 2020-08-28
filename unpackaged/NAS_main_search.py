from argparse import Namespace as APNamespace, _SubParsersAction,ArgumentParser
from pathlib import Path
import os
import platform
# import logging
#import torch.backends.cudnn as cudnn
import numpy as np
import torch

from train_help import *
import  global_vars as GLOBALS

if __name__ == '__main__':

    parser = ArgumentParser(description=__doc__)
    args(parser)
    args_true = parser.parse_args()
    train_loader,test_loader,device,optimizer,scheduler,output_path,starting_conv_sizes = initialize(args_true,0)
    print(GLOBALS.CONFIG['delta_threshold_values'])
    for i in GLOBALS.CONFIG['delta_threshold_values']: #Where threshold_values is a list of threshold values we want to iterate over?
        GLOBALS.FIRST_INIT = True
        parser = ArgumentParser(description=__doc__)
        args(parser)
        args_true = parser.parse_args()
        train_loader,test_loader,device,optimizer,scheduler,output_path,starting_conv_sizes = initialize(args_true,0,new_threshold=i)

        output_path = output_path / f"conv_{GLOBALS.CONFIG['init_conv_setting']}_deltaThresh={GLOBALS.CONFIG['delta_threshold']}_minScaleLimit={GLOBALS.CONFIG['min_scale_limit']}_beta={GLOBALS.CONFIG['beta']}_epochpert={GLOBALS.CONFIG['epochs_per_trial']}_adaptnum={GLOBALS.CONFIG['adapt_trials']}"
        GLOBALS.OUTPUT_PATH_STRING = str(output_path)

        if not os.path.exists(GLOBALS.OUTPUT_PATH_STRING):
            os.mkdir(GLOBALS.OUTPUT_PATH_STRING)

        print('~~~Initialization Complete. Beginning first training~~~')

        epochs = range(0, GLOBALS.CONFIG['epochs_per_trial'])
        full_train_epochs = range(0, GLOBALS.CONFIG['max_epoch'])
        if platform.system == 'Windows':
            slash = '\\'
        else:
            slash = '/'
        output_path_string_trials = GLOBALS.OUTPUT_PATH_STRING +slash+ 'Trials'
        output_path_string_modelweights = GLOBALS.OUTPUT_PATH_STRING +slash+ 'model_weights'
        output_path_string_graph_files = GLOBALS.OUTPUT_PATH_STRING +slash+ 'graph_files'
        output_path_string_full_train = GLOBALS.OUTPUT_PATH_STRING +slash+ 'full_train'
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

        if GLOBALS.CONFIG['full_train_only']==False:
            print('Starting Trials')
            kernel_data,conv_data,rank_final_data,rank_stable_data,output_sizes,kernel_sizes,delta_info,delta_info_kernel=run_trials(train_loader,test_loader,device,optimizer,scheduler,epochs,output_path_train,new_threshold=i)
            create_trial_data_file(kernel_data,conv_data,delta_info_kernel,delta_info,rank_final_data,rank_stable_data,output_path_string_trials,output_path_string_graph_files,output_path_string_modelweights)
            print('Done Trials.')
        else:
            try:
                #print(int('booger'))
                output_sizes=get_output_sizes(output_path_string_trials+'\\'+'adapted_architectures.xlsx')
            except:
                output_sizes=[[32,32,32,32,32,32,32],[32,32,32,32,32,32,32,32],[32,32,32,32,32,32,32,32,32,32,32,32],[32,32,32,32,32,32]] #WHATEVER WE WANT.

        #output_sizes=[[64,64,64,64,64],[64,64,64,64],[64,64,64,64],[64,64,64,64],[64,64,64,64]]

        run_fresh_full_train(train_loader,test_loader,device,output_sizes,kernel_sizes,full_train_epochs,output_path_fulltrain)

        create_full_data_file(GLOBALS.NET,output_path_string_full_train+'\\'+f"StepLR_last_iter_fulltrain_trial=0_net={GLOBALS.CONFIG['network']}_dataset={GLOBALS.CONFIG['dataset']}.xlsx",
                                     output_path_string_full_train+'\\'+f"StepLR_fresh_fulltrain_trial=0_net={GLOBALS.CONFIG['network']}_dataset={GLOBALS.CONFIG['dataset']}.xlsx",
                                     output_path_string_full_train)

    print('Done Full Train')
