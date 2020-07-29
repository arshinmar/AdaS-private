from argparse import Namespace as APNamespace, _SubParsersAction,ArgumentParser
from pathlib import Path
import os
# import logging

import copy
import math
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import torch
import yaml
import shutil
from models.own_network import AdaptiveNet
from early_stop import EarlyStop
from train_support import run_epochs, get_ranks, get_max_ranks_by_layer
from optim import get_optimizer_scheduler

from utils import parse_config
from metrics import Metrics
from models import get_net
from data import get_data
from AdaS import AdaS
import  global_vars as GLOBALS
from adaptive_channels import prototype

from adaptive_graph import create_adaptive_graphs,create_layer_plot
from ptflops import get_model_complexity_info

def args(sub_parser: _SubParsersAction):
    # print("\n---------------------------------")
    # print("AdaS Train Args")
    # print("---------------------------------\n")
    # sub_parser.add_argument(
    #     '-vv', '--very-verbose', action='store_true',
    #     dest='very_verbose',
    #     help="Set flask debug mode")
    # sub_parser.add_argument(
    #     '-v', '--verbose', action='store_true',
    #     dest='verbose',
    #     help="Set flask debug mode")
    # sub_parser.set_defaults(verbose=False)
    # sub_parser.set_defaults(very_verbose=False)
    sub_parser.add_argument(
        '--config', dest='config',
        default='config.yaml', type=str,
        help="Set configuration file path: Default = 'config.yaml'")
    sub_parser.add_argument(
        '--data', dest='data',
        default='.adas-data', type=str,
        help="Set data directory path: Default = '.adas-data'")
    sub_parser.add_argument(
        '--output', dest='output',
        default='adas_search', type=str,
        help="Set output directory path: Default = '.adas-output'")
    sub_parser.add_argument(
        '--checkpoint', dest='checkpoint',
        default='.adas-checkpoint', type=str,
        help="Set checkpoint directory path: Default = '.adas-checkpoint")
    sub_parser.add_argument(
        '--root', dest='root',
        default='.', type=str,
        help="Set root path of project that parents all others: Default = '.'")
    sub_parser.set_defaults(resume=False)
    sub_parser.add_argument(
        '--cpu', action='store_true',
        dest='cpu',
        help="Flag: CPU bound training")
    sub_parser.set_defaults(cpu=False)

def initialize(args: APNamespace):
    def get_loss(loss: str) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss() if loss == 'cross_entropy' else None

    root_path = Path(args.root).expanduser()
    config_path = Path(args.config).expanduser()
    data_path = root_path / Path(args.data).expanduser()
    output_path = root_path / Path(args.output).expanduser()
    # global checkpoint_path, config
    GLOBALS.CHECKPOINT_PATH = root_path / Path(args.checkpoint).expanduser()
    #checks
    if not config_path.exists():
        # logging.critical(f"AdaS: Config path {config_path} does not exist")
        raise ValueError(f"AdaS: Config path {config_path} does not exist")
    if not data_path.exists():
        print(f"AdaS: Data dir {data_path} does not exist, building")
        data_path.mkdir(exist_ok=True, parents=True)
    if not output_path.exists():
        print(f"AdaS: Output dir {output_path} does not exist, building")
        output_path.mkdir(exist_ok=True, parents=True)
    if not GLOBALS.CHECKPOINT_PATH.exists():
        if args.resume:
            raise ValueError(f"AdaS: Cannot resume from checkpoint without " +
                             "specifying checkpoint dir")
        GLOBALS.CHECKPOINT_PATH.mkdir(exist_ok=True, parents=True)
    #parse from yaml
    with config_path.open() as f:
        GLOBALS.CONFIG = parse_config(yaml.load(f))
        print('~~~GLOBALS.CONFIG:~~~')
        print(GLOBALS.CONFIG)
    print("Adas: Argument Parser Options")
    print("-"*45)
    print(f"    {'config':<20}: {args.config:<40}")
    print(f"    {'data':<20}: {str(Path(args.root) / args.data):<40}")
    print(f"    {'output':<20}: {str(Path(args.root) / args.output):<40}")
    #print(f"    {'checkpoint':<20}: " + No checkpoints used
    #      f"{str(Path(args.root) / args.checkpoint):<40}")
    print(f"    {'root':<20}: {args.root:<40}")
    #print(f"    {'resume':<20}: {'True' if args.resume else 'False':<20}") No checkpoints / resumes used
    print("\nAdas: Train: Config")
    print(f"    {'Key':<20} {'Value':<20}")
    print("-"*45)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for k, v in GLOBALS.CONFIG.items():
        if isinstance(v, list):
            print(f"    {k:<20} {v}")
        else:
            print(f"    {k:<20} {v:<20}")
    print(f"AdaS: Pytorch device is set to {device}")
    # global best_acc
    GLOBALS.BEST_ACC = 0  # best test accuracy

    start_epoch = 0
    '''
    Early stopping stuff
    '''
    if np.less(float(GLOBALS.CONFIG['early_stop_threshold']), 0):
        print(
            "AdaS: Notice: early stop will not be used as it was set to " +
            f"{GLOBALS.CONFIG['early_stop_threshold']}, training till " +
            "completion")
    elif GLOBALS.CONFIG['optim_method'] != 'SGD' and \
            GLOBALS.CONFIG['lr_scheduler'] != 'AdaS':
        print(
            "AdaS: Notice: early stop will not be used as it is not SGD with" +
            " AdaS, training till completion")
        GLOBALS.CONFIG['early_stop_threshold'] = -1

    train_loader, test_loader = get_data(
                root=data_path,
                dataset=GLOBALS.CONFIG['dataset'],
                mini_batch_size=GLOBALS.CONFIG['mini_batch_size'])
    GLOBALS.PERFORMANCE_STATISTICS = {}
    #Gets initial conv size list (string) from config yaml file and converts into int list
    init_conv = [int(conv_size) for conv_size in GLOBALS.CONFIG['init_conv_setting'].split(',')]
    GLOBALS.NET = get_net(
                GLOBALS.CONFIG['network'], num_classes=10 if
                GLOBALS.CONFIG['dataset'] == 'CIFAR10' else 100 if
                GLOBALS.CONFIG['dataset'] == 'CIFAR100'
                else 1000, init_adapt_conv_size=init_conv)

    GLOBALS.METRICS = Metrics(list(GLOBALS.NET.parameters()),
                                      p=GLOBALS.CONFIG['p'])

    GLOBALS.NET = GLOBALS.NET.to(device)

    GLOBALS.CRITERION = get_loss(GLOBALS.CONFIG['loss'])

    optimizer, scheduler = get_optimizer_scheduler(
            net_parameters=GLOBALS.NET.parameters(),
            listed_params=list(GLOBALS.NET.parameters()),
            # init_lr=learning_rate,
            # optim_method=GLOBALS.CONFIG['optim_method'],
            # lr_scheduler=GLOBALS.CONFIG['lr_scheduler'],
            train_loader_len=len(train_loader),
            config=GLOBALS.CONFIG)

    GLOBALS.EARLY_STOP = EarlyStop(
                patience=int(GLOBALS.CONFIG['early_stop_patience']),
                threshold=float(GLOBALS.CONFIG['early_stop_threshold']))

    GLOBALS.OPTIMIZER = optimizer
    if device == 'cuda':
            GLOBALS.NET = torch.nn.DataParallel(GLOBALS.NET)
            cudnn.benchmark = True

    return train_loader,test_loader,device,optimizer,scheduler,output_path,init_conv

def new_output_sizes(current_conv_sizes,ranks,threshold):
    scaling_factor=np.subtract(ranks,threshold)
    new_conv_sizes = np.multiply(current_conv_sizes,np.add(1,scaling_factor))
    new_conv_sizes = [int(i) for i in new_conv_sizes]
    print(scaling_factor,'Scaling Factor')
    print(current_conv_sizes, 'CURRENT CONV SIZES')
    print(new_conv_sizes,'NEW CONV SIZES')
    return new_conv_sizes

def create_graphs(accuracy_data_file_name,conv_data_file_name,out_folder):
    create_adaptive_graphs(accuracy_data_file_name,GLOBALS.CONFIG['epochs_per_trial'],GLOBALS.CONFIG['adapt_trials'],out_folder)
    create_layer_plot(conv_data_file_name,GLOBALS.CONFIG['adapt_trials'],out_folder)
    return True

'''
input_ranks = 36-long list
output_ranks = 36-long list

conv_size_list = 33-long list

output_conv_size_list = 33-long list
'''

'''[3,3,3,3,3]'''

def even_round(number):
    return int(round(number/2)*2)

def calculate_correct_output_sizes(input_ranks,output_ranks,conv_size_list,shortcut_indexes,threshold):
    #Note that input_ranks/output_ranks may have a different size than conv_size_list
    #threshold=GLOBALS.CONFIG['adapt_rank_threshold']

    input_ranks_layer_1, output_ranks_layer_1 = input_ranks[0], output_ranks[0]
    input_ranks_superblock_1, output_ranks_superblock_1 = input_ranks[1:shortcut_indexes[0]], output_ranks[1:shortcut_indexes[0]]
    input_ranks_superblock_2, output_ranks_superblock_2 = input_ranks[shortcut_indexes[0]+1:shortcut_indexes[1]], output_ranks[shortcut_indexes[0]+1:shortcut_indexes[1]]
    input_ranks_superblock_3, output_ranks_superblock_3 = input_ranks[shortcut_indexes[1]+1:shortcut_indexes[2]], output_ranks[shortcut_indexes[1]+1:shortcut_indexes[2]]
    input_ranks_superblock_4, output_ranks_superblock_4 = input_ranks[shortcut_indexes[2]+1:shortcut_indexes[3]], output_ranks[shortcut_indexes[2]+1:shortcut_indexes[3]]
    input_ranks_superblock_5, output_ranks_superblock_5 = input_ranks[shortcut_indexes[3]+1:], output_ranks[shortcut_indexes[3]+1:]

    new_input_ranks = [input_ranks_superblock_1] + [input_ranks_superblock_2] + [input_ranks_superblock_3] + [input_ranks_superblock_4] + [input_ranks_superblock_5]
    new_output_ranks = [output_ranks_superblock_1] + [output_ranks_superblock_2] + [output_ranks_superblock_3] + [output_ranks_superblock_4] + [output_ranks_superblock_5]

    block_averages=[]
    block_averages_input=[]
    block_averages_output=[]
    grey_list_input=[]
    grey_list_output=[]

    for i in range(0,len(new_input_ranks),1):
        block_averages+=[[]]
        block_averages_input+=[[]]
        block_averages_output+=[[]]
        grey_list_input+=[[]]
        grey_list_output+=[[]]
        temp_counter=0
        for j in range(1,len(new_input_ranks[i]),2):
            block_averages_input[i]=block_averages_input[i]+[new_input_ranks[i][j]]
            block_averages_output[i]=block_averages_output[i]+[new_output_ranks[i][j-1]]

            grey_list_input[i]=grey_list_input[i]+[new_input_ranks[i][j-1]]
            grey_list_output[i]=grey_list_output[i]+[new_output_ranks[i][j]]

        block_averages_input[i]=block_averages_input[i]+[np.average(np.array(grey_list_input[i]))]
        block_averages_output[i]=block_averages_output[i]+[np.average(np.array(grey_list_output[i]))]
        block_averages[i]=np.average(np.array([block_averages_input[i],block_averages_output[i]]),axis=0)

    print(conv_size_list,'CONV SIZE LIST')
    output_conv_size_list=copy.deepcopy(conv_size_list)
    for i in range(0,len(block_averages)):
        for j in range(0,len(conv_size_list[i])):
            if (i==0):
                if (j%2==0):
                    scaling_factor=block_averages[i][-1]-threshold
                else:
                    scaling_factor=block_averages[i][int((j-1)/2)]-threshold
            else:
                if (j%2==1):
                    scaling_factor=block_averages[i][-1]-threshold
                else:
                    scaling_factor=block_averages[i][int(j/2)]-threshold
            output_conv_size_list[i][j]=even_round(output_conv_size_list[i][j]*(1+scaling_factor))

    GLOBALS.super1_idx = output_conv_size_list[0]
    GLOBALS.super2_idx = output_conv_size_list[1]
    GLOBALS.super3_idx = output_conv_size_list[2]
    GLOBALS.super4_idx = output_conv_size_list[3]
    GLOBALS.super5_idx = output_conv_size_list[4]
    GLOBALS.index = output_conv_size_list[0] + output_conv_size_list[1] + output_conv_size_list[2] + output_conv_size_list[3] + output_conv_size_list[4]

    print(output_conv_size_list,'OUTPUT CONV SIZE LIST')
    return output_conv_size_list

def network_initialize(new_network):
    GLOBALS.NET = torch.nn.DataParallel(new_network.cuda())
    GLOBALS.NET_RAW = new_network.cuda()
    cudnn.benchmark = True

    optimizer, scheduler = get_optimizer_scheduler(
            net_parameters=GLOBALS.NET.parameters(),
            listed_params=list(GLOBALS.NET.parameters()),
            train_loader_len=len(train_loader),
            config=GLOBALS.CONFIG)
    return optimizer, scheduler

def update_network(output_sizes):
    new_model_state_dict = prototype(GLOBALS.NET.state_dict(),output_sizes)
    new_network=AdaptiveNet(num_classes=10,new_output_sizes=output_sizes)
    new_network.load_state_dict(new_model_state_dict)
    return new_network

def create_data_file(new_network,full_save_file,full_fresh_file,output_path_string_full_train):
    parameter_data = pd.DataFrame(columns=['Accuracy (%)','Training Loss','GMacs','GFlops','Parameter Count (M)'])

    full_save_dfs=pd.read_excel(full_save_file)
    full_fresh_dfs=pd.read_excel(full_fresh_file)

    final_epoch_save=full_save_dfs.columns[-1][(full_save_dfs.columns[-1].index('epoch_')+6):]
    final_epoch_fresh=full_fresh_dfs.columns[-1][(full_fresh_dfs.columns[-1].index('epoch_')+6):]

    full_save_accuracy = full_save_dfs['test_acc_epoch_'+str(final_epoch_save)][0]*100
    full_fresh_accuracy = full_fresh_dfs['test_acc_epoch_'+str(final_epoch_fresh)][0]*100
    full_save_loss = full_save_dfs['train_loss_epoch_'+str(final_epoch_save)][0]
    full_fresh_loss = full_fresh_dfs['train_loss_epoch_'+str(final_epoch_fresh)][0]

    macs, params = get_model_complexity_info(new_network, (3,32,32), as_strings=False,print_per_layer_stat=False, verbose=True)

    save_parameter_size_list = [full_save_accuracy,full_save_loss,int(macs)/1000000000,2*int(macs)/1000000000,int(params)/1000000]
    fresh_parameter_size_list = [full_fresh_accuracy,full_fresh_loss,int(macs)/1000000000,2*int(macs)/1000000000,int(params)/1000000]
    parameter_data.loc[len(parameter_data)] = save_parameter_size_list
    parameter_data.loc[len(parameter_data)] = fresh_parameter_size_list
    parameter_data.to_excel(output_path_string_full_train+'\\'+'adapted_parameters.xlsx')

    return True

if __name__ == '__main__':

    parser = ArgumentParser(description=__doc__)
    args(parser)
    args = parser.parse_args()
    train_loader,test_loader,device,optimizer,scheduler,output_path,starting_conv_sizes = initialize(args)
    output_path = output_path / f"conv_{GLOBALS.CONFIG['init_conv_setting']}_thresh={GLOBALS.CONFIG['adapt_rank_threshold']}_beta={GLOBALS.CONFIG['beta']}_epochpert={GLOBALS.CONFIG['epochs_per_trial']}_adaptnum={GLOBALS.CONFIG['adapt_trials']}"
    GLOBALS.OUTPUT_PATH_STRING = str(output_path)

    if not os.path.exists(GLOBALS.OUTPUT_PATH_STRING):
        os.mkdir(GLOBALS.OUTPUT_PATH_STRING)

    print('~~~Initialization Complete. Beginning first training~~~')

    epochs = range(0, GLOBALS.CONFIG['epochs_per_trial'])

    conv_data = pd.DataFrame(columns=['superblock1','superblock2','superblock3','superblock4','superblock5'])
    #conv_data = pd.Series(index=['superblock1','superblock2','superblock3','superblock4','superblock5'])

    conv_size_list=[GLOBALS.super1_idx,GLOBALS.super2_idx,GLOBALS.super3_idx,GLOBALS.super4_idx,GLOBALS.super5_idx]
    conv_data.loc[0] = conv_size_list

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
    '''
    run_epochs(0, epochs, train_loader, test_loader,
                           device, optimizer, scheduler, output_path_train)

    print('~~~First run_epochs done.~~~')

    for i in range(1,GLOBALS.CONFIG['adapt_trials']):

        input_ranks, output_ranks = get_max_ranks_by_layer(path=GLOBALS.EXCEL_PATH)
        shortcut_indexes=[7,14,21,28]

        index_conv_size_list=GLOBALS.index
        output_sizes=calculate_correct_output_sizes(input_ranks,output_ranks,conv_size_list,shortcut_indexes,GLOBALS.CONFIG['adapt_rank_threshold'])
        conv_size_list=copy.deepcopy(output_sizes)
        conv_data.loc[i] = output_sizes

        print('~~~Starting Conv Adjustments~~~')
        new_network=update_network(output_sizes)
        optimizer,scheduler=network_initialize(new_network)

        print('~~~Training with new model~~~')

        epochs = range(0, GLOBALS.CONFIG['epochs_per_trial'])
        run_epochs(i, epochs, train_loader, test_loader,
                               device, optimizer, scheduler, output_path_train)


    for param_tensor in GLOBALS.NET.state_dict():
        val=param_tensor.find('bn')
        if val==-1:
            continue
        print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor].size(), 'OLD NETWORK')
        print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor], 'OLD NETWORK')
        break;

    #parameter_data.to_excel(output_path_string_trials+'\\'+'adapted_parameters.xlsx')
    conv_data.to_excel(output_path_string_trials+'\\'+'adapted_architectures.xlsx')
    create_graphs(GLOBALS.EXCEL_PATH,output_path_string_trials+'\\'+'adapted_architectures.xlsx',output_path_string_graph_files)
    torch.save(GLOBALS.NET.state_dict(), output_path_string_modelweights+'\\'+'model_state_dict')

    print('done')
    '---------------------------------------------------------------------------- LAST TRIAL FULL TRAIN ----------------------------------------------------------------------------------'
    GLOBALS.CONFIG['beta'] = 0.95
    GLOBALS.FULL_TRAIN = True
    GLOBALS.FULL_TRAIN_MODE = 'last_trial'
    GLOBALS.PERFORMANCE_STATISTICS = {}
    new_network=update_network(output_sizes)
    new_model_state_dict = prototype(GLOBALS.NET.state_dict(),output_sizes)
    #new_model_state_dict = prototype(torch.load('model_weights'+'\\'+'model_state_dict_32,32,32,32,32_thresh=0.3'),output_sizes)
    new_network=AdaptiveNet(num_classes=10, new_output_sizes=output_sizes)
    new_network.load_state_dict(new_model_state_dict)

    optimizer,scheduler=network_initialize(new_network)

    print('Using Early stopping of thresh 0.001')
    GLOBALS.EARLY_STOP = EarlyStop(
            patience=int(GLOBALS.CONFIG['early_stop_patience']),
            threshold=0.001)

    for param_tensor in GLOBALS.NET.state_dict():
        val=param_tensor.find('bn')
        if val==-1:
            continue
        print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor].size(), 'OLD NETWORK FULL TRAIN')
        print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor], 'OLD NETWORK FULL TRAIN')
        break;

    epochs = range(0,250)
    run_epochs(0, epochs, train_loader, test_loader,device, optimizer, scheduler, output_path_fulltrain)
    '''
    '--------------------------------------------------------------------------- FRESH NETWORK FULL TRAIN ----------------------------------------------------------------------------------'
    GLOBALS.PERFORMANCE_STATISTICS = {}
    GLOBALS.FULL_TRAIN = True
    GLOBALS.FULL_TRAIN_MODE = 'fresh'
    GLOBALS.EXCEL_PATH = ''
    GLOBALS.CONFIG['beta'] = 0.95


    #torch.save(GLOBALS.NET.state_dict(), 'model_weights/'+'model_state_dict_'+GLOBALS.CONFIG['init_conv_setting']+'_thresh='+str(GLOBALS.CONFIG['adapt_rank_threshold']))
    #new_model_state_dict = prototype(GLOBALS.NET.state_dict(),output_sizes)
    new_network=AdaptiveNet(num_classes=10,new_output_sizes=None)
    #new_network.load_state_dict(GLOBALS.NET.state_dict())

    optimizer,scheduler=network_initialize(new_network)

    print('Using Early stopping of thresh 0.001')
    GLOBALS.EARLY_STOP = EarlyStop(
            patience=int(GLOBALS.CONFIG['early_stop_patience']),
            threshold=0.001)

    for param_tensor in GLOBALS.NET.state_dict():
        val=param_tensor.find('bn')
        if val==-1:
            continue
        print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor].size(), 'FRESH')
        print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor], 'FRESH')
        break;

    epochs = range(0,250)

    run_epochs(0, epochs, train_loader, test_loader, device, optimizer, scheduler, output_path_fulltrain)

    '----------------------------------------------------------------------------===========================----------------------------------------------------------------------------------'
    #parameter count for fresh, full train
    #Parameters, macs, flops, accuracy, training loss
    create_data_file(GLOBALS.NET,output_path_string_full_train+'\\'+f"AdaS_last_iter_fulltrain_trial=0_net={GLOBALS.CONFIG['network']}_dataset={GLOBALS.CONFIG['dataset']}.xlsx",
                                 output_path_string_full_train+'\\'+f"AdaS_fresh_fulltrain_trial=0_net={GLOBALS.CONFIG['network']}_dataset={GLOBALS.CONFIG['dataset']}.xlsx",
                                 output_path_string_full_train)
    print('Done')
