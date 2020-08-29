from typing import Tuple
from argparse import Namespace as APNamespace, _SubParsersAction,ArgumentParser
from pathlib import Path
from early_stop import EarlyStop
import os
import platform
import time
import copy
import pandas as pd
import numpy as np
import global_vars as GLOBALS
from profiler import Profiler
from AdaS import AdaS
from test import test_main
from optim.sls import SLS
from optim.sps import SPS
from optim import get_optimizer_scheduler
from early_stop import EarlyStop
import sys
from adaptive_graph import create_adaptive_graphs,create_plot,adapted_info_graph,trial_info_graph, stacked_bar_plot
from ptflops import get_model_complexity_info
from models.own_network import DASNet34,DASNet50
import copy
import torch
import torch.backends.cudnn as cudnn
from scaling_algorithms import *
import ast
import matplotlib.pyplot as plt
from utils import parse_config
from metrics import Metrics
from models import get_net
from data import get_data
import yaml
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

def initialize(args: APNamespace, new_network, beta=None, new_threshold=None, new_threshold_kernel=None, scheduler=None, trial=-1):
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

    if scheduler != None:
        GLOBALS.CONFIG['lr_scheduler']='StepLR'
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

    '''if GLOBALS.CONFIG['blocks_per_superblock']==2:
        GLOBALS.super1_idx = [64,64,64,64,64]
        GLOBALS.super2_idx = [64,64,64,64]
        GLOBALS.super3_idx = [64,64,64,64]
        GLOBALS.super4_idx = [64,64,64,64]
    else:
        GLOBALS.super1_idx = [64,64,64,64,64,64,64]
        GLOBALS.super2_idx = [64,64,64,64,64,64]
        GLOBALS.super3_idx = [64,64,64,64,64,64]
        GLOBALS.super4_idx = [64,64,64,64,64,64]'''

    GLOBALS.index_used = GLOBALS.super1_idx + GLOBALS.super2_idx + GLOBALS.super3_idx + GLOBALS.super4_idx

    if GLOBALS.FIRST_INIT == True:
        print('FIRST_INIT==True, GETTING NET FROM CONFIG')
        GLOBALS.NET = get_net(
                    GLOBALS.CONFIG['network'], num_classes=10 if
                    GLOBALS.CONFIG['dataset'] == 'CIFAR10' else 100 if
                    GLOBALS.CONFIG['dataset'] == 'CIFAR100'
                    else 1000, init_adapt_conv_size=init_conv)
        GLOBALS.FIRST_INIT = False
    else:
        print('GLOBALS.FIRST_INIT IS FALSE LOADING IN NETWORK FROM UPDATE (Fresh weights)')
        GLOBALS.NET = new_network
        GLOBALS.NET_RAW = new_network.cuda()


    GLOBALS.METRICS = Metrics(list(GLOBALS.NET.parameters()),p=GLOBALS.CONFIG['p'])
    GLOBALS.NET = GLOBALS.NET.to(device)
    GLOBALS.CRITERION = get_loss(GLOBALS.CONFIG['loss'])

    if beta != None:
        GLOBALS.CONFIG['beta']=beta

    if new_threshold != None:
        GLOBALS.CONFIG['delta_threshold']=new_threshold

    if new_threshold_kernel != None:
        GLOBALS.CONFIG['delta_threshold_kernel']=new_threshold_kernel

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

def get_max_ranks_by_layer(path=GLOBALS.EXCEL_PATH):
    '''
    -returns 2 36-lists of the maximum value of the input and output ranks from the datafile produced after one adapting trial
    '''
    sheet = pd.read_excel(path,index_col=0)
    out_rank_col = [col for col in sheet if col.startswith('out_rank')]
    in_rank_col = [col for col in sheet if col.startswith('in_rank')]

    out_ranks = sheet[out_rank_col]
    in_ranks = sheet[in_rank_col]

    out_max_ranks = out_ranks.max(axis=1)
    in_max_ranks = in_ranks.max(axis=1)

    out_max_ranks = out_max_ranks.tolist()
    in_max_ranks = in_max_ranks.tolist()

    return in_max_ranks,out_max_ranks

def new_output_sizes(current_conv_sizes,ranks,threshold):
    scaling_factor=np.subtract(ranks,threshold)
    new_conv_sizes = np.multiply(current_conv_sizes,np.add(1,scaling_factor))
    new_conv_sizes = [int(i) for i in new_conv_sizes]
    print(scaling_factor,'Scaling Factor')
    print(current_conv_sizes, 'CURRENT CONV SIZES')
    print(new_conv_sizes,'NEW CONV SIZES')
    return new_conv_sizes

def nearest_upper_odd(list_squared_kernel_size):
    for superblock in range(len(list_squared_kernel_size)):
        list_squared_kernel_size[superblock] = (np.ceil(np.sqrt(list_squared_kernel_size[superblock])) // 2 * 2 + 1).tolist()
    return list_squared_kernel_size

def update_network(new_channel_sizes,new_kernel_sizes):
    if GLOBALS.CONFIG['network']=='DASNet34':
        new_network=DASNet34(num_classes=10,new_output_sizes=new_channel_sizes,new_kernel_sizes=new_kernel_sizes)
    elif GLOBALS.CONFIG['network']=='DASNet50':
        new_network=DASNet50(num_classes=10,new_output_sizes=new_channel_sizes,new_kernel_sizes=new_kernel_sizes)
    return new_network

def create_full_data_file(new_network,full_save_file,full_fresh_file,output_path_string_full_train):
    parameter_data = pd.DataFrame(columns=['Accuracy (%)','Training Loss','GMacs','GFlops','Parameter Count (M)'])

    #full_save_dfs=pd.read_excel(full_save_file)
    full_fresh_dfs=pd.read_excel(full_fresh_file)

    #final_epoch_save=full_save_dfs.columns[-1][(full_save_dfs.columns[-1].index('epoch_')+6):]
    final_epoch_fresh=full_fresh_dfs.columns[-1][(full_fresh_dfs.columns[-1].index('epoch_')+6):]

    #full_save_accuracy = full_save_dfs['test_acc_epoch_'+str(final_epoch_save)][0]*100
    full_fresh_accuracy = full_fresh_dfs['test_acc_epoch_'+str(final_epoch_fresh)][0]*100
    #full_save_loss = full_save_dfs['train_loss_epoch_'+str(final_epoch_save)][0]
    full_fresh_loss = full_fresh_dfs['train_loss_epoch_'+str(final_epoch_fresh)][0]

    macs, params = get_model_complexity_info(new_network, (3,32,32), as_strings=False,print_per_layer_stat=False, verbose=True)

    #save_parameter_size_list = [full_save_accuracy,full_save_loss,int(macs)/1000000000,2*int(macs)/1000000000,int(params)/1000000]
    fresh_parameter_size_list = [full_fresh_accuracy,full_fresh_loss,int(macs)/1000000000,2*int(macs)/1000000000,int(params)/1000000]
    #parameter_data.loc[len(parameter_data)] = save_parameter_size_list
    parameter_data.loc[len(parameter_data)] = fresh_parameter_size_list
    if platform.system() == 'Windows':
        parameter_data.to_excel(output_path_string_full_train+'\\'+'adapted_parameters.xlsx')
    else:
         parameter_data.to_excel(output_path_string_full_train+'/'+'adapted_parameters.xlsx')

    return True

def run_fresh_full_train(train_loader,test_loader,device,output_sizes,kernel_sizes,epochs,output_path_fulltrain):

    if GLOBALS.CONFIG['network']=='DASNet34':
        new_network=DASNet34(num_classes=10,new_output_sizes=output_sizes,new_kernel_sizes=kernel_sizes)
    elif GLOBALS.CONFIG['network']=='DASNet50':
        new_network=DASNet50(num_classes=10,new_output_sizes=output_sizes,new_kernel_sizes=kernel_sizes)

    GLOBALS.FIRST_INIT = False

    #optimizer,scheduler=network_initialize(new_network,train_loader)
    parser = ArgumentParser(description=__doc__)
    args(parser)
    args_true = parser.parse_args()
    train_loader,test_loader,device,optimizer,scheduler,output_path,starting_conv_sizes = initialize(args_true,new_network,beta=GLOBALS.CONFIG['beta_full'],scheduler='StepLR')

    GLOBALS.FULL_TRAIN = True
    GLOBALS.PERFORMANCE_STATISTICS = {}
    GLOBALS.FULL_TRAIN_MODE = 'fresh'
    GLOBALS.EXCEL_PATH = ''

    for param_tensor in GLOBALS.NET.state_dict():
        val=param_tensor.find('bn')
        if val==-1:
            continue
        print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor].size(), 'FRESH')
        #print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor], 'FRESH')
        break;

    run_epochs(0, epochs, train_loader, test_loader, device, optimizer, scheduler, output_path_fulltrain)
    return True

def create_graphs(trial_info_file_name,adapted_kernel_file_name,adapted_conv_file_name,rank_final_file_name,rank_stable_file_name,out_folder):
    if platform.system == "Windows":
        slash = '\\'
    else:
        slash = '/'
    create_adaptive_graphs(trial_info_file_name,GLOBALS.CONFIG['epochs_per_trial'],GLOBALS.CONFIG['adapt_trials'],out_folder)
    kernel_path=out_folder+slash+'dynamic_kernel_Size_Plot.png'
    conv_path=out_folder+slash+'dynamic_layer_Size_Plot.png'
    rank_final_path=out_folder+slash+'dynamic_rank_final.png'
    rank_stable_path=out_folder+slash+'dynamic_rank_stable.png'
    output_condition_path=out_folder+slash+'dynamic_output_condition.png'
    input_condition_path=out_folder+slash+'dynamic_input_condition.png'
    network_visualize_path=out_folder+slash+'dynamic_network_Size_Plot.png'
    '''create_layer_plot(conv_data_file_name,GLOBALS.CONFIG['adapt_trials'],conv_path, 'Layer Size')
    #create_layer_plot(rank_final_file_name,GLOBALS.CONFIG['adapt_trials'],rank_final_path, 'Final Rank')
    #create_layer_plot(rank_stable_file_name,GLOBALS.CONFIG['adapt_trials'],rank_stable_path, 'Stable Rank')'''

    last_epoch=GLOBALS.CONFIG['epochs_per_trial']-1
    stable_epoch=GLOBALS.CONFIG['stable_epoch']

    shortcut_indexes=[]
    old_conv_size_list=[GLOBALS.super1_idx,GLOBALS.super2_idx,GLOBALS.super3_idx,GLOBALS.super4_idx]
    counter=-1
    for j in old_conv_size_list:
        if len(shortcut_indexes)==len(old_conv_size_list)-1:
            break
        counter+=len(j) + 1
        shortcut_indexes+=[counter]
    plt.clf()
    stacked_bar_plot(adapted_conv_file_name, network_visualize_path)
    if GLOBALS.CONFIG['kernel_adapt']!=0:
        plt.clf()
        adapted_info_graph(adapted_kernel_file_name,GLOBALS.CONFIG['adapt_trials_kernel'],kernel_path,'Kernel Size',last_epoch)
    plt.clf()
    adapted_info_graph(adapted_conv_file_name,GLOBALS.CONFIG['adapt_trials'],conv_path,'Layer Size',last_epoch)
    plt.clf()
    trial_info_graph(trial_info_file_name, GLOBALS.total_trials, len(GLOBALS.index_used)+3, rank_final_path,'Final Rank', 'out_rank_epoch_',shortcut_indexes,last_epoch)
    plt.clf()
    trial_info_graph(trial_info_file_name, GLOBALS.total_trials, len(GLOBALS.index_used)+3, rank_stable_path,'Stable Rank', 'out_rank_epoch_',shortcut_indexes,stable_epoch)
    plt.clf()
    trial_info_graph(trial_info_file_name, GLOBALS.total_trials, len(GLOBALS.index_used)+3, output_condition_path,'Output Condition', 'out_condition_epoch_',shortcut_indexes,last_epoch)
    plt.clf()
    trial_info_graph(trial_info_file_name, GLOBALS.total_trials, len(GLOBALS.index_used)+3, input_condition_path,'Input Condition', 'in_condition_epoch_',shortcut_indexes,last_epoch)
    plt.clf()
    return True

def run_trials(train_loader,test_loader,device,optimizer,scheduler,epochs,output_path_train, new_threshold=None):
    last_operation,factor_scale,delta_percentage,last_operation_kernel,factor_scale_kernel,delta_percentage_kernel=[],[],[],[],[],[]
    parameter_type=GLOBALS.CONFIG['parameter_type']

    kernel_begin_trial=0
    def check_last_operation(last_operation,last_operation_kernel,kernel_begin_trial):
        all_channels_stopped=True
        for blah in last_operation:
            for inner in blah:
                if inner!=0:
                    all_channels_stopped=False
        all_kernels_stopped=True
        #if kernel_begin_trial!=0:
        for blah in last_operation_kernel:
            for inner in blah:
                if inner!=0:
                    all_kernels_stopped=False
        return all_channels_stopped,all_kernels_stopped
    def get_shortcut_indexes(conv_size_list):
        shortcut_indexes=[]
        counter=-1
        for j in conv_size_list:
            if len(shortcut_indexes)==len(conv_size_list)-1:
                break
            counter+=len(j) + 1
            shortcut_indexes+=[counter]
        return shortcut_indexes
    def initialize_dataframes_and_lists():
        conv_data = pd.DataFrame(columns=['superblock1','superblock2','superblock3','superblock4'])
        kernel_data = pd.DataFrame(columns=['superblock1','superblock2','superblock3','superblock4'])
        rank_final_data = pd.DataFrame(columns=['superblock1','superblock2','superblock3','superblock4'])
        rank_stable_data = pd.DataFrame(columns=['superblock1','superblock2','superblock3','superblock4'])
        conv_size_list=[GLOBALS.super1_idx,GLOBALS.super2_idx,GLOBALS.super3_idx,GLOBALS.super4_idx]
        kernel_size_list=[GLOBALS.super1_kernel_idx,GLOBALS.super2_kernel_idx,GLOBALS.super3_kernel_idx,GLOBALS.super4_kernel_idx]

        conv_data.loc[0] = conv_size_list
        kernel_data.loc[0] = kernel_size_list
        delta_info = pd.DataFrame(columns=['delta_percentage','factor_scale','last_operation'])
        delta_info_kernel = pd.DataFrame(columns=['delta_percentage_kernel','factor_scale_kernel','last_operation_kernel'])
        return conv_data,kernel_data,rank_final_data,rank_stable_data,delta_info,delta_info_kernel,conv_size_list,kernel_size_list
    def should_break(i,all_channels_stopped,all_kernels_stopped,kernel_begin_trial,parameter_type):
        break_loop=False
        if (all_channels_stopped==True and kernel_begin_trial==0) or i==GLOBALS.CONFIG['adapt_trials']:
            GLOBALS.CONFIG['adapt_trials']=i
            parameter_type='kernel'
            kernel_begin_trial=i
            if GLOBALS.CONFIG['adapt_trials_kernel']==0 or GLOBALS.CONFIG['kernel_adapt']==0:
                print('ACTIVATED IF STATEMENT 1 FOR SOME STUPID REASON')
                break_loop=True

        if all_kernels_stopped==True or i==kernel_begin_trial+GLOBALS.CONFIG['adapt_trials_kernel']:# and kernel_begin_trial!=0:
            print('ACTIVATED IF STATEMENT 2 FOR SOME EVEN STUPIDER REASON')
            GLOBALS.total_trials=i
            break_loop=True
        return kernel_begin_trial,parameter_type,break_loop

    #####################################################################################################################################
    conv_data,kernel_data,rank_final_data,rank_stable_data,delta_info,delta_info_kernel,conv_size_list,kernel_size_list=initialize_dataframes_and_lists()
    shortcut_indexes=get_shortcut_indexes(conv_size_list)
    run_epochs(0, epochs, train_loader, test_loader,device, optimizer, scheduler, output_path_train)
    print('~~~First run_epochs done.~~~')

    if (GLOBALS.CONFIG['kernel_adapt']==0):
        GLOBALS.CONFIG['adapt_trials_kernel']=0

    GLOBALS.total_trials=GLOBALS.CONFIG['adapt_trials']+GLOBALS.CONFIG['adapt_trials_kernel']
    for i in range(1,GLOBALS.total_trials):
        if (GLOBALS.CONFIG['kernel_adapt']==0):
            GLOBALS.CONFIG['adapt_trials_kernel']=0
        if kernel_begin_trial!=0:
            if (i > (GLOBALS.total_trials//2 - kernel_begin_trial)) and all_channels_stopped==True:
                GLOBALS.min_kernel_size_1=GLOBALS.CONFIG['min_kernel_size']
                GLOBALS.CONFIG['min_kernel_size']=GLOBALS.CONFIG['min_kernel_size_2']
        '------------------------------------------------------------------------------------------------------------------------------------------------'
        last_operation, last_operation_kernel, factor_scale, factor_scale_kernel, new_channel_sizes, new_kernel_sizes, delta_percentage, delta_percentage_kernel, rank_averages_final, rank_averages_stable = delta_scaling(conv_size_list,kernel_size_list,shortcut_indexes,last_operation, factor_scale, delta_percentage, last_operation_kernel, factor_scale_kernel, delta_percentage_kernel,parameter_type=parameter_type)
        '------------------------------------------------------------------------------------------------------------------------------------------------'
        print(last_operation_kernel, 'LAST OPERATION KERNEL FOR TRIAL '+str(i))
        all_channels_stopped, all_kernels_stopped = check_last_operation(last_operation,last_operation_kernel,kernel_begin_trial)
        print(all_channels_stopped,all_kernels_stopped, 'BREAK VALUES!')
        kernel_begin_trial,parameter_type,break_loop = should_break(i,all_channels_stopped,all_kernels_stopped,kernel_begin_trial,parameter_type)
        if break_loop==True: break

        last_operation_copy, factor_scale_copy, delta_percentage_copy, rank_averages_final_copy, rank_averages_stable_copy = copy.deepcopy(last_operation),copy.deepcopy(factor_scale),copy.deepcopy(delta_percentage),copy.deepcopy(rank_averages_final),copy.deepcopy(rank_averages_stable)
        last_operation_kernel_copy, factor_scale_kernel_copy, delta_percentage_kernel_copy = copy.deepcopy(last_operation_kernel), copy.deepcopy(factor_scale_kernel), copy.deepcopy(delta_percentage_kernel)
        conv_size_list=copy.deepcopy(new_channel_sizes)
        kernel_size_list=copy.deepcopy(new_kernel_sizes)

        print('~~~Writing to Dataframe~~~')
        if parameter_type=='channel' or parameter_type=='both' or GLOBALS.CONFIG['kernel_adapt']==0:
            conv_data.loc[i] = new_channel_sizes
            delta_info.loc[i] = [delta_percentage_copy,factor_scale_copy,last_operation_copy]
        elif parameter_type=='kernel' or parameter_type=='both':
            kernel_data.loc[i-kernel_begin_trial] = new_kernel_sizes
            delta_info_kernel.loc[i-kernel_begin_trial] = [delta_percentage_kernel_copy,factor_scale_kernel_copy,last_operation_kernel_copy]
        rank_final_data.loc[i] = rank_averages_final_copy
        rank_stable_data.loc[i] = rank_averages_stable_copy

        print('~~~Starting Conv parameter_typements~~~')
        new_network=update_network(new_channel_sizes,new_kernel_sizes)

        print('~~~Initializing the new model~~~')
        parser = ArgumentParser(description=__doc__)
        args(parser)
        args_true = parser.parse_args()
        train_loader,test_loader,device,optimizer,scheduler,output_path,starting_conv_sizes = initialize(args_true,new_network,new_threshold=new_threshold)
        epochs = range(0, GLOBALS.CONFIG['epochs_per_trial'])

        print('~~~Training with new model~~~')
        run_epochs(i, epochs, train_loader, test_loader,device, optimizer, scheduler, output_path_train)

    return kernel_data,conv_data,rank_final_data,rank_stable_data,new_channel_sizes,new_kernel_sizes,delta_info, delta_info_kernel

def create_trial_data_file(kernel_data,conv_data,delta_info_kernel,delta_info,rank_final_data,rank_stable_data,output_path_string_trials,output_path_string_graph_files,output_path_string_modelweights):
    #parameter_data.to_excel(output_path_string_trials+'\\'+'adapted_parameters.xlsx')
    if platform.system == 'Windows':
        slash = '\\'
    else:
        slash = '/'
    delta_info_kernel.to_excel(output_path_string_trials+slash+'adapted_delta_info_kernel.xlsx')
    delta_info.to_excel(output_path_string_trials+slash+'adapted_delta_info.xlsx')
    kernel_data.to_excel(output_path_string_trials+slash+'adapted_kernels.xlsx')
    conv_data.to_excel(output_path_string_trials+slash+'adapted_architectures.xlsx')
    rank_final_data.to_excel(output_path_string_trials+slash+'adapted_rank_final.xlsx')
    rank_stable_data.to_excel(output_path_string_trials+slash+'adapted_rank_stable.xlsx')
    create_graphs(GLOBALS.EXCEL_PATH,output_path_string_trials+slash+'adapted_kernels.xlsx',output_path_string_trials+slash+'adapted_architectures.xlsx',output_path_string_trials+slash+'adapted_rank_final.xlsx',output_path_string_trials+slash+'adapted_rank_stable.xlsx',output_path_string_graph_files)
    #torch.save(GLOBALS.NET.state_dict(), output_path_string_modelweights+'\\'+'model_state_dict')

def get_output_sizes(file_name):
    outputs=pd.read_excel(file_name)
    output_sizes=outputs.iloc[-1,1:]
    output_sizes=output_sizes.tolist()
    output_sizes_true=[ast.literal_eval(i) for i in output_sizes]
    print(output_sizes_true,'Output sizes frome excel')
    return output_sizes_true


def run_epochs(trial, epochs, train_loader, test_loader,
               device, optimizer, scheduler, output_path):
    if platform.system == 'Windows':
        slash = '\\'
    else:
        slash = '/'
    print('------------------------------' + slash)
    if GLOBALS.CONFIG['lr_scheduler'] == 'AdaS':
        if GLOBALS.FULL_TRAIN == False:
            xlsx_name = \
                slash + f"AdaS_adapt_trial={trial}_" +\
                f"net={GLOBALS.CONFIG['network']}_" +\
                f"{GLOBALS.CONFIG['init_lr']}_dataset=" +\
                f"{GLOBALS.CONFIG['dataset']}.xlsx"
        else:
            if GLOBALS.FULL_TRAIN_MODE == 'last_trial':
                xlsx_name = \
                    slash + f"AdaS_last_iter_fulltrain_trial={trial}_" +\
                    f"net={GLOBALS.CONFIG['network']}_" +\
                    f"dataset=" +\
                    f"{GLOBALS.CONFIG['dataset']}.xlsx"
            elif GLOBALS.FULL_TRAIN_MODE == 'fresh':
                xlsx_name = \
                    slash + f"AdaS_fresh_fulltrain_trial={trial}_" +\
                    f"net={GLOBALS.CONFIG['network']}_" +\
                    f"beta={GLOBALS.CONFIG['beta']}_" +\
                    f"dataset=" +\
                    f"{GLOBALS.CONFIG['dataset']}.xlsx"
            else:
                print('ERROR: INVALID FULL_TRAIN_MODE | Check that the correct full_train_mode strings have been initialized in main file | Should be either fresh, or last_trial')
                sys.exit()
    else:
        if GLOBALS.FULL_TRAIN == False:
            xlsx_name = \
                slash + f"StepLR_adapt_trial={trial}_" +\
                f"net={GLOBALS.CONFIG['network']}_" +\
                f"{GLOBALS.CONFIG['init_lr']}_dataset=" +\
                f"{GLOBALS.CONFIG['dataset']}.xlsx"
        else:
            if GLOBALS.FULL_TRAIN_MODE == 'last_trial':
                xlsx_name = \
                    slash + f"StepLR_last_iter_fulltrain_trial={trial}_" +\
                    f"net={GLOBALS.CONFIG['network']}_" +\
                    f"dataset=" +\
                    f"{GLOBALS.CONFIG['dataset']}.xlsx"
            elif GLOBALS.FULL_TRAIN_MODE == 'fresh':
                xlsx_name = \
                    slash + f"StepLR_fresh_fulltrain_trial={trial}_" +\
                    f"net={GLOBALS.CONFIG['network']}_" +\
                    f"dataset=" +\
                    f"{GLOBALS.CONFIG['dataset']}.xlsx"
            else:
                print('ERROR: INVALID FULL_TRAIN_MODE | Check that the correct full_train_mode strings have been initialized in main file | Should be either fresh, or last_trial')
                sys.exit()
    if platform.system == 'Windows':
        slash = '\\'
    else:
        slash = '/'
    xlsx_path = str(output_path) +slash+ xlsx_name

    if GLOBALS.FULL_TRAIN == False:
        filename = \
            slash + f"stats_net={GLOBALS.CONFIG['network']}_AdaS_trial={trial}_" +\
            f"beta={GLOBALS.CONFIG['beta']}_initlr={GLOBALS.CONFIG['init_lr']}_" +\
            f"dataset={GLOBALS.CONFIG['dataset']}.csv"
    else:
        if GLOBALS.FULL_TRAIN_MODE == 'last_trial':
            filename = \
                slash + f"stats_last_iter_net={GLOBALS.CONFIG['network']}_StepLR_trial={trial}_" +\
                f"beta={GLOBALS.CONFIG['beta']}_" +\
                f"dataset={GLOBALS.CONFIG['dataset']}.csv"
        elif GLOBALS.FULL_TRAIN_MODE == 'fresh':
            filename = \
                slash + f"stats_fresh_net={GLOBALS.CONFIG['network']}_StepLR_trial={trial}_" +\
                f"beta={GLOBALS.CONFIG['beta']}_" +\
                f"dataset={GLOBALS.CONFIG['dataset']}.csv"
    Profiler.filename = output_path / filename
    GLOBALS.EXCEL_PATH = xlsx_path
    print(GLOBALS.EXCEL_PATH,'SET GLOBALS EXCEL PATH')

    for epoch in epochs:
        start_time = time.time()
        # print(f"AdaS: Epoch {epoch}/{epochs[-1]} Started.")
        train_loss, train_accuracy, test_loss, test_accuracy = \
            epoch_iteration(trial,train_loader, test_loader,epoch, device, optimizer, scheduler)

        end_time = time.time()

        if GLOBALS.CONFIG['lr_scheduler'] == 'StepLR':
            scheduler.step()
        total_time = time.time()
        print(
            f"AdaS: Trial {trial}/{GLOBALS.total_trials - 1} | " +
            f"Epoch {epoch}/{epochs[-1]} Ended | " +
            "Total Time: {:.3f}s | ".format(total_time - start_time) +
            "Epoch Time: {:.3f}s | ".format(end_time - start_time) +
            "~Time Left: {:.3f}s | ".format(
                (total_time - start_time) * (epochs[-1] - epoch)),
            "Train Loss: {:.4f}% | Train Acc. {:.4f}% | ".format(
                train_loss,
                train_accuracy) +
            "Test Loss: {:.4f}% | Test Acc. {:.4f}%".format(test_loss,
                                                            test_accuracy))
        df = pd.DataFrame(data=GLOBALS.PERFORMANCE_STATISTICS)

        df.to_excel(xlsx_path)
        if GLOBALS.EARLY_STOP(train_loss):
            print("AdaS: Early stop activated.")
            break

#@Profiler
def epoch_iteration(trial, train_loader, test_loader, epoch: int,
                    device, optimizer,scheduler) -> Tuple[float, float]:
    # logging.info(f"Adas: Train: Epoch: {epoch}")
    # global net, performance_statistics, metrics, adas, config
    GLOBALS.NET.train()
    train_loss = 0
    correct = 0
    total = 0
    """train CNN architecture"""
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if GLOBALS.CONFIG['lr_scheduler'] == 'CosineAnnealingWarmRestarts':
            scheduler.step(epoch + batch_idx / len(train_loader))
        optimizer.zero_grad()
        # if GLOBALS.CONFIG['optim_method'] == 'SLS':
        if isinstance(optimizer, SLS):
            def closure():
                outputs = GLOBALS.NET(inputs)
                loss = GLOBALS.CRITERION(outputs, targets)
                return loss, outputs
            loss, outputs = optimizer.step(closure=closure)
        else:
            outputs = GLOBALS.NET(inputs)
            loss = GLOBALS.CRITERION(outputs, targets)
            loss.backward()
            # if GLOBALS.ADAS is not None:
            #     optimizer.step(GLOBALS.METRICS.layers_index_todo,
            #                    GLOBALS.ADAS.lr_vector)
            if isinstance(scheduler, AdaS):
                optimizer.step(GLOBALS.METRICS.layers_index_todo,
                               scheduler.lr_vector)
            # elif GLOBALS.CONFIG['optim_method'] == 'SPS':
            elif isinstance(optimizer, SPS):
                optimizer.step(loss=loss)
            else:
                optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        GLOBALS.TRAIN_LOSS = train_loss
        GLOBALS.TRAIN_CORRECT = correct
        GLOBALS.TRAIN_TOTAL = total

        if GLOBALS.CONFIG['lr_scheduler'] == 'OneCycleLR':
            scheduler.step()
        #Update optimizer
        GLOBALS.OPTIMIZER = optimizer

        # progress_bar(batch_idx, len(train_loader),
        #              'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1),
        #                  100. * correct / total, correct, total))
    GLOBALS.PERFORMANCE_STATISTICS[f'train_acc_epoch_{epoch}'] = \
        float(correct / total)
    GLOBALS.PERFORMANCE_STATISTICS[f'train_loss_epoch_{epoch}'] = \
        train_loss / (batch_idx + 1)

    io_metrics = GLOBALS.METRICS.evaluate(epoch)
    GLOBALS.PERFORMANCE_STATISTICS[f'in_S_epoch_{epoch}'] = \
        io_metrics.input_channel_S
    GLOBALS.PERFORMANCE_STATISTICS[f'out_S_epoch_{epoch}'] = \
        io_metrics.output_channel_S
    GLOBALS.PERFORMANCE_STATISTICS[f'mode12_S_epoch_{epoch}'] = \
        io_metrics.mode_12_channel_S
    GLOBALS.PERFORMANCE_STATISTICS[f'fc_S_epoch_{epoch}'] = \
        io_metrics.fc_S
    GLOBALS.PERFORMANCE_STATISTICS[f'in_rank_epoch_{epoch}'] = \
        io_metrics.input_channel_rank
    GLOBALS.PERFORMANCE_STATISTICS[f'out_rank_epoch_{epoch}'] = \
        io_metrics.output_channel_rank
    GLOBALS.PERFORMANCE_STATISTICS[f'mode12_rank_epoch_{epoch}'] = \
        io_metrics.mode_12_channel_rank
    GLOBALS.PERFORMANCE_STATISTICS[f'fc_rank_epoch_{epoch}'] = \
        io_metrics.fc_rank
    GLOBALS.PERFORMANCE_STATISTICS[f'in_condition_epoch_{epoch}'] = \
        io_metrics.input_channel_condition
    GLOBALS.PERFORMANCE_STATISTICS[f'out_condition_epoch_{epoch}'] = \
        io_metrics.output_channel_condition
    GLOBALS.PERFORMANCE_STATISTICS[f'mode12_condition_epoch_{epoch}'] = \
        io_metrics.mode_12_channel_condition
    # if GLOBALS.ADAS is not None:

    if isinstance(scheduler, AdaS):
        lrmetrics = scheduler.step(epoch, GLOBALS.METRICS)
        GLOBALS.PERFORMANCE_STATISTICS[f'rank_velocity_epoch_{epoch}'] = \
            lrmetrics.rank_velocity
        GLOBALS.PERFORMANCE_STATISTICS[f'learning_rate_epoch_{epoch}'] = \
            lrmetrics.r_conv
    else:
        # if GLOBALS.CONFIG['optim_method'] == 'SLS' or \
        #         GLOBALS.CONFIG['optim_method'] == 'SPS':
        if isinstance(optimizer, SLS) or isinstance(optimizer, SPS):
            GLOBALS.PERFORMANCE_STATISTICS[f'learning_rate_epoch_{epoch}'] = \
                optimizer.state['step_size']
        else:
            GLOBALS.PERFORMANCE_STATISTICS[f'learning_rate_epoch_{epoch}'] = \
                optimizer.param_groups[0]['lr']
    test_loss, test_accuracy = test_main(test_loader, epoch, device)

    return (train_loss / (batch_idx + 1), 100. * correct / total,
            test_loss, test_accuracy)
