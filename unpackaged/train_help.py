from typing import Tuple
from argparse import Namespace as APNamespace, _SubParsersAction,ArgumentParser
from pathlib import Path
from early_stop import EarlyStop
import os
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
from adaptive_channels import prototype
from adaptive_graph import create_adaptive_graphs,create_layer_plot
from ptflops import get_model_complexity_info
from models.own_network import AdaptiveNet
import copy
import torch
import torch.backends.cudnn as cudnn
from scaling_algorithms import *

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

def initialize(args: APNamespace, new_network):
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

    if GLOBALS.CONFIG['blocks_per_superblock']==2:
        GLOBALS.super1_idx = [64,64,64,64,64]
        GLOBALS.super2_idx = [64,64,64,64]
        GLOBALS.super3_idx = [64,64,64,64]
        GLOBALS.super4_idx = [64,64,64,64]
        GLOBALS.super5_idx = [64,64,64,64]
    else:
        GLOBALS.super1_idx = [64,64,64,64,64,64,64]
        GLOBALS.super2_idx = [64,64,64,64,64,64]
        GLOBALS.super3_idx = [64,64,64,64,64,64]
        GLOBALS.super4_idx = [64,64,64,64,64,64]
        GLOBALS.super5_idx = [64,64,64,64,64,64]

    GLOBALS.index_used = GLOBALS.super1_idx + GLOBALS.super2_idx + GLOBALS.super3_idx + GLOBALS.super4_idx + GLOBALS.super5_idx

    if GLOBALS.FIRST_INIT == True:
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

def update_network(output_sizes):
    #new_model_state_dict = prototype(GLOBALS.NET.state_dict(),output_sizes)
    new_network=AdaptiveNet(num_classes=10,new_output_sizes=output_sizes)
    #new_network.load_state_dict(new_model_state_dict)
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
    parameter_data.to_excel(output_path_string_full_train+'\\'+'adapted_parameters.xlsx')

    return True

def run_saved_weights_full_train(train_loader,test_loader,device,output_sizes,epochs,output_path_fulltrain):

    new_network=update_network(output_sizes)
    new_model_state_dict = prototype(GLOBALS.NET.state_dict(),output_sizes)
    #new_model_state_dict = prototype(torch.load('model_weights'+'\\'+'model_state_dict_32,32,32,32,32_thresh=0.3'),output_sizes)
    new_network=AdaptiveNet(num_classes=10, new_output_sizes=output_sizes)
    new_network.load_state_dict(new_model_state_dict)

    optimizer,scheduler=network_initialize(new_network,train_loader)

    print('Using Early stopping of thresh 0.001')
    GLOBALS.EARLY_STOP = EarlyStop(
            patience=int(GLOBALS.CONFIG['early_stop_patience']),
            threshold=0.001)
    GLOBALS.CONFIG['beta'] = 0.95
    GLOBALS.FULL_TRAIN = True
    GLOBALS.FULL_TRAIN_MODE = 'last_trial'
    GLOBALS.EXCEL_PATH = ''
    GLOBALS.PERFORMANCE_STATISTICS = {}

    for param_tensor in GLOBALS.NET.state_dict():
        val=param_tensor.find('bn')
        if val==-1:
            continue
        print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor].size(), 'OLD NETWORK FULL TRAIN')
        print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor], 'OLD NETWORK FULL TRAIN')
        break;

    run_epochs(0, epochs, train_loader, test_loader,device, optimizer, scheduler, output_path_fulltrain)

def run_fresh_full_train(train_loader,test_loader,device,output_sizes,epochs,output_path_fulltrain):
    #torch.save(GLOBALS.NET.state_dict(), 'model_weights/'+'model_state_dict_'+GLOBALS.CONFIG['init_conv_setting']+'_thresh='+str(GLOBALS.CONFIG['adapt_rank_threshold']))
    #new_model_state_dict = prototype(GLOBALS.NET.state_dict(),output_sizes)
    new_network=AdaptiveNet(num_classes=10,new_output_sizes=output_sizes)
    #new_network.load_state_dict(GLOBALS.NET.state_dict())

    #optimizer,scheduler=network_initialize(new_network,train_loader)
    parser = ArgumentParser(description=__doc__)
    args(parser)
    args_true = parser.parse_args()
    train_loader,test_loader,device,optimizer,scheduler,output_path,starting_conv_sizes = initialize(args_true,new_network)

    print('Using Early stopping of thresh 0.001')
    GLOBALS.EARLY_STOP = EarlyStop(patience=int(GLOBALS.CONFIG['early_stop_patience']),threshold=0.001)
    GLOBALS.FULL_TRAIN = True
    GLOBALS.PERFORMANCE_STATISTICS = {}
    GLOBALS.FULL_TRAIN_MODE = 'fresh'
    GLOBALS.EXCEL_PATH = ''
    GLOBALS.CONFIG['beta'] = 0.95

    for param_tensor in GLOBALS.NET.state_dict():
        val=param_tensor.find('bn')
        if val==-1:
            continue
        print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor].size(), 'FRESH')
        print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor], 'FRESH')
        break;
    print('~~~FRESH TRAINING, USING BETA: {} ~~~'.format(GLOBALS.CONFIG['beta']))
    run_epochs(0, epochs, train_loader, test_loader, device, optimizer, scheduler, output_path_fulltrain)
    return True

def create_graphs(accuracy_data_file_name,conv_data_file_name,rank_final_file_name,rank_stable_file_name,out_folder):
    create_adaptive_graphs(accuracy_data_file_name,GLOBALS.CONFIG['epochs_per_trial'],GLOBALS.CONFIG['adapt_trials'],out_folder)
    conv_path=out_folder+'\\'+'dynamic_layer_Size_Plot.png'
    rank_final_path=out_folder+'\\'+'dynamic_rank_final.png'
    rank_stable_path=out_folder+'\\'+'dynamic_rank_stable.png'
    create_layer_plot(conv_data_file_name,GLOBALS.CONFIG['adapt_trials'],conv_path, 'Layer Size')
    create_layer_plot(rank_final_file_name,GLOBALS.CONFIG['adapt_trials'],rank_final_path, 'Final Rank')
    create_layer_plot(rank_stable_file_name,GLOBALS.CONFIG['adapt_trials'],rank_stable_path, 'Stable Rank')
    return True

def run_trials(train_loader,test_loader,device,optimizer,scheduler,epochs,output_path_train):
    conv_data = pd.DataFrame(columns=['superblock1','superblock2','superblock3','superblock4','superblock5'])
    rank_final_data = pd.DataFrame(columns=['superblock1','superblock2','superblock3','superblock4','superblock5'])
    rank_stable_data = pd.DataFrame(columns=['superblock1','superblock2','superblock3','superblock4','superblock5'])

    conv_size_list=[GLOBALS.super1_idx,GLOBALS.super2_idx,GLOBALS.super3_idx,GLOBALS.super4_idx,GLOBALS.super5_idx]
    conv_data.loc[0] = conv_size_list
    delta_info = pd.DataFrame(columns=['delta_percentage','factor_scale','last_operation'])


    run_epochs(0, epochs, train_loader, test_loader,
                           device, optimizer, scheduler, output_path_train)

    print('~~~First run_epochs done.~~~')
    last_operation,factor_scale,delta_percentage=[],[],[]
    for i in range(1,GLOBALS.CONFIG['adapt_trials']):

        input_ranks, output_ranks = get_max_ranks_by_layer(path=GLOBALS.EXCEL_PATH)
        counter=-1
        shortcut_indexes=[]
        for j in conv_size_list:
            if len(shortcut_indexes)==len(conv_size_list)-1:
                break
            counter+=len(j) + 1
            shortcut_indexes+=[counter]
        index_conv_size_list=GLOBALS.index
        print('GLOBALS.EXCEL_PATH:{}'.format(GLOBALS.EXCEL_PATH))
        start=time.time()
        '------------------------------------------------------------------------------------------------------------------------------------------------'
        #output_sizes=calculate_correct_output_sizes(input_ranks,output_ranks,conv_size_list,shortcut_indexes,GLOBALS.CONFIG['adapt_rank_threshold'])[0]
        #output_sizes=calculate_correct_output_sizes_averaged(input_ranks,output_ranks,conv_size_list,shortcut_indexes,GLOBALS.CONFIG['adapt_rank_threshold'])
        last_operation, factor_scale, output_sizes, delta_percentage, rank_averages_final, rank_averages_stable = delta_scaling(conv_size_list,GLOBALS.CONFIG['delta_threshold'],GLOBALS.CONFIG['min_scale_limit'],GLOBALS.CONFIG['adapt_trials'],shortcut_indexes,last_operation, factor_scale, delta_percentage)
        '------------------------------------------------------------------------------------------------------------------------------------------------'
        end=time.time()
        print((end-start),'Time ELAPSED FOR SCALING in TRIAL '+str(i))
        last_operation_copy, factor_scale_copy, delta_percentage_copy, rank_averages_final_copy, rank_averages_stable_copy = copy.deepcopy(last_operation),copy.deepcopy(factor_scale),copy.deepcopy(delta_percentage),copy.deepcopy(rank_averages_final),copy.deepcopy(rank_averages_stable)
        conv_size_list=copy.deepcopy(output_sizes)
        conv_data.loc[i] = output_sizes
        rank_final_data.loc[i] = rank_averages_final_copy
        rank_stable_data.loc[i] = rank_averages_stable_copy
        delta_info.loc[i] = [delta_percentage_copy,factor_scale_copy,last_operation_copy]
        print('~~~Starting Conv Adjustments~~~')
        new_network=update_network(output_sizes)

        print('~~~Initializing the new model~~~')
        parser = ArgumentParser(description=__doc__)
        args(parser)
        args_true = parser.parse_args()
        train_loader,test_loader,device,optimizer,scheduler,output_path,starting_conv_sizes = initialize(args_true,new_network)

        epochs = range(0, GLOBALS.CONFIG['epochs_per_trial'])

        print('~~~Training with new model~~~')
        run_epochs(i, epochs, train_loader, test_loader,
                               device, optimizer, scheduler, output_path_train)

    return conv_data,rank_final_data,rank_stable_data,output_sizes,delta_info

def create_trial_data_file(conv_data,delta_info,rank_final_data,rank_stable_data,output_path_string_trials,output_path_string_graph_files,output_path_string_modelweights):
    #parameter_data.to_excel(output_path_string_trials+'\\'+'adapted_parameters.xlsx')
    delta_info.to_excel(output_path_string_trials+'\\'+'adapted_delta_info.xlsx')
    conv_data.to_excel(output_path_string_trials+'\\'+'adapted_architectures.xlsx')
    rank_final_data.to_excel(output_path_string_trials+'\\'+'adapted_rank_final.xlsx')
    rank_stable_data.to_excel(output_path_string_trials+'\\'+'adapted_rank_stable.xlsx')
    create_graphs(GLOBALS.EXCEL_PATH,output_path_string_trials+'\\'+'adapted_architectures.xlsx',output_path_string_trials+'\\'+'adapted_rank_final.xlsx',output_path_string_trials+'\\'+'adapted_rank_stable.xlsx',output_path_string_graph_files)
    torch.save(GLOBALS.NET.state_dict(), output_path_string_modelweights+'\\'+'model_state_dict')

def run_epochs(trial, epochs, train_loader, test_loader,
               device, optimizer, scheduler, output_path):
    if GLOBALS.CONFIG['lr_scheduler'] == 'AdaS':
        if GLOBALS.FULL_TRAIN == False:
            xlsx_name = \
                f"AdaS_adapt_trial={trial}_" +\
                f"net={GLOBALS.CONFIG['network']}_" +\
                f"{GLOBALS.CONFIG['init_lr']}_dataset=" +\
                f"{GLOBALS.CONFIG['dataset']}.xlsx"
        else:
            if GLOBALS.FULL_TRAIN_MODE == 'last_trial':
                xlsx_name = \
                    f"AdaS_last_iter_fulltrain_trial={trial}_" +\
                    f"net={GLOBALS.CONFIG['network']}_" +\
                    f"dataset=" +\
                    f"{GLOBALS.CONFIG['dataset']}.xlsx"
            elif GLOBALS.FULL_TRAIN_MODE == 'fresh':
                xlsx_name = \
                    f"AdaS_fresh_fulltrain_trial={trial}_" +\
                    f"net={GLOBALS.CONFIG['network']}_" +\
                    f"dataset=" +\
                    f"{GLOBALS.CONFIG['dataset']}.xlsx"
            else:
                print('ERROR: INVALID FULL_TRAIN_MODE | Check that the correct full_train_mode strings have been initialized in main file | Should be either fresh, or last_trial')
                sys.exit()
    else:
        xlsx_name = \
            f"{GLOBALS.CONFIG['optim_method']}_" +\
            f"{GLOBALS.CONFIG['lr_scheduler']}_" +\
            f"trial={trial}_initlr={GLOBALS.CONFIG['init_lr']}" +\
            f"net={GLOBALS.CONFIG['network']}_dataset=" +\
            f"{GLOBALS.CONFIG['dataset']}.xlsx"
    xlsx_path = str(output_path) +'\\'+ xlsx_name

    if GLOBALS.FULL_TRAIN == False:
        filename = \
            f"stats_net={GLOBALS.CONFIG['network']}_AdaS_trial={trial}_" +\
            f"beta={GLOBALS.CONFIG['beta']}_initlr={GLOBALS.CONFIG['init_lr']}_" +\
            f"dataset={GLOBALS.CONFIG['dataset']}.csv"
    else:
        if GLOBALS.FULL_TRAIN_MODE == 'last_trial':
            filename = \
                f"stats_last_iter_net={GLOBALS.CONFIG['network']}_AdaS_trial={trial}_" +\
                f"beta={GLOBALS.CONFIG['beta']}_" +\
                f"dataset={GLOBALS.CONFIG['dataset']}.csv"
        elif GLOBALS.FULL_TRAIN_MODE == 'fresh':
            filename = \
                f"stats_fresh_net={GLOBALS.CONFIG['network']}_AdaS_trial={trial}_" +\
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
            f"AdaS: Trial {trial}/{GLOBALS.CONFIG['n_trials'] - 1} | " +
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
    GLOBALS.PERFORMANCE_STATISTICS[f'fc_S_epoch_{epoch}'] = \
        io_metrics.fc_S
    GLOBALS.PERFORMANCE_STATISTICS[f'in_rank_epoch_{epoch}'] = \
        io_metrics.input_channel_rank
    GLOBALS.PERFORMANCE_STATISTICS[f'out_rank_epoch_{epoch}'] = \
        io_metrics.output_channel_rank
    GLOBALS.PERFORMANCE_STATISTICS[f'fc_rank_epoch_{epoch}'] = \
        io_metrics.fc_rank
    GLOBALS.PERFORMANCE_STATISTICS[f'in_condition_epoch_{epoch}'] = \
        io_metrics.input_channel_condition

    GLOBALS.PERFORMANCE_STATISTICS[f'out_condition_epoch_{epoch}'] = \
        io_metrics.output_channel_condition
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
