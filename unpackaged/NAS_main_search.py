
from argparse import Namespace as APNamespace, _SubParsersAction,ArgumentParser
from pathlib import Path
import os
# import logging

import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import torch
import yaml
import shutil
from models.own_network import AdaptiveNet

from train_support import run_epochs, get_ranks
from optim import get_optimizer_scheduler

from utils import parse_config
from metrics import Metrics
from models import get_net
from data import get_data
from AdaS import AdaS
import  global_vars as GLOBALS
from adaptive_channels import prototype

from adaptive_graph import create_adaptive_graphs,create_layer_plot

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
    NOTE: no early stopping functionality considered
    '''
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

def create_graphs(accuracy_data_file_name,conv_data_file_name):
    create_adaptive_graphs(accuracy_data_file_name,GLOBALS.CONFIG['epochs_per_trial'],GLOBALS.CONFIG['adapt_trials'])
    create_layer_plot(conv_data_file_name,GLOBALS.CONFIG['adapt_trials'])
    return True


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    args(parser)
    args = parser.parse_args()
    train_loader,test_loader,device,optimizer,scheduler,output_path,starting_conv_sizes = initialize(args)
    print('~~~Initialization Complete. Beginning first training~~~')
    epochs = range(0, GLOBALS.CONFIG['epochs_per_trial'])

    conv_data = pd.DataFrame(columns=['superblock1','superblock2','superblock3','superblock4','superblock5'])
    conv_data.loc[len(conv_data)] = starting_conv_sizes

    run_epochs(0, epochs, train_loader, test_loader,
                           device, optimizer, scheduler, output_path)

    print('~~~First run_epochs done.~~~')

    for i in range(1,GLOBALS.CONFIG['adapt_trials']):

        ranks = get_ranks(max=True)
        output_sizes=new_output_sizes(starting_conv_sizes,ranks,GLOBALS.CONFIG['adapt_rank_threshold'])
        conv_data.loc[len(conv_data)] = output_sizes

        starting_conv_sizes = output_sizes
        
        print('~~~Starting Conv Adjustments~~~')
        new_model_state_dict = prototype(GLOBALS.NET.state_dict(),output_sizes)
        new_network=AdaptiveNet(num_classes=10,new_output_sizes=output_sizes)
        new_network.load_state_dict(new_model_state_dict)

        print('~~~~~~~~~~~~~~~~~~~~~~~~NEW NET BEFORE INIT ~~~~~~~~~~~~~~~~~~~~~')

        GLOBALS.NET = torch.nn.DataParallel(new_network.cuda())
        cudnn.benchmark = True

        optimizer, scheduler = get_optimizer_scheduler(
                net_parameters=GLOBALS.NET.parameters(),
                listed_params=list(GLOBALS.NET.parameters()),
                train_loader_len=len(train_loader),
                config=GLOBALS.CONFIG)

        print('~~~Training with new model~~~')

        for param_tensor in GLOBALS.NET.state_dict():
            val=param_tensor.find('conv')
            if val==-1:
                continue
            print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor].size(),'NEW Network')
            print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor],'NEW Network')
            break;

        for param_tensor in GLOBALS.NET.state_dict():
            val=param_tensor.find('bn')
            if val==-1:
                continue
            print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor].size(),'NEW Network')
            print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor],'NEW Network')
            break;

        epochs = range(0, GLOBALS.CONFIG['epochs_per_trial'])
        run_epochs(i, epochs, train_loader, test_loader,
                               device, optimizer, scheduler, output_path)

        for param_tensor in GLOBALS.NET.state_dict():
            val=param_tensor.find('conv')
            if val==-1:
                continue
            print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor].size(), 'OLD NETWORK')
            print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor], 'OLD NETWORK')
            break;

        for param_tensor in GLOBALS.NET.state_dict():
            val=param_tensor.find('bn')
            if val==-1:
                continue
            print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor].size(), 'OLD NETWORK')
            print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor], 'OLD NETWORK')
            break;



    conv_data.to_excel(str(output_path)+'\\'+'adapted_architectures.xlsx')

    create_graphs(GLOBALS.EXCEL_PATH,str(output_path)+'\\'+'adapted_architectures.xlsx')

    print('Done')
