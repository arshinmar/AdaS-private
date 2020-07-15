
from argparse import Namespace as APNamespace, _SubParsersAction,ArgumentParser
from pathlib import Path

# import logging

import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import torch
import yaml

from train_support import run_epochs
from optim import get_optimizer_scheduler

from utils import parse_config
#from profiler import Profiler
from metrics import Metrics
from models import get_net
from data import get_data
from AdaS import AdaS
import  global_vars as GLOBALS

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
        default='.adas-output', type=str,
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
    GLOBALS.NET = get_net(
                GLOBALS.CONFIG['network'], num_classes=10 if
                GLOBALS.CONFIG['dataset'] == 'CIFAR10' else 100 if
                GLOBALS.CONFIG['dataset'] == 'CIFAR100'
                else 1000 if GLOBALS.CONFIG['dataset'] == 'ImageNet' else 10)
    GLOBALS.METRICS = Metrics(list(GLOBALS.NET.parameters()),
                                      p=GLOBALS.CONFIG['p'])

    GLOBALS.NET = GLOBALS.NET.to(device)

    GLOBALS.CRITERION = get_loss(GLOBALS.CONFIG['loss'])

    optimizer, scheduler = get_optimizer_scheduler(
                net_parameters=GLOBALS.NET.parameters(),
                #listed_params=list(GLOBALS.NET.parameters()),
                init_lr=GLOBALS.CONFIG['init_lr'],
                optim_method=GLOBALS.CONFIG['optim_method'],
                lr_scheduler=GLOBALS.CONFIG['lr_scheduler'],
                train_loader_len=len(train_loader),
                max_epochs=int(GLOBALS.CONFIG['max_epoch']))
    GLOBALS.OPTIMIZER = optimizer
    if device == 'cuda':
            GLOBALS.NET = torch.nn.DataParallel(GLOBALS.NET)
            cudnn.benchmark = True

    return train_loader,test_loader,device,optimizer,scheduler,output_path




if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    args(parser)
    args = parser.parse_args()
    train_loader,test_loader,device,optimizer,scheduler,output_path = initialize(args)
    print('Beginning first training')
    epochs = range(0, 3)
    run_epochs(0, epochs, train_loader, test_loader,
                           device, optimizer, scheduler, output_path)
    cnt = 0
    for param in GLOBALS.NET.parameters():
        print(param)
        cnt += 1
        if cnt == 6:
            break

    print('Finished first training, run_epochs function complete')

    epochs = range(0, 3)
    run_epochs(0, epochs, train_loader, test_loader,
                           device, optimizer, scheduler, output_path)
    print('~~~~~final~~~~~')
    cnt = 0
    for param in GLOBALS.NET.parameters():
        print(param)
        cnt += 1
        if cnt == 6:
            break



    print('Done')
