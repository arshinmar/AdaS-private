from models.own_network import Network, TestNetwork
import torch, torchvision
import numpy as np
from numpy import linalg as LA

from train_support import run_epochs
#from . import global_vars as GLOBALS
import global_vars as GLOBALS


def L1_norm(matrix):
    main=matrix.numpy()
    #print(main.shape)
    comp=torch.zeros([1,3,3,3]).numpy()
    print(LA.norm(main-comp))
    print(np.max(np.sum(np.abs(main-comp),axis=0)))
    #print(comp.shape)
    #print(np.linalg.norm((matrix - comp), ord=1))

def test():
    net = TestNetwork()
    ##print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)

    # Data
    # logging.info("Adas: Preparing Data")
    GLOBALS.CONFIG['init_lr'] = learning_rates[lr_idx]
    print(f"Using LR: {GLOBALS.CONFIG['init_lr']}")
    train_loader, test_loader = get_data(
        root=data_path,
        dataset=GLOBALS.CONFIG['dataset'],
        mini_batch_size=GLOBALS.CONFIG['mini_batch_size'])
    # global performance_statistics, net, metrics, adas
    GLOBALS.PERFORMANCE_STATISTICS = {}

    # logging.info("AdaS: Building Model")
    GLOBALS.NET = get_net(
        GLOBALS.CONFIG['network'], num_classes=10 if
        GLOBALS.CONFIG['dataset'] == 'CIFAR10' else 100 if
        GLOBALS.CONFIG['dataset'] == 'CIFAR100'
        else 1000 if GLOBALS.CONFIG['dataset'] == 'ImageNet' else 10)
    GLOBALS.METRICS = Metrics(list(GLOBALS.NET.parameters()),
                              p=GLOBALS.CONFIG['p'])

    GLOBALS.NET = GLOBALS.NET.to(device)

    # global criterion
    GLOBALS.CRITERION = get_loss(GLOBALS.CONFIG['loss'])

    optimizer, scheduler = get_optimizer_scheduler(
        net_parameters=GLOBALS.NET.parameters(),
        listed_params=list(GLOBALS.NET.parameters()),
        # init_lr=learning_rate,
        # optim_method=GLOBALS.CONFIG['optim_method'],
        # lr_scheduler=GLOBALS.CONFIG['lr_scheduler'],
        train_loader_len=len(train_loader),
        config=GLOBALS.CONFIG)
    # max_epochs=int(GLOBALS.CONFIG['max_epoch']))
    GLOBALS.EARLY_STOP = EarlyStop(
        patience=int(GLOBALS.CONFIG['early_stop_patience']),
        threshold=float(GLOBALS.CONFIG['early_stop_threshold']))

    if device == 'cuda':
        GLOBALS.NET = torch.nn.DataParallel(GLOBALS.NET)
        cudnn.benchmark = True
    # Data
    # logging.info("Adas: Preparing Data")
    GLOBALS.CONFIG['init_lr'] = learning_rates[lr_idx]
    print(f"Using LR: {GLOBALS.CONFIG['init_lr']}")
    train_loader, test_loader = get_data(
        root=data_path,
        dataset=GLOBALS.CONFIG['dataset'],
        mini_batch_size=GLOBALS.CONFIG['mini_batch_size'])
    # global performance_statistics, net, metrics, adas
    GLOBALS.PERFORMANCE_STATISTICS = {}

    # logging.info("AdaS: Building Model")
    GLOBALS.NET = get_net(
        GLOBALS.CONFIG['network'], num_classes=10 if
        GLOBALS.CONFIG['dataset'] == 'CIFAR10' else 100 if
        GLOBALS.CONFIG['dataset'] == 'CIFAR100'
        else 1000 if GLOBALS.CONFIG['dataset'] == 'ImageNet' else 10)
    GLOBALS.METRICS = Metrics(list(GLOBALS.NET.parameters()),
                              p=GLOBALS.CONFIG['p'])

    GLOBALS.NET = GLOBALS.NET.to(device)

    # global criterion
    GLOBALS.CRITERION = get_loss(GLOBALS.CONFIG['loss'])

    optimizer, scheduler = get_optimizer_scheduler(
        net_parameters=GLOBALS.NET.parameters(),
        listed_params=list(GLOBALS.NET.parameters()),
        # init_lr=learning_rate,
        # optim_method=GLOBALS.CONFIG['optim_method'],
        # lr_scheduler=GLOBALS.CONFIG['lr_scheduler'],
        train_loader_len=len(train_loader),
        config=GLOBALS.CONFIG)
    # max_epochs=int(GLOBALS.CONFIG['max_epoch']))
    GLOBALS.EARLY_STOP = EarlyStop(
        patience=int(GLOBALS.CONFIG['early_stop_patience']),
        threshold=float(GLOBALS.CONFIG['early_stop_threshold']))

    if device == 'cuda':
        GLOBALS.NET = torch.nn.DataParallel(GLOBALS.NET)
        cudnn.benchmark = True

    run_epochs(0, epochs, train_loader, test_loader,
               device, optimizer, scheduler, auto_lr_path)

    Profiler.stream = None

def prototype():

    net = TestNetwork()
    #print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)

    # Print model's state_dict
    counter=0
    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        if counter==0:
            print(net.state_dict()[param_tensor][0][0][0][0])
            net.state_dict()[param_tensor][0][0][0][0]=1
            print(net.state_dict()[param_tensor][0][0][0][0])
            counter+=1
            break

test()
