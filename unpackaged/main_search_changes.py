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

    # ###### LR RANGE STUFF #######
    min_lr = 1e-4
    max_lr = 0.1
    num_split = 20
    learning_rates = np.geomspace(min_lr, max_lr, num_split)
    rank_history = list()
    lr_idx = 0
    min_delta = 5e-2
    rank_thresh = 0.93 * 0
    exit_counter = 0
    lr_delta = 3e-5
    output_history = list()
    first_run = True
    epochs = range(0, 5)
    set_threshold = False
    set_threshold_list = []
    historical_rate_of_change = list()
    ##############################
    # while lr_idx < len(learning_rates):
    cur_rank = 0.0
    auto_lr_path = output_path / 'auto-lr'
    auto_lr_path.mkdir(exist_ok=True)
    while True:
        if lr_idx == len(learning_rates) and not set_threshold:
            min_lr = learning_rates[-2]
            max_lr = float(learning_rates[-1]) * 1.5
            print("LR Range Test: Reached End")
            learning_rates = np.geomspace(min_lr, max_lr, num_split)
            rank_history = list()
            output_history.append(
                (learning_rates[lr_idx - 1], -1, 'end-reached'))
            historical_rate_of_change.append(
                (learning_rates[lr_idx - 1], -1, 'end-reached'))
            lr_idx = 0
            continue
        if np.less(np.abs(np.subtract(min_lr, max_lr)), lr_delta) and not set_threshold:
            print(
                f"LR Range Test Complete: LR Delta: Final LR Range is {min_lr}-{max_lr}")
            output_history.append(
                (learning_rates[lr_idx], cur_rank, 'exit-delta'))
            historical_rate_of_change.append(
                (learning_rates[lr_idx], cur_rank, 'exit-delta'))
            break
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
