from models.own_network import Network, TestNetwork
import torch, torchvision
import numpy as np
from numpy import linalg as LA

from train_support import run_epochs
from collections import OrderedDict
#from . import global_vars as GLOBALS
import global_vars as GLOBALS
import time
import copy
from models.own_network import TestNetwork



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
    ##############################GLOBAL VARIABLES#####################
    #CONFIG=NONE



    # while lr_idx < len(learning_rates):
    cur_rank = 0.0
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
        GLOBALS.CONFIG['init_lr'] = 0.1# learning_rates[lr_idx]
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



def prototype(net_state_dict,new_output_sizes):
    '''RETURNS L1 NORM'''
    def L1_norm(matrix):
        main=matrix.numpy()
        main=main.reshape((1,main.shape[0],main.shape[1],main.shape[2]))
        main_shape=main.shape
        if shape!=():
            comp=torch.zeros(main_shape).numpy()
            return np.sum(np.abs(main-comp))

    '''RETURNS the CHANNELS with the HIGHEST L1 VALUES (the number of channels is size new_output_size)'''
    def return_channel_numbers(L1_values,new_output_size):
        L1_values.sort(key=lambda tup: tup[0])
        post_output_rank_sorting=L1_values[len(L1_values)-new_output_size:]
        post_channel_rank_sorting=sorted(post_output_rank_sorting, key=lambda tup: tup[1])
        channel_numbers=[i[1] for i in post_channel_rank_sorting]
        return channel_numbers

    '''RETURNS A RANDOM KERNEL OF SIZE Out x In x Width x Height'''
    def create_weights(out_channels,in_channels,width,height):
        goal=torch.randn(out_channels,in_channels,width,height)
        #print(goal.shape, 'RANDOM KERNEL SHAPE')
        return goal

    '''Concatenates previous_weights with that new random_kernel'''
    def adjust_weights(prev_weight, old_out_channels, out_channels, in_channels, width, height):
        difference=out_channels-old_out_channels
        print(prev_weight.shape, 'PREV WEIGHT SHAPE in ADJUST WEIGHTS')
        goal=create_weights(difference,in_channels,width,height)
        print(goal.shape,'RANDOM KERNEL SHAPE in ADJUST WEIGHTS')
        final=torch.cat((prev_weight,goal),0)
        print(final.shape,'FINAL ADJUSTED SHAPE in ADJUST WEIGHTS')
        return final

    '''Initialise new network with CORRECT OUTPUT SIZES'''
    model=TestNetwork(new_output_sizes=new_output_sizes)

    L1_values=[]
    counter=0
    #net_copy=copy.deepcopy(net).cpu()
    # Print model's state_dict
    print("Model's state_dict:")
    start=time.time()

    '''FOR EACH LAYER IN THE NETWORK'''
    for param_tensor in net_state_dict:
        '''IF NOT A CONV WEIGHT, SKIP!'''
        val=param_tensor.find('conv')
        if val==-1:
            #If not a conv parameter, skip
            continue

        '''Extract ONE conv layer weights'''

        weights=net_state_dict[param_tensor]
        print(weights.shape, 'PREV WEIGHTS')
        new_weights=model.state_dict()[param_tensor]
        print(new_weights.shape, 'NEW WEIGHTS')

        old_output_channel_size=weights.shape[0]
        new_output_channel_size=new_weights.shape[0]

        new_input_channel_size=new_weights.shape[1]

        width=new_weights.shape[2]
        height=new_weights.shape[3]

        '''
        (64,64,3,3)
        (54,54,3,3)

        take the 54 most important channels, the size of this "important channel" is (1,64,3,3)
        (54,64,3,3)

        POTENTIAL SOLUTIONS:
         -- AVERAGE the 64,3,3 into a 54,3,3, retaining as much usefulness as possible.
         -- Take the first 54 input channels and the accompanying weights
         -----------------------------------------
         Size of Initial Weights:(64,64,3,3) --> (out, in, kernel_size_1, kernel_size_2)
          TO
         Size of New Weights:  (128,128,3,3)

        Concatenation issues?
        '''

        if old_output_channel_size>new_output_channel_size:

            '''Add L1 Norm Values for Each Channel's Weights'''
            for i in weights:
                L1_values+=[(L1_norm(i),counter)]
                counter+=1

            '''Get X MOST IMPORTANT channel numbers"'''
            channel_numbers=return_channel_numbers(L1_values,new_output_channel_size)
            '''Store weights of those X MOST IMPORTANT channel numbers (with some reshaping done)'''
            main_tensors=[weights[i].reshape(1,weights[i].shape[0],weights[i].shape[1], weights[i].shape[2]) for i in channel_numbers]
            '''Concatenate those tensors'''
            final=torch.cat(main_tensors,0)
            new_output_size_counter+=1
            break

        elif old_output_channel_size<=new_output_channel_size:
            final=adjust_weights(weights,old_output_channel_size,new_output_channel_size,new_input_channel_size,width,height)

        '''LOAD NEW KERNEL IN!'''
        new_state_dict = OrderedDict({str(param_tensor): final})
        model.load_state_dict(new_state_dict, strict=False)

    end=time.time()
    print(end-start, 'TIME')

    return model

net = TestNetwork()

x=torch.randn(1,32,32)
model=prototype(net.state_dict(),[128,128,128,128,128])
y=model(x)
print(y.shape)
