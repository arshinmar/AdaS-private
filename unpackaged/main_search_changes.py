'''for j in i:
    print(j.shape, 'J')
    initial_L1_values+=[(initial_L1_norm(j),initial_counter)]
    initial_counter+=1

print(initial_L1_values, 'initial_L1')

initial_channel_numbers=return_channel_numbers(initial_L1_values,new_input_channel_size)
initial_main_tensors=[i[j].reshape(1,i[j].shape[0], i[j].shape[1]) for j in initial_channel_numbers]
initial_final=torch.cat(initial_main_tensors,0)
print(initial_final.shape)
initial_L1_values=[]
i=initial_final

'''

'''
(64,64,3,3)
(54,54,3,3)

take the 54 most important channels, the size of this "important channel" is (1,64,3,3)
(54,64,3,3)

POTENTIAL SOLUTIONS:
 -- AVERAGE the 64,3,3 into a 54,3,3, retaining as much usefulness as possible.
 -- Take the best 54 input channels and the accompanying weights
 -----------------------------------------
 Size of Initial Weights:(64,64,3,3) --> (out, in, kernel_size_1, kernel_size_2)
  TO
 Size of New Weights:  (128,128,3,3)

Concatenation issues?

(64,64,3,3)
to
(64,60,3,3)
to
(128,60,3,3)
'''

from models.own_network import Network, TestNetwork
import torch, torchvision
import numpy as np

from train_support import run_epochs
from collections import OrderedDict
#from . import global_vars as GLOBALS
import global_vars as GLOBALS
import time, copy
from models.own_network import TestNetwork

def prototype(net_state_dict,new_output_sizes):
    '''CONVERTS LIST TO TUPLE'''
    def convert(list):
        return tuple(list)

    def initial_L1_norm(matrix):
        main=matrix.numpy()
        main=main.reshape((1,main.shape[0],main.shape[1]))
        main_shape=main.shape
        if main_shape!=():
            comp=torch.zeros(main_shape).numpy()
            return np.sum(np.abs(main-comp))

    #RETURNS L1 NORM of a WEIGHTS OF A SINGLE OUTPUT CHANNEL
    def L1_norm(matrix):
        main=matrix.numpy()
        main=main.reshape((1,main.shape[0],main.shape[1],main.shape[2]))
        main_shape=main.shape
        if main_shape!=():
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
    def create_weights(new_output_channel_size,new_input_channel_size,width,height):
        goal=torch.randn(new_output_channel_size,new_input_channel_size,width,height)
        #print(goal.shape, 'RANDOM KERNEL SHAPE')
        return goal

    '''Makes the weights have the size new_input_channel_size'''
    def make_same_input(new_input_channel_size,weights):
        our_tensors=[]
        for i in weights:
            new_tensor=i[:new_input_channel_size]
            new_tensor=new_tensor.reshape(1,new_tensor.shape[0],new_tensor.shape[1], new_tensor.shape[2])
            our_tensors+=[new_tensor]
        weights=torch.cat(our_tensors,0)
        return weights

    '''Concatenates previous_weights with that new random_kernel'''
    def adjust_weights(weights,
                       old_output_channel_size, old_input_channel_size,
                       new_output_channel_size, new_input_channel_size,
                       width, height):

        output_difference=new_output_channel_size-old_output_channel_size
        input_difference=new_input_channel_size-old_input_channel_size

        L1_values=[]
        initial_L1_values=[]

        initial_counter=0
        counter=0

        #SHRINK OUTPUT
        if output_difference<0:

            print(weights.shape, 'NEW PREV WEIGHT SIZE')
            #Iterating through a layer's weights
            '''Add L1 Norm Values for Each Channel's Weights'''
            for i in weights:

                L1_values+=[(L1_norm(i),counter)]
                counter+=1

            '''Get X MOST IMPORTANT channel numbers"'''
            channel_numbers=return_channel_numbers(L1_values,new_output_channel_size)
            '''Store weights of those X MOST IMPORTANT channel numbers (with some reshaping done)'''
            try:
                best_tensors=[weights[i].reshape(1,weights[i].shape[0],weights[i].shape[1], weights[i].shape[2]) for i in channel_numbers]
            except:
                print(channel_numbers)
            '''Concatenate those tensors'''
            final=torch.cat(best_tensors,0)

        #EXPAND OUTPUT
        else:
            goal=create_weights(output_difference,old_input_channel_size,width,height)
            final=torch.cat((weights,goal),0)

        #SHRINK INPUT
        if input_difference<0:

            final=make_same_input(new_input_channel_size,final)



        #EXPAND INPUT
        else:
            goal2=create_weights(new_output_channel_size,input_difference,width,height)
            final=torch.cat((final,goal2),1)




















        if output_difference<0:

                goal2=create_weights(old_output_channel_size,input_difference,width,height)
                weights=torch.cat((weights,goal2),1)

            #SHRINK INPUT
            else:

                weights=make_same_input(new_input_channel_size,weights)



        #EXPAND OUTPUT
        else:
            #EXPAND INPUT
            if input_difference>0:
                #print(weights.shape, 'PREV WEIGHT SHAPE in ADJUST WEIGHTS')

                goal=create_weights(output_difference,old_input_channel_size,width,height)
                #print(goal.shape, 'RANDOM KERNEL FROM OUTPUT DIFFERENCE')
                goal2=create_weights(new_output_channel_size,input_difference,width,height)
                #print(goal2.shape, 'RANDOM KERNEL FROM INPUT DIFFERENCE')

                output_final=torch.cat((weights,goal),0)
                #print(output_final.shape, 'ASODNASDIJADSOIJSADOIJSADOIASDJOI')
                input_final=torch.cat((output_final,goal2),1)
                #print(input_final.shape,'FINAL ADJUSTED w')
            else:
                new_weights=make_same_input(new_input_channel_size,weights)
                print(weights.shape, 'iubgwihb')
                goal=create_weights(output_difference,new_input_channel_size,width,height)
                input_final=torch.cat((new_weights,goal),0)
        return input_final


    '''Initialise new network with CORRECT OUTPUT SIZES'''
    model=TestNetwork(new_output_sizes=new_output_sizes)

    start=time.time()

    '''FOR EACH LAYER IN THE NETWORK'''
    for param_tensor in net_state_dict:
        '''IF NOT A CONV WEIGHT, SKIP!'''
        val=param_tensor.find('conv')
        if val==-1:
            continue

        '''Extract ONE conv layer weights'''

        weights=net_state_dict[param_tensor]
        print(weights.shape, 'PREV WEIGHTS')
        new_weights=model.state_dict()[param_tensor]
        print(new_weights.shape, 'NEW WEIGHTS')

        old_output_channel_size=weights.shape[0]
        new_output_channel_size=new_weights.shape[0]

        new_input_channel_size=new_weights.shape[1]
        old_input_channel_size=weights.shape[1]

        width=new_weights.shape[2]
        height=new_weights.shape[3]

        final=adjust_weights(weights,old_output_channel_size,old_input_channel_size,new_output_channel_size,new_input_channel_size,width,height)

        elif old_output_channel_size<=new_output_channel_size:


        print(final.shape, 'FINAL ATTEMPTED LOADING SHAPE')
        print('_______________________________________')
        '''LOAD NEW KERNEL IN!'''
        new_state_dict = OrderedDict({str(param_tensor): final})
        model.load_state_dict(new_state_dict, strict=False)
        #break

    end=time.time()
    print(end-start, 'TIME')

    return model

net = TestNetwork()
#OLD
#'''
for param_tensor in net.state_dict():
    val=param_tensor.find('conv')
    if val==-1:
        #If not a conv parameter, skip
        continue
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())
#'''
x=torch.randn(1,3,32,32)
model=prototype(net.state_dict(),[60,128,20,128,128])
#NEW
#'''
for param_tensor in model.state_dict():
    val=param_tensor.find('conv')
    if val==-1:
        #If not a conv parameter, skip
        continue
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#'''
y=model(x)
print(y.shape)
