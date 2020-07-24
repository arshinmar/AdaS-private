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

from models.own_network import Network, AdaptiveNet
import torch, torchvision
import numpy as np
import heapq

#from train_support import run_epochs
from collections import OrderedDict
#from . import global_vars as GLOBALS
#import global_vars as GLOBALS
import time, copy

def prototype(net_state_dict,new_output_sizes):
    '''CONVERTS LIST TO TUPLE'''
    def convert(list):
        return tuple(list)

    def initial_L1_norm(matrix):
        main=matrix
        main=main.reshape((1,main.shape[0],main.shape[1]))
        main_shape=main.shape
        if main_shape!=():
            comp=torch.zeros(main_shape)
            return torch.sum(torch.abs(main.cuda()-comp.cuda()))

    #RETURNS L1 NORM of a WEIGHTS OF A SINGLE OUTPUT CHANNEL
    def L1_norm(matrix):
        main=matrix
        main=main.reshape((1,main.shape[0],main.shape[1],main.shape[2]))
        main_shape=main.shape
        if main_shape!=():
            comp=torch.zeros(main_shape)
            return torch.sum(torch.abs(main.cuda()-comp.cuda()))

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
        ##print(goal.shape, 'RANDOM KERNEL SHAPE')
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


    def adjust_bn_weights(bn_weights, new_bn_weights, old_output_channel_size,new_output_channel_size,param_tensor):
        output_difference=new_output_channel_size - old_output_channel_size
        final = []
        if output_difference<0:
            bn_L1_values=[]
            counter=0
            bn_weights_temp=bn_weights.tolist()
            for i in bn_weights_temp:
                bn_L1_values+=[(i,counter)]
                counter+=1

            bn_channel_numbers=return_channel_numbers(bn_L1_values,new_output_channel_size)
            bn_values=[bn_weights_temp[i] for i in bn_channel_numbers]

            final=torch.FloatTensor(bn_values)
        else:
            if param_tensor.find('weight')!=-1:
                ones = [1] * output_difference
                ones = torch.FloatTensor(ones)
                final = torch.cat((bn_weights.cuda(),ones.cuda()), 0)
            elif param_tensor.find('bias')!=-1:
                zeros = [0] * output_difference
                zeros = torch.FloatTensor(zeros)
                final = torch.cat((bn_weights.cuda(),zeros.cuda()), 0)
            else:
                final = new_bn_weights
            '''elif param_tensor.find('running')!=-1:
                random=torch.randn(output_difference)
                final=torch.cat((bn_weights.cuda(),random.cuda()),0)'''
        return final

    '''Expands/shrinks output and input channels to get desired weights for replacement.'''

    def adjust_conv_weights(weights,new_weights,old_output_channel_size, new_output_channel_size, param_tensor):

        old_input_channel_size, new_input_channel_size=weights.shape[1], new_weights.shape[1]
        width,height=weights.shape[2],weights.shape[3]

        output_difference=new_output_channel_size-old_output_channel_size
        input_difference=new_input_channel_size-old_input_channel_size

        #SHRINK OUTPUT
        L1_values=[]
        counter=0
        if output_difference<0:
            '''Add L1 Norm Values for Each Channel's Weights'''
            for i in weights:
                L1_values+=[(L1_norm(i),counter)]
                counter+=1

            '''Get X MOST IMPORTANT channel numbers"'''
            channel_numbers=return_channel_numbers(L1_values,new_output_channel_size)
            '''Store weights of those X MOST IMPORTANT channel numbers (with some reshaping done)'''
            best_tensors=[weights[i].reshape(1,weights[i].shape[0],weights[i].shape[1], weights[i].shape[2]) for i in channel_numbers]
            '''Concatenate those tensors'''
            final=torch.cat(best_tensors,0)

        #EXPAND OUTPUT
        else:
            goal=create_weights(output_difference,old_input_channel_size,width,height)
            final=torch.cat((weights.cuda(),goal.cuda()),0)

        #SHRINK INPUT
        full_form=[]
        if input_difference<0:
            full_input_final=[]

            initial_L1_values=[]
            initial_counter=0
            '''For each output channel weight Y'''
            for i in final:
                #print(i.shape)
                '''Get L1 norm values for each tensor in Y'''
                for j in i:
                    initial_L1_values+=[(initial_L1_norm(j),initial_counter)]
                    initial_counter+=1
                '''Get the best tensors in Y and put them in a list'''
                initial_channel_numbers=return_channel_numbers(initial_L1_values,new_input_channel_size)
                initial_best_tensors=[i[k].reshape(1,1,i[k].shape[0],i[k].shape[1]) for k in initial_channel_numbers]
                initial_final=torch.cat(initial_best_tensors,1)
                full_form+=[initial_final]

                '''Reset!'''
                initial_final=[]
                initial_L1_values=[]
                initial_counter=0
                #print(new_tensor.shape, 'NEW')
            #final=new_tensor
            final=torch.cat(full_form,0)
            #print(final.shape, 'FINALITY')
            #final=make_same_input(new_input_channel_size,final)

        #EXPAND INPUT
        else:
            goal2=create_weights(new_output_channel_size,input_difference,width,height)
            final=torch.cat((final.cuda(),goal2.cuda()),1)
        return final

    def adjust_shortcut_weights(weights,new_weights, old_output_channel_size, new_output_channel_size, param_tensor):
        if len(new_weights.shape)!=1:
            final=adjust_conv_weights(weights,new_weights,old_output_channel_size,new_output_channel_size,param_tensor)
        else:
            final=adjust_bn_weights(weights,new_weights,old_output_channel_size,new_output_channel_size,param_tensor)
        return final

    #Initialise new network with CORRECT OUTPUT SIZES
    model=AdaptiveNet(new_output_sizes=new_output_sizes)

    start=time.time()
    '''for param_tensor in net_state_dict:
        print(param_tensor, "\t", net_state_dict[param_tensor].size())'''
    for param_tensor in net_state_dict:
        if (param_tensor.find('num_batches_tracked')!=-1):
            continue
        weights=net_state_dict[param_tensor]
        try:
            new_weights=model.state_dict()[param_tensor]
        except:
            new_weights=model.state_dict()[param_tensor[7:]]

        old_output_channel_size=weights.shape[0]
        new_output_channel_size=new_weights.shape[0]

        if (param_tensor.find('conv')!=-1):
            '''---------------------------------'''
            final=adjust_conv_weights(weights,new_weights, old_output_channel_size,new_output_channel_size,param_tensor)
            '''---------------------------------'''
            '''
        elif (param_tensor.find('shortcut')!=-1):

            final=adjust_shortcut_weights(weights,new_weights, old_output_channel_size,new_output_channel_size,param_tensor)

            '''
        elif (param_tensor.find('bn')!=-1):
            '''---------------------------------'''
            final=adjust_bn_weights(weights,new_weights, old_output_channel_size, new_output_channel_size, param_tensor)
            '''---------------------------------'''
        else:
            continue

        new_state_dict = OrderedDict({param_tensor[7:]: final})
        model.load_state_dict(new_state_dict, strict=False)

    end=time.time()
    print(end-start, 'Time Elapsed')
    return model.state_dict()

def test():
    net = AdaptiveNet()
    #OLD WEIGHTS
    #'''
    for param_tensor in net.state_dict():
        val=param_tensor.find('conv')
        #if val==-1:
        #    continue
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    #'''
    x=torch.randn(1,3,32,32)
    model=prototype(net.state_dict(),[100,10,100,10,508])
    #NEW WEIGHTS
    #'''
    for param_tensor in model.state_dict():
        val=param_tensor.find('conv')
        if val==-1:
            continue
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #'''
    y=model(x)
    #print(y.shape)

#test()
