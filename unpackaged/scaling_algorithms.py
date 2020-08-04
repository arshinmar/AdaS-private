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

def even_round(number):
    return int(round(number/2)*2)

def get_ranks(path = GLOBALS.EXCEL_PATH, epoch_number = -1):
    '''
    - Read from .adas-output excel file
    - Get Final epoch ranks
    '''
    sheet = pd.read_excel(path,index_col=0)
    out_rank_col = [col for col in sheet if col.startswith('out_rank')]
    in_rank_col = [col for col in sheet if col.startswith('in_rank')]

    out_ranks = sheet[out_rank_col]
    in_ranks = sheet[in_rank_col]

    last_rank_col_out = out_ranks.iloc[:,epoch_number]
    last_rank_col_in = in_ranks.iloc[:,epoch_number]

    last_rank_col_in = last_rank_col_in.tolist()
    last_rank_col_out = last_rank_col_out.tolist()

    return last_rank_col_in, last_rank_col_out

def delta_scaling(conv_size_list,delta_threshold,min_scale_limit,num_trials,shortcut_indexes,last_operation,factor_scale,delta_percentage):
    #print('GLOBALS EXCEL PATH IN DELTA_SCALING FUNCTION:{}'.format(GLOBALS.EXCEL_PATH))
    input_ranks_final,output_ranks_final = get_ranks(path=GLOBALS.EXCEL_PATH,epoch_number=-1)
    input_ranks_stable,output_ranks_stable = get_ranks(path=GLOBALS.EXCEL_PATH,epoch_number=0)

    #print("FINAL EPOCH RANKS")
    rank_averages_final=calculate_correct_output_sizes(input_ranks_final, output_ranks_final, conv_size_list, shortcut_indexes, GLOBALS.CONFIG['delta_threshold'],final=False)[1]
    #print("STABLE EPOCH RANKS")
    rank_averages_stable=calculate_correct_output_sizes(input_ranks_stable,output_ranks_stable, conv_size_list, shortcut_indexes, GLOBALS.CONFIG['delta_threshold'],final=False)[1]

    #print(rank_averages_final, 'RANKS AVERAGES FINAL')
    #print(rank_averages_stable,'RANK AVERAGES STABLE')

    EXPAND,SHRINK,STOP = 1,-1,0
    new_channel_sizes=copy.deepcopy(conv_size_list)

    FIRST_TIME=False

    if last_operation==[]:
        FIRST_TIME = True
        for i in conv_size_list:
            factor_scale.append([0.1]*len(i))
            last_operation.append([1]*len(i))
            delta_percentage.append([0]*len(i))
    for superblock in range(len(new_channel_sizes)):
        for layer in range(0,len(new_channel_sizes[superblock])):
            if (last_operation[superblock][layer] == STOP):
                continue
            delta_percentage[superblock][layer] = round((rank_averages_final[superblock][layer]-rank_averages_stable[superblock][layer])/rank_averages_final[superblock][layer],5)

            current_operation = EXPAND if delta_percentage[superblock][layer] >= delta_threshold else SHRINK

            if (last_operation[superblock][layer] != current_operation and FIRST_TIME==True):
                if (factor_scale[superblock][layer] < min_scale_limit):
                    current_operation = STOP
                factor_scale[superblock][layer] = factor_scale[superblock][layer]/2

            last_operation[superblock][layer] = current_operation
            new_channel_sizes[superblock][layer] = even_round(conv_size_list[superblock][layer] * (1 + factor_scale[superblock][layer]*last_operation[superblock][layer]))

    print("Delta Percentage:{}".format(delta_percentage))
    print(factor_scale,'FACTOR SCALE')
    print(new_channel_sizes,'OUTPUT CONV SIZE LIST')

    return last_operation,factor_scale,new_channel_sizes,delta_percentage, rank_averages_final, rank_averages_stable

def calculate_correct_output_sizes_averaged(input_ranks,output_ranks,conv_size_list,shortcut_indexes,threshold):
    output_ranks_layer_1 = output_ranks[0]
    scaling_factor=[0,0,0,0,0]
    output_ranks_superblock_1 = output_ranks[1:shortcut_indexes[0]]
    output_ranks_superblock_2 = output_ranks[shortcut_indexes[0]+1:shortcut_indexes[1]]
    output_ranks_superblock_3 = output_ranks[shortcut_indexes[1]+1:shortcut_indexes[2]]
    output_ranks_superblock_4 = output_ranks[shortcut_indexes[2]+1:shortcut_indexes[3]]
    output_ranks_superblock_5 = output_ranks[shortcut_indexes[3]+1:]

    super_block_1_val=conv_size_list[0][0]
    super_block_2_val=conv_size_list[1][0]
    super_block_3_val=conv_size_list[2][0]
    super_block_4_val=conv_size_list[3][0]
    super_block_5_val=conv_size_list[4][0]

    scaling_factor[0] = np.average(output_ranks_superblock_1)-threshold
    scaling_factor[1] = np.average(output_ranks_superblock_2)-threshold
    scaling_factor[2] = np.average(output_ranks_superblock_3)-threshold
    scaling_factor[3] = np.average(output_ranks_superblock_4)-threshold
    scaling_factor[4] = np.average(output_ranks_superblock_5)-threshold

    super_block_1 = [even_round(super_block_1_val*(1+scaling_factor[0]))] * (len(output_ranks_superblock_1)+1)
    super_block_2 = [even_round(super_block_2_val*(1+scaling_factor[1]))] * len(output_ranks_superblock_2)
    super_block_3 = [even_round(super_block_3_val*(1+scaling_factor[2]))] * len(output_ranks_superblock_3)
    super_block_4 = [even_round(super_block_4_val*(1+scaling_factor[3]))] * len(output_ranks_superblock_4)
    super_block_5 = [even_round(super_block_5_val*(1+scaling_factor[4]))] * len(output_ranks_superblock_5)

    output_conv_size_list=[super_block_1]+[super_block_2]+[super_block_3]+[super_block_4]+[super_block_5]
    print(output_conv_size_list)

    return output_conv_size_list

def calculate_correct_output_sizes(input_ranks,output_ranks,conv_size_list,shortcut_indexes,threshold,final=True):
    #Note that input_ranks/output_ranks may have a different size than conv_size_list
    #threshold=GLOBALS.CONFIG['adapt_rank_threshold']
    '''
    input_ranks_layer_1, output_ranks_layer_1 = input_ranks[0], output_ranks[0]

    input_ranks_superblock_1, output_ranks_superblock_1 = input_ranks[1:shortcut_indexes[0]], output_ranks[1:shortcut_indexes[0]]
    input_ranks_superblock_2, output_ranks_superblock_2 = input_ranks[shortcut_indexes[0]+1:shortcut_indexes[1]], output_ranks[shortcut_indexes[0]+1:shortcut_indexes[1]]
    input_ranks_superblock_3, output_ranks_superblock_3 = input_ranks[shortcut_indexes[1]+1:shortcut_indexes[2]], output_ranks[shortcut_indexes[1]+1:shortcut_indexes[2]]
    input_ranks_superblock_4, output_ranks_superblock_4 = input_ranks[shortcut_indexes[2]+1:shortcut_indexes[3]], output_ranks[shortcut_indexes[2]+1:shortcut_indexes[3]]
    input_ranks_superblock_5, output_ranks_superblock_5 = input_ranks[shortcut_indexes[3]+1:], output_ranks[shortcut_indexes[2]+1:shortcut_indexes[3]]'''

    temp_shortcut_indexes=[0]+shortcut_indexes+[len(input_ranks)]
    new_input_ranks=[]
    new_output_ranks=[]
    for i in range(0,len(temp_shortcut_indexes)-1,1):
        new_input_ranks+=[input_ranks[temp_shortcut_indexes[i]+1:temp_shortcut_indexes[i+1]]]
        new_output_ranks+=[output_ranks[temp_shortcut_indexes[i]+1:temp_shortcut_indexes[i+1]]]

    #new_input_ranks = [input_ranks_superblock_1] + [input_ranks_superblock_2] + [input_ranks_superblock_3] + [input_ranks_superblock_4] + [input_ranks_superblock_5]
    #new_output_ranks = [output_ranks_superblock_1] + [output_ranks_superblock_2] + [output_ranks_superblock_3] + [output_ranks_superblock_4] + [output_ranks_superblock_5]

    #print(new_input_ranks,'INPUT RANKS WITHOUT SHORTCUTS')
    #print(new_output_ranks,'OUTPUT RANKS WITHOUT SHORTCUTS')

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

    #print(conv_size_list,'CONV SIZE LIST')
    output_conv_size_list=copy.deepcopy(conv_size_list)
    rank_averages = copy.deepcopy(conv_size_list)
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
            rank_averages[i][j] = scaling_factor + threshold

    if final==True:
        GLOBALS.super1_idx = output_conv_size_list[0]
        GLOBALS.super2_idx = output_conv_size_list[1]
        GLOBALS.super3_idx = output_conv_size_list[2]
        GLOBALS.super4_idx = output_conv_size_list[3]
        GLOBALS.super5_idx = output_conv_size_list[4]
        GLOBALS.index = output_conv_size_list[0] + output_conv_size_list[1] + output_conv_size_list[2] + output_conv_size_list[3] + output_conv_size_list[4]

    #print(output_conv_size_list,'OUTPUT CONV SIZE LIST')
    return output_conv_size_list, rank_averages
