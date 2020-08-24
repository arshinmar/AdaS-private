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
from ptflops import get_model_complexity_info
from models.own_network import DASNet34,DASNet50
import copy
import torch
import torch.backends.cudnn as cudnn
from adaptive_graph import adaptive_stop, slope
def even_round(number):
    return int(round(number/2)*2)

def get_info(info,path = GLOBALS.EXCEL_PATH, epoch_number = -1):
    '''
    - Read from .adas-output excel file
    - Get Final epoch ranks
    '''
    sheet = pd.read_excel(path,index_col=0)
    out_condition_col = [col for col in sheet if col.startswith('out_'+info)]
    in_condition_col = [col for col in sheet if col.startswith('in_'+info)]

    out_condition = sheet[out_condition_col]
    in_condition = sheet[in_condition_col]

    last_condition_col_out = out_condition.iloc[:,epoch_number]
    last_condition_col_in = in_condition.iloc[:,epoch_number]

    last_condition_col_out = last_condition_col_out.tolist()
    last_condition_col_in = last_condition_col_in.tolist()


    return last_condition_col_in, last_condition_col_out


def delta_scaling(conv_size_list,delta_threshold,mapping_threshold,min_scale_limit,num_trials,shortcut_indexes,last_operation,factor_scale,delta_percentage):
    #print('GLOBALS EXCEL PATH IN DELTA_SCALING FUNCTION:{}'.format(GLOBALS.EXCEL_PATH))
    input_ranks_final,output_ranks_final = get_info('rank',path=GLOBALS.EXCEL_PATH,epoch_number=-1)
    input_ranks_stable,output_ranks_stable = get_info('rank',path=GLOBALS.EXCEL_PATH,epoch_number=GLOBALS.CONFIG['stable_epoch'])
    in_conditions,out_conditions = get_info('condition',path=GLOBALS.EXCEL_PATH,epoch_number=-1)
    rank_averages_final=calculate_correct_output_sizes(input_ranks_final, output_ranks_final, conv_size_list, shortcut_indexes, GLOBALS.CONFIG['delta_threshold'],final=False)[1]
    rank_averages_stable=calculate_correct_output_sizes(input_ranks_stable,output_ranks_stable, conv_size_list, shortcut_indexes, GLOBALS.CONFIG['delta_threshold'],final=False)[1]

    mapping_conditions=convert_format(out_conditions,shortcut_indexes)
    mapping_conditions[0] = [out_conditions[0]]+mapping_conditions[0]

    EXPAND,SHRINK,STOP = 1,-1,0
    new_channel_sizes=copy.deepcopy(conv_size_list)

    FIRST_TIME=False

    slope_averages=[]
    if last_operation==[]:
        FIRST_TIME = True
        for i in conv_size_list:
            factor_scale.append([GLOBALS.CONFIG['factor_scale']]*len(i))
            last_operation.append([1]*len(i))
            delta_percentage.append([0]*len(i))

    for superblock in range(len(new_channel_sizes)):
        for layer in range(0,len(new_channel_sizes[superblock])):
            if (last_operation[superblock][layer] == STOP):
                continue
            #delta_percentage[superblock][layer] = round((rank_averages_final[superblock][layer]-rank_averages_stable[superblock][layer])/rank_averages_final[superblock][layer],5)
            epoch_num=[i for i in range(GLOBALS.CONFIG['epochs_per_trial'])]
            yaxis=[]
            for k in range(GLOBALS.CONFIG['epochs_per_trial']):
                input_ranks,output_ranks=get_info('rank',path=GLOBALS.EXCEL_PATH,epoch_number=k)
                rank_averages=calculate_correct_output_sizes(input_ranks, output_ranks, conv_size_list, shortcut_indexes, 0.1,final=False)[1]
                yaxis+=[rank_averages[superblock][layer]]
            break_point = adaptive_stop(epoch_num,yaxis,0.005,4)
            delta_percentage[superblock][layer] = slope(yaxis,break_point)

            #CHANNEL SIZE MANIPULATION
            current_operation=EXPAND
            if ((delta_percentage[superblock][layer]<delta_threshold) or (mapping_conditions[superblock][layer] >= mapping_threshold)) and (conv_size_list[superblock][layer] > GLOBALS.CONFIG['min_conv_size']):
                current_operation = SHRINK

            #KERNEL SIZE MANIPULATION


            '''
            if (delta_percentage[superblock][layer] >= delta_threshold):
                current_operation = EXPAND
            elif (conv_size_list[superblock][layer] > GLOBALS.CONFIG['min_conv_size']):
                current_operation = SHRINK

            if (mapping_conditions[superblock][layer] >= mapping_threshold) and (conv_size_list[superblock][layer] > GLOBALS.CONFIG['min_conv_size']):
                current_operation = SHRINK

            if (current_operation==None):
                current_operation = EXPAND
            '''

            if (last_operation[superblock][layer] != current_operation and FIRST_TIME==False):
                if (factor_scale[superblock][layer] < min_scale_limit):
                    current_operation = STOP
                factor_scale[superblock][layer] = factor_scale[superblock][layer]/2

            last_operation[superblock][layer] = current_operation

            new_channel_sizes[superblock][layer] = even_round(conv_size_list[superblock][layer] * (1 + factor_scale[superblock][layer]*last_operation[superblock][layer]))

    print(factor_scale,'FACTOR SCALE')
    print(conv_size_list, 'OLD OUTPUT CONV SIZE LIST')
    print(new_channel_sizes,'NEW OUTPUT CONV SIZE LIST')

    return last_operation,factor_scale,new_channel_sizes,new_kernel_sizes,delta_percentage, rank_averages_final, rank_averages_stable

def calculate_correct_output_sizes_averaged(input_ranks,output_ranks,conv_size_list,shortcut_indexes,threshold):
    output_ranks_layer_1 = output_ranks[0]
    scaling_factor=[0,0,0,0]

    output_ranks_superblock_1 = output_ranks[1:shortcut_indexes[0]]
    output_ranks_superblock_2 = output_ranks[shortcut_indexes[0]+1:shortcut_indexes[1]]
    output_ranks_superblock_3 = output_ranks[shortcut_indexes[1]+1:shortcut_indexes[2]]
    output_ranks_superblock_4 = output_ranks[shortcut_indexes[2]+1:shortcut_indexes[3]]

    super_block_1_val=conv_size_list[0][0]
    super_block_2_val=conv_size_list[1][0]
    super_block_3_val=conv_size_list[2][0]
    super_block_4_val=conv_size_list[3][0]

    scaling_factor[0] = np.average(output_ranks_superblock_1)-threshold
    scaling_factor[1] = np.average(output_ranks_superblock_2)-threshold
    scaling_factor[2] = np.average(output_ranks_superblock_3)-threshold
    scaling_factor[3] = np.average(output_ranks_superblock_4)-threshold

    super_block_1 = [even_round(super_block_1_val*(1+scaling_factor[0]))] * (len(output_ranks_superblock_1)+1)
    super_block_2 = [even_round(super_block_2_val*(1+scaling_factor[1]))] * len(output_ranks_superblock_2)
    super_block_3 = [even_round(super_block_3_val*(1+scaling_factor[2]))] * len(output_ranks_superblock_3)
    super_block_4 = [even_round(super_block_4_val*(1+scaling_factor[3]))] * len(output_ranks_superblock_4)

    output_conv_size_list=[super_block_1]+[super_block_2]+[super_block_3]+[super_block_4]
    print(output_conv_size_list)

    return output_conv_size_list


def convert_format(full_list, temp_shortcut_indexes):
    final=[]
    shortcut_indexes=[0]+temp_shortcut_indexes+[len(full_list)]
    for i in range(0,len(shortcut_indexes)-1,1):
        final+=[full_list[shortcut_indexes[i]+1:shortcut_indexes[i+1]]]
        #print(final,'final in covert_format loop')
    return final

def calculate_correct_output_sizes(input_ranks,output_ranks,conv_size_list,shortcut_indexes,threshold,final=True):
    #Note that input_ranks/output_ranks may have a different size than conv_size_list
    #threshold=GLOBALS.CONFIG['adapt_rank_threshold']

    new_input_ranks=[]
    new_output_ranks=[]

    new_input_ranks=convert_format(input_ranks,shortcut_indexes)
    new_output_ranks=convert_format(output_ranks,shortcut_indexes)

    #new_input_ranks = [input_ranks_superblock_1] + [input_ranks_superblock_2] + [input_ranks_superblock_3] + [input_ranks_superblock_4] + [input_ranks_superblock_5]
    #new_output_ranks = [output_ranks_superblock_1] + [output_ranks_superblock_2] + [output_ranks_superblock_3] + [output_ranks_superblock_4] + [output_ranks_superblock_5]

    #print(new_input_ranks,'INPUT RANKS WITHOUT SHORTCUTS')
    #print(new_output_ranks,'OUTPUT RANKS WITHOUT SHORTCUTS')

    block_averages=[]
    block_averages_input=[]
    block_averages_output=[]
    grey_list_input=[]
    grey_list_output=[]
    increment = 0
    if GLOBALS.BLOCK_TYPE=='BasicBlock':
        increment=2
    elif GLOBALS.BLOCK_TYPE=='Bottleneck':
        increment=3

    for i in range(0,len(new_input_ranks),1):
        block_averages+=[[]]
        block_averages_input+=[[]]
        block_averages_output+=[[]]
        grey_list_input+=[[]]
        grey_list_output+=[[]]
        temp_counter=0

        for j in range(1,len(new_input_ranks[i]),increment):

            block_averages_input[i]=block_averages_input[i]+[new_input_ranks[i][j]]
            if GLOBALS.BLOCK_TYPE=='Bottleneck':
                block_averages_input[i]=block_averages_input[i]+[new_input_ranks[i][j+1]]
            block_averages_output[i]=block_averages_output[i]+[new_output_ranks[i][j-1]]

            if GLOBALS.BLOCK_TYPE=='Bottleneck':
                block_averages_output[i]=block_averages_output[i]+[new_output_ranks[i][j]]
                grey_list_output[i]=grey_list_output[i]+[new_output_ranks[i][j+1]]
            else:
                grey_list_output[i]=grey_list_output[i]+[new_output_ranks[i][j]]

            grey_list_input[i]=grey_list_input[i]+[new_input_ranks[i][j-1]]


        block_averages_input[i]=block_averages_input[i]+[np.average(np.array(grey_list_input[i]))]
        block_averages_output[i]=block_averages_output[i]+[np.average(np.array(grey_list_output[i]))]
        block_averages[i]=np.average(np.array([block_averages_input[i],block_averages_output[i]]),axis=0)

    #print(conv_size_list,'CONV SIZE LIST')
    output_conv_size_list=copy.deepcopy(conv_size_list)
    rank_averages = copy.deepcopy(conv_size_list)
    '''
    self.superblock1_indexes_50=['32',32,32,'32',32,32,'32',32,32,'32']
    self.superblock2_indexes_50=[32, 32,'32',32, 32, '32',32, 32, '32',32, 32, '32',]
    self.superblock3_indexes_50=[32,32,'32',32,32,'32',32,32,'32',32,32,'32',32,32,'32',32,32,'32']
    self.superblock4_indexes_50=[32,32,'32',32,32,'32',32,32,'32']
    '''

    count=0
    for i in range(0,len(block_averages)):
        for j in range(0,len(conv_size_list[i])):
            if (i==0):
                if (j%increment==0):
                    scaling_factor=block_averages[i][-1]-threshold
                else:
                    scaling_factor=block_averages[i][count]-threshold
                    count+=1
            else:
                if ((j+1)%increment==0):
                    scaling_factor=block_averages[i][-1]-threshold
                else:
                    scaling_factor=block_averages[i][count]-threshold
                    count+=1
            output_conv_size_list[i][j]=even_round(output_conv_size_list[i][j]*(1+scaling_factor))
            rank_averages[i][j] = scaling_factor + threshold
        count=0

    if final==True:
        GLOBALS.super1_idx = output_conv_size_list[0]
        GLOBALS.super2_idx = output_conv_size_list[1]
        GLOBALS.super3_idx = output_conv_size_list[2]
        GLOBALS.super4_idx = output_conv_size_list[3]
        GLOBALS.index = output_conv_size_list[0] + output_conv_size_list[1] + output_conv_size_list[2] + output_conv_size_list[3]

    #print(output_conv_size_list,'OUTPUT CONV SIZE LIST')
    return output_conv_size_list, rank_averages
