import time
import copy
import pandas as pd
import numpy as np
import global_vars as GLOBALS
from models.own_network import DASNet34,DASNet50
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

def get_info_singular(info,path = GLOBALS.EXCEL_PATH, epoch_number = -1):
    '''
    - Read from .adas-output excel file
    - Get specified column

    return: 1D list
    '''
    sheet = pd.read_excel(path,index_col=0)
    item_col = [col for col in sheet if col.startswith(info)]
    item_col = sheet[item_col]
    last_item = item_col.iloc[:,epoch_number]
    last_item = last_item.tolist()
    return last_item

def convert_format(full_list, temp_shortcut_indexes):
    final=[]
    shortcut_indexes=[0]+temp_shortcut_indexes+[len(full_list)]
    #shortcut_indexes=[0,7,16,29,36]?
    for i in range(0,len(shortcut_indexes)-1,1):
        final+=[full_list[shortcut_indexes[i]+1:shortcut_indexes[i+1]]]
        #final=[[7 elements],[]]
        #print(final,'final in covert_format loop')
    return final

def retrieve_layer_ranks(conv_size_list,shortcut_indexes,superblock,layer):
    epoch_num=[i for i in range(GLOBALS.CONFIG['epochs_per_trial'])]
    yaxis=[]
    yaxis_kernel=[]
    for k in range(GLOBALS.CONFIG['epochs_per_trial']):
        input_ranks,output_ranks=get_info('rank',path=GLOBALS.EXCEL_PATH,epoch_number=k)
        if GLOBALS.CONFIG['kernel_adapt'] == 1:
            kernel_ranks_flat = get_info_singular('mode12_rank',path=GLOBALS.EXCEL_PATH,epoch_number=k)
            kernel_ranks = convert_format(kernel_ranks_flat,shortcut_indexes)
            kernel_ranks[0] = [kernel_ranks_flat[0]]+kernel_ranks[0]
            yaxis_kernel+=[kernel_ranks[superblock][layer]]
        rank_averages=calculate_correct_output_sizes(input_ranks, output_ranks, conv_size_list, shortcut_indexes, 0.1,final=False)[1]
        yaxis+=[rank_averages[superblock][layer]]
    return epoch_num,yaxis,yaxis_kernel

def adjust_gray_values(new_channel_sizes,increment,gray_values):
    gray_averages=[]
    for superblock in range(len(gray_values)):
        gray_averages+=[even_round(np.average(gray_values[superblock]))]

    for superblock in range(len(new_channel_sizes)):
        if superblock==0:
            starter=0
        else:
            starter=1
        for layer in range(starter,len(new_channel_sizes[superblock]),increment):
            new_channel_sizes[superblock][layer]=gray_averages[superblock]
    return new_channel_sizes,gray_averages

def delta_scaling(conv_size_list,kernel_size_list,shortcut_indexes,last_operation,factor_scale,delta_percentage,last_operation_kernel,factor_scale_kernel,delta_percentage_kernel,parameter_type='channel'):
    #print('GLOBALS EXCEL PATH IN DELTA_SCALING FUNCTION:{}'.format(GLOBALS.EXCEL_PATH))
    print('*****MIN KERNEL SIZE USED IN DELTA SCALING: {}'.format(GLOBALS.CONFIG['min_kernel_size']))

    input_ranks_final,output_ranks_final = get_info('rank',path=GLOBALS.EXCEL_PATH,epoch_number=-1)
    input_ranks_stable,output_ranks_stable = get_info('rank',path=GLOBALS.EXCEL_PATH,epoch_number=GLOBALS.CONFIG['stable_epoch'])
    in_conditions,out_conditions = get_info('condition',path=GLOBALS.EXCEL_PATH,epoch_number=-1)
    kernel_rank_final_flat=get_info_singular('mode12_rank',path=GLOBALS.EXCEL_PATH,epoch_number=-1)
    kernel_rank_stable_flat=get_info_singular('mode12_rank',path=GLOBALS.EXCEL_PATH,epoch_number=0)
    '---------------------------------------------------------------------------------------------------------------------------'
    rank_averages_final=calculate_correct_output_sizes(input_ranks_final, output_ranks_final, conv_size_list, shortcut_indexes, GLOBALS.CONFIG['delta_threshold'],final=False)[1]
    rank_averages_stable=calculate_correct_output_sizes(input_ranks_stable,output_ranks_stable, conv_size_list, shortcut_indexes, GLOBALS.CONFIG['delta_threshold'],final=False)[1]
    '---------------------------------------------------------------------------------------------------------------------------'
    mapping_conditions=convert_format(out_conditions,shortcut_indexes)
    mapping_conditions[0] = [out_conditions[0]]+mapping_conditions[0]
    '---------------------------------------------------------------------------------------------------------------------------'

    delta_threshold = GLOBALS.CONFIG['delta_threshold']
    delta_threshold_kernel = GLOBALS.CONFIG['delta_threshold_kernel']
    mapping_threshold = GLOBALS.CONFIG['mapping_condition_threshold']
    min_scale_limit = GLOBALS.CONFIG['min_scale_limit']

    EXPAND,SHRINK,STOP = 1,-1,0
    new_channel_sizes=copy.deepcopy(conv_size_list)
    new_kernel_sizes=copy.deepcopy(kernel_size_list)

    if GLOBALS.BLOCK_TYPE=='BasicBlock':
        increment=2
    elif GLOBALS.BLOCK_TYPE=='Bottleneck':
        increment=3

    #Initialize Parameters
    FIRST_TIME=False
    slope_averages=[]
    gray_values=[]
    gray_averages=[]
    for i in conv_size_list:
        gray_values+=[[]]
    if last_operation==[]:
        FIRST_TIME = True
        for i in conv_size_list:
            factor_scale.append([GLOBALS.CONFIG['factor_scale']]*len(i))
            last_operation.append([1]*len(i))
            delta_percentage.append([0]*len(i))

            factor_scale_kernel.append([GLOBALS.CONFIG['factor_scale_kernel']]*len(i))
            last_operation_kernel.append([-1]*len(i))
            delta_percentage_kernel.append([0]*len(i))

    #ITERATE THROUGH LAYERS
    for superblock in range(len(new_channel_sizes)):
        for layer in range(0,len(new_channel_sizes[superblock])):
            channel_stop = False
            kernel_stop = False

            #CHANGE MIN KERNEL SIZE HALF WAY TO SECOND MIN
            if (GLOBALS.CONFIG['min_kernel_size']==GLOBALS.CONFIG['min_kernel_size_2'] and kernel_size_list[superblock][layer]==GLOBALS.min_kernel_size_1):
                last_operation_kernel[superblock][layer]=-2

            #DO NOT GIVE IT THE OPTION TO EXPAND/SHRINK WHEN THE LAST OPERATION WAS SET TO STOP
            if (last_operation[superblock][layer] == STOP):
                channel_stop = True
                current_operation=STOP
            if (last_operation_kernel[superblock][layer] == STOP):
                kernel_stop = True
                current_operation_kernel=STOP

            epoch_num,yaxis,yaxis_kernel=retrieve_layer_ranks(conv_size_list,shortcut_indexes,superblock,layer)
            '-----------------------------------------------CHANNEL ADAPT--------------------------------------------------------------------------------'
            if (parameter_type=='channel' or parameter_type=='both'):
                break_point = adaptive_stop(epoch_num,yaxis,0.005,4)
                delta_percentage[superblock][layer] = slope(yaxis,break_point)
                #CHANNEL SIZE OPERATION
                if (channel_stop==False):
                    current_operation=EXPAND
                    if (((delta_percentage[superblock][layer]<delta_threshold) or (mapping_conditions[superblock][layer] >= mapping_threshold)) and (conv_size_list[superblock][layer] > GLOBALS.CONFIG['min_conv_size'])) or (conv_size_list[superblock][layer]>=GLOBALS.CONFIG['max_conv_size']):
                        current_operation = SHRINK
                #CHANNEL SIZE FACTOR
                if (last_operation[superblock][layer] != current_operation and FIRST_TIME==False):
                    factor_scale[superblock][layer] = factor_scale[superblock][layer]/2
                #CHANNEL SIZE STOP
                if (factor_scale[superblock][layer] < min_scale_limit):
                    current_operation = STOP
                #ASSIGN LAST OPERATION CHANNEL
                last_operation[superblock][layer] = current_operation
                #CALCULATE NEW SIZES
                new_channel_sizes[superblock][layer] = even_round(conv_size_list[superblock][layer] * (1 + factor_scale[superblock][layer]*last_operation[superblock][layer]))
                new_kernel_sizes[superblock][layer] = kernel_size_list[superblock][layer]
            '-----------------------------------------------KERNEL ADAPT--------------------------------------------------------------------------------'
            if (GLOBALS.CONFIG['kernel_adapt'] == 1 and (parameter_type=='kernel' or parameter_type=='both')):
                break_point_kernel = adaptive_stop(epoch_num,yaxis_kernel,0.005,4)
                delta_percentage_kernel[superblock][layer] = slope(yaxis_kernel,break_point_kernel)
                #KERNEL SIZE OPERATION
                if (kernel_stop==False):
                    current_operation_kernel=EXPAND
                    if ((delta_percentage_kernel[superblock][layer] < delta_threshold_kernel) and (kernel_size_list[superblock][layer] > GLOBALS.CONFIG['min_kernel_size'])) or (kernel_size_list[superblock][layer]>=GLOBALS.CONFIG['max_kernel_size']):
                        current_operation_kernel = SHRINK
                #KERNEL SIZE FACTOR
                if (last_operation_kernel[superblock][layer] != current_operation_kernel and FIRST_TIME==False):
                    factor_scale_kernel[superblock][layer] = factor_scale_kernel[superblock][layer]/2
                #KERNEL SIZE STOP
                if (factor_scale_kernel[superblock][layer] <= GLOBALS.CONFIG['factor_scale_kernel']/32): #If the operation has alternated 3 times
                    current_operation_kernel = STOP
                #ASSIGN LAST OPERATION KERNEL
                last_operation_kernel[superblock][layer] = current_operation_kernel
                #CALCULATE NEW SIZES
                new_kernel_sizes[superblock][layer] = int(kernel_size_list[superblock][layer] + (last_operation_kernel[superblock][layer]*2))
                new_channel_sizes[superblock][layer] = conv_size_list[superblock][layer]
            '-----------------------------------------------STORE CORRECT GRAY VALUES--------------------------------------------------------------------------------'
            if (superblock==0):
                if layer%increment==0:
                    gray_values[superblock]+=[new_channel_sizes[superblock][layer]]
            else:
                if (layer-1)%increment==0:
                    gray_values[superblock]+=[new_channel_sizes[superblock][layer]]
            '----------------------------------------------------------------------------------------------------------------------------------------'

    new_channel_sizes,gray_averages=adjust_gray_values(new_channel_sizes,increment,gray_values)

    print('------------------------------------------------------------------------------------------------')
    print(gray_values, 'GRAY VALUES')
    print(gray_averages, 'GRAY AVERAGES')
    print('------------------------------------------------------------------------------------------------')
    print(factor_scale,'FACTOR SCALE')
    print(conv_size_list, 'OLD CONV SIZE LIST')
    print(new_channel_sizes,'NEW CONV SIZE LIST')
    print('------------------------------------------------------------------------------------------------')
    print(factor_scale_kernel,'FACTOR SCALE KERNEL')
    print(kernel_size_list, 'OLD KERNEL SIZE LIST')
    print(new_kernel_sizes,'NEW KERNEL SIZE LIST')
    print('------------------------------------------------------------------------------------------------')

    return last_operation, last_operation_kernel, factor_scale, factor_scale_kernel, new_channel_sizes,new_kernel_sizes, delta_percentage, delta_percentage_kernel, rank_averages_final, rank_averages_stable

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
    #print(increment, 'INCREMENT IN calculate_correct_output_sizes')
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

    #print(output_conv_size_list, "OUTPUT CONV SIZE LIST IN CALC_OUTPUT_SIZES FUNCTION~~~~~~~~~~~~~~~~~~~~~~!!!")

    if final==True:
        GLOBALS.super1_idx = output_conv_size_list[0]
        GLOBALS.super2_idx = output_conv_size_list[1]
        GLOBALS.super3_idx = output_conv_size_list[2]
        GLOBALS.super4_idx = output_conv_size_list[3]
        GLOBALS.index = output_conv_size_list[0] + output_conv_size_list[1] + output_conv_size_list[2] + output_conv_size_list[3]

    #print(output_conv_size_list,'OUTPUT CONV SIZE LIST')
    return output_conv_size_list, rank_averages
