import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_vars as GLOBALS
import ast
import copy
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
def compile_adaptive_files(file_name,num_trials):
    #CHANGE THIS VALUE FOR NUMBER OF TRIALS
    num_trials=num_trials
    adaptive_set=[]
    manipulate_index=file_name.find('trial')+6
    try:
        blah=int(file_name[manipulate_index+1])
        shift=2
    except:
        shift=1
    for trial_num in range (0,num_trials):
        adaptive_set.append(file_name[0:manipulate_index]+str(trial_num)+file_name[manipulate_index+shift:])

    return adaptive_set

def create_adaptive_graphs(file_name,num_epochs,num_trials,out_folder):
    #CHANGE THIS VALUE FOR NUMBER OF EPOCHS PER TRIAL
    total_num_epochs=num_epochs
    accuracies=[]
    epoch_num=[]
    count=0

    adaptive_set=compile_adaptive_files(file_name,num_trials)
    #print(adaptive_set,'adaptive_set')
    for trial in adaptive_set:
        dfs=pd.read_excel(trial)
        #print(dfs)
        for epoch in range (0,total_num_epochs):
            epoch_num.append(epoch+count)
            accuracies.append(dfs['test_acc_epoch_'+str(epoch)][0]*100)
            new_trial_indic=''
        count+=total_num_epochs
#    print(epoch_num)
#    print(accuracies)

    fig=plt.plot(epoch_num,accuracies, label='accuracy vs epoch', marker='o', color='r')
    #figure=plt.gcf()
    #figure.set_size_inches(16, 9)
    plt.xticks(np.arange(min(epoch_num), max(epoch_num)+1, total_num_epochs))
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Dynamic AdaptiveNet: Test Accuracy vs Epoch (init_conv_size='+GLOBALS.CONFIG['init_conv_setting']+' thresh='+str(GLOBALS.CONFIG['adapt_rank_threshold'])+')')
    plt.savefig(out_folder+'\\'+'dynamic_accuracy_plot.png',bbox_inches='tight')
    #plt.show()

#create_adaptive_graphs()

def remove_brackets(value):
    check=']'
    val=''
    for i in range(0,len(value),1):
        if i==len(value)-1:
            val+=']'
            break
        if value[i]==check:
            if check==']':
                check='['
                if value[i+1]==check:
                    val+=', '
                    i+=2
            else:
                check=']'
        else:
            val+=value[i]
    return val

def create_layer_plot(file_name,num_trials,path,evo_type):
    layers_info=pd.read_excel(file_name)
    layers_size_list=[]

    for i in range(len(layers_info.iloc[:,0].to_numpy())):
        temp=''
        main=layers_info.iloc[i,1:].to_numpy()
        for i in main:
            temp+=i[:]
        temp=ast.literal_eval(remove_brackets(temp))
        layers_size_list+=[temp]

    barWidth=0.5
    layers_list=[[]]
    if num_trials<=10:
        mult_val,temp_val=6,5
        for i in range(1,32,1):
            layers_list[0]+=[6*i]
    else:
        mult_val,temp_val=12,10
        for i in range(1,32,1):
            layers_list[0]+=[12*i]

    for i in range(1,len(layers_size_list),1):
        temp=[x + barWidth for x in layers_list[i-1]]
        layers_list+=[temp]

    colors=['#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045']
    plt.figure()
    for i in range(0,len(layers_list),1):
        plt.bar(layers_list[i],layers_size_list[i],color=colors[i],width=barWidth, edgecolor='white',label=str('Trial '+str(i+1)))

    plt.xlabel('SuperBlock',fontweight='bold')
    plt.ylabel('Layer Size',fontweight='bold')
    plt.title('AdaptiveNet:' + evo_type + ' Evolution w.r.t. Trial (init_conv_size='+GLOBALS.CONFIG['init_conv_setting']+' thresh='+str(GLOBALS.CONFIG['adapt_rank_threshold'])+')')
    if num_trials<=10:
        plt.xticks([mult_val*r + temp_val*barWidth + 3 + num_trials*0.3 for r in range(len(temp))], [str(i) for i in range(len(temp))])
    else:
        plt.xticks([mult_val*r + temp_val*barWidth + 6 + num_trials*0.3 for r in range(len(temp))], [str(i) for i in range(len(temp))])

    plt.legend(loc='upper right')
    figure=plt.gcf()
    figure.set_size_inches(11.4, 5.34)
    #addition=str(GLOBALS.CONFIG['adapt_rank_threshold'])+'_conv_size='+GLOBALS.CONFIG['init_conv_setting']+'_epochpertrial='+str(GLOBALS.CONFIG['epochs_per_trial'])+'_beta='+str(GLOBALS.CONFIG['beta'])
    plt.savefig(path,bbox_inches='tight')
    return True


def even_round(number):
    return int(round(number/2)*2)

def rank_line_graph(conv_size_list,shortcut_indexes,path=GLOBALS.EXCEL_PATH):
    layer_num,superblock=0,0
    yaxis=[]
    for i in range(25):
        input_ranks,output_ranks=get_ranks(path=path,epoch_number=i)
        rank_averages=calculate_correct_output_sizes(input_ranks, output_ranks, conv_size_list, shortcut_indexes, 0.1,final=False)[1]
        yaxis+=[rank_averages[superblock][layer_num]]
    epoch_num=[i for i in range(25)]
    fig=plt.plot(epoch_num,yaxis, label='ranks vs epoch', marker='o', color='r')
    plt.show()
    return True

#shortcut_indexes=[7,14,21,28]
#conv_size_list=[[32, 32, 32, 32, 32, 32, 32],[32, 32, 32, 32, 32, 32],[32, 32, 32, 32, 32, 32],[32, 32, 32, 32, 32, 32],[32, 32, 32, 32, 32, 32]]
#rank_line_graph(conv_size_list,shortcut_indexes,path='AdaS_adapt_trial=0_net=AdaptiveNet_0.1_dataset=CIFAR10.xlsx')
