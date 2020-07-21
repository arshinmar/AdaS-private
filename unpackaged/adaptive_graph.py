import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_vars as GLOBALS
def compile_adaptive_files(file_name,num_trials):
    #CHANGE THIS VALUE FOR NUMBER OF TRIALS
    num_trials=num_trials
    adaptive_set=[]
    manipulate_index=file_name.find('trial')+6
    for trial_num in range (0,num_trials):
        adaptive_set.append(file_name[0:manipulate_index]+str(trial_num)+file_name[manipulate_index+1:])
    return adaptive_set

def create_adaptive_graphs(file_name,num_epochs,num_trials):
    #CHANGE THIS VALUE FOR NUMBER OF EPOCHS PER TRIAL
    total_num_epochs=num_epochs
    accuracies=[]
    epoch_num=[]
    count=0
    new_trial_indic='*'

    adaptive_set=compile_adaptive_files(file_name,num_trials)
    #print(adaptive_set,'adaptive_set')

    for trial in adaptive_set:
        dfs=pd.read_excel(trial)
        #print(dfs)
        for epoch in range (0,total_num_epochs):
            epoch_num.append(str(epoch+count)+new_trial_indic)
            accuracies.append(dfs['test_acc_epoch_'+str(epoch)][0]*100)
            new_trial_indic=''
        count+=total_num_epochs
        new_trial_indic='*'
    #print(epoch_num)
    #print(accuracies)
    fig=plt.figure()
    fig=plt.plot(epoch_num,accuracies, label='accuracy vs epoch', marker='o', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('AdaptiveNet: Test Accuracy vs Epoch (init_conv_size='+GLOBALS.CONFIG['init_conv_setting']+' thresh='+GLOBALS.CONFIG['adapt_rank_threshold']+')')
    plt.savefig('graph_files/accuracy_plot.png')
    #plt.show()

#create_adaptive_graphs()

def create_layer_plot(file_name,num_trials):
    layers_info=pd.read_excel(file_name)
    layers_size_list=[]

    for i in range(len(layers_info.iloc[:,0].to_numpy())):
        main=layers_info.iloc[i,1:].to_numpy()
        layers_size_list+=[main]

    barWidth=0.5
    layers_list=[[6,12,18,24,30]]
    for i in range(1,len(layers_size_list),1):
        temp=[x + barWidth for x in layers_list[i-1]]
        layers_list+=[temp]

    colors=['#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045']
    plt.figure()
    for i in range(0,len(layers_list),1):
        plt.bar(layers_list[i],layers_size_list[i],color=colors[i],width=barWidth, edgecolor='white',label=str('Trial '+str(i+1)))

    plt.xlabel('SuperBlock',fontweight='bold')
    plt.ylabel('Layer Size',fontweight='bold')
    plt.title('AdaptiveNet: Evolution of Layer Size Vs Trial (init_conv_size='+GLOBALS.CONFIG['init_conv_setting']+' thresh='+GLOBALS.CONFIG['adapt_rank_threshold']+')')
    plt.xticks([6*r + 5*barWidth + 3 + num_trials*0.3 for r in range(len(layers_size_list[0]))], [str(i) for i in range(len(layers_size_list[0]))])

    plt.legend(loc='upper right')
    plt.savefig('graph_files/'+'Layer_Size_Plot.png')

create_layer_plot('adapted_architectures/adapted_architectures.xlsx',10)
