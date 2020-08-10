import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_layer_plot(file_name,num_trials):
    layers_info=pd.read_excel(file_name)
    print(layers_info.head())
    layers_size_list=[[]]

    counter=0
    for i in range(len(layers_info.iloc[:,0].to_numpy())):
        if counter==0:
            counter+=1
            continue
        main=layers_info.iloc[i,1:].to_numpy()
        print(main)
        layers_size_list+=[main]

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
    print(layers_list, 'wefo0[ihfwiuh]')
    print(layers_size_list, 'LAYERS SIZE LIST')

    colors=['#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045']
    plt.figure()
    for i in range(0,len(layers_list),1):
        plt.bar(layers_list[i],layers_size_list[i],color=colors[i],width=barWidth, edgecolor='white',label=str('Trial '+str(i+1)))

    plt.xlabel('SuperBlock',fontweight='bold')
    plt.ylabel('Layer Size',fontweight='bold')
    #plt.title('AdaptiveNet: Evolution of Layer Size Vs Trial (init_conv_size='+GLOBALS.CONFIG['init_conv_setting']+' thresh='+str(GLOBALS.CONFIG['adapt_rank_threshold'])+')')
    if num_trials<=10:
        plt.xticks([mult_val*r + temp_val*barWidth + 3 + num_trials*0.3 for r in range(len(layers_size_list[0]))], [str(i) for i in range(len(layers_size_list[0]))])
    else:
        plt.xticks([mult_val*r + temp_val*barWidth + 6 + num_trials*0.3 for r in range(len(layers_size_list[0]))], [str(i) for i in range(len(layers_size_list[0]))])

    plt.legend(loc='upper right')
    figure=plt.gcf()
    figure.set_size_inches(25, 9)
    plt.show()
    #plt.savefig('graph_files/Layer_Size_Plot_thresh='+str(GLOBALS.CONFIG['adapt_rank_threshold'])+'_conv_size='+GLOBALS.CONFIG['init_conv_setting']+'_epochpertrial='+str(GLOBALS.CONFIG['epochs_per_trial'])+'_beta='+str(GLOBALS.CONFIG['beta'])+'.png',bbox_inches='tight')

    return True

create_layer_plot('adapted_architectures/adapted_architectures_6464646464_thresh0.29.xlsx',1)
