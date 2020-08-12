import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_name='adapted_architectures.xlsx'
def create_layer_plot(file_name):
    layers_info=pd.read_excel(file_name)
    layers_size_list=[]
    #print(layers_info.iloc[0,1:].to_numpy(), 'wefuihfw')

    for i in range(len(layers_info.iloc[:,0].to_numpy())):
        main=layers_info.iloc[i,1:].to_numpy()
        #print(main,'dun')
        layers_size_list+=[main]

    #print(layers_size_list)

    barWidth=0.5
    layers_list=[[6,12,18,24,30]]
    for i in range(1,len(layers_size_list),1):
        temp=[x + barWidth for x in layers_list[i-1]]
        layers_list+=[temp]
    #print(layers_list)

    colors=['#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045']

    #print(len(layers_list),'ewqoinefwiu')
    for i in range(0,len(layers_list),1):
        plt.bar(layers_list[i],layers_size_list[i],color=colors[i],width=barWidth, edgecolor='white',label=str('Trial '+str(i+1)))

    plt.xlabel('SuperBlock',fontweight='bold')
    plt.ylabel('Layer Size',fontweight='bold')
    plt.title('AdaptiveNet: Evolution of Layer Size Vs Trial')
    plt.xticks([6*r + 5*barWidth + 6 for r in range(len(layers_size_list[0]))], [str(i) for i in range(len(layers_size_list[0]))])

    plt.legend(loc='upper right')
    plt.show()
    ##print(superblock_1)

create_layer_plot(file_name)
