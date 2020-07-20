import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_name='adapted_architectures/adapted_architectures (1).xlsx'
def create_layer_plot(file_name):
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

    colors=['#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045','#4d4d4e','#b51b1b','#1f639b','#1bb5b5','#fcb045']

    for i in range(0,len(layers_list),1):
        plt.bar(layers_list[i],layers_size_list[i],color=colors[i],width=barWidth, edgecolor='white',label=str('Trial '+str(i+1)))

    plt.xlabel('SuperBlock',fontweight='bold')
    plt.ylabel('Layer Size',fontweight='bold')
    plt.title('AdaptiveNet: Evolution of Layer Size Vs Trial')
    plt.xticks([6*r + 5*barWidth + 6 for r in range(len(layers_size_list[0]))], [str(i) for i in range(len(layers_size_list[0]))])

    plt.legend(loc='upper right')
    plt.savefig('graph_files/'+'VGG_knowledge_gain_graph_scaled.png')

create_layer_plot(file_name)
