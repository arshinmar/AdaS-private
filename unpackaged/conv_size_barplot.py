import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def create_layer_barplot(file_name):

    layers_info=pd.read_excel(file_name)
    layers=[]
    temp_counter=0

    superblock_1=layers_info.iloc[:,0].to_numpy()
    superblock_2=layers_info.iloc[:,1].to_numpy()
    superblock_3=layers_info.iloc[:,2].to_numpy()
    superblock_4=layers_info.iloc[:,3].to_numpy()
    superblock_5=layers_info.iloc[:,4].to_numpy()

    for i in layers_info.iloc[:,0]:
        temp_counter+=1
        layers+=[temp_counter]

    layers=np.array(layers)


    superblock_1=layers_info.iloc[:,0].to_numpy()


    layers=layers_info.iloc[1:,0].to_numpy()
    barWidth=0.5
    #X-data, separated by Bar Width
    vgg_r0_baseline = 4*vgg_layers
    vgg_r1_baseline = [x + barWidth for x in vgg_r0_baseline]
    vgg_r2_baseline = [x + barWidth for x in vgg_r1_baseline]
    vgg_r3_baseline = [x + barWidth for x in vgg_r2_baseline]
    vgg_r4_baseline = [x + barWidth for x in vgg_r3_baseline]
    vgg_r5_baseline = [x + barWidth for x in vgg_r4_baseline]


    #VGG plot
    plt.bar(vgg_r0_baseline, vgg_data[0], color='#4d4d4e', width=barWidth, edgecolor='white', label='Baseline')
    plt.bar(vgg_r1_baseline, vgg_data[1], color='#b51b1b', width=barWidth, edgecolor='white', label='Scaled 20%')
    plt.bar(vgg_r2_baseline, vgg_data[2], color='#1f639b', width=barWidth, edgecolor='white', label='Scaled 40%')
    plt.bar(vgg_r3_baseline, vgg_data[3], color='#1bb5b5', width=barWidth, edgecolor='white', label='Scaled 60%')
    plt.bar(vgg_r4_baseline, vgg_data[4], color='#fcb045', width=barWidth, edgecolor='white', label='Scaled 80%')
    plt.bar(vgg_r5_baseline, vgg_data[5], color='#aaaaaa', width=barWidth, edgecolor='white', label='Scaled 100%')

    plt.xlabel('Layers', fontweight='bold')
    plt.ylabel('Knowledge Gain (%)', fontweight='bold')
    plt.title('VGG: Knowledge Gained w.r.t. Layers')
    plt.xticks([4*r + 5*barWidth + 2.75 for r in range(len(vgg_data[0]))], ['1', '2', '3', '4', '5','6', '7', '8', '9', '10','11', '12', '13'])

    plt.legend(loc='upper center')
    #plt.show()
    plt.savefig('graph_files/'+'VGG_knowledge_gain_graph_scaled.png')

    return True
