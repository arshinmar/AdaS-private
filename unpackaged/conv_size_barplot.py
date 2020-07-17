import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_name='adapted_architectures.xlsx'
layers_info=pd.read_excel(file_name)
print(layers_info.head())
print(layers_info.iloc[:,0], 'dsaiu')

layers=layers_info.iloc[:,0].to_numpy()
superblock_1=layers_info.iloc[:,1].to_numpy()
superblock_2=layers_info.iloc[:,2].to_numpy()
superblock_3=layers_info.iloc[:,3].to_numpy()
superblock_4=layers_info.iloc[:,4].to_numpy()
superblock_5=layers_info.iloc[:,5].to_numpy()

print(superblock_1)

barWidth=0.5
main_0_layers = 4*layers
print(main_0_layers)
main_1_layers = [x + barWidth for x in main_0_layers]
main_2_layers = [x + barWidth for x in main_1_layers]
main_3_layers = [x + barWidth for x in main_2_layers]
main_4_layers = [x + barWidth for x in main_3_layers]

plt.bar(main_0_layers, superblock_1, color='#4d4d4e', width=barWidth, edgecolor='white', label='Baseline')
plt.bar(main_1_layers, superblock_2, color='#b51b1b', width=barWidth, edgecolor='white', label='Scaled 20%')
plt.bar(main_2_layers, superblock_3, color='#1f639b', width=barWidth, edgecolor='white', label='Scaled 40%')
plt.bar(main_3_layers, superblock_4, color='#1bb5b5', width=barWidth, edgecolor='white', label='Scaled 60%')
plt.bar(main_4_layers, superblock_5, color='#fcb045', width=barWidth, edgecolor='white', label='Scaled 80%')

plt.xlabel('Trial', fontweight='bold')
plt.ylabel('Layer Size', fontweight='bold')
plt.title('AdaptiveNet: Evolution of Layer Size Vs Trial')
plt.xticks([4*r + 5*barWidth + 2.75 for r in range(len(superblock_1))], [str(i) for i in range(len(superblock_1))])

plt.legend(loc='upper right')
plt.show()

def create_layer_barplot(file_name):

    layers_info=pd.read_excel(file_name)

    layers=layers_info.iloc[:,0].to_numpy()
    superblock_1=layers_info.iloc[:,0].to_numpy()
    superblock_2=layers_info.iloc[:,1].to_numpy()
    superblock_3=layers_info.iloc[:,2].to_numpy()
    superblock_4=layers_info.iloc[:,3].to_numpy()
    superblock_5=layers_info.iloc[:,4].to_numpy()

    barWidth=0.5
    #X-data, separated by Bar Width
    main_0_layers = 4*layers
    main_1_layers = [x + barWidth for x in main_0_layers]
    main_2_layers = [x + barWidth for x in main_1_layers]
    main_3_layers = [x + barWidth for x in main_2_layers]
    main_4_layers = [x + barWidth for x in main_3_layers]
    main_5_layers = [x + barWidth for x in main_4_layers]


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
