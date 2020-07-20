import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compile_adaptive_files():
    #CHANGE THIS VALUE FOR NUMBER OF TRIALS
    num_trials=3
    adaptive_set=[]
    for trial_num in range (1,num_trials):
        adaptive_set.append('trial'+str(trial_num)+'.xlsx')
    return adaptive_set

def create_adaptive_graphs():
    #CHANGE THIS VALUE FOR NUMBER OF EPOCHS PER TRIAL
    total_num_epochs=10
    accuracies=[]
    epoch_num=[]
    count=0
    new_trial_indic='*'

    adaptive_set=compile_adaptive_files()
    print(adaptive_set,'adaptive_set')

    for trial in adaptive_set:
        dfs=pd.read_excel(trial)
        print(dfs)
        for epoch in range (0,total_num_epochs):
            epoch_num.append(str(epoch+count)+new_trial_indic)
            accuracies.append(dfs['test_acc_epoch_'+str(epoch)][0]*100)
            new_trial_indic=''
        count+=total_num_epochs
        new_trial_indic='*'
    print(epoch_num)
    print(accuracies)
    fig=plt.figure()
    fig=plt.plot(epoch_num,accuracies, label='accuracy vs epoch', marker='o', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('AdaptiveNet: Test Accuracy vs Epoch')
    plt.show()

create_adaptive_graphs()
