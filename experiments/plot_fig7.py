import matplotlib.pyplot as plt
import torch
import numpy as np
import json

datasets = ['colored_MNIST', 'double_colored_MNIST', 'wildlife_MNIST']
with open("json_data.json") as jsonFile:
    dataset = json.load(jsonFile)

def plot(all_data, dataset):
    data = all_data[dataset]
    L1, L2, L3, L4 = [], [], [], []
    for size in data:
        L1.append(data[size]["1"][0])
        L2.append(data[size]["5"][0])
        L3.append(data[size]["10"][0])
        if dataset != 'colored_MNIST':    
            L4.append(data[size]["20"][0])
        
    sizes = [10**4, 10**5, 10**6]
    plt.plot(sizes, L1, 'r-')
    plt.plot(sizes, L2, 'm-')    
    plt.plot(sizes, L3, 'g-')
    if dataset != 'colored_MNIST':    
        plt.plot(sizes, L4, 'c-')

    plt.plot(sizes, L1, 'ro', label="CF ratio = 1")
    plt.plot(sizes, L2, 'mo', label="CF ratio = 5")    
    plt.plot(sizes, L3, 'go', label="CF ratio = 10")
    if dataset != 'colored_MNIST':    
        plt.plot(sizes, L4, 'co', label="CF ratio = 20")

    plt.xscale('log')
    plt.title(dataset)
    plt.xlabel("Num counterfactual datapoints")
    plt.ylabel("Test Accuracy [%]")
    plt.legend()
    plt.show()

for set in dataset:
    plot(dataset, set)