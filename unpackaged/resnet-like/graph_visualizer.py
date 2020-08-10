import torch
import torchvision.models
import hiddenlayer as hl
import matplotlib.pyplot as plt

# VGG16 with BatchNorm
model = torchvision.models.vgg16()

# Build HiddenLayer graph
# Jupyter Notebook renders it automatically
temp=hl.build_graph(model, torch.zeros([1, 3, 224, 224]))
temp.save('temp.pdf')
