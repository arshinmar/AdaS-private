"""
MIT License

Copyright (c) 2017 liukuang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software withx restriction, including withx limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHx WARRANTY OF ANY KIND, Ex PRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
x OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, x iangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arx iv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torchviz import make_dot, make_dot_from_trace

from graphviz import Digraph
import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import torchvision.models as models

import torch.onnx
from ptflops import get_model_complexity_info
import sys
sys.path.append("..")
import global_vars as GLOBALS
class BasicBlock(nn.Module):


    def __init__(self, in_planes, intermediate_planes, out_planes,stride=1):
        self.in_planes=in_planes
        self.intermediate_planes=intermediate_planes
        self.out_planes=out_planes

        super(BasicBlock,self).__init__()
        '''if in_planes!=intermediate_planes:
            #print('shortcut_needed')
            stride=2
        else:
            stride=stride'''
        self.conv1=nn.Conv2d(
                in_planes,
                intermediate_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
        )
        self.bn1=nn.BatchNorm2d(intermediate_planes)
        self.conv2=nn.Conv2d(
                intermediate_planes,
                out_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
        )
        self.bn2=nn.BatchNorm2d(out_planes)
        self.relu=nn.ReLU()
        self.shortcut=nn.Sequential()
        if stride!=1 or in_planes!=out_planes:
            #print('shortcut_made')
            self.shortcut=nn.Sequential(
                nn.Conv2d(
                        in_planes,
                        out_planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False
                ),
                nn.BatchNorm2d(out_planes),
                #nn.ReLU()
            )

    def forward(self,y):
        x = self.conv1(y)
        #print(x.shape,'post conv1 block')
        x = self.bn1(x)
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        #print(x.shape,'post conv2 block')
        #if self.shortcut!=nn.Sequential():
            #print('shortcut_made')
        #print(self.shortcut)
        #print(x.shape)
        #print(y.shape)
        #print(self.shortcut(y).shape)
        x += self.shortcut(y)
        #print(x.shape,'post conv3 block')
        x = self.relu(x)
        return x

class Network(nn.Module):

    def __init__(self, block, image_channels=3,new_output_sizes=None,num_classes=10):
        super(Network, self).__init__()

        ################################################################################## AdaS ##################################################################################
        '''self.shortcut_1_index = 7 #Number on excel corresponding to shortcut 1
        self.shortcut_2_index = 14 #Number on excel corresponding to shortcut 2
        self.shortcut_3_index = 21 #Number on excel corresponding to shortcut 2
        self.shortcut_4_index = 28'''
        ####################### O% ########################
        self.superblock1_indexes=[32,32,32,32,32,32,32] #7
        self.superblock2_indexes=[32,32,32,32,32,32,32,32]
        self.superblock3_indexes=[32,32,32,32,32,32,32,32,32,32,32,32]
        self.superblock4_indexes=[32,32,32,32,32,32]

        #self.superblock1_indexes=[64, 2, 64, 2, 64, 2, 64]
        #self.superblock2_indexes=[2, 128, 2, 128, 2, 128]
        #self.superblock3_indexes=[256, 256, 64, 64, 64, 64]
        #self.superblock4_indexes=[64, 64, 64, 64, 64, 64]
        #self.superblock5_indexes=[64, 64, 64, 64, 64, 64]
        #previous shortcut indexes = [7,14,21,28]
        #new shortcut indexes= [7,16,29]

        '''if new_output_sizes!=None:
            self.superblock1_indexes=new_output_sizes[0]
            self.superblock2_indexes=new_output_sizes[1]
            self.superblock3_indexes=new_output_sizes[2]
            self.superblock4_indexes=new_output_sizes[3]'''

        shortcut_indexes=[]
        counter=-1
        conv_size_list=[self.superblock1_indexes,self.superblock2_indexes,self.superblock3_indexes,self.superblock4_indexes]
        for j in conv_size_list:
            if len(shortcut_indexes)==len(conv_size_list)-1:
                break
            counter+=len(j) + 1
            shortcut_indexes+=[counter]

        print(shortcut_indexes)

        self.shortcut_1_index = shortcut_indexes[0]
        self.shortcut_2_index = shortcut_indexes[1]
        self.shortcut_3_index = shortcut_indexes[2]

        self.index=self.superblock1_indexes+self.superblock2_indexes+self.superblock3_indexes+self.superblock4_indexes

        self.num_classes=num_classes
        self.conv1 = nn.Conv2d(image_channels, self.index[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.index[0])
        self.network=self._create_network(block)
        self.linear=nn.Linear(self.index[len(self.index)-1],num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu=nn.ReLU()

    def _create_network(self,block):
        layers=[]
        layers.append(block(self.index[0],self.index[1],self.index[2],stride=1))
        for i in range(2,len(self.index)-2,2):
            #print(self.index [i],self.index [i+1],self.index [i+2],'for loop ',i)
            #if (self.index[i]!=self.index[i+2] or self.index[i]!=self.index[i+1]) and output_size>4:
            if (i+1==self.shortcut_1_index or i+2==self.shortcut_2_index or i+3==self.shortcut_3_index):
                stride=2
                print(i, 'shortcut')
            else:
                stride=1
        #    if i==len(self.index)-4:
            #    self.linear=nn.Linear(self.index[len(self.index)-2],self.num_classes)
            layers.append(block(self.index[i],self.index[i+1],self.index[i+2],stride=stride))
        #    #print(i, 'i')
        #print(len(self.index),'len index')
        return nn.Sequential(*layers)

    def forward(self, y):
        #print(self.index )
        x = self.conv1(y)
        #print(x.shape, 'conv1')
        x = self.bn1(x)
        #print(x.shape, 'bn1')
        x = self.relu(x)
        #print(x.shape, 'relu')
        #x = self.maxpool(x)
        ##print(x.shape, 'max pool')
        x = self.network(x)
        #print(x.shape, 'post bunch of blocks')
        x = self.avgpool(x)
        #print(x.shape, 'post avgpool')
        x = x.view(x.size(0), -1)
        #print(x.shape, 'post reshaping')
        x = self.linear(x)
        #print(x.shape, 'post fc')
        return x


def AdaptiveNet(num_classes = 10,new_output_sizes=None):
    return Network(BasicBlock, 3, num_classes=10, new_output_sizes=new_output_sizes)

def test():
    #writer = SummaryWriter('runs/resnet34_1')
    net = AdaptiveNet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

    macs, params = get_model_complexity_info(net, (3,32,32), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    #print(net)
    g=make_dot(y)
    g.view()
    #g.view()
    torch.save(net.state_dict(),'temp_resnet.onnx')
    dummy_input = Variable(torch.randn(4, 3, 32, 32))
    torch.onnx.export(net, dummy_input, "model.onnx")


test()
