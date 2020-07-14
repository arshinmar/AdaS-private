#ResNet Implementation Video Documentation
#Sepehr Hosseini
#Supervisor: Mahdi Hosseini, June 22, 2020
#This file provides comments and documentation for every step performed in the ResNet implementation as per https://www.youtube.com/watch?v=DkNIBBBvcPs.

import torch
import torch.nn as nn

#This class is responsible for creating the blocks used in the ResNet architecture.
#It is composed of the init and forward function, both of which are described in the comments above their corresponding functions.
class block(nn.Module):

    #This function is responsible for defining the operations used in the forward pass of ResNet.
    #The ResNet class instantiates these blocks in the _make_layer function in the ResNet class in order to carry out the respective convolutions and batch normalizations for each block.

    #This function takes in the following parameters:

    #in_channels: This parameter passes in the size of the input channel. It is important to note that the input channel size is equal in size to the previous layer's output.

    #out_channels: This parameter passes in the size of the output channel for the convolution operation.

    #identity_downsample:  A convolution layer which may need to be performed depending on if the input size or number of channels is changed.
    #The purpose for this identity_downsample is to adapt the identity so it can be added later on after a few convolution layers.

    #stride: This indicates the stride performed in the convolutions. Note that it is default set to 1 meaning that the filters are moved 1 pixel at a time.
    #However, it is important to realize that at times, a stride of 2 will be passed in from the ResNet class which will cut the output size in half.
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        #This line passes the parameters to the constructor of nn.Module.
        super(block, self).__init__()
        self.expansion = 4
        #This assigns the operation for the first convolution in the block. It uses the values for in_channels and out_channels which are passed in from the _make_layer function which initially recieves the values from the init function in the ResNet class.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        #This assigns the operation for the batch normalization. It is important to perform batch normalization on the output channels to ensure that stabilize the network and ensure the network is consistent.
        self.bn1 = nn.BatchNorm2d(out_channels)
        #This assigns the operation for the next convolution in the block. The stride used here is the one inputted to the init function.
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        #This assigns the operation for the batch normalization on the second convolution which follows the same motivation to that of the first.
        self.bn2 = nn.BatchNorm2d(out_channels)
        #This assigns the third convolution used within each block. The key difference is the expansion to the out channels with the self.expansion variable.
        #This operation is performed as defined in Table 1. of the ResNet paper in the 50-layer column corresponding to ResNet50.
        #Essentially, the only difference is that the channel in the last layer of each block is expanded by a factor of 4.
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        #This is the batch normalization for the third convolution layer. The same methodology as previous batch normalizations applies to this one.
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        #ReLU is then performed as the activation function as per standard networks.
        self.relu = nn.ReLU()
        #This is a convolutional layer performed to the identity mapping so the same shape is maintained in the later layers.
        self.identity_downsample = identity_downsample

    #This is the forward operation which performs the network computations (defined in the init function) on x.
    def forward(self, x):
        #This sets the identity x which is responsible for performing the skip connection.
        identity = x
        #This line performs the conv1 operation (as defined in the init function) on x.
        x = self.conv1(x)
        #This line performs the bn1 operation (as defined in the init function) on x.
        x = self.bn1(x)
        #This line performs the relu operation (as defined in the init function) on x.
        x = self.relu(x)
        #This line performs the conv2 operation (as defined in the init function) on x.
        x = self.conv2(x)
        #This line performs the bn2 operation (as defined in the init function) on x.
        x = self.bn2(x)
        #This line performs the relu operation (as defined in the init function) on x.
        x = self.relu(x)
        #This line performs the conv3 operation (as defined in the init function) on x.
        x = self.conv3(x)
        #This line performs the bn3 operation (as defined in the init function) on x.
        x = self.bn3(x)

        #If the shape of the convolution needs to be changed due to the conditions defined in the comments above the init function (if identity_downsample is None) then the downsample operation is performed.
        #Note that the identity_downsample is set to None by default in init. The computation for this conditional is performed in the _make_layer function in the ResNet class.
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        #This line performs the skip connection.
        x += identity
        #This line performs the relu operation (as defined in the init function) on x.
        x = self.relu(x)
        return x

#The motivation behind the ResNet class and architecture is the skip connections that are introduced within the architecture.
#These skip connections allow for the network to learn new things, but at the same time not forget the information which was learned before.
#This retrieval of information means that in theory, the network should not become worse as it becomes deeper, which is what in many cases has been observed.
class ResNet(nn.Module):

    #This function is responsible for instantiating the ResNet class. There are different types of ResNet, specifically ResNet50, ResNet101, and ResNet152 are the ones defined in this code.

    #This function takes in the follwoing parameters:

    #block: The blocks that compose the ResNet architecture. This object is created using the block class whose parameters are defined when making the layers in the _make_layer function.

    #layers: This parameter yields an array of which each index corresponds to the number of blocks used within that layer.

    #image_channels: This parameter corresponds to the number of image channels as defined by each type of ResNet class respectively.
    #It is used as the input channel size for the initial convolution (conv1) in the init function for this class.

    #num_classes: The number of classes used for the network. This number depends on the specific architecture that is being defined (ResNet50, ResNet101, ResNet152).
    #Note that this variable is primarily used when creating the fully connected layer.
    def __init__(self, block, layers, image_channels, num_classes):
        #This line passes the parameters to the constructor of nn.Module.
        super(ResNet, self).__init__()
        #This is the input channel for the intial convolution (conv1).
        #Note that in Table 1 of the ResNet paper (row 2), regardless of the type of ResNet, all networks have an input channel size of 64.
        self.in_channels = 64
        #This is the initial convolution performed for ResNet. As stated above, all networks follow the same convolution regardless of the type of ResNet.
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        #The batch normalization operation is performed with the same reasoning mentioned in earlier convolutions.
        self.bn1 = nn.BatchNorm2d(64)
        #RelU activation function is used as per same reasoning mentioned in earlier convolutions.
        self.relu = nn.ReLU()
        #Finally, a max pooling is performed on the initial convolution to reduce the size whilst maintaining important features of the input.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #The below four lines perform essentially the entire ResNet network operations.

        #4 different layers are created, each with a different number of blocks (which was passed into the function with the layers variable) used.
        #For example, layers[0] corresponds to 3 blocks for ResNet50 for its number of residual blocks.
        #The output channels are defined as per Table 1 in the ResNet paper along with their strides.
        #Notice that layers 2,3, and 4 have a stride of 2 which halves the output size.

        #The _make_layer function is used to create these layers which is defined below.

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        #An average pooling is performed to extract outstanding features of output.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #Fully connected layer is used to flatten out the layers.
        self.fc = nn.Linear(512 * 4, num_classes)

    #Similar to the forward function used for the block class, this function peforms the network computations for ResNet as a whole using the assignemnts above.
    def forward(self, x):
        #This line performs the initial conv1 operation (as defined in the init function) on x.
        x = self.conv1(x)
        #This line performs the initial bn1 operation (as defined in the init function) on x.
        x = self.bn1(x)
        #This line performs the initial relu operation (as defined in the init function) on x.
        x = self.relu(x)
        #This line performs the initial maxpool operation (as defined in the init function) on x.
        x = self.maxpool(x)
        #This line performs the layer1 computation (as defined in the init function) on x.
        x = self.layer1(x)
        #This line performs the layer2 computation (as defined in the init function) on x.
        x = self.layer2(x)
        #This line performs the layer3 computation (as defined in the init function) on x.
        x = self.layer3(x)
        #This line performs the layer4 computation (as defined in the init function) on x.
        x = self.layer4(x)

        #This line performs the avgpool operation (as defined in the init function) on x.
        x = self.avgpool(x)
        #This line performs the reshape operation (as defined in the init function) on x.
        x = x.reshape(x.shape[0], -1)
        #This line performs the fc operation (as defined in the init function) on x.
        x = self.fc(x)

        return x

    #This function is responsible for creating the layers used for the ResNet class which is composed of several blocks as described earlier.
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        #identity_downsample is intially set to None but its assignment will change based on the computations performed below.
        identity_downsample = None
        #Layers is intially defined as empty but is populated later in this method with the blocks.
        layers = []

        #If the input size is changed (e.g. 56x56 -> 28x28) or number of channels is changed, the identity or skip connection needs to be adapted using the below convolution. A batch normalization is also performed.
        #This is so that it is able to be added in the subsequent layers.
        if stride != 1 or self.in_channels != out_channels * 4:
            print ('shortcut')
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * 4,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels * 4),
            )

        #This is the initial layer that changes the number of channels.
        #The only case where the input and number of channels is changed is within this first block.
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))

        #At the end of the last created block in the intial layer, the output channel is expanded by a factor of 4 (with conv3) so this line changes the size of the in_channels by also increasing it by a factor of 4.
        self.in_channels = out_channels * 4

        #This part of the code is responsible for simply appending each block that composes the layer to the overall layer.
        #The for loop runs for the number of residual blocks as specified by each index in the the layers array which was initially passed in to the definition functions for each type of ResNet (the functions below).
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        #Finally, the list is unpacked so pytorch knows each one will come one after one another.
        return nn.Sequential(*layers)

#Below are the function definitions for each different type of ResNet. Note that each type of ResNet is distinguished by the specfic parameters that it is passed when instatiating the ResNet class with its corresponding parameters.
#Each type of ResNet varies by its block (all have same type of block in this case), layers (number of blocks within each layer), img_channel (same for all in this case), and number of classes (same for all in this case).
def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)
def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)
def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)

#This function is used to test whether or not the ResNet class works as intended.
def test():
    net = ResNet152()
    x=torch.randn(2,3,224,224)
    y=net(x)
    print(y.size())

test()
