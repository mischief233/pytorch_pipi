# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 09:12:51 2019

@author: wayne
"""

#第一种方法里，小编修改了resnet的后两层，即只要了第一层到第-2层，然后接上了平均池化层和线性层，是不是很神奇呀！

base_model=models.resnet50(pretrained=True)
#print(type(base_model))
batch_size=15
class M1501(nn.Module):
        def __init__(self , model):
            super(M1501, self).__init__()
            #取掉model的后两层
            self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
            #self.transion_layer = nn.ConvTranspose2d(2048, 2048, kernel_size=14, stride=3)
            #self.pool_layer = nn.MaxPool2d(32)
            self.avgpool=nn.AvgPool2d(7,stride=1)

            self.Linear_layer = nn.Linear(2048, 751)

        def forward(self, x):
            x = self.resnet_layer(x)
            x = self.avgpool(x)
            x1 = x.view(x.size(0), -1) 
            x2 = self.Linear_layer(x1) 
            return x1,x2

model=M1501(base_model).to(device)


#第二种方法
#第二种方法里，小编依然修改的是resnet50，但是这里不同的是，加载模型后，把平均池化改为了自适应平均池化，并在最后定义了自己的分类器。

import torch
import torch.nn as nn
from torchvision import models

# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, class_num = 751):
        super(ft_net, self).__init__()
        #load the model
        model_ft = models.resnet50(pretrained=True) 
        # change avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num) #define our classifier.

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x) #use our classifier.
        return x