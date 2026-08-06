import torch
import torch.nn as nn
import re
import matplotlib.pyplot as plt
import numpy as np
import math



#Defines ListNet neural network
#Follows same structure as RankNet to help make accurate comparisons
class ListNet(nn.Module):

    #Initialise layers of the network
    def __init__(self,input_features,hl1,out):
        super().__init__()
        self.fc1 = nn.Linear(input_features,hl1)
        self.out = nn.Linear(hl1,out)


    #Forward function to define how input is computed 
    # and activation functions used to reach output
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.out()

        return x

