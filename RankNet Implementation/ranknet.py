import torch
import torch.nn as nn
import torch.functional as F
import re
import numpy as np

#Path to dataset being used
dataset_name= "placeholder"


#To store data entries (which consist of query IDs, features and relevancy labels)
array=[]


#Open file and iterate through and add to array
with open(dataset_name,"r") as f:
    #Variable to count number of lines read
    i=0


    for line in f:
        #Add line to array
        array.append(line)

        #Update variable to indicate one more line read
        i=i+1


        #Used to set limit on number of lines read to not overload resources
        if i>=1000:
            #Exit loop once enough lines read
            break
    

#Arrays to store required information about query-doc pairs
relevancy_labels=[]
queries=[] #Query IDs stored
features=[] #Array of arrays, with each entry representing a document