import torch
import torch.nn as nn
import re
import numpy as np
import matplotlib.pyplot as plt


#Path to dataset being used
dataset_name= "/Users/ritesh/Desktop/project-extra/MQ2008/Fold1/train.txt"

#Function to get pairs
def get_pairs_tensors(dataset_name : str, data_limit  : int = None):
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
            if data_limit != None:
                if i>=data_limit:
                    #Exit loop once enough lines read
                    break



    #Arrays to store required information about query-doc pairs
    relevancy_labels = []
    queries = [] #Query IDs stored
    features = [] #Array of arrays, with each entry representing a document



    for val in array:
        #First char of each entry is relevancy label, so extracted, converted to int and stored
        relevancy_labels.append(int(val[0]))

        #Finds substring that matched "qid:" prefix
        if re.search(r"qid:([0-9]+)",val):
            #Only append group from regular expression that contains qid
            queries.append(re.search(r"qid:([0-9]+)",val).group(1))

            #Find all features in entry and only extracts numerical terms that refer to feature values
            feature_vals=re.findall(r"[0-9]+:([^ ]*)" , val) 

            #Convert all data extracted here to floats for future operations
            features.append(list( map(float,feature_vals)))


    #Create dictionary where keys are query IDs and value per key is list of tuples where each tuple contains a linked document's features and its relevancy to query
    dictionary_query_docs = {}


    #Iterate through all prior arrays at once to extract required information to store in dictionary
    for label,qid,f in zip(relevancy_labels,queries,features):
        #If query not already a key in dictionary, create new entry
        if qid not in dictionary_query_docs:
            dictionary_query_docs[qid] = [(f,label)]
    
        #If query already present, add tuple to list
        else:
            dictionary_query_docs[qid].append((f,label))

    #Create array to store document pairs, where leftmost entry is more relevant to a document that one on the right
    pairs=[]



    #Iterate through per query at a time to ensure any 2 documents in a pair are related to the same query
    for qid,docs in dictionary_query_docs.items():
        #Iterate through documents
        for i in range(len(docs)):
            for j in range(i+1, len(docs)):
                #Do not create pair if 2 docs have same pair, as purpose of training to understand when a doc is more relevant than another
                if docs[i][1] != docs[j][1]:

                    #Condition ensures more relevant doc first in tuple
                    if docs[i][1] > docs[j][1]:
                        if (docs[i][0], docs[j][0]) not in pairs:
                            pairs.append((docs[i][0], docs[j][0]))
                        
                
                
                    else:
                        if (docs[j][0], docs[i][0]) not in pairs:
                            pairs.append((docs[j][0], docs[i][0]))



    #Convert pairs into 2 separate arrays to keep more relevant and less relevant queries separate
    #Tensors are also useful as they automatically manage own memory for gradients (useful for optimiser steps)
    features_1 = torch.tensor([v[0] for v in pairs],dtype=torch.float32)
    features_2 = torch.tensor([v[1] for v in pairs],dtype=torch.float32)


    #Targets are equal to one as the more relevant doc is one the left side, so target is one
    target = torch.ones(len(features_1),1)


    #Outputs needed values
    return features_1,features_2,target



#Defines RankNet model
class RankNet(nn.Module):

    #Initialises layers in the neural network
    def __init__(self,input_features,hl1,out):
        super().__init__()
        self.fc1 =nn.Linear(input_features,hl1)
        self.out=nn.Linear(hl1,out)


    #Forward pass to compute output from the input
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x= self.out(x)

        return x

#Runs operation once to save time
data = get_pairs_tensors(dataset_name)

#Pairs extracted and stored separately
features_1 = data[0]
features_2 = data[1]

#Target values stored
target= data[2]


#Initialise model by specifying input size (equal to number of features per doc), number of neurons in middle layer and one output
model=RankNet(len(features_1[0]),30,1)

#Defines loss function used for the model
critierion = nn.BCELoss()

#Used to update weights based on information obtained
optimiser = torch.optim.Adam(model.parameters(),lr=0.01)

#How many times to run loop
epochs = 100

#Keeps track of loss function outputs to see how if changes over a number of iterations (to plot graph later)
losses = []

#Train loop
for i in range(epochs):
    #Outputs obtained as docs are input into network
    output_1 = model.forward(features_1)
    output_2 = model.forward(features_2)

    #Calculates difference  and its sigmoid value befor inputting into loss function
    diff = output_1 - output_2 
    diff = torch.sigmoid(diff)


    #Calculate loss and add to array
    loss = critierion(diff,target)

    losses.append(loss.detach().numpy())


    

    #Compute gradients
    loss.backward()

    #Weight adjustment from optimiser
    optimiser.step()

    #Prepare gradients for next loop by resetting current ones
    optimiser.zero_grad()

    

#Graph to show how loss changes based on iterations
plt.ylabel("Losses")
plt.xlabel("Epochs")
plt.plot(range(epochs),losses)
plt.show()