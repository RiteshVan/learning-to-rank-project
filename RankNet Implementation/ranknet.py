import torch
import torch.nn as nn
import re
import numpy as np
import matplotlib.pyplot as plt
import math


#Path to dataset being used
dataset_name = "/Users/ritesh/Desktop/project-extra/MQ2008/Fold1/train.txt"


#Path to validation set
val_dataset_name = "/Users/ritesh/Desktop/project-extra/MQ2008/Fold1/vali.txt"






#Function to get dictionary to iterate through specific docs from each query
#In particular created to help with calculating nDCG during validation
#Adapted from get_pairs_tensors() function
def get_dict(dataset_name : str, data_limit : int = None):
    #To store data entries (which consist of query IDs, features and relevancy labels)
    array = []


    #Open file and iterate through and add to array
    with open(dataset_name, "r") as f:
        #Variable to count number of lines read
        i = 0

        for line in f:
            #Add line to array
            array.append(line)
            #Update variable to indicate one more line read
            i=i+1


            #Used to set limit on number of lines read to not overload resources
            if data_limit != None:
                if i >= data_limit:
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


    #Create dictionary where keys are query IDs and value per key is 
    # list of tuples where each tuple contains a linked document's features and its relevancy to query
    dictionary_query_docs = {}


    #Iterate through all prior arrays at once to extract required information to store in dictionary
    for label,qid,f in zip(relevancy_labels,queries,features):
        #If query not already a key in dictionary, create new entry
        if qid not in dictionary_query_docs:
            dictionary_query_docs[qid] = [(f,label)]
    
        #If query already present, add tuple to list
        else:
            dictionary_query_docs[qid].append((f,label))

    
    #Function returns dictionary
    return dictionary_query_docs








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




#Runs operation once on validation set to save time
val_set = get_pairs_tensors(val_dataset_name)




#Pairs extracted and stored separately
val_features_1 = val_set[0]
val_features_2 = val_set[1]


#Target values stored
val_target=val_set[2]





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

#Keep track ofloss on validation set
val_losses = []



#Create dictionary for validation set
val_dict = get_dict(val_dataset_name)


#Keep track of nDCG values over the course of execution
avg_nDCG_vals = []



#Train loop
for i in range(epochs):
    #Let model know that coming steps are for training and ensures layers operate as intended
    model.train()

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


    #Disable model gradient calculations to save memory for inference tasks
    with torch.no_grad():
        #Prepares neural network for evaluation tasks
        model.eval()


        #Get outputs from docs from validation set
        val_output_1 = model.forward(val_features_1)
        val_output_2 = model.forward(val_features_2)


        #Get difference and apply sigmoid to input into loss function
        val_diff = val_output_1 - val_output_2
        val_diff = torch.sigmoid(val_diff)


        #Calculate loss and store into validation set losses
        val_loss = critierion(val_diff, val_target)


        #Update array to store loss value
        val_losses.append(val_loss)





        #Keep track of nDCG values for each query
        nDCG_per_query = []


        #Iterate through each query in validation set
        for query in val_dict.keys():

            #To keep track of actual relevancy labels (to be ordered with a sorted function later)
            actual_rank_order = []

            #Store tuple 
            #FORMAT:(actual_label,features,predicted_label)  
            #Later ordered based on scores given
            predicted_rank_order=[]


            #Iterate through each doc per query
            for item in val_dict[query]:

                #Append actual label to list
                actual_rank_order.append(item[1])


                #Get predicted score for doc
                #Convert to tensor so model can take doc as input
                predicted_score = model.forward( torch.tensor(item[0],dtype=torch.float32) )


                #Add tuple to keep track of real label, prediction and document itself
                #Important to keep tuple format for ordering based on predicted scores and making calculations based on actual labels 
                predicted_rank_order.append((item[1],item[0],float(predicted_score)))


            #Sort labels in order to find ideal-DCG
            #Only take top 10 values for NCDG calculation
            actual_rank_order = sorted(actual_rank_order,reverse=True)[:10]


            #Order based predicted score
            #Only take top 10 values for NCDG calculation
            predicted_rank_order = sorted(predicted_rank_order,reverse=True,key= lambda x : x[2])[:10]


            #Keep track of overall values
            iDCG = 0

            dcg = 0


            #Log calculations start at log2(2) then rise iteratively do set variable to 2
            denominator_val = 2

            #Iterate through actual order vals to find iDCG value
            for val in actual_rank_order:
                calc = ((2**val)-1)/math.log2(denominator_val)


                #Add to sum
                iDCG += calc


                #Iterate to prepare for next position term
                denominator_val +=1


            #Reset for next loop
            denominator_val = 2

            #Iterate through predicted order vals to find DCG value
            for val in predicted_rank_order:
                #Only take vals from first index as these represent real labels
                calc = ((2**val[0])-1)/math.log2(denominator_val)


                #Add to sum
                dcg += calc
            

                #Iterate to prepare for next position term
                denominator_val += 1


                #Checks in case query has no relevant docs connected to it
                if iDCG != 0 :
                    #Calculation to find nDCG and add to array
                    nDCG = dcg /iDCG

                    nDCG_per_query.append(nDCG)


        #Append to array overall nDCG val for the query
        avg_nDCG_vals.append(sum(nDCG_per_query)/len(nDCG_per_query))




        



#Graph to show how loss changes based on iterations
# plt.ylabel("Losses")
# plt.xlabel("Epochs")
# plt.plot(range(epochs),losses,'r-',label="Training set losses")
# plt.plot(range(epochs),val_losses,'b-',label="Validation set losses")
# plt.legend()
# plt.show()


plt.plot(range(epochs),avg_nDCG_vals)
plt.show()