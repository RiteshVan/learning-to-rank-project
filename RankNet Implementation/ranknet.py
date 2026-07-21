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
        dictionary_query_docs[qid]= [(f,label)]
    
    #If query already present, add tuple to list
    else:
        dictionary_query_docs[qid].append((f,label))