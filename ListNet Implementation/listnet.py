import torch
import torch.nn as nn
import re
import matplotlib.pyplot as plt
import numpy as np
import math


#Path to training dataset
train_dataset_name = "/Users/ritesh/Desktop/project-extra/MQ2008/Fold1/train.txt"


#Path to validation dataset
val_dataset_name = "/Users/ritesh/Desktop/project-extra/MQ2008/Fold1/vali.txt"


#Path to test dataset
test_dataset_name = "/Users/ritesh/Desktop/project-extra/MQ2008/Fold1/test.txt"



#Function to organise documents linked to each query
def get_network_input(dataset_name : str):

    #Used to store lines from dataset (corresponds to doc-query pair)
    array = []


    #Open file and add lines to array
    with open(dataset_name,"r") as f:
        #Go through file and add lines to array
        for line in f:
            array.append(line)



    #Arrays to store  information about query-doc pairs
    relevancy_labels = []
    queries = [] #Query IDs
    features = [] #Array of arrays each of which representing a document



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

    

    network_input=[]


    for query in dictionary_query_docs.keys():
        query_docs=[]
        query_targets=[]

        for document,target in dictionary_query_docs[query]:
            query_docs.append(document)
            query_targets.append(target)


        query_docs = torch.tensor(query_docs,dtype=torch.float32)

        query_targets = torch.tensor(query_targets,dtype=torch.float32)

        network_input.append((query_docs,query_targets))


    return network_input







#Similar to prior function just need to get dictionary for NDCG calculations
#Function to organise documents linked to each query
def get_dict(dataset_name : str):

    #Used to store lines from dataset (corresponds to doc-query pair)
    array = []


    #Open file and add lines to array
    with open(dataset_name,"r") as f:
        #Go through file and add lines to array
        for line in f:
            array.append(line)



    #Arrays to store  information about query-doc pairs
    relevancy_labels = []
    queries = [] #Query IDs
    features = [] #Array of arrays each of which representing a document



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



    return dictionary_query_docs
















#Defines ListNet neural network
#Follows same structure as RankNet to help make accurate comparisons
class ListNet(nn.Module):

    #Initialise layers of the network
    def __init__(self,input_features,hl1,hl2,out):
        super().__init__()
        self.fc1 = nn.Linear(input_features,hl1)
        self.fc2 = nn.Linear(hl1,hl2)
        self.out = nn.Linear(hl2,out)


    #Forward function to define how input is computed 
    # and activation functions used to reach output
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x= self.out(x)

        return x

model = ListNet(46,5,5,1)


training_input = get_network_input(train_dataset_name)


validation_input = get_network_input(val_dataset_name)




val_dict = get_dict(val_dataset_name)



optimiser = torch.optim.Adam(model.parameters(),lr=0.001)

epochs = 100


train_losses=[]

val_losses=[]


val_NDCG_tracking =[]


max_ndcg_val=-1


for i in range(epochs):
    train_loss_avg=[]
    for j in range(len(training_input)):
        model.train()

        documents = training_input[j][0]

        targets = training_input[j][1]


        scores = model.forward(documents)

        scores = torch.flatten(scores)

        scores = torch.softmax(scores,-1)

        targets = torch.softmax(targets,-1)



        loss = -torch.sum(targets * torch.log2(scores))


        train_loss_avg.append(loss.detach().numpy())

        loss.backward()

        optimiser.step()


        optimiser.zero_grad()

    train_losses.append(sum(train_loss_avg)/len(train_loss_avg))


    val_loss_avg=[]
    with torch.no_grad():
        model.eval()
        for k in range(len(validation_input)):

            documents = validation_input[k][0]

            targets = validation_input[k][1]

            scores = model.forward(documents)

            scores = torch.flatten(scores)

            scores=torch.softmax(scores,-1)


            targets = torch.softmax(targets,-1)

            val_loss =  -torch.sum(targets * torch.log2(scores))

            val_loss_avg.append(val_loss.detach().numpy())

        val_losses.append(sum(val_loss_avg)/len(val_loss_avg))




        ndcg_per_query = []

        for query in val_dict.keys():


            actual_rank_order = []


            predicted_rank_order = []



            for item in val_dict[query]:
                actual_rank_order.append(item[1])


                predicted_score = model.forward(torch.tensor(item[0],dtype=torch.float32))

                predicted_rank_order.append((item[1],item[0],float(predicted_score)))



            actual_rank_order = sorted(actual_rank_order,reverse=True)[:10]

            predicted_rank_order = sorted(predicted_rank_order,reverse=True,key= lambda x : x[2])[:10]


            idcg = 0

            dcg = 0


            denominator_val =2


            for val in actual_rank_order:
                calc = ((2**val) - 1)/math.log2(denominator_val)


                idcg += calc

                denominator_val += 1



            denominator_val=2

            for val in predicted_rank_order:
                calc = ((2**val[0])-1)/math.log2(denominator_val)

                dcg+=calc

                denominator_val+=1


            if idcg != 0:
                ndcg = dcg/idcg


                ndcg_per_query.append(ndcg)


        ndcg_overall = sum(ndcg_per_query)/len(ndcg_per_query)

        val_NDCG_tracking.append(ndcg_overall)


        if max_ndcg_val < ndcg_overall:
            max_ndcg_val = ndcg_overall

            best_epoch = i

            torch.save(model.state_dict(),"model-weights.pth")



print(best_epoch)
plt.plot(range(epochs),val_NDCG_tracking)
plt.show()






