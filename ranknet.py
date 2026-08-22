import torch
import torch.nn as nn
import re
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os


#Function to obtain data with no pairs that have equal relevancy labels
def get_data_without_equal_pairs(fold : int):
    #MQ2008 Dataset from LETOR 4.0
    #Path to dataset being used
    dataset_name = os.getcwd() + "/MQ2008/Fold" +  str(fold)  + "/train.txt"



    #Path to validation set
    val_dataset_name =  os.getcwd() + "/MQ2008/Fold"+ str(fold) + "/vali.txt"


    #Path to test set
    test_dataset_name = os.getcwd() + "/MQ2008/Fold"+ str(fold) + "/test.txt"

    #Function to get dictionary to iterate through specific docs from each query
    #In particular created to help with calculating nDCG and MRR during validation
    #Adapted from get_pairs_tensors() function
    def get_dict(dataset_name : str):
        #To store data entries (which consist of relevancy labels, query IDs and features)
        array = []


        #Open file and iterate through and add to array
        with open(dataset_name, "r") as f:
            for line in f:
                #Add line to array
                array.append(line)
                
                



        #Arrays to store required information about query-doc pairs
        relevancy_labels = [] #Stores document relevancy labels
        queries = [] #Query IDs stored
        features = [] #Array of arrays, with each entry representing a document



        #Iterates through each data entry
        for val in array:
            #First char of each entry is relevancy label, so extracted, converted to int and stored
            relevancy_labels.append(int(val[0]))

            #Finds substring that matches "qid:" prefix
            if re.search(r"qid:([0-9]+)",val):
                #Only append group from regular expression that contains qid
                #Queries can be left as strings as not used for computations
                queries.append(re.search(r"qid:([0-9]+)",val).group(1))

                #Find all features in entry and only extracts numerical terms that refer to feature values
                feature_vals  =re.findall(r"[0-9]+:([^ ]*)" , val) 

                #Convert all data extracted here to floats for future operations
                features.append(list( map(float,feature_vals)))


        #Create dictionary where keys are query IDs and key values is 
        #lists of tuples where each tuple contains a linked document's features and its relevancy to query
        dictionary_query_docs = {}


        #Iterate through all prior arrays at once to extract required information to store in dictionary
        for label,qid,f in zip(relevancy_labels,queries,features):
            #If query not already a key in dictionary, create new entry
            if qid not in dictionary_query_docs:
                #Initialise list
                dictionary_query_docs[qid] = [(f,label)]
    
            #If query already present, add tuple to list
            else:
                dictionary_query_docs[qid].append((f,label))

    
        #Function returns dictionary
        return dictionary_query_docs








    #Function to get pairs
    def get_pairs_tensors(dataset_name : str):
        #To store data entries (which consist of query IDs, features and relevancy labels)
        array=[]


        #Open file and iterate through and add to array
        with open(dataset_name,"r") as f:
            for line in f:
                #Add line to array
                array.append(line)
            


        #Arrays to store required information about query-doc pairs
        relevancy_labels = [] #Stores document relevancy labels
        queries = [] #Query IDs stored
        features = [] #Array of arrays, with each entry representing a document


        #Iterate through each data entry
        for val in array:
            #First char of each entry is relevancy label, so extracted, converted to int and stored
            relevancy_labels.append(int(val[0]))

            #Finds substring that matches "qid:" prefix
            if re.search(r"qid:([0-9]+)",val):
                #Only append group from regular expression that contains qid
                queries.append(re.search(r"qid:([0-9]+)",val).group(1))

                #Find all features in entry and only extracts numerical terms that refer to feature values
                feature_vals = re.findall(r"[0-9]+:([^ ]*)" , val) 

                #Convert all data extracted here to floats for future operations
                features.append(list( map(float,feature_vals)))


        #Create dictionary where keys are query IDs and value per key is list of tuples where each tuple contains a linked document's features and its relevancy to query
        dictionary_query_docs = {}


        #Iterate through all prior arrays at once to extract required information to store in dictionary
        for label,qid,f in zip(relevancy_labels,queries,features):
            #If query not already a key in dictionary, create new entry
            if qid not in dictionary_query_docs:
                #Initialise list
                dictionary_query_docs[qid] = [(f,label)]
    
            #If query already present, add tuple to list
            else:
                dictionary_query_docs[qid].append((f,label))

        #Create array to store document pairs, where leftmost entry is more relevant to a document that one on the right
        pairs=[]

        target=[]

        #Iterate through per query at a time to ensure any 2 documents in a pair are related to the same query
        for qid in dictionary_query_docs.keys():
            #Iterate through documents
            for i in range(len(dictionary_query_docs[qid])):
                for j in range(i+1, len(dictionary_query_docs[qid])):
                    #Condition ensures more relevant doc first in tuple
                    if dictionary_query_docs[qid][i][1] > dictionary_query_docs[qid][j][1]:     
                        pairs.append((dictionary_query_docs[qid][i][0], dictionary_query_docs[qid][j][0]))
                        target.append([1])

                    #Condition checks and ensure less relevant document placed second in tuple
                    elif dictionary_query_docs[qid][i][1] < dictionary_query_docs[qid][j][1]:
                        pairs.append((dictionary_query_docs[qid][j][0], dictionary_query_docs[qid][i][0]))
                        target.append([1])

                
                    #Skip pairs that have equal relevancy
                    else:               
                        continue



        #Convert pairs into 2 separate arrays to keep more relevant and less relevant queries separate
        #Tensors are also useful as they automatically manage own memory for gradients (useful for optimiser steps)
        features_1 = torch.tensor([v[0] for v in pairs],dtype=torch.float32)
        features_2 = torch.tensor([v[1] for v in pairs],dtype=torch.float32)

        
        #Targets are equal to one as the more relevant doc is one the left side, so target is one
        target = torch.tensor(target,dtype=torch.float32)

        
        #Outputs needed values
        return features_1,features_2,target


    #Run functions to obtain data required 
    training_pairs = get_pairs_tensors(dataset_name)

    val_pairs = get_pairs_tensors(val_dataset_name)


    val_dict = get_dict(val_dataset_name)

    test_dict= get_dict(test_dataset_name)

    #Return data so can be passed to run() function
    return training_pairs, val_pairs, val_dict, test_dict














#Function to obtain data with pairs that have equal relevancy labels
def get_data_with_equal_pairs(fold : int):
    #MQ2008 Dataset from LETOR 4.0
    #Path to dataset being used
    dataset_name = os.getcwd() + "/MQ2008/Fold" +  str(fold)  + "/train.txt"



    #Path to validation set
    val_dataset_name = os.getcwd() + "/MQ2008/Fold"+ str(fold) + "/vali.txt"


    #Path to test set
    test_dataset_name = os.getcwd() + "/MQ2008/Fold"+ str(fold) + "/test.txt"

    #Function to get dictionary to iterate through specific docs from each query
    # #In particular created to help with calculating nDCG during validation
    # #Adapted from get_pairs_tensors() function
    def get_dict(dataset_name : str):
        #To store data entries (which consist of query IDs, features and relevancy labels)
        array = []


    #Open file and iterate through and add to array
        with open(dataset_name, "r") as f:

            for line in f:
                #Add line to array
                array.append(line)
                

                



        #Arrays to store required information about query-doc pairs
        relevancy_labels = [] #Stores document relevancy labels
        queries = [] #Query IDs stored
        features = [] #Array of arrays, with each entry representing a document



        #Iterate through data entries
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
                #Initialise list
                dictionary_query_docs[qid] = [(f,label)]
    
            #If query already present, add tuple to list
            else:
                dictionary_query_docs[qid].append((f,label))

    
        #Function returns dictionary
        return dictionary_query_docs








    #Function to get pairs
    def get_pairs_tensors(dataset_name : str):
        #To store data entries (which consist of query IDs, features and relevancy labels)
        array=[]


        #Open file and iterate through and add to array
        with open(dataset_name,"r") as f:
            for line in f:
                #Add line to array
                array.append(line)
                



        #Arrays to store required information about query-doc pairs
        relevancy_labels = []
        queries = [] #Query IDs stored
        features = [] #Array of arrays, with each entry representing a document


        #Iterate through data entries
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
                #Initialise list
                dictionary_query_docs[qid] = [(f,label)]
    
            #If query already present, add tuple to list
            else:
                dictionary_query_docs[qid].append((f,label))

        #Create array to store document pairs, where leftmost entry is more relevant to a document that one on the right
        pairs=[]

        target=[]

        #Iterate through per query at a time to ensure any 2 documents in a pair are related to the same query
        for qid,docs in dictionary_query_docs.items():
            #Iterate through documents
            for i in range(len(docs)):
                for j in range(i+1, len(docs)):
                    #Condition ensures more relevant doc first in tuple
                    if docs[i][1] > docs[j][1]:
                        
                        pairs.append((docs[i][0], docs[j][0]))
                        target.append([1])


                    elif docs[i][1] < docs[j][1]:
                        
                        pairs.append((docs[j][0], docs[i][0]))
                        target.append([1])

                
                
                    else:
                        
                        pairs.append((docs[j][0], docs[i][0]))
                        target.append([0.5])
                            



        #Convert pairs into 2 separate arrays to keep more relevant and less relevant queries separate
        #Tensors are also useful as they automatically manage own memory for gradients (useful for optimiser steps)
        features_1 = torch.tensor([v[0] for v in pairs],dtype=torch.float32)
        features_2 = torch.tensor([v[1] for v in pairs],dtype=torch.float32)

        # print(target)
        #Targets are equal to one as the more relevant doc is one the left side, so target is one
        target = torch.tensor(target,dtype=torch.float32)


        #Outputs needed values
        return features_1,features_2,target


    #Run functions to obtain data required 
    training_pairs = get_pairs_tensors(dataset_name)

    val_pairs = get_pairs_tensors(val_dataset_name)


    val_dict = get_dict(val_dataset_name)

    test_dict= get_dict(test_dataset_name)

    #Return data so can be passed to run() function
    return training_pairs, val_pairs, val_dict, test_dict




















#Function to run main code and get results related to NDCG values
def run_ndcg(training_pairs_data,val_pairs_data,val_dict_data,test_dict_data,ndcg_k : int):
    #Defines RankNet model
    class RankNet(nn.Module):

        #Initialises layers in the neural network
        def __init__(self,input_features,hl1,hl2,out):
            super().__init__()
            self.fc1 = nn.Linear(input_features,hl1)
            self.fc2= nn.Linear(hl1,hl2)
            self.out= nn.Linear(hl2,out)


        #Forward pass to compute output from the input
        def forward(self,x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.out(x)

            return x


    


    #Train data stored in one variable
    data = training_pairs_data



    



    #Pairs extracted and stored separately
    features_1 = data[0]
    features_2 = data[1]

    #Target values stored
    target= data[2]



    



    #Stores all validation data to one variable
    val_set = val_pairs_data


    



    #Pairs extracted and stored separately
    val_features_1 = val_set[0]
    val_features_2 = val_set[1]


    #Target values stored
    val_target=val_set[2]





    #Initialise model by specifying input size (equal to number of features per doc), number of neurons in middle layer and one output
    model = RankNet(len(features_1[0]),5,5,1)

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
    val_dict = val_dict_data


    #Keep track of nDCG values over the course of execution
    avg_nDCG_vals = []







    #   Used in order to save the model that gives the best nDCG score
    #Set to negative one as the nDCG values will always be above this
    max_ndcg_val = -1



    #Check epoch at which best model is seen
    best_epoch = -1


    #Used to check how long training and validation takes
    train_time_start = time.time()


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
            val_losses.append(val_loss.detach().numpy())





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
                    predicted_rank_order.append((item[1],float(predicted_score)))


                #Sort labels in order to find ideal-DCG
                #Only take top 10 values for NCDG calculation
                actual_rank_order = sorted(actual_rank_order,reverse=True)[:ndcg_k]


                #Order based predicted score
                #Only take top 10 values for NCDG calculation
                predicted_rank_order = sorted(predicted_rank_order,reverse=True,key= lambda x : x[1])[:ndcg_k]


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
                    nDCG = dcg / iDCG

                    nDCG_per_query.append(nDCG)


            #Append to array overall nDCG val for this iteration
            nDCG_overall = sum(nDCG_per_query) / len(nDCG_per_query)
            avg_nDCG_vals.append(nDCG_overall)


            #Check if model is new best in terms of NDCG
            if max_ndcg_val < nDCG_overall:
                max_ndcg_val = nDCG_overall

                #Keep track of when this new max NDCG value is seen
                best_epoch = i

                #Save only model weights so can be recerated easily for testing
                torch.save(model.state_dict(), "model-weights.pth")
            
        

    #Outputs best NDCG value seen
    print("Best Validation NDCG value obtained is: ", max_ndcg_val)
    print("Value seen after epoch ", best_epoch)


    #Empty lines printed for visual clarity
    print()
    print()
    print()


    #Outputs time taken for training and validation process to complete
    print("Time taken for validation and training to be completed (in seconds): " , time.time() - train_time_start)


    #Testing phase

    #Load in the stored model that gave the best nDCG score during validation

    #Keep same characteristics as model used for training (same layers, neurons)
    test_model = RankNet(len(features_1[0]), 5, 5, 1)

    #Apply saved weights to model
    test_model.load_state_dict(torch.load("model-weights.pth"))


    #Load in dictionary that keeps documents associated with each query
    test_dict = test_dict_data


    #This stores the nDCG values of all queries
    overall_nDCG_to_calc = []


    #Prep for task with no gradient calculations
    with torch.no_grad():
        test_model.eval()


        #Iterate through dictionary query by query
        for query in test_dict.keys():

            #Array to store real order for calculations (sorted later)
            actual_order = []

            #Store predicted orders
            pred_order = []


            #Iterate through docs for a given query
            for doc in test_dict[query]:
                #Add document label to array (to sort later)
                actual_order.append(doc[1])


                #Get document relevancy score from the model
                doc_score = test_model.forward(torch.tensor(doc[0] ,dtype=torch.float32))

                #Append as tuple format (pred_relevancy,actual relevancy) 
                # This so array can be sorted by predicted order, but NDCG calculations 
                # can be done on actual labels
                pred_order.append((doc_score,doc[1]))


            #Order true relevancy labels
            actual_order = sorted(actual_order, reverse=True)[:ndcg_k]

            #Order based on predicted scores
            #Key used to sort by first term in tuple 
            pred_order = sorted(pred_order, reverse=True, key= lambda x : x[0])[:ndcg_k]


            #Calculations here done with similar operations to those in validation step
            iDCG =0

            denominator_val=2
            for val in actual_order:
                calc = ((2**val)-1)/math.log2(denominator_val)

                iDCG += calc

                denominator_val +=1


            dcg = 0

            denominator_val=2
            for val in pred_order:
                calc =((2**val[1])-1)/math.log2(denominator_val)

                dcg += calc

                denominator_val+=1

            #This is done to check in case some queries have no relevant labels 
            # so must be skipped from final calculation
            if iDCG !=0:
                overall_nDCG_to_calc.append(dcg/iDCG)



    #Add up all NDCG vals for all queries and 
    # then divide by total number of queries to get average
    test_ndcg = sum(overall_nDCG_to_calc)/len(overall_nDCG_to_calc)

    #Print final NDCG val obtained on test set
    print("Test NDCG: ", test_ndcg)


    

    return max_ndcg_val, best_epoch, test_ndcg





#Function used to run main code and get MRR values
def run_mrr(training_pairs_data,val_pairs_data,val_dict_data,test_dict_data):
    #Defines RankNet model
    class RankNet(nn.Module):
    
        #Initialises layers in the neural network
        def __init__(self,input_features,hl1,hl2,out):
            super().__init__()
            self.fc1 = nn.Linear(input_features,hl1)
            self.fc2= nn.Linear(hl1,hl2)
            self.out= nn.Linear(hl2,out)
    
    
        #Forward pass to compute output from the input
        def forward(self,x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.out(x)
    
            return x


    #Train data stored in one variable
    data = training_pairs_data


    #Pairs extracted and stored separately
    features_1 = data[0]
    features_2 = data[1]

    #Target values stored
    target= data[2]


    #Stores all validation data to one variable
    val_set = val_pairs_data




    #Pairs extracted and stored separately
    val_features_1 = val_set[0]
    val_features_2 = val_set[1]


    #Target values stored
    val_target = val_set[2]





    #Initialise model by specifying input size (equal to number of features per doc), number of neurons in middle layer and one output
    model=RankNet(len(features_1[0]), 5, 5, 1)

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
    val_dict = val_dict_data




    #   Used in order to save the model that gives the best MRR score
    #Set to negative one as the MRR values will always be above this
    max_mrr_val = -1



    #Check epoch at which best model is seen
    best_epoch = -1


    #Used to check how long training and validation takes
    train_time_start = time.time()


    #To track how MRR values change over time
    avg_mrr_values=[]

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
            val_losses.append(val_loss.detach().numpy())





            #Keep track of MRR values for each query
            mrr_per_query = []


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
                    predicted_score = model.forward( torch.tensor(item[0],dtype = torch.float32) )


                    #Add tuple to keep track of real label, prediction and document itself
                    #Important to keep tuple format for ordering based on predicted scores and making calculations based on actual labels 
                    predicted_rank_order.append((item[1],item[0],float(predicted_score)))


                #Order real labels to get true best possible order
                actual_rank_order = sorted(actual_rank_order,reverse = True)

                #Order true labels based on scores from neural net
                predicted_rank_order = sorted(predicted_rank_order,reverse = True, key = lambda x : x[2])




                #In place to track position of most relevant doc in list
                top_label_position = -1


                #Iterate through the ordering from predicted scores
                for l in range(len(predicted_rank_order)):

                    #Checks if item at index i is the most relevant doc to a query and if relevancy label is not zero 
                    if (predicted_rank_order[l][0] == actual_rank_order[0]) and (actual_rank_order[0] != 0):

                        #Note position (add one as to get actual position not index)
                        top_label_position = l + 1

                        #Break to ensure the loop does not check lower ranked position
                        # as there may be other docs with same top relevancy label but are ranked lower
                        # Only need topmost ranked item
                        break   

                #If top label is zero then do not consider the query for calculation
                if (actual_rank_order[0] != 0):
                    print("MRR val for query: ", 1/top_label_position)

                    mrr_per_query.append(1/top_label_position)

            #Get overall average MRR considering all queries at particular epoch
            avg_epoch_mrr = sum(mrr_per_query)/len(mrr_per_query)

            #Add to list to track how MRR changes
            avg_mrr_values.append(avg_epoch_mrr)


            #If MRR value is highest seen so far, store model weights to be used for testing
            if avg_epoch_mrr > max_mrr_val:
                #Update new maximum MRR value 
                max_mrr_val = avg_epoch_mrr

                #Track epoch at which best model weights are obtained
                best_epoch = i

                #Save model weights so they can be loaded later
                torch.save(model.state_dict(),"model-weights.pth")


    
    #Create test model that is structurally identical to one from prior
    test_model = RankNet(len(features_1[0]), 5, 5, 1)


    #Load in best model weights from prior
    test_model.load_state_dict(torch.load("model-weights.pth"))

    #Load in data from test dataset
    test_dict = test_dict_data


    #Used to track MRRs for all queries to get average at end
    overall_mrr_to_calc = []

    #Disable gradient computations
    with torch.no_grad():
        #Set model to evaluation mode
        test_model.eval()


        #Iterate through keys (queries) in dictionary
        for query in test_dict.keys():

            #Used to keep track of orders of labels
            actual_rank_order = []

            predicted_rank_order = []


            #Iterates through docs that belong to single query
            for doc in test_dict[query]:


                #Get labels of docs and add to array
                actual_rank_order.append(doc[1])


                #Get model score for document and append to array
                doc_score = test_model.forward(torch.tensor(doc[0],dtype=torch.float32))

                predicted_rank_order.append((doc_score,doc[1]))


            #Order list of relevancy labels
            actual_rank_order = sorted(actual_rank_order,reverse=True)

            #Order list of relevancy labels but based on scores from model
            predicted_rank_order = sorted(predicted_rank_order,reverse=True, key= lambda x : x[0])


            #MRR process similar to that from validation step
            top_label_position = -1

            for i in range(len(predicted_rank_order)):
                if (predicted_rank_order[i][1] == actual_rank_order[0]) and (actual_rank_order[0] != 0):
                    top_label_position = i + 1
                    break

            if (actual_rank_order[0] != 0):
                overall_mrr_to_calc.append(1 / top_label_position)


    #Calculate overall MRR on test set and print
    test_mrr = sum(overall_mrr_to_calc) / len(overall_mrr_to_calc)

    print("Test MRR: ", test_mrr)


    #Function returns values so that they can be used to compute averages
    return max_mrr_val, best_epoch, test_mrr




data = get_data_without_equal_pairs(1)[0][0]
data_equal = get_data_with_equal_pairs(1)[0][0]

print(len(data))

print(len(data_equal))