import torch
import torch.nn as nn
import re
import matplotlib.pyplot as plt
import numpy as np
import math
import os

#Function used to obtain all data needed for ListNet algorithm
#Contains subfunctions so that data for neural network and metrics can be obtained in one step
def get_data(fold : int,doc_count : int = 0):

    #Uses current directory string to get datasets
    #Path to training dataset
    train_dataset_name = os.getcwd() + "/MQ2008/Fold" +  str(fold) + "/train.txt"


    #Path to validation dataset
    val_dataset_name = os.getcwd() + "/MQ2008/Fold" +  str(fold) + "/vali.txt"


    #Path to test dataset
    test_dataset_name = os.getcwd() + "/MQ2008/Fold" +  str(fold) + "/test.txt"



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
        relevancy_labels = [] #Relevance labels of documents
        queries = [] #Query IDs
        features = [] #Array of arrays each of which represent a document



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

    
        #To store items to be input to neural network
        network_input=[]

        #Iterate through queries in dictionary
        for query in dictionary_query_docs.keys():

            #Arrays to store values on per query basis
            query_docs=[]
            query_targets=[]

            if len(dictionary_query_docs[query]) > doc_count:

                #Iterate through document and label in query list 
                for document,target in dictionary_query_docs[query]:
                    #Add required values to arrays
                    query_docs.append(document)
                    query_targets.append(target)

                #Convert to tensors
                query_docs = torch.tensor(query_docs,dtype=torch.float32)

                query_targets = torch.tensor(query_targets,dtype=torch.float32)


                #Each tuple appended related to all information for one query
                network_input.append((query_docs,query_targets))

        #Return network input
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
        relevancy_labels = [] #Store relevancy labels of documents
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
                #Initialise list
                dictionary_query_docs[qid] = [(f,label)]
    
            #If query already present, add tuple to list
            else:
                dictionary_query_docs[qid].append((f,label))


        for query in dictionary_query_docs.copy().keys():
            if len(dictionary_query_docs[query]) <= doc_count:
                del dictionary_query_docs[query]


        #Return dictionary
        return dictionary_query_docs

    
    #Obtain required data
    training_input = get_network_input(train_dataset_name)


    validation_input = get_network_input(val_dataset_name)

    val_dict = get_dict(val_dataset_name)

    test_dict = get_dict(test_dataset_name)


    network_input_len = len(training_input) + len(validation_input) + len(get_network_input(test_dataset_name))
    
    print("Dict size: ",len(val_dict.keys()) + len(test_dict.keys()) + len(get_dict(train_dataset_name).keys()) )
    return training_input,validation_input,val_dict,test_dict,network_input_len












#Function run to obtain NDCG values
def run_ndcg(training_input,validation_input,val_dict,test_dict,ndcg_k):
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
            x = self.out(x)

            return x

    #The length of input is equal to number of features in document
    #Two hidden layers of size 5 and output of size 1
    model = ListNet(len(training_input[0][0][0]),5,5,1)


    #Used to update the model weights based on loss
    optimiser = torch.optim.Adam(model.parameters(),lr=0.01)


    #Sets number of times to run loop
    epochs = 100

    #To keep track of training losses
    train_losses=[]


    #To keep track of validation losses
    val_losses=[]

    #Track how NDCG changes
    val_NDCG_tracking =[]

    #Track the best validation NDCG value to choose test weights
    max_ndcg_val = -1

    #To track iteration at which best NDCG value weights are seen
    best_epoch = -1


    for i in range(epochs):
        #Get average loss for epoch based on loss found for each query's data
        train_loss_avg=[]

        #Set model to train mode so its layers operate as intended
        model.train()

        #Iterate through items query by query
        for j in range(len(training_input)):
            
            #Extract documents list linked to query
            documents = training_input[j][0]


            #Store target values for documents
            targets = training_input[j][1]

            #Get model scores for all documents
            scores = model.forward(documents)


            #Ensure all scores kept in one dimension tensor array
            scores = torch.flatten(scores)

            #Calculate top-1 probability over targets and predicted scores
            scores = torch.softmax(scores,-1)  

            targets = torch.softmax(targets,-1)


            #Input into loss function
            loss = -torch.sum(targets * torch.log2(scores))

            #Add to array to track losses and take average for epoch at end of loop
            train_loss_avg.append(loss.detach().numpy())


            #Compute gradients
            loss.backward()


            #Adjust parameters based on gradients
            optimiser.step()

            #Clear gradients for next epoch
            optimiser.zero_grad()


        #Get average losses among all queries to get average epoch train loss
        train_losses.append(sum(train_loss_avg)/len(train_loss_avg))

        #To find validation loss at epoch
        val_loss_avg=[]

        #Disable gradient tracking to save resources
        with torch.no_grad():
            #Setup neural network for evaluation tasks
            model.eval()

            #Following is same process as training to get val losses
            #Iterate through tuples related to each query
            for k in range(len(validation_input)):

                #Extract documents list linked to query
                documents = validation_input[k][0]

                #Store target values for documents
                targets = validation_input[k][1]


                #Get model scored for documents
                scores = model.forward(documents)


                #Store scores for all documents in one dimension tensor array 
                scores = torch.flatten(scores)


                #Calculate top-1 probability for targets and scores
                scores = torch.softmax(scores,-1)

                targets = torch.softmax(targets,-1)

                #Calculate loss and append to list to take average
                val_loss =  -torch.sum(targets * torch.log2(scores))

                val_loss_avg.append(val_loss.detach().numpy())

            #Add average val epoch loss to array to track
            val_losses.append(sum(val_loss_avg)/len(val_loss_avg))



            #Track NDCG value obtained per query
            ndcg_per_query = []

            #Iterate through dictionary query-wise
            for query in val_dict.keys():

                #Setup arrays to track real and predicted order
                #Uses sorted() function after they are populated
                actual_rank_order = []


                predicted_rank_order = []


                #Iterate through each (doc,label) tuple per query list
                for item in val_dict[query]:
                    #Add relevancy label to list
                    actual_rank_order.append(item[1])

                    #Use document features to get score for document
                    predicted_score = model.forward(torch.tensor(item[0],dtype=torch.float32))


                    #Add predicted score to list (label,model_score)
                    predicted_rank_order.append((item[1],float(predicted_score)))


                #Order list
                actual_rank_order = sorted(actual_rank_order,reverse=True)[:ndcg_k]

                #Order list based on ordering of scores from model
                predicted_rank_order = sorted(predicted_rank_order,reverse=True,key= lambda x : x[1])[:ndcg_k]


                #Track DCG and IDCG to calculate NDCG later
                idcg = 0

                dcg = 0


                #Set start denominator to 2 then iterate for each position down the list (discounting)
                denominator_val = 2


                #Go through actual rankings first to get IDCG
                for val in actual_rank_order:
                    #Get score for item at given position
                    calc = ((2**val) - 1)/math.log2(denominator_val)

                    #Update overall idcg
                    idcg += calc

                    #Iterate to find score for next item down
                    denominator_val += 1


                #Set denominator to 2 and iterate for each position going down the list (discounting)
                denominator_val = 2


                #Go through labels of documents based on the order the model placed them
                for val in predicted_rank_order:
                    #Get score for item at given position
                    calc = ((2**val[0])-1)/math.log2(denominator_val)


                    #Update overall DCG score
                    dcg+=calc


                    #Iterate to setup to calculate next position down
                    denominator_val+=1

                #To prevent division by zero
                #If query has no docs associated to it, skip
                if idcg != 0:
                    ndcg = dcg/idcg


                    ndcg_per_query.append(ndcg)

            #Take average across queries
            ndcg_overall = sum(ndcg_per_query)/len(ndcg_per_query)

            #Track how NDCG changes over iterations
            val_NDCG_tracking.append(ndcg_overall)

            #If NDCG val is highest seen so far, update max_ndcg_val variable
            #Track epoch
            #Store model weights
            if max_ndcg_val < ndcg_overall:
                max_ndcg_val = ndcg_overall

                best_epoch = i

                torch.save(model.state_dict(),"model-weights.pth")








    #Recreate model same as model used for training and validation steps
    test_model = ListNet(len(training_input[0][0][0]),5,5,1)

    #Load in model weights that were saved prior
    test_model.load_state_dict(torch.load("model-weights.pth"))

    
    #Track ndcg for all queries to get average at end
    overall_ndcg_to_calc = []

    #Disable gradient tracking to save resources
    with torch.no_grad():
        #Set model layers to prepare for evaluation steps
        test_model.eval()

        #Process here is same as for validation
        for query in test_dict.keys():


            actual_rank_order = []

            predicted_rank_order = []


            for item in test_dict[query]:

                actual_rank_order.append(item[1])

                predicted_score = test_model.forward(torch.tensor(item[0],dtype=torch.float32))

                predicted_rank_order.append((predicted_score,item[1]))



            actual_rank_order = sorted(actual_rank_order,reverse=True)[:ndcg_k]


            predicted_rank_order = sorted(predicted_rank_order,reverse=True, key= lambda x : x[0])[:ndcg_k]



            idcg = 0

            dcg = 0


            denominator_val = 2


            for val in actual_rank_order:
                calc = ((2**val)-1)/math.log2(denominator_val)

                idcg += calc

                denominator_val+= 1




            denominator_val = 2 
            for val in predicted_rank_order:
                calc = ((2**val[1])-1)/math.log2(denominator_val)


                dcg += calc

                denominator_val += 1



            if idcg != 0:
                overall_ndcg_to_calc.append(dcg/idcg)


    #For this test dataset there is at least one query with at least one document allocated to it 
    #so no need to check if denominator is zero
    test_ndcg = sum(overall_ndcg_to_calc)/len(overall_ndcg_to_calc)

    #Print obtained values
    print("Best Validation NDCG: ", max_ndcg_val)
    print("Best epoch: ", best_epoch)
    print("Test NDCG: ", test_ndcg)


    plt.plot(range(epochs),val_NDCG_tracking)
    plt.show()

    

    return max_ndcg_val,best_epoch,test_ndcg







#Code here similar to prior function, only change is metric used
def run_mrr(training_input,validation_input,val_dict,test_dict):
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
            x = self.out(x)

            return x

    #The length of input is equal to number of features in document
    #Two hidden layers of size 5 and output of size 1
    model = ListNet(len(training_input[0][0][0]),5,5,1)



    



    optimiser = torch.optim.Adam(model.parameters(),lr=0.01)

    epochs = 100


    train_losses=[]

    val_losses=[]


    max_mrr_val = -1


    best_epoch = -1

    avg_mrr_values = []

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


            mrr_per_query = []

            for query in val_dict.keys():


                actual_rank_order = []         


                predicted_rank_order = []         



                for item in val_dict[query]:


                    actual_rank_order.append(item[1])


                    predicted_score = model.forward(torch.tensor(item[0],dtype = torch.float32))

                    predicted_rank_order.append((item[1],predicted_score))




                actual_rank_order = sorted(actual_rank_order,reverse = True)


                predicted_rank_order = sorted(predicted_rank_order,reverse = True, key = lambda x : x[1])



                top_label_position = -1



                for l in range(len(predicted_rank_order)):
                    if (predicted_rank_order[l][0] == actual_rank_order[0]) and (actual_rank_order[0] != 0):
                        top_label_position = l + 1

                        break


                if (actual_rank_order[0] != 0) and (top_label_position > 0):
                    mrr_per_query.append(1/top_label_position)


            avg_epoch_mrr = sum(mrr_per_query)/len(mrr_per_query)

            avg_mrr_values.append(avg_epoch_mrr)

            if avg_epoch_mrr > max_mrr_val:
                max_mrr_val = avg_epoch_mrr

                best_epoch = i

                torch.save(model.state_dict(),"model-weights.pth")




    test_model = ListNet(len(training_input[0][0][0]),5,5,1)


    test_model.load_state_dict(torch.load("model-weights.pth"))


    


    overall_mrr_to_calc = []



    with torch.no_grad():
        test_model.eval()

        for query in test_dict.keys():

            actual_rank_order = []

            predicted_rank_order = []


            for item in test_dict[query]:
                actual_rank_order.append(item[1])

                predicted_score = test_model.forward(torch.tensor(item[0],dtype = torch.float32))


                predicted_rank_order.append((item[1],predicted_score))


            actual_rank_order = sorted(actual_rank_order,reverse=True)

            predicted_rank_order = sorted(predicted_rank_order,reverse=True, key= lambda x : x[1])


            top_label_position = -1


            for i in range(len(predicted_rank_order)):
                if (predicted_rank_order[i][0] == actual_rank_order[0]) and (actual_rank_order[0] != 0):
                    top_label_position = i + 1

                    break



            if (actual_rank_order[0] != 0) and (top_label_position > 0): 
                overall_mrr_to_calc.append(1/top_label_position)


        #For this test dataset there is at least one query with at least one document allocated to it 
        #so no need to check if denominator is zero
        test_mrr = sum(overall_mrr_to_calc)/len(overall_mrr_to_calc)

        print("Test MRR: ", test_mrr)

        return max_mrr_val, best_epoch, test_mrr
    


data = get_data(1)


run_ndcg(data[0],data[1],data[2],data[3],10)