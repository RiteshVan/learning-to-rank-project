import torch
import torch.nn as nn
import re
import matplotlib.pyplot as plt
import numpy as np
import math

def get_data(fold : int):
    #Path to training dataset
    train_dataset_name = "/Users/ritesh/Desktop/project-extra/MQ2008/Fold" +  str(fold) + "/train.txt"


    #Path to validation dataset
    val_dataset_name = "/Users/ritesh/Desktop/project-extra/MQ2008/Fold" +  str(fold) + "/vali.txt"


    #Path to test dataset
    test_dataset_name = "/Users/ritesh/Desktop/project-extra/MQ2008/Fold" +  str(fold) + "/test.txt"



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

    

    training_input = get_network_input(train_dataset_name)


    validation_input = get_network_input(val_dataset_name)

    val_dict = get_dict(val_dataset_name)

    test_dict = get_dict(test_dataset_name)


    return training_input,validation_input,val_dict,test_dict













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
            x= self.out(x)

            return x

    #The length of input is equal to number of features in document
    #Two hidden layers of size 5 and output of size 1
    model = ListNet(len(training_input[0][0][0]),5,5,1)


    training_input = training_input


    validation_input = validation_input




    val_dict = val_dict



    optimiser = torch.optim.Adam(model.parameters(),lr=0.001)

    epochs = 100


    train_losses=[]

    val_losses=[]


    val_NDCG_tracking =[]


    max_ndcg_val=-1

    best_epoch = -1

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



                actual_rank_order = sorted(actual_rank_order,reverse=True)[:ndcg_k]

                predicted_rank_order = sorted(predicted_rank_order,reverse=True,key= lambda x : x[2])[:ndcg_k]


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









    test_model =ListNet(len(training_input[0][0][0]),5,5,1)


    test_model.load_state_dict(torch.load("model-weights.pth"))

    test_dict = test_dict

    overall_ndcg_to_calc = []


    with torch.no_grad():
        test_model.eval()

        for query in test_dict.keys():


            actual_rank_order = []

            predicted_rank_order = []


            for item in test_dict[query]:

                actual_rank_order.append(item[1])

                predicted_score = model.forward(torch.tensor(item[0],dtype=torch.float32))

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



    test_ndcg = sum(overall_ndcg_to_calc)/len(overall_ndcg_to_calc)


    print("Best Validation NDCG: ", max_ndcg_val)
    print("Best epoch: ", best_epoch)
    print("Test NDCG: ", test_ndcg)


    

    return max_ndcg_val,best_epoch,test_ndcg








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
            x= self.out(x)

            return x

    #The length of input is equal to number of features in document
    #Two hidden layers of size 5 and output of size 1
    model = ListNet(len(training_input[0][0][0]),5,5,1)



    training_input = training_input


    validation_input = validation_input




    val_dict = val_dict



    optimiser = torch.optim.Adam(model.parameters(),lr=0.001)

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



                for i in range(len(predicted_score)):
                    if (predicted_rank_order[i][0] == actual_rank_order[0]) and (actual_rank_order[0] != 0):
                        top_label_position = i+1

                        break


                if (actual_rank_order[0] != 0):
                    mrr_per_query.append(1/top_label_position)


            avg_epoch_mrr = sum(mrr_per_query)/len(mrr_per_query)

            avg_mrr_values.append(avg_epoch_mrr)

            if avg_epoch_mrr > max_mrr_val:
                max_mrr_val = avg_epoch_mrr

                best_epoch = i

                torch.save(model.state_dict(),"model-weights.pth")




    test_model = ListNet(len(training_input[0][0][0]),5,5,1)


    test_model.load_state_dict(torch.load("model-weights.pth"))


    test_dict = test_dict


    overall_mrr_to_calc = []



    with torch.no_grad():
        test_model.eval()

        for query in test_dict.keys():

            actual_rank_order = []

            predicted_rank_order = []


            for item in test_dict[query]:
                actual_rank_order.append(item[1])

                predicted_score = model.forward(torch.tensor(item[0],dtype = torch.float32))


                predicted_rank_order.append((item[1],predicted_score))


            actual_rank_order = sorted(actual_rank_order,reverse=True)

            predicted_rank_order = sorted(predicted_rank_order,reverse=True, key= lambda x : x[1])


            top_label_position = -1


            for i in range(len(predicted_rank_order)):
                if (predicted_rank_order[i][0] == actual_rank_order[0]) and (actual_rank_order[0] != 0):
                    top_label_position = i + 1

                    break



            if (actual_rank_order[0] != 0):
                overall_mrr_to_calc.append(1/top_label_position)



        test_mrr = sum(overall_mrr_to_calc)/len(overall_mrr_to_calc)

        print("Test MRR: ", test_mrr)

        return max_mrr_val, best_epoch, test_mrr