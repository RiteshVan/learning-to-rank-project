from ranknet import get_data_with_equal_pairs,get_data_without_equal_pairs,run_ndcg, run_mrr
import time
import numpy as np


#This is the testing file used to obtain results for the RankNet implementation.
#To start with all the required data is loaded in.
#Next, runs of the algorithm are carried out to obtain results for the metrics


#Metrics obtained are validation set NDCG scores,
#epochs taken to obtain best validation NDCG scores 
#and test set NDCG scores


#File created to track times to taken to carry out certain aspects of training
time_results_file = open("time_data.txt","w")
time_results_file.write("RESULTS:\n\n")


#Get start time for when the main part of the script is executed
results_run_time = time.time()


#Load in all required data from all folds where no document pairs of equal relevance are included
#Also times how long it takes to obtain data needed and writes this to the time data file
start_time = time.time()
fold_1_data = get_data_without_equal_pairs(1)
fold_2_data = get_data_without_equal_pairs(2)
fold_3_data = get_data_without_equal_pairs(3)
fold_4_data = get_data_without_equal_pairs(4)
fold_5_data = get_data_without_equal_pairs(5)
folds_data_without_equal_pairs = [fold_1_data,fold_2_data,fold_3_data,fold_4_data,fold_5_data]
time_results_file.write("Time taken to load in data without equal pairs: " + str(time.time() - start_time) + "\n")


#Load in data where document pairs of equal relevance are included
#Writes time taken to load in data to time data text file
start_time = time.time()
fold_1_data = get_data_with_equal_pairs(1)
fold_2_data = get_data_with_equal_pairs(2)
fold_3_data = get_data_with_equal_pairs(3)
fold_4_data = get_data_with_equal_pairs(4)
fold_5_data = get_data_with_equal_pairs(5)
folds_data_with_equal_pairs = [fold_1_data,fold_2_data,fold_3_data,fold_4_data,fold_5_data]
time_results_file.write("Time taken to load in data with equal pairs: " +  str(time.time() - start_time) + "\n")





#Creates file store NDCG scores for input data where no equal pairs are included
results_file = open("results_ranknet_ndcg_without_equal_pairs.txt" , "w")
results_file.write("RESULTS:\n")
results_file.write("\n")
results_file.write("\n")



#Create list of all NDCG@k metrics to obtain results for
ndcg_vals = [1,2,5,10,20]

#Iterate through NDCG cutoff points
for val in ndcg_vals:
    results_file.write("NDCG@" + str(val) +" outputs\n")
    results_file.write("\n")

    #Used to calculated averages across folds
    fold_avg_vali_ndcg = []
    fold_avg_epoch = []
    fold_avg_test_ndcg = []

    #Iterate through all folds
    for i in range(len(folds_data_without_equal_pairs)):
        #Arrays used to find average values
        vali_ndcg_avg = []
        avg_epoch = []
        test_ndcg_avg = []

        #Run function 10 times
        for _ in range(10):
            values = run_ndcg(folds_data_without_equal_pairs[i][0],folds_data_without_equal_pairs[i][1],folds_data_without_equal_pairs[i][2],folds_data_without_equal_pairs[i][3],val)

            #Store values obtained in arrays
            vali_ndcg_avg.append(values[0])
            avg_epoch.append(values[1])
            test_ndcg_avg.append(values[2])

        #Update fold average arrays to calculate average at end
        fold_avg_vali_ndcg.append(sum(vali_ndcg_avg)/len(vali_ndcg_avg))

        fold_avg_epoch.append(sum(avg_epoch)/len(avg_epoch))


        fold_avg_test_ndcg.append(sum(test_ndcg_avg)/len(test_ndcg_avg))


        #Write data obtained to text file
        results_file.write("Fold "+ str(i+1) + " val NDCG: " + str(sum(vali_ndcg_avg)/len(vali_ndcg_avg)) + "\n")
        results_file.write("Fold "+ str(i+1) + " epochs: " + str(sum(avg_epoch)/len(avg_epoch))+ "\n")
        results_file.write("Fold "+ str(i+1) + " test NDCG: " + str(sum(test_ndcg_avg)/len(test_ndcg_avg))+ "\n")
        results_file.write("\n")



    #Calulate and write averages across folds
    results_file.write("AVG FOLD val NDCG: " + str(sum(fold_avg_vali_ndcg)/ len(fold_avg_vali_ndcg)) + "\n" )
    results_file.write("AVG FOLD epochs: " + str(sum(fold_avg_epoch)/ len(fold_avg_epoch) ) + "\n" )
    results_file.write("AVD FOLD test NDCG: " + str(sum(fold_avg_test_ndcg)/ len(fold_avg_test_ndcg)) + "\n" )
    results_file.write("\n")


    #Calculate standard deviations across folds
    results_file.write("ST. DEVIATION val NDCG: " + str(np.std(fold_avg_vali_ndcg)) + "\n")
    results_file.write("ST. DEVIATION epochs: " + str(np.std(fold_avg_epoch)) + "\n")
    results_file.write("ST. DEVIATION test NDCG: " + str(np.std(fold_avg_test_ndcg)) + "\n")

#Close to free up resources
results_file.close()








#Create text file to store NDCG scores for input data where document pairs of equal relevance are included
results_file = open("results_ranknet_ndcg_with_equal_pairs.txt" , "w")
results_file.write("RESULTS:\n")
results_file.write("\n")
results_file.write("\n")




#NDCG@k cutoffs stored
ndcg_vals = [1,2,5,10,20]

#Iterate through all NDCG cutoff points
for val in ndcg_vals:
    results_file.write("NDCG@" + str(val) +" outputs\n")
    results_file.write("\n")


    #Used to calculated averages across folds
    fold_avg_vali_ndcg = []
    fold_avg_epoch = []
    fold_avg_test_ndcg = []


    #Iterate through all folds
    for i in range(len(folds_data_with_equal_pairs)):

        #Arrays to calculate average values
        vali_ndcg_avg = []
        avg_epoch = []
        test_ndcg_avg = []

        #Run function ten times
        for _ in range(10):
            values = run_ndcg(folds_data_with_equal_pairs[i][0],folds_data_with_equal_pairs[i][1],folds_data_with_equal_pairs[i][2],folds_data_with_equal_pairs[i][3],val)

            #Store values obtained in arrays
            vali_ndcg_avg.append(values[0])
            avg_epoch.append(values[1])
            test_ndcg_avg.append(values[2])


        #Update fold average arrays to calculate average at end
        fold_avg_vali_ndcg.append(sum(vali_ndcg_avg)/len(vali_ndcg_avg))
        
        fold_avg_epoch.append(sum(avg_epoch)/len(avg_epoch))
        
        
        fold_avg_test_ndcg.append(sum(test_ndcg_avg)/len(test_ndcg_avg))


        #Write results obtained for a fold
        results_file.write("Fold "+ str(i+1) + " val NDCG: " + str(sum(vali_ndcg_avg)/len(vali_ndcg_avg)) + "\n")
        results_file.write("Fold "+ str(i+1) + " epochs: " + str(sum(avg_epoch)/len(avg_epoch))+ "\n")
        results_file.write("Fold "+ str(i+1) + " test NDCG: " + str(sum(test_ndcg_avg)/len(test_ndcg_avg))+ "\n")
        results_file.write("\n")


    #Wrtte average values obtained across folds
    results_file.write("AVG FOLD val NDCG: " + str(sum(fold_avg_vali_ndcg)/ len(fold_avg_vali_ndcg)) + "\n" )
    results_file.write("AVG FOLD epochs: " + str(sum(fold_avg_epoch)/ len(fold_avg_epoch) ) + "\n" )
    results_file.write("AVD FOLD test NDCG: " + str(sum(fold_avg_test_ndcg)/ len(fold_avg_test_ndcg)) + "\n" )
    results_file.write("\n")

    #Write standard deviation values obtained across folds
    results_file.write("ST. DEVIATION val NDCG: " + str(np.std(fold_avg_vali_ndcg)) + "\n")
    results_file.write("ST. DEVIATION epochs: " + str(np.std(fold_avg_epoch)) + "\n")
    results_file.write("ST. DEVIATION test NDCG: " + str(np.std(fold_avg_test_ndcg)) + "\n")


#Close file to free memory
results_file.close()




#Create file for MRR results for input that does not contain document pairs with equal relevancy labels
results_file = open("results_ranknet_MRR_without_equal_pairs.txt" , "w")
results_file.write("RESULTS:\n")
results_file.write("\n")
results_file.write("\n")




#Used to calculate averages across folds
fold_avg_vali_mrr = []
fold_avg_epoch = []
fold_avg_test_mrr = []


#Iterate through all folds
for i in range(len(folds_data_without_equal_pairs)):

    #Used to store averages 
    vali_mrr_avg = []
    avg_epoch = []
    test_mrr_avg = []
    for _ in range(10):
        values = run_mrr(folds_data_without_equal_pairs[i][0],folds_data_without_equal_pairs[i][1],folds_data_without_equal_pairs[i][2],folds_data_without_equal_pairs[i][3])

        #Add values to array
        vali_mrr_avg.append(values[0])
        avg_epoch.append(values[1])
        test_mrr_avg.append(values[2])


    #Calculate average values obtained for current fold
    #and add to arrays to a later find averages across folds
    fold_avg_vali_mrr.append(sum(vali_mrr_avg)/len(vali_mrr_avg))

    fold_avg_epoch.append(sum(avg_epoch)/len(avg_epoch))

    fold_avg_test_mrr.append(sum(test_mrr_avg)/len(test_mrr_avg))



    #Write averages obtained to text file
    results_file.write("Fold "+ str(i+1) + " val MRR: " + str(sum(vali_mrr_avg)/len(vali_mrr_avg)) + "\n")
    results_file.write("Fold "+ str(i+1) + " epochs: " + str(sum(avg_epoch)/len(avg_epoch))+ "\n")
    results_file.write("Fold "+ str(i+1) + " test MRR: " + str(sum(test_mrr_avg)/len(test_mrr_avg))+ "\n")
    results_file.write("\n")


#Write averages obtained across folds to text file
results_file.write("AVG FOLD val MRR: " + str(sum(fold_avg_vali_mrr)/ len(fold_avg_vali_mrr)) + "\n" )
results_file.write("AVG FOLD epochs: " + str(sum(fold_avg_epoch)/ len(fold_avg_epoch) ) + "\n" )
results_file.write("AVD FOLD test MRR: " + str(sum(fold_avg_test_mrr)/ len(fold_avg_test_mrr)) + "\n" )
results_file.write("\n")


#Write standard deviation across folds to text file
results_file.write("ST. DEVIATION val MRR: " + str(np.std(fold_avg_vali_mrr)) + "\n")
results_file.write("ST. DEVIATION epochs: " + str(np.std(fold_avg_epoch)) + "\n")
results_file.write("ST. DEVIATION test MRR: " + str(np.std(fold_avg_test_mrr)) + "\n")


#Close file to free resources
results_file.close()









#Create text file to store NDCG scores for input data that contains documents pairs with equal relevancy labels
results_file = open("results_ranknet_MRR_with_equal_pairs.txt" , "w")
results_file.write("RESULTS:\n")
results_file.write("\n")
results_file.write("\n")






#Used to calculated averages across folds
fold_avg_vali_mrr = []
fold_avg_epoch = []
fold_avg_test_mrr = []


#Iterate through folds
for i in range(len(folds_data_with_equal_pairs)):

    #Arrays to calculate average values
    vali_mrr_avg = []
    avg_epoch = []
    test_mrr_avg = []

    #Run function to get results ten times
    for _ in range(10):
        values = run_mrr(folds_data_with_equal_pairs[i][0],folds_data_with_equal_pairs[i][1],folds_data_with_equal_pairs[i][2],folds_data_with_equal_pairs[i][3])

        #Add values to prior arrays
        vali_mrr_avg.append(values[0])
        avg_epoch.append(values[1])
        test_mrr_avg.append(values[2])


    #Calculate averages and ad to fold average arrays 
    #to later find average values across folds
    fold_avg_vali_mrr.append(sum(vali_mrr_avg)/len(vali_mrr_avg))
    
    fold_avg_epoch.append(sum(avg_epoch)/len(avg_epoch))
    
    fold_avg_test_mrr.append(sum(test_mrr_avg)/len(test_mrr_avg))



    #Write fold results obtained to text file
    results_file.write("Fold "+ str(i+1) + " val MRR: " + str(sum(vali_mrr_avg)/len(vali_mrr_avg)) + "\n")
    results_file.write("Fold "+ str(i+1) + " epochs: " + str(sum(avg_epoch)/len(avg_epoch))+ "\n")
    results_file.write("Fold "+ str(i+1) + " test MRR: " + str(sum(test_mrr_avg)/len(test_mrr_avg))+ "\n")
    results_file.write("\n")


#Write averages across folds to text file
results_file.write("AVG FOLD val MRR: " + str(sum(fold_avg_vali_mrr)/ len(fold_avg_vali_mrr)) + "\n" )
results_file.write("AVG FOLD epochs: " + str(sum(fold_avg_epoch)/ len(fold_avg_epoch) ) + "\n" )
results_file.write("AVD FOLD test MRR: " + str(sum(fold_avg_test_mrr)/ len(fold_avg_test_mrr)) + "\n" )
results_file.write("\n")


#Write standard deviations across folds to text file
results_file.write("ST. DEVIATION val MRR: " + str(np.std(fold_avg_vali_mrr)) + "\n")
results_file.write("ST. DEVIATION epochs: " + str(np.std(fold_avg_epoch)) + "\n")
results_file.write("ST. DEVIATION test MRR: " + str(np.std(fold_avg_test_mrr)) + "\n")


#Close file to free resources 
results_file.close()


#Calculate overall time to complete process fo obtaining results 
#and write to text file
time_results_file.write("Time taken to run all tests: " +  str(time.time() - results_run_time) + "\n")
time_results_file.close()