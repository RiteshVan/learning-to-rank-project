from ranknet import get_data_without_equal_pairs,get_data_with_equal_pairs,run_ndcg
import numpy as np


#For this file the testing operations are similar to those from testing.py
#Key difference is that process is only carried out at NDCG@20 for queries with above 10 documents associated to them



#Get all data for folds without equal pairs 
folds_1_data = get_data_without_equal_pairs(1,10)
folds_2_data = get_data_without_equal_pairs(2,10)
folds_3_data = get_data_without_equal_pairs(3,10)
folds_4_data = get_data_without_equal_pairs(4,10)
folds_5_data = get_data_without_equal_pairs(5,10)
folds_without_equal_pairs = [folds_1_data,folds_2_data,folds_3_data,folds_4_data,folds_5_data]


#Get all data for folds with equal pairs 
folds_1_data = get_data_with_equal_pairs(1,10)
folds_2_data = get_data_with_equal_pairs(2,10)
folds_3_data = get_data_with_equal_pairs(3,10)
folds_4_data = get_data_with_equal_pairs(4,10)
folds_5_data = get_data_with_equal_pairs(5,10)
folds_with_equal_pairs = [folds_1_data,folds_2_data,folds_3_data,folds_4_data,folds_5_data]



#create file to store results
results_file = open("filtered_results_ndcg20_without_equal_pairs.txt","w")


#Write title then add space for clarity
results_file.write("RESULTS NDCG20 FILTERED WITHOUT EQUAL PAIRS:\n")
results_file.write("\n")
results_file.write("\n")


#Used to calculate averages
fold_avg_vali_ndcg = []
fold_avg_epochs = []
fold_avg_test_ndcg = []


for i in range(len(folds_without_equal_pairs)):
    vali_ndcg_avg = []
    epoch_avg = []
    test_ndcg_avg = []
    for _ in range(10):
        values = run_ndcg(folds_without_equal_pairs[i][0],
                          folds_without_equal_pairs[i][1],
                          folds_without_equal_pairs[i][2],
                          folds_without_equal_pairs[i][3],20)
        vali_ndcg_avg.append(values[0])
        epoch_avg.append(values[1])
        test_ndcg_avg.append(values[2])



    fold_avg_vali_ndcg.append(sum(vali_ndcg_avg)/len(vali_ndcg_avg))

    fold_avg_epochs.append(sum(epoch_avg)/len(epoch_avg))

    fold_avg_test_ndcg.append(sum(test_ndcg_avg)/len(test_ndcg_avg))


    results_file.write("Fold " + str(i + 1) + " val NDCG: " + str(sum(vali_ndcg_avg)/len(vali_ndcg_avg)) + "\n")

    results_file.write("Fold " + str(i + 1) + " epochs: " + str(sum(epoch_avg)/len(epoch_avg)) + "\n")

    results_file.write("Fold " + str(i + 1) + " test NDCG: " + str(sum(test_ndcg_avg)/len(test_ndcg_avg)) + "\n")
    results_file.write("\n")




results_file.write("AVG FOLD val NDCG: " + str(sum(fold_avg_vali_ndcg)/len(fold_avg_vali_ndcg)) + "\n" )

results_file.write("AVG FOLD epochs: " + str(sum(fold_avg_epochs)/len(fold_avg_epochs)) + "\n")

results_file.write("AVG FOLD test NDCG: " + str(sum(fold_avg_test_ndcg)/len(fold_avg_test_ndcg)) + "\n" )
results_file.write("\n")


results_file.write("ST DEVIATION val NDCG: " + str(np.std(fold_avg_vali_ndcg)) + "\n")

results_file.write("ST DEVIATION epoch: " + str(np.std(fold_avg_epochs)) + "\n")

results_file.write("ST DEVIATION test NDCG: " + str(np.std(fold_avg_test_ndcg)) + "\n")



results_file.close()

















#create file to store results
results_file = open("filtered_results_ndcg20_with_equal_pairs.txt","w")


#Write title then add space for clarity
results_file.write("RESULTS NDCG20 FILTERED WITH EQUAL PAIRS:\n")
results_file.write("\n")
results_file.write("\n")


#Used to calculate averages
fold_avg_vali_ndcg = []
fold_avg_epochs = []
fold_avg_test_ndcg = []


for i in range(len(folds_with_equal_pairs)):
    vali_ndcg_avg = []
    epoch_avg = []
    test_ndcg_avg = []
    for _ in range(10):
        values = run_ndcg(folds_with_equal_pairs[i][0],
                          folds_with_equal_pairs[i][1],
                          folds_with_equal_pairs[i][2],
                          folds_with_equal_pairs[i][3],20)
        vali_ndcg_avg.append(values[0])
        epoch_avg.append(values[1])
        test_ndcg_avg.append(values[2])



    fold_avg_vali_ndcg.append(sum(vali_ndcg_avg)/len(vali_ndcg_avg))

    fold_avg_epochs.append(sum(epoch_avg)/len(epoch_avg))

    fold_avg_test_ndcg.append(sum(test_ndcg_avg)/len(test_ndcg_avg))


    results_file.write("Fold " + str(i + 1) + " val NDCG: " + str(sum(vali_ndcg_avg)/len(vali_ndcg_avg)) + "\n")

    results_file.write("Fold " + str(i + 1) + " epochs: " + str(sum(epoch_avg)/len(epoch_avg)) + "\n")

    results_file.write("Fold " + str(i + 1) + " test NDCG: " + str(sum(test_ndcg_avg)/len(test_ndcg_avg)) + "\n")
    results_file.write("\n")




results_file.write("AVG FOLD val NDCG: " + str(sum(fold_avg_vali_ndcg)/len(fold_avg_vali_ndcg)) + "\n" )

results_file.write("AVG FOLD epochs: " + str(sum(fold_avg_epochs)/len(fold_avg_epochs)) + "\n")

results_file.write("AVG FOLD test NDCG: " + str(sum(fold_avg_test_ndcg)/len(fold_avg_test_ndcg)) + "\n" )
results_file.write("\n")


results_file.write("ST DEVIATION val NDCG: " + str(np.std(fold_avg_vali_ndcg)) + "\n")

results_file.write("ST DEVIATION epoch: " + str(np.std(fold_avg_epochs)) + "\n")

results_file.write("ST DEVIATION test NDCG: " + str(np.std(fold_avg_test_ndcg)) + "\n")



results_file.close()


