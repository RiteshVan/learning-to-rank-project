from listnet import get_data, run_mrr,run_ndcg
import numpy as np

fold_1_data = get_data(1,10)
fold_2_data = get_data(2,10)
fold_3_data = get_data(3,10)
fold_4_data = get_data(4,10)
fold_5_data = get_data(5,10)
folds_data = [fold_1_data, fold_2_data, fold_3_data, fold_4_data, fold_5_data]



results_file = open("filtered_listnet_results_ndcg20.txt","w")

results_file.write("NDCG@20 RESULTS FILTERED:\n")
results_file.write("\n")

avg_fold_vali_ndcg = []
avg_fold_epochs = []
avg_fold_test_ndcg = []

for i in range(len(folds_data)):

    fold_vali_ndcg = []
    fold_epochs = []
    fold_test_ndcg = []

    for _ in range(10):
        data = run_ndcg(folds_data[i][0],folds_data[i][1],folds_data[i][2],folds_data[i][3],20)
        fold_vali_ndcg.append(data[0])
        fold_epochs.append(data[1])
        fold_test_ndcg.append(data[2])


    results_file.write("Fold " + str(i + 1) + " val NDCG: " +  str(sum(fold_vali_ndcg)/len(fold_vali_ndcg))   + "\n")
    results_file.write("Fold " + str(i + 1) + " epochs: " + str(sum(fold_epochs)/len(fold_epochs)) + "\n")
    results_file.write("Fold " + str(i + 1) + " test NDCG: " +  str(sum(fold_test_ndcg)/len(fold_test_ndcg))   + "\n")
    results_file.write("\n")


    avg_fold_vali_ndcg.append(sum(fold_vali_ndcg)/len(fold_vali_ndcg) )
    avg_fold_epochs.append(sum(fold_epochs)/len(fold_epochs))
    avg_fold_test_ndcg.append(sum(fold_test_ndcg)/len(fold_test_ndcg))

results_file.write("AVG FOLD val NDCG: " + str(sum(avg_fold_vali_ndcg)/len(avg_fold_vali_ndcg))+ "\n")
results_file.write("AVG FOLD epochs: " + str(sum(avg_fold_epochs)/len(avg_fold_epochs)) + "\n")
results_file.write("AVG FOLD test NDCG: " + str(sum(avg_fold_test_ndcg)/len(avg_fold_test_ndcg)) + "\n")
results_file.write("\n")

results_file.write("ST. DEVIATION val NDCG: " + str(np.std(avg_fold_vali_ndcg)) + "\n")
results_file.write("ST. DEVIATION epochs: " + str(np.std(avg_fold_epochs)) + "\n")
results_file.write("ST. DEVIATION test NDCG: " + str(np.std(avg_fold_test_ndcg)) + "\n")



results_file.close()