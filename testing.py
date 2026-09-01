from listnet import run_ndcg,get_data,run_mrr
import time
import numpy as np


#Similar to testing file used in RankNet implementation

time_results = open("times_taken.txt","w")

execution_start_time = time.time()


data_load_time = time.time()

fold_1_data = get_data(1)
fold_2_data = get_data(2)
fold_3_data = get_data(3)
fold_4_data = get_data(4)
fold_5_data = get_data(5)
folds_data = [fold_1_data,fold_2_data,fold_3_data,fold_4_data,fold_5_data]


time_results.write("Time taken to load in all data: " + str(time.time() - data_load_time)  + "\n")


ndcg_calculations_time = time.time()


results_file = open("listnet_results_ndcg.txt","w")

results_file.write("LISTNET NDCG RESULTS:\n")
results_file.write("\n")
results_file.write("\n")


ndcg_vals = [1,2,5,10,20]




for val in ndcg_vals:
    results_file.write("NDCG@" + str(val) +" outputs\n")
    results_file.write("\n")


    avg_fold__vali_ndcg = [] 
    avg_fold_epoch = []
    avg_fold__test_ndcg = [] 



    for i in range(len(folds_data)):
        vali_ndcg_avg = []
        epoch_avg = []
        test_ndcg_avg = []
        for _ in range(10):
            values = run_ndcg(folds_data[i][0],folds_data[i][1],folds_data[i][2],folds_data[i][3],val)
            vali_ndcg_avg.append(values[0])
            epoch_avg.append(values[1])
            test_ndcg_avg.append(values[2])

        results_file.write("Fold "+ str(i+1) + " val NDCG: " + str(sum(vali_ndcg_avg)/len(vali_ndcg_avg)) + "\n")
        results_file.write("Fold "+ str(i+1) + " epochs: " + str(sum(epoch_avg)/len(epoch_avg))+ "\n")
        results_file.write("Fold "+ str(i+1) + " test NDCG: " + str(sum(test_ndcg_avg)/len(test_ndcg_avg))+ "\n")
        results_file.write("\n")


        avg_fold__vali_ndcg.append(sum(vali_ndcg_avg)/len(vali_ndcg_avg))
        avg_fold_epoch.append(sum(epoch_avg)/len(epoch_avg))
        avg_fold__test_ndcg.append(sum(test_ndcg_avg)/len(test_ndcg_avg))

    results_file.write("AVG fold vali NDCG: " + str(sum(avg_fold__vali_ndcg)/len(avg_fold__vali_ndcg)) + '\n' )
    results_file.write("AVG fold epochs: " + str(sum(avg_fold_epoch)/len(avg_fold_epoch)) + '\n' )
    results_file.write("AVG fold test NDCG: " + str(sum(avg_fold__test_ndcg)/len(avg_fold__test_ndcg)) + '\n' )
    results_file.write("\n")


    results_file.write("ST. DEVIATION vali NDCG: " + str(np.std(avg_fold__vali_ndcg)) + "\n")
    results_file.write("ST. DEVIATION epochs: " + str( np.std(avg_fold_epoch) ) + "\n")
    results_file.write("ST. DEVIATION test NDCG: " + str(np.std(avg_fold__test_ndcg)) + "\n")
    results_file.write("\n")


    


results_file.close()




time_results.write("Time taken to get NDCG results: " + str(time.time() - ndcg_calculations_time )  + "\n")





mrr_calculations_time = time.time()


results_file = open("listnet_results_mrr.txt","w")


results_file.write("LISTNET MRR RESULTS:\n")
results_file.write("\n")
results_file.write("\n")



avg_fold__vali_mrr = [] 
avg_fold_epoch = []
avg_fold__test_mrr = [] 



for i in range(len(folds_data)):
    vali_mrr_avg = []
    epoch_avg = []
    test_mrr_avg = []
    for _ in range(10):
        values = run_mrr(folds_data[i][0],folds_data[i][1],folds_data[i][2],folds_data[i][3])
        vali_mrr_avg.append(values[0])
        epoch_avg.append(values[1])
        test_mrr_avg.append(values[2])

    results_file.write("Fold "+ str(i+1) + " val MRR: " + str(sum(vali_mrr_avg)/len(vali_mrr_avg)) + "\n")
    results_file.write("Fold "+ str(i+1) + " epochs: " + str(sum(epoch_avg)/len(epoch_avg))+ "\n")
    results_file.write("Fold "+ str(i+1) + " test MRR: " + str(sum(test_mrr_avg)/len(test_mrr_avg))+ "\n")
    results_file.write("\n")

    avg_fold__vali_mrr.append(sum(vali_mrr_avg)/len(vali_mrr_avg))
    avg_fold_epoch.append(sum(epoch_avg)/len(epoch_avg))
    avg_fold__test_mrr.append(sum(test_mrr_avg)/len(test_mrr_avg))

results_file.write("AVG fold val MRR: " + str(sum(avg_fold__vali_mrr)/len(avg_fold__vali_mrr)) + "\n" )
results_file.write("AVG fold epochs: " + str(sum(avg_fold_epoch)/len(avg_fold_epoch)) + "\n" )
results_file.write("AVG fold test MRR: " + str(sum(avg_fold__test_mrr)/len(avg_fold__test_mrr)) + "\n")
results_file.write("\n")




results_file.write("ST. DEVIATION val MRR: " + str(np.std(avg_fold__vali_mrr)) + "\n")
results_file.write("ST. DEVIATION epochs: " + str(np.std(avg_fold_epoch)) + "\n")
results_file.write("ST. DEVIATION test MRR: " + str(np.std(avg_fold__test_mrr)) + "\n" )
results_file.write("\n")

results_file.close()



time_results.write("Time taken to get MRR results: " + str(time.time() - mrr_calculations_time ) + "\n")


time_results.write("Time take for whole execution: " + str(time.time() - execution_start_time)  + "\n")


time_results.close()