from ranknet import get_data_with_equal_pairs,get_data_without_equal_pairs,run_ndcg, run_mrr
import time


# results_file = open("results.txt" , "w")
# results_file.write("RESULTS:\n")
# results_file.write("\n")
# results_file.write("\n")


# ndcg_vals = [1,2,5,10]

# folds = [1,2,3,4,5]


# for val in ndcg_vals:
#     results_file.write("NDCG@" + str(val) +" outputs\n")
#     results_file.write("\n")
#     for fold in folds:
#         vali_ndcg_avg = []
#         avg_epoch = []
#         test_ndcg_avg = []
#         for i in range(10):
#             values = run(fold,val)
#             vali_ndcg_avg.append(values[0])
#             avg_epoch.append(values[1])
#             test_ndcg_avg.append(values[2])        
            

#         results_file.write("Fold "+ str(fold) + " val NDCG: " + str(sum(vali_ndcg_avg)/len(vali_ndcg_avg)) + "\n")
#         results_file.write("Fold "+ str(fold) + " epochs: " + str(sum(avg_epoch)/len(avg_epoch))+ "\n")
#         results_file.write("Fold "+ str(fold) + " test NDCG: " + str(sum(test_ndcg_avg)/len(test_ndcg_avg))+ "\n")
#         results_file.write("\n")

        

    



# results_file.close()


time_results_file = open("time_data.txt","w")
time_results_file.write("RESULTS:\n\n")


results_run_time = time.time()


start_time = time.time()
fold_1_data = get_data_without_equal_pairs(1)
fold_2_data = get_data_without_equal_pairs(2)
fold_3_data = get_data_without_equal_pairs(3)
fold_4_data = get_data_without_equal_pairs(4)
fold_5_data = get_data_without_equal_pairs(5)
folds_data_without_equal_pairs = [fold_1_data,fold_2_data,fold_3_data,fold_4_data,fold_5_data]
time_results_file.write("Time taken to load in data without equal pairs: " + str(time.time() - start_time) + "\n")



start_time = time.time()
fold_1_data = get_data_with_equal_pairs(1)
fold_2_data = get_data_with_equal_pairs(2)
fold_3_data = get_data_with_equal_pairs(3)
fold_4_data = get_data_with_equal_pairs(4)
fold_5_data = get_data_with_equal_pairs(5)
folds_data_with_equal_pairs = [fold_1_data,fold_2_data,fold_3_data,fold_4_data,fold_5_data]
time_results_file.write("Time taken to load in data with equal pairs: " +  str(time.time() - start_time) + "\n")






results_file = open("results_ranknet_ndcg_without_equal_pairs.txt" , "w")
results_file.write("RESULTS:\n")
results_file.write("\n")
results_file.write("\n")




ndcg_vals = [1,2,5,10,20]

for val in ndcg_vals:
    results_file.write("NDCG@" + str(val) +" outputs\n")
    results_file.write("\n")

    #Used to calculated averages across folds
    fold_avg_vali_ndcg = []
    fold_avg_epoch = []
    fold_avg_test_ndcg = []


    for i in range(len(folds_data_without_equal_pairs)):
        vali_ndcg_avg = []
        avg_epoch = []
        test_ndcg_avg = []
        for _ in range(10):
            values = run_ndcg(folds_data_without_equal_pairs[i][0],folds_data_without_equal_pairs[i][1],folds_data_without_equal_pairs[i][2],folds_data_without_equal_pairs[i][3],val)
            vali_ndcg_avg.append(values[0])
            avg_epoch.append(values[1])
            test_ndcg_avg.append(values[2])

        #Update fold average arrays to calculate average at end
        fold_avg_vali_ndcg.append(sum(vali_ndcg_avg)/len(vali_ndcg_avg))

        fold_avg_epoch.append(sum(avg_epoch)/len(avg_epoch))


        fold_avg_test_ndcg.append(sum(test_ndcg_avg)/len(test_ndcg_avg))

        results_file.write("Fold "+ str(i+1) + " val NDCG: " + str(sum(vali_ndcg_avg)/len(vali_ndcg_avg)) + "\n")
        results_file.write("Fold "+ str(i+1) + " epochs: " + str(sum(avg_epoch)/len(avg_epoch))+ "\n")
        results_file.write("Fold "+ str(i+1) + " test NDCG: " + str(sum(test_ndcg_avg)/len(test_ndcg_avg))+ "\n")
        results_file.write("\n")


    results_file.write("AVG FOLD val NDCG: " + str(sum(fold_avg_vali_ndcg)/ len(fold_avg_vali_ndcg)) + "\n" )
    results_file.write("AVG FOLD epochs: " + str(sum(fold_avg_epoch)/ len(fold_avg_epoch) ) + "\n" )
    results_file.write("AVD FOLD test NDCG: " + str(sum(fold_avg_test_ndcg)/ len(fold_avg_test_ndcg)) + "\n" )
    results_file.write("\n")


results_file.close()









results_file = open("results_ranknet_ndcg_with_equal_pairs.txt" , "w")
results_file.write("RESULTS:\n")
results_file.write("\n")
results_file.write("\n")





ndcg_vals = [1,2,5,10,20]

for val in ndcg_vals:
    results_file.write("NDCG@" + str(val) +" outputs\n")
    results_file.write("\n")


    #Used to calculated averages across folds
    fold_avg_vali_ndcg = []
    fold_avg_epoch = []
    fold_avg_test_ndcg = []



    for i in range(len(folds_data_with_equal_pairs)):
        vali_ndcg_avg = []
        avg_epoch = []
        test_ndcg_avg = []
        for _ in range(10):
            values = run_ndcg(folds_data_with_equal_pairs[i][0],folds_data_with_equal_pairs[i][1],folds_data_with_equal_pairs[i][2],folds_data_with_equal_pairs[i][3],val)
            vali_ndcg_avg.append(values[0])
            avg_epoch.append(values[1])
            test_ndcg_avg.append(values[2])


        #Update fold average arrays to calculate average at end
        fold_avg_vali_ndcg.append(sum(vali_ndcg_avg)/len(vali_ndcg_avg))
        
        fold_avg_epoch.append(sum(avg_epoch)/len(avg_epoch))
        
        
        fold_avg_test_ndcg.append(sum(test_ndcg_avg)/len(test_ndcg_avg))

        results_file.write("Fold "+ str(i+1) + " val NDCG: " + str(sum(vali_ndcg_avg)/len(vali_ndcg_avg)) + "\n")
        results_file.write("Fold "+ str(i+1) + " epochs: " + str(sum(avg_epoch)/len(avg_epoch))+ "\n")
        results_file.write("Fold "+ str(i+1) + " test NDCG: " + str(sum(test_ndcg_avg)/len(test_ndcg_avg))+ "\n")
        results_file.write("\n")


    results_file.write("AVG FOLD val NDCG: " + str(sum(fold_avg_vali_ndcg)/ len(fold_avg_vali_ndcg)) + "\n" )
    results_file.write("AVG FOLD epochs: " + str(sum(fold_avg_epoch)/ len(fold_avg_epoch) ) + "\n" )
    results_file.write("AVD FOLD test NDCG: " + str(sum(fold_avg_test_ndcg)/ len(fold_avg_test_ndcg)) + "\n" )
    results_file.write("\n")


results_file.close()





results_file = open("results_ranknet_MRR_without_equal_pairs.txt" , "w")
results_file.write("RESULTS:\n")
results_file.write("\n")
results_file.write("\n")




#Used to calculated averages across folds
fold_avg_vali_mrr = []
fold_avg_epoch = []
fold_avg_test_mrr = []


for i in range(len(folds_data_without_equal_pairs)):
    vali_mrr_avg = []
    avg_epoch = []
    test_mrr_avg = []
    for _ in range(10):
        values = run_mrr(folds_data_without_equal_pairs[i][0],folds_data_without_equal_pairs[i][1],folds_data_without_equal_pairs[i][2],folds_data_without_equal_pairs[i][3])
        vali_mrr_avg.append(values[0])
        avg_epoch.append(values[1])
        test_mrr_avg.append(values[2])


    fold_avg_vali_mrr.append(sum(vali_mrr_avg)/len(vali_mrr_avg))

    fold_avg_epoch.append(sum(avg_epoch)/len(avg_epoch))

    fold_avg_test_mrr.append(sum(test_mrr_avg)/len(test_mrr_avg))


    results_file.write("Fold "+ str(i+1) + " val MRR: " + str(sum(vali_mrr_avg)/len(vali_mrr_avg)) + "\n")
    results_file.write("Fold "+ str(i+1) + " epochs: " + str(sum(avg_epoch)/len(avg_epoch))+ "\n")
    results_file.write("Fold "+ str(i+1) + " test MRR: " + str(sum(test_mrr_avg)/len(test_mrr_avg))+ "\n")
    results_file.write("\n")


results_file.write("AVG FOLD val MRR: " + str(sum(fold_avg_vali_mrr)/ len(fold_avg_vali_mrr)) + "\n" )
results_file.write("AVG FOLD epochs: " + str(sum(fold_avg_epoch)/ len(fold_avg_epoch) ) + "\n" )
results_file.write("AVD FOLD test MRR: " + str(sum(fold_avg_test_mrr)/ len(fold_avg_test_mrr)) + "\n" )
results_file.write("\n")



results_file.close()










results_file = open("results_ranknet_MRR_with_equal_pairs.txt" , "w")
results_file.write("RESULTS:\n")
results_file.write("\n")
results_file.write("\n")






#Used to calculated averages across folds
fold_avg_vali_mrr = []
fold_avg_epoch = []
fold_avg_test_mrr = []


for i in range(len(folds_data_with_equal_pairs)):
    vali_mrr_avg = []
    avg_epoch = []
    test_mrr_avg = []
    for _ in range(10):
        values = run_mrr(folds_data_with_equal_pairs[i][0],folds_data_with_equal_pairs[i][1],folds_data_with_equal_pairs[i][2],folds_data_with_equal_pairs[i][3])
        vali_mrr_avg.append(values[0])
        avg_epoch.append(values[1])
        test_mrr_avg.append(values[2])


    fold_avg_vali_mrr.append(sum(vali_mrr_avg)/len(vali_mrr_avg))
    
    fold_avg_epoch.append(sum(avg_epoch)/len(avg_epoch))
    
    fold_avg_test_mrr.append(sum(test_mrr_avg)/len(test_mrr_avg))

    results_file.write("Fold "+ str(i+1) + " val MRR: " + str(sum(vali_mrr_avg)/len(vali_mrr_avg)) + "\n")
    results_file.write("Fold "+ str(i+1) + " epochs: " + str(sum(avg_epoch)/len(avg_epoch))+ "\n")
    results_file.write("Fold "+ str(i+1) + " test MRR: " + str(sum(test_mrr_avg)/len(test_mrr_avg))+ "\n")
    results_file.write("\n")


results_file.write("AVG FOLD val MRR: " + str(sum(fold_avg_vali_mrr)/ len(fold_avg_vali_mrr)) + "\n" )
results_file.write("AVG FOLD epochs: " + str(sum(fold_avg_epoch)/ len(fold_avg_epoch) ) + "\n" )
results_file.write("AVD FOLD test MRR: " + str(sum(fold_avg_test_mrr)/ len(fold_avg_test_mrr)) + "\n" )
results_file.write("\n")


results_file.close()


time_results_file.write("Time taken to run all tests: " +  str(time.time() - results_run_time) + "\n")
time_results_file.close()