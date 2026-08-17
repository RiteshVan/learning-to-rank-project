from ranknet import get_data_with_equal_pairs,get_data_without_equal_pairs,run_ndcg
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





results_file = open("results_ranknet_ndcg_without_equal_pairs.txt" , "w")
results_file.write("RESULTS:\n")
results_file.write("\n")
results_file.write("\n")

start_time = time.time()
fold_1_data = get_data_without_equal_pairs(1)
fold_2_data = get_data_without_equal_pairs(2)
fold_3_data = get_data_without_equal_pairs(3)
fold_4_data = get_data_without_equal_pairs(4)
fold_5_data = get_data_without_equal_pairs(5)
folds_data = [fold_1_data,fold_2_data,fold_3_data,fold_4_data,fold_5_data]
print(time.time()-start_time)


ndcg_vals = [1,2,5,10,20]

for val in ndcg_vals:
    results_file.write("NDCG@" + str(val) +" outputs\n")
    results_file.write("\n")
    for i in range(len(folds_data)):
        vali_ndcg_avg = []
        avg_epoch = []
        test_ndcg_avg = []
        for _ in range(10):
            values = run_ndcg(folds_data[i][0],folds_data[i][1],folds_data[i][2],folds_data[i][3],val)
            vali_ndcg_avg.append(values[0])
            avg_epoch.append(values[1])
            test_ndcg_avg.append(values[2])


        results_file.write("Fold "+ str(i+1) + " val NDCG: " + str(sum(vali_ndcg_avg)/len(vali_ndcg_avg)) + "\n")
        results_file.write("Fold "+ str(i+1) + " epochs: " + str(sum(avg_epoch)/len(avg_epoch))+ "\n")
        results_file.write("Fold "+ str(i+1) + " test NDCG: " + str(sum(test_ndcg_avg)/len(test_ndcg_avg))+ "\n")
        results_file.write("\n")


results_file.close()









results_file = open("results_ranknet_ndcg_with_equal_pairs.txt" , "w")
results_file.write("RESULTS:\n")
results_file.write("\n")
results_file.write("\n")


start_time = time.time()
fold_1_data = get_data_with_equal_pairs(1)
fold_2_data = get_data_with_equal_pairs(2)
fold_3_data = get_data_with_equal_pairs(3)
fold_4_data = get_data_with_equal_pairs(4)
fold_5_data = get_data_with_equal_pairs(5)
folds_data = [fold_1_data,fold_2_data,fold_3_data,fold_4_data,fold_5_data]
print(time.time()-start_time)


ndcg_vals = [1,2,5,10,20]

for val in ndcg_vals:
    results_file.write("NDCG@" + str(val) +" outputs\n")
    results_file.write("\n")
    for i in range(len(folds_data)):
        vali_ndcg_avg = []
        avg_epoch = []
        test_ndcg_avg = []
        for _ in range(10):
            values = run(folds_data[i][0],folds_data[i][1],folds_data[i][2],folds_data[i][3],val)
            vali_ndcg_avg.append(values[0])
            avg_epoch.append(values[1])
            test_ndcg_avg.append(values[2])


        results_file.write("Fold "+ str(i+1) + " val NDCG: " + str(sum(vali_ndcg_avg)/len(vali_ndcg_avg)) + "\n")
        results_file.write("Fold "+ str(i+1) + " epochs: " + str(sum(avg_epoch)/len(avg_epoch))+ "\n")
        results_file.write("Fold "+ str(i+1) + " test NDCG: " + str(sum(test_ndcg_avg)/len(test_ndcg_avg))+ "\n")
        results_file.write("\n")


results_file.close()





results_file = open("results_ranknet_MRR_without_equal_pairs.txt" , "w")
results_file.write("RESULTS:\n")
results_file.write("\n")
results_file.write("\n")


start_time = time.time()
fold_1_data = get_data_without_equal_pairs(1)
fold_2_data = get_data_without_equal_pairs(2)
fold_3_data = get_data_without_equal_pairs(3)
fold_4_data = get_data_without_equal_pairs(4)
fold_5_data = get_data_without_equal_pairs(5)
folds_data = [fold_1_data,fold_2_data,fold_3_data,fold_4_data,fold_5_data]
print(time.time()-start_time)



for i in range(len(folds_data)):
    vali_mrr_avg = []
    avg_epoch = []
    test_mrr_avg = []
    for _ in range(10):
        values = run(folds_data[i][0],folds_data[i][1],folds_data[i][2],folds_data[i][3],val)
        vali_mrr_avg.append(values[0])
        avg_epoch.append(values[1])
        test_mrr_avg.append(values[2])


    results_file.write("Fold "+ str(i+1) + " val NDCG: " + str(sum(vali_mrr_avg)/len(vali_mrr_avg)) + "\n")
    results_file.write("Fold "+ str(i+1) + " epochs: " + str(sum(avg_epoch)/len(avg_epoch))+ "\n")
    results_file.write("Fold "+ str(i+1) + " test NDCG: " + str(sum(test_mrr_avg)/len(test_mrr_avg))+ "\n")
    results_file.write("\n")


results_file.close()










results_file = open("results_ranknet_MRR_with_equal_pairs.txt" , "w")
results_file.write("RESULTS:\n")
results_file.write("\n")
results_file.write("\n")


start_time = time.time()
fold_1_data = get_data_with_equal_pairs(1)
fold_2_data = get_data_with_equal_pairs(2)
fold_3_data = get_data_with_equal_pairs(3)
fold_4_data = get_data_with_equal_pairs(4)
fold_5_data = get_data_with_equal_pairs(5)
folds_data = [fold_1_data,fold_2_data,fold_3_data,fold_4_data,fold_5_data]
print(time.time()-start_time)



for i in range(len(folds_data)):
    vali_mrr_avg = []
    avg_epoch = []
    test_mrr_avg = []
    for _ in range(10):
        values = run(folds_data[i][0],folds_data[i][1],folds_data[i][2],folds_data[i][3],val)
        vali_mrr_avg.append(values[0])
        avg_epoch.append(values[1])
        test_mrr_avg.append(values[2])


    results_file.write("Fold "+ str(i+1) + " val NDCG: " + str(sum(vali_mrr_avg)/len(vali_mrr_avg)) + "\n")
    results_file.write("Fold "+ str(i+1) + " epochs: " + str(sum(avg_epoch)/len(avg_epoch))+ "\n")
    results_file.write("Fold "+ str(i+1) + " test NDCG: " + str(sum(test_mrr_avg)/len(test_mrr_avg))+ "\n")
    results_file.write("\n")


results_file.close()