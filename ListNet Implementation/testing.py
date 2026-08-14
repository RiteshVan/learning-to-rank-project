from listnet import run,get_data



results_file = open("listnet_results.txt","w")

results_file.write("LISTNET RESULTS:\n")
results_file.write("\n")
results_file.write("\n")

fold_1_data = get_data(1)
fold_2_data = get_data(2)
fold_3_data = get_data(3)
fold_4_data = get_data(4)
fold_5_data = get_data(5)
folds_data = [fold_1_data,fold_2_data,fold_3_data,fold_4_data,fold_5_data]

ndcg_vals = [1,2,5,10,20]




for val in ndcg_vals:
    results_file.write("NDCG@" + str(val) +" outputs\n")
    results_file.write("\n")
    for i in range(len(folds_data)):
        vali_ndcg_avg = []
        epoch_avg = []
        test_ndcg_avg = []
        for _ in range(10):
            values = run(folds_data[i][0],folds_data[i][1],folds_data[i][2],folds_data[i][3],val)
            vali_ndcg_avg.append(values[0])
            epoch_avg.append(values[1])
            test_ndcg_avg.append(values[2])

        results_file.write("Fold "+ str(i+1) + " val NDCG: " + str(sum(vali_ndcg_avg)/len(vali_ndcg_avg)) + "\n")
        results_file.write("Fold "+ str(i+1) + " epochs: " + str(sum(epoch_avg)/len(epoch_avg))+ "\n")
        results_file.write("Fold "+ str(i+1) + " test NDCG: " + str(sum(test_ndcg_avg)/len(test_ndcg_avg))+ "\n")
        results_file.write("\n")



results_file.close()
