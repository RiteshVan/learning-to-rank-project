from ranknet import run



results_file = open("results_ranknet_with_equal_pairs.txt" , "w")
results_file.write("RESULTS:\n")
results_file.write("\n")
results_file.write("\n")


ndcg_vals = [1,2,5,10]

folds = [1,2,3,4,5]


for val in ndcg_vals:
    results_file.write("NDCG@" + str(val) +" outputs\n")
    results_file.write("\n")
    for fold in folds:
        vali_ndcg_avg = []
        avg_epoch = []
        test_ndcg_avg = []
        for i in range(10):
            values = run(fold,val)
            vali_ndcg_avg.append(values[0])
            avg_epoch.append(values[1])
            test_ndcg_avg.append(values[2])        
            

        results_file.write("Fold "+ str(fold) + " val NDCG: " + str(sum(vali_ndcg_avg)/len(vali_ndcg_avg)) + "\n")
        results_file.write("Fold "+ str(fold) + " epochs: " + str(sum(avg_epoch)/len(avg_epoch))+ "\n")
        results_file.write("Fold "+ str(fold) + " test NDCG: " + str(sum(test_ndcg_avg)/len(test_ndcg_avg))+ "\n")
        results_file.write("\n")

        

    



results_file.close()



# vali_ndcg_avg = []
# avg_epoch = []
# test_ndcg_avg = []


# for i in range(10):
#     values = run()
#     vali_ndcg_avg.append(values[0])
#     avg_epoch.append(values[1])
#     test_ndcg_avg.append(values[2])


# print("Avg val NDCG : " , sum(vali_ndcg_avg)/len(vali_ndcg_avg))
# print("Avg epochs : " , sum(avg_epoch)/len(avg_epoch))
# print("Avg test NDCG : " , sum(test_ndcg_avg)/len(test_ndcg_avg))