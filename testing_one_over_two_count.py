from ranknet import run_ndcg,get_data_with_equal_pairs

#Test to calculate on average the number of queries (which are associated with 1 and 2 relevancy label documents) 
#where a document withe label 1 is ranked over a document with label 2

#Array used to find average
avg_count_one_over_two_labels = []

#Only fold one data used as sample for this experiment
data = get_data_with_equal_pairs(1)

for _ in range(10):
    #Large ndcg_k val used as purpose here is to check entire ordering
    count = run_ndcg(data[0],data[1],data[2],data[3],1000000000)[3]
    avg_count_one_over_two_labels.append(count)


print(sum(avg_count_one_over_two_labels)/len(avg_count_one_over_two_labels))