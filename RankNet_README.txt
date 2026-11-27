This folder contains all files that relate to the RankNet implementation.

Instructions:
Please install the PyTorch library to run the implementation.

Please open the RankNet implementation folder in an IDE.

The ranknet.py file contains then implementation.

In order to obtain results, run the testing.py file, which automatically creates text files to store output.


The testing_ndcg20.py file is used as part of an experiment where only queries with above ten documents are used as input to the neural network. Metric scores are calculated at NDCG@20. 

The results file is created automatically here also.


The testing_one_over_two_count.py file is used to find the average number of queries (which are associated with both 1 and 2 relevancy label documents) for which the model places the document with label 1 at the top of a list over a document with the label 2.

The MQ2008 dataset is included here and contains all folds and data needed for implementation 


The results from a prior run of the algorithm are included in this folder
 