import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import re
import time

#Initialise directed graph to visualise connections for PageRank
g = nx.DiGraph()

#Create connections between nodes

#Used during initial testing, switched to use real dataset
# #Small graph to showcase how PageRank works

# g.add_edge(0,1)
# g.add_edge(1,1)
# g.add_edge(2,1)
# g.add_edge(0,2)




# g.add_edge(1,3)

# g.add_edge(3,2)
# g.add_edge(3,4)
# g.add_edge(4,1)
# g.add_edge(4,2)
# g.add_edge(4,3)

# g.add_edge(4,0)


# g.add_edge(0,4)


#Dataset from this website URL: https://www.kaggle.com/datasets/linhbngo/hollins-dataset 
#Load in dataset
file = "/Users/ritesh/Desktop/VSCODE-project/final-project/PageRank Implementation/hollins.dat"

#Array used to store node connections as list of tuples
array = []

#Add tuples to array
with open(file,"r") as f:
    for line in f:
        #Only take into account lines that relate to nodes connections based on URL IDs
        x = re.search(r"([0-9]+)\s([0-9]+)",line)
        if x:
            #Append groups to array such that  leftmost item in tuple --> rightmost item in tuple
            array.append((int(x.group(1)),int(x.group(2))))


#First line corresponds to total no. nodes and edges, so must be skipped when creating graph
array = array[1:]

#Creates graph based on array
for val in array:
    g.add_edge(val[0],val[1])

#Contains main implementation
def pagerank(graph : nx.classes.digraph.DiGraph ,iter : int,damping : float,convergence_threshold : float):
    

    #Create matrixes for algorithm

    #Transpose such that (x,y) entry shows if x -> y is connected
    adj_matrix = np.array(nx.adjacency_matrix(graph,dtype=float).toarray()).transpose()


    #Create transition matrix based on number of outlinks per node
    to_divide = np.sum(adj_matrix,axis=0,dtype=float)

    tr_matrix = np.array(np.divide(adj_matrix,to_divide,dtype=float,out=np.zeros_like(adj_matrix),where=to_divide!=0 )).transpose()


    #Modify matrix such that dangling nodes link to all other nodes equally
    for row in tr_matrix:
        #Checks if points to no other nodes
        if sum(row) == 0:
            #Updates values such that transition matrix gives equal probability to all other nodes
            for i in range(len(row)):
                row[i] = 1/len(row)

    #Transpostion done to simplify later calculations with node_values
    tr_matrix=tr_matrix.transpose()
    
    

    #Final matrix to use after taking damping and random clicks into account
    final_matrix = (np.dot(damping,tr_matrix)) + np.dot((np.full((len(tr_matrix),len(tr_matrix)),(1/len(adj_matrix))  ,dtype=float) ), (1-damping) )

    #Create matrix for initial PageRank values for nodes
    node_values = np.array([[1/len(adj_matrix)]]*len(adj_matrix))

    
    #Run for specified number of iterations to get final PageRank values
    for i in range(iter):
        #Multiplication to find new values
        new_node_values = np.dot(final_matrix,node_values)

        #Check difference between prior and new values (to check convergence threshold in next step)
        diff = np.absolute(new_node_values-node_values)
        
        #Check to see if all values change by less than threshold
        if np.all(diff < convergence_threshold):
            
            #If condition met make update and end loop
            node_values = new_node_values
            break

        #Update values
        node_values=new_node_values

    #Update number of iterations taken
    iter_taken = i+1

    #Formulate output into readable array
    output = []
    for i in range(len(node_values)):
        
        #Add tuples to array such that leftmost entry is nodeID and rightmost tuple is its PageRank value
        output.append((list(graph.nodes)[i],float(node_values[i][0])))
    

    # print("Number of iterations till convergence: ",iter_taken)

    #Output number of iterations taken and pagerank values
    return iter_taken,output


# #Print results from own implementation
# print("Own implementation: ")
# # start = time.time()
# print(sorted(pagerank(g,100,0.85,0.00001)[1][:20], key= lambda x : x[1],reverse=True ) )
# # print("Time taken: ", time.time()-start)

# #Extra print lines for clarity and separation
# print()


# #Print results for built-in function
# #Built-in function also sets damping = 0.85
# print("Compared to networkx built-in function: " )
# print(sorted(list(nx.pagerank(g).items())[:20], key = lambda x : x[1],reverse=True  ))



#Extra print lines for clarity and separation
print()
print()
print()



#Experiment(s)


#Test algorithm with many thresholds to check number of iterations
test_convergence_vals =[0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001]


# #arrays to keep track of values to plot
# plot_x=[] #Stores convergence thresholds
# plot_y=[]  #Stores iterations taken at each threshold


# #Iterate through selected thresholds (from prior array) and run pagerank at each threshold (keeping damping the same throughout for fair testing)
# for val in test_convergence_vals:
#     #Add current threshold value to array
#     plot_x.append(val)

#     #Add current pagerank iterations taken count to array
#     plot_y.append(pagerank(g,100,0.85,val)[0])

# #Plot and label graph
# plt.plot(plot_x,plot_y)
# plt.xscale("log")
# plt.xlabel("Convergence threshold")
# plt.ylabel("Iterations till convergence")
# plt.title("Iterations till convergence against convergence threshold")
# plt.show()



#Visualise graph in circular way for clarity
#nx.draw_circular(g,with_labels=True)
#plt.show()



# avg_times_taken=[]

# for val in test_convergence_vals:

#     per_val_times=[]
#     for i in range(30):
#         start= time.time()
#         pagerank(g,100,0.85,val)
#         per_val_times.append(time.time() - start)


#     avg_times_taken.append(sum(per_val_times)/len(per_val_times))
        

# print(avg_times_taken)
# print(test_convergence_vals)
# plt.xscale("log")
# plt.plot(test_convergence_vals,avg_times_taken)
# plt.xlabel("Convergence threshold")
# plt.ylabel("Average time taken to convergence (in seconds)")
# plt.show()




#Experiment to find total and average distance from NetworkX results

pagerank_results= list(map(lambda x : x[1] ,sorted(pagerank(g,100,0.85,0.00001)[1],key= lambda x : x[0])))

networkx_results= list(map(lambda x : x[1] ,sorted(list(nx.pagerank(g).items()),key= lambda x : x[0])))

total_diff = np.absolute(np.array(pagerank_results)-np.array(networkx_results))


print(sum(total_diff))

print(sum(total_diff)/ len(pagerank_results))