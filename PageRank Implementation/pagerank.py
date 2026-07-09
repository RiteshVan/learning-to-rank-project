import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


#Initialise directed graph to visualise connections for PageRank
g = nx.DiGraph()

#Create connections between nodes
#Small graph to showcase how PageRank works

g.add_edge(0,1)
g.add_edge(1,1)
g.add_edge(2,1)
g.add_edge(0,2)




g.add_edge(1,3)

g.add_edge(3,2)
g.add_edge(3,4)
g.add_edge(4,1)
g.add_edge(4,2)
g.add_edge(4,3)

g.add_edge(4,0)


g.add_edge(0,4)

print(g.nodes())

def pagerank(graph : nx.classes.digraph.DiGraph ,iter : int,damping : float):
    

    #Create matrixes for algorithm

    #Transpose such that (x,y) entry shows if x -> y is connected
    adj_matrix = np.array(nx.adjacency_matrix(graph,dtype=float).toarray()).transpose()


    #Create transition matrix based on number of outlinks per node
    to_divide = np.sum(adj_matrix,axis=0,dtype=float)

    tr_matrix = np.array(np.divide(adj_matrix,to_divide,dtype=float,out=np.zeros_like(adj_matrix),where=to_divide!=0 )).transpose()


    #Modify matrix such that dangling nodes link to all other nodes equally

    for row in tr_matrix:
        if sum(row)==0:
            for i in range(len(row)):
                row[i]=1/len(row)


    tr_matrix=tr_matrix.transpose()
    
    

    #Final matrix to use after taking damping and random clicks into account
    final_matrix = (np.dot(damping,tr_matrix)) + np.dot((np.full((len(tr_matrix),len(tr_matrix)),(1/len(adj_matrix))  ,dtype=float) ), (1-damping) )

    #Create matrix for initial PageRank values for nodes
    node_values = np.array([[1/len(adj_matrix)]]*len(adj_matrix))

    #Run for specified number of iterations to get final PageRank values
    for i in range(iter):
        node_values = np.dot(final_matrix,node_values)

    output=[]
    for i in range(len(node_values)):
        
        output.append((list(graph.nodes)[i],float(node_values[i][0])))
        

    return output

#Print results from own implementation
print("Own implementation: ")
print(pagerank(g,10,0.85))


print()
print()
print()

#Print results for built-in function
#Built-in function also sets damping = 0.85
print("Compared to networkx built-in function: " )
print(nx.pagerank(g))






#Visualise graph in circular way for clarity
#nx.draw_circular(g,with_labels=True)
#plt.show()