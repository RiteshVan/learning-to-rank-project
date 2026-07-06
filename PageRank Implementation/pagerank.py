import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


#Initialise directed graph to visualise connections for PageRank
g = nx.DiGraph()

#Create connections between nodes
#Small graph to showcase how PageRank works
g.add_edge(0,1)
g.add_edge(0,2)
g.add_edge(0,4)

g.add_edge(1,1)
g.add_edge(1,3)
g.add_edge(2,1)
g.add_edge(3,2)
g.add_edge(3,4)
g.add_edge(4,1)
g.add_edge(4,2)
g.add_edge(4,3)

g.add_edge(4,0)


#Visualise graph in circular way for clarity
nx.draw_circular(g,with_labels=True)
plt.show()