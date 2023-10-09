import networkx as nx
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def network_plot_3D(G, angle, save=False):

    # Get node positions
    #pos = nx.spring_layout(G, dim=3)
    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()

    # Get the redius of the graph around node 0
    radius_max = max([nx.shortest_path_length(G,source=0,target=i) for i in range(n)])
    print('max radius: ' + str(radius_max))
    # Define node color range proportional the distance from 0, edge colors by the type of edge.

    colors = [plt.cm.plasma(nx.shortest_path_length(G,source=0,target=i)/float(radius_max)) for i in range(n)] 
    edgeColors = {'vein':'green', 'strut':'purple', 'ring':'red'}  
    

    # 3D network plot
    with plt.style.context(('ggplot')):
        
        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)
        
        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]
            
            # Scatter plot
            ax.scatter(xi, yi, zi, c=colors[key], s=5+2*G.degree(key), edgecolors='k', alpha=0.7)
        
        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i,j in enumerate(G.edges()):

            col = edgeColors[G.edges[j]['edge_type']]

            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))
        
        # Plot the connecting lines
            ax.plot(x, y, z, c=col, alpha=0.5)
    
    # Set the initial view
    ax.view_init(30, angle)

    # Hide the axes
    ax.set_axis_off()

    if save is not False:
         plt.savefig("./growMeshImages/immersed"+str(radius)+str(angle).zfill(3)+".png")
         plt.close('all')
    else:
          plt.show()
    
    return




G = nx.Graph()
radius = 5
G.add_node(0, layer=0, node_type='start')
for i in range(1,8):
    G.add_node(i, node_type='convex', layer=1)
    G.add_edge(0,i, edge_type='vein')
G.add_edge(1,2, edge_type='ring')
G.add_edge(2,3, edge_type='ring')
G.add_edge(3,4, edge_type='ring')
G.add_edge(4,5, edge_type='ring')
G.add_edge(5,6, edge_type='ring')
G.add_edge(6,7, edge_type='ring')
G.add_edge(7,1, edge_type='ring')
    
boundary = range(1,8)
maxNode = 7
pos = nx.spring_layout(G, dim = 3)

for x in range(2,radius+1):

    B=G.subgraph(boundary)
    print('boundary ' + str(x-1) + ': ' + str(nx.find_cycle(B)))
    
    #print('boundary'+str(boundary))
    newBoundary=[]

    G.add_node("firstNode", node_type='concave')
    lastNode="firstNode" #lastNode doesn't exist for the first element of boundary

    for y in boundary: 
        n=maxNode
        
       
       
        if G.nodes[y]['node_type']=='convex':
            G.add_node(n+1, node_type='convex', layer=x)
            G.add_node(n+2, node_type='convex', layer=x)
            G.add_node(n+3, node_type='concave', layer=x)
            newBoundary.extend([n+1,n+2,n+3])
            G.add_edge(y, lastNode, edge_type='strut') 
            G.add_edge(n+1, lastNode, edge_type='ring')
            G.add_edge(y, n+1, edge_type='vein')
            G.add_edge(y, n+2, edge_type='vein')
            G.add_edge(n+1, n+2, edge_type='ring')
            G.add_edge(y, n+3, edge_type='strut')
            G.add_edge(n+2,n+3, edge_type='ring')
            lastNode=n+3
        elif G.nodes[y]['node_type']=='concave':
            G.add_node(n+1, node_type='convex', layer=x)
            G.add_node(n+2, node_type='concave', layer=x)
            newBoundary.extend([n+1,n+2])
            G.add_edge(y, lastNode, edge_type='strut')
            G.add_edge(n+1, lastNode, edge_type='ring')
            G.add_edge(y, n+1, edge_type='vein')
            G.add_edge(n+1, n+2, edge_type='ring')
            G.add_edge(y, n+2, edge_type='strut')
            lastNode=n+2
        else:
            print('node_type error')
        
        maxNode=lastNode

    G=nx.contracted_nodes(G, lastNode, "firstNode")
    boundary=newBoundary
    pos = nx.spring_layout(G, pos = pos, dim = 3)
    
print(nx.number_of_nodes(G), 'nodes')
print(nx.number_of_edges(G), 'edges')
all_cliques= nx.enumerate_all_cliques(G)
triangles=[x for x in all_cliques if len(x)==3 ]
print(len(triangles), 'triangles')

#for n in G:
#    print(len(G[n])),

pos = nx.kamada_kawai_layout(G, pos=pos, dim=3)

nx.set_node_attributes(G, pos, 'pos')

network_plot_3D(G,0)


#for k in range(20,201,1):

 #   angle = (k-20)*360/(200-20)
    
  #  network_plot_3D(G,angle, save=True)

   # print(angle)
