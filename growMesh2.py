import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from scipy.spatial import KDTree

def Diff(li1, li2):
    return (list(set(li1) - set(li2)))


def network_plot_3D(G, angle, save=False, labels=False):

    # Get node positions
    #pos = nx.spring_layout(G, dim=3)
    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()

    # Get the radius of the graph around node 0
    radius_max = max([nx.shortest_path_length(G,source=0,target=i) for i in range(n)])
    #print(radius_max)

    # Define node color range proportional the distance from 0, edge colors by the type of edge.

    colors = [plt.cm.plasma(nx.shortest_path_length(G,source=0,target=i)/float(radius_max)) for i in range(n)]
    #edgeColors = {'vein':'green', 'strut':'purple', 'ring':'red'}  


    # 3D network plot
    with plt.style.context('ggplot'):

        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, c=colors[key], s=5+2*G.degree(key), edgecolors='k', alpha=0.7)
            if labels == True:
                ax.text(xi, yi, zi, '%s' % (str(key)), size=20, zorder=1,
                        color='k')


        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i,j in enumerate(G.edges()):

            #col = edgeColors[G.edges[j]['edge_type']]
            col='black'

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
        plt.savefig("./growMeshImages/immersed"+str(maxnode)+".png")
        plt.close('all')
    else:
        plt.show()

    return



# Initialize the graph
G = nx.Graph()

radius = 5
G.add_node(0)

print('Build the first ring')
for i in range(1,8):
    G.add_node(i)
    G.add_edge(0,i)
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(3,4)
G.add_edge(4,5)
G.add_edge(5,6)
G.add_edge(6,7)
G.add_edge(7,1)

all_cliques = nx.enumerate_all_cliques(G)
triangles = [x for x in all_cliques if len(x) == 3]
print('triangles: '+str(triangles))

pos = nx.spring_layout(G, dim = 3)
nx.set_node_attributes(G, pos, 'pos')

maxnode = len(G.nodes) - 1

print('define the boundary')
boundary=[x for x in G.nodes if G.degree[x]<7]
B=G.subgraph(boundary)
print('boundary:'+str(boundary))
#print(list(nx.get_node_attributes(B, 'pos').values()))

# function to check for appropriate distance nodes
def check_for_link(G,x):
    print('checking '+str(x)+' for link')
    z=[]
    boundary=[x for x in G.nodes() if G.degree[x]<7]
    B=G.subgraph(boundary)
    print('B.nodes',B.nodes)
    for e in B.edges():
        if B.degree[e[0]] > 2 and B.degree[e[1]] > 2:
            newB = nx.Graph(B)
            newB.remove_edge(e[0], e[1])
            print('removing edge ' + str(e) + ' from boundary')
            B = newB
    print('B.edges: '+str(B.edges))
    for b in B.nodes:
        if B.degree[b] != 2:
            print('point ' + str(b) + ' has boundary degree ' + str(B.degree[b]))
    #print('boundary positions: '+ str(list(nx.get_node_attributes(B, 'pos').values())))
    tree = KDTree(list(nx.get_node_attributes(B, 'pos').values()))
    print('outside: '+str([list(B.nodes)[p] for p in tree.query_ball_point(pos[x], 1.01)])+', inside: ' + str([list(B.nodes)[p] for p in tree.query_ball_point(pos[x], 0.99)]))
    close_pts = Diff(Diff([list(B.nodes)[p] for p in tree.query_ball_point(pos[x], 1.01)], [list(B.nodes)[p] for p in tree.query_ball_point(pos[x], 0.99)]),list(B[x]))
    for p in close_pts:
        if any([q in B[x] for q in B[p]]):
            close_pts.remove(p)
    close_dists = [distance.sqeuclidean(pos[x], pos[y]) for y in close_pts]
    print('x: ' + str(x) +
          ' close_pts: '+ str(close_pts) +
          ' close_dists: ' + str(close_dists) +
          ' B[x]: '+str(list(B[x])))
    #  close_pts = close_pts - [z for z in close_pts if G.degree(z) + G.degree(x) > 7 ]
    if close_dists:
        z = close_pts[np.argmin([(x-1)**2 for x in close_dists])]
    return z

counter = 1


while maxnode < 100 and boundary:
    counter = counter + 1
    network_plot_3D(G, angle=0, labels=True)
    #network_plot_3D(G, angle=counter * 5, save=True, labels=True)
    print('radius: '+str(min([nx.shortest_path_length(G, source=0, target=z) for z in B.nodes])))
    print('radius node: ' + str(list(B.nodes)[np.argmin([nx.shortest_path_length(G, source=0, target=z) for z in B.nodes])]))
    x = list(B.nodes)[np.argmin([nx.shortest_path_length(G, source=0, target=z) for z in B.nodes])]
    print('degree of x:',G.degree[x])
    openptsx = B[x]
    if G.degree[x] == 6:
        link = check_for_link(G, x)
        if link and G.degree[link] < 5:
            print('linking '+str(x)+' to '+str(link))
            G.add_edge(x, link)
            for y in openptsx:
                G.add_edge(y, link)
            triangles.extend([[x,link,y] for y in openptsx])

        else:
            print('adding one node: '+str(maxnode+1))
            maxnode = maxnode + 1
            G.add_node(maxnode)
            G.add_edge(x, maxnode)
            for y in openptsx:
                G.add_edge(y, maxnode)
            triangles.extend([[x, maxnode, y] for y in openptsx])

    elif G.degree[x] <= 5:
        link = check_for_link(G, x)
        if link:
            #print('x: ' + str(x) + ' link: ' + str(link) + ' B[link]: ' + str(B[link]) + str([G.degree(y) < 6 for y in B[link]]))
            if all([G.degree[y] < 5 for y in B[link]]):
                openptsl = B[link]
                print('openptsl: ' + str(openptsl))
                print('linking ' + str(x) + ' to ' + str(link))
                G.add_edge(x, link)
                # print('open points: '+str(B[link]))
                distsl = [distance.sqeuclidean(pos[x], pos[y]) for y in openptsl]
                zl = list(openptsl)[np.argmin([(s - 1) ** 2 for s in distsl])]
                distsx = [distance.sqeuclidean(pos[link], pos[y]) for y in openptsx]
                zx = list(openptsx)[np.argmin([(s - 1) ** 2 for s in distsx])]
                zzx = list(openptsx)[np.argmax([(s - 1) ** 2 for s in distsx])]
                G.add_edge(link, zx)
                G.add_edge(x, zl)
                triangles.extend([[x, link, zx], [x,link,zl]])
                if G.degree[x] == 5:
                    G.add_edge(zzx,zl)
                    triangles.extend([[x,zx,zzx]])

        else:
            k = 7 - G.degree[x]
            print('adding '+str(k)+' nodes: '+str([maxnode + i for i in range(1,k+1)]))
            for i in range(1, k + 1):
                G.add_node(maxnode + i)
                G.add_edge(x, maxnode + i)
            for i in range(1,k):
                G.add_edge(maxnode+i, maxnode+i+1)
                triangles.append([x,maxnode+i, maxnode+i+1])

            #print('x: ' + str(x) + ' B[x][0]: ' + str(list(B[x])[0]))
            G.add_edge(list(openptsx)[0], maxnode + 1)
            triangles.append([x,list(openptsx)[0], maxnode + 1])
            G.add_edge(list(openptsx)[1], maxnode + k)
            triangles.append([x, list(openptsx)[1], maxnode + k])
            maxnode = len(G.nodes) - 1


    else:
        print('node degree error: ' + str(x) + ', ' + str(G.degree[x]))

    boundary = [x for x in G.nodes() if G.degree[x] < 7]
    B = G.subgraph(boundary)
    for e in B.edges():
        if B.degree[e[0]] > 2 and B.degree[e[1]] > 2:
            newB = nx.Graph(B)
            newB.remove_edge(e[0], e[1])
            B = newB
        elif len(e) < 2:
            print('edge ' + str(e) + ' has too few boundary components')
    print('B.edges', B.edges)
    print('triangles: '+str(triangles))

    pos = nx.spring_layout(G, pos=pos, dim=3)
    pos = nx.kamada_kawai_layout(G, pos=pos, dim=3)
    nx.set_node_attributes(G, pos, 'pos')

print(nx.number_of_nodes(G), 'nodes')
print(nx.number_of_edges(G), 'edges')
print(len(triangles), 'triangles')

for n in G:
    print(len(G[n])),

pos = nx.kamada_kawai_layout(G, pos=pos, dim=3)
nx.set_node_attributes(G, pos, 'pos')

network_plot_3D(G, 0)

    # for k in range(20,201,1):

    #   angle = (k-20)*360/(200-20)

    #  network_plot_3D(G,angle, save=True)

    # print(angle)
