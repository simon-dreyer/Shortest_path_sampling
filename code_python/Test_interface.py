import networkx as nx
import matplotlib.pyplot as plt
from grave import plot_network
from grave.style import use_attributes

import osmnx as ox


def loadVille(nom_fichier):
    base_filepath = "C:/Users/Dreyer Simon/Documents/Travail/5A/Stage de recherche/code_python/data/"
    filepath = base_filepath + nom_fichier + ".graphml"

    return ox.load_graphml(filepath)


def rename_nodes(G):
    return nx.convert_node_labels_to_integers(G)



Piedmont = loadVille("piedmont")
Grid8x8 = rename_nodes(nx.grid_2d_graph(8,8))
pos8x8 = {}
for i in range(64):
    pos8x8[i] = [i//8, i%8]



def hilighter(event):
    # if we did not hit a node, bail
    if not hasattr(event, 'nodes') or not event.nodes:
        return

    # pull out the graph,
    graph = event.artist.graph

    # clear any non-default color on nodes
    for node, attributes in graph.nodes.data():
        attributes.pop('color', None)

    for u, v, attributes in graph.edges.data():
        attributes.pop('width', None)

    for node in event.nodes:
        graph.nodes[node]['color'] = 'C1'

        for edge_attribute in graph[node].values():
            edge_attribute['width'] = 3

    # update the screen
    event.artist.stale = True
    event.artist.figure.canvas.draw_idle()


def test1():
    #graph = nx.barbell_graph(10, 14)
    graph = Grid8x8

    fig, ax = plt.subplots()
    art = plot_network(graph,layout=lambda G: {node: (node//8, node%8) for node in G}, ax=ax, node_style=use_attributes(),
                    edge_style=use_attributes())

    art.set_picker(10)
    ax.set_title('Click on the nodes!')
    fig.canvas.mpl_connect('pick_event', hilighter)
    plt.show()



def test2():
    G = Piedmont

    # define the colors to use for different edge types
    hwy_colors = {'footway': 'skyblue',
                'residential': 'paleturquoise',
                'cycleway': 'orange',
                'service': 'sienna',
                'living street': 'lightgreen',
                'secondary': 'grey',
                'pedestrian': 'lightskyblue'}

    # return edge IDs that do not match passed list of hwys
    def find_edges(G, hwys):
        edges = []
        for u, v, k, data in G.edges(keys=True, data='highway'):
            check1 = isinstance(data, str) and data not in hwys
            check2 = isinstance(data, list) and all([d not in hwys for d in data])
            if check1 or check2:
                edges.append((u, v, k))
        return set(edges)

    # first plot all edges that do not appear in hwy_colors's types
    G_tmp = G.copy()
    G_tmp.remove_edges_from(G.edges - find_edges(G, hwy_colors.keys()))
    m = ox.plot_graph_folium(G_tmp, popup_attribute='highway', weight=5, color='black')

    # then plot each edge type in hwy_colors one at a time
    for hwy, color in hwy_colors.items():
        G_tmp = G.copy()
        G_tmp.remove_edges_from(find_edges(G_tmp, [hwy]))
        if G_tmp.edges:
            m = ox.plot_graph_folium(G_tmp,
                                    graph_map=m,
                                    popup_attribute='highway',
                                    weight=5,
                                    color=color)
    m