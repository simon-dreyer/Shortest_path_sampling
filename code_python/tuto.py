import networkx as nx
import matplotlib.pyplot as plt


G = nx.cycle_graph(6)

pred, dist = nx.dijkstra_predecessor_and_distance(G, 0)

for node in [0, 1, 2, 3, 4]:
    print(f"{node}: {length[node]}")