#!/usr/bin/env python
# coding: utf-8

# In[192]:


import os
import networkx as nx
import pickle

folder_path = 'datasets'
# ULONG_MAX 18446744073709551615


# In[193]:


def file_exists(file_path):
    """
    Checks if a file exists.

    :param file_path: Path of the file to check
    :return: True if the file exists, False otherwise
    """
    return os.path.isfile(file_path)

def save_dic(d,s):
    with open(s+'.pickle', 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def read_dic(s):
    with open(s+'.pickle', 'rb') as handle:
        b = pickle.load(handle)
        return b


# In[66]:


l = [["astro_ph","u"], ["power_grid", "u"], ["hamster_full","u"], ["linux_mail","d"], ["slashdot", "d"], ["milan", "d"], ["vienna","d"], ["paris","d"] ]
#l = [["power_grid", "u"], ["hamster_full","u"], ["aachen", "d"]]


# In[67]:


#launch pre-computations
import subprocess
algos = ["b-unrank", "ob-unrank", "i-unrank", "alias-unrank"]
j = 0
for x in l:
    file_path = folder_path + "/" + x[0] + ".edge"
    #g = nx.read_edgelist(file_path, create_using=nx.Graph, data=False)
    for alg in algos:
        print("filepath : ",file_path)
        subprocess.run(["./main", file_path ,x[1], alg, str(0), str(1), str(1), 'c']) 
    j += 1


# In[68]:


info = dict()
for x in l:
    info[x[0]] = dict()
    file_path = folder_path + "/" + x[0] + ".edge"
    with open(file_path,'r') as f:
        edges = set()
        nodes = set()
        nb_edges = 0
        for line in f:
            nb_edges += 1
            r = line.split()
            edges.add((r[0],r[1]))
            nodes.add(r[0])
            nodes.add(r[1])
    info[x[0]] = (len(nodes), len(edges))
info


# In[88]:


import random
def random_pairs(n, infos, x, max_tries_per_node = 1000, more_than_one = 0):
    nb = 0
    res = []
    while nb < n:
        print("nb", nb, end = " ")
        j = random.randint(0, infos[x[0]][0]-1)
        file_path = folder_path + "/" + x[0] + "_b-unrank/nb_paths_" + str(j) + ".csv"
        file_path2 = folder_path + "/" + x[0] + "_b-unrank/distances_" + str(j) + ".csv"
        ll = read_integers_from_file(file_path)
        ll2 = read_integers_from_file(file_path2)
        tent = 0
        while tent < max_tries_per_node:
            k = random.randint(0, infos[x[0]][0]-1)
            #print("len ll ", len(ll), "k", k)
            if ll[k] != 0 and k!=j:
                if more_than_one:
                    if ll[k] > 1:
                        break
                else:
                    break
            tent += 1
        if tent < max_tries_per_node:
            res.append([[j,k], ll2[k], ll[k]])
            nb += 1
    return res
            
def random_pairs_exact(n, V):
    res = []
    for i in range(n):
        lV = list(V)
        pair = random.sample(lV, k = 2)
        res.append(pair)
    return res

def read_float_from_file(file_path):
    """
    Reads a single float value from a file.

    :param file_path: Path to the file
    :return: The float value read from the file
    """
    with open(file_path, 'r') as file:
        value = file.readline().strip()
        print("val", value)
        return float(value)
    
    
def read_integer_from_file(file_path):
    """
    Reads a single float value from a file.

    :param file_path: Path to the file
    :return: The float value read from the file
    """
    with open(file_path, 'r') as file:
        value = file.readline().strip()
        print("val", value)
        return int(value)
    
import numpy as np

def calculate_mean_and_std(array):
    """
    Calculates the mean and standard deviation of an array.

    :param array: Input array
    :return: A tuple containing mean and standard deviation
    """
    mean = np.mean(array)
    std_dev = np.std(array)
    return mean, std_dev


# In[89]:


def read_integers_from_file(file_path):
    """
    Reads a file containing one integer on each line into an array.

    :param file_path: Path to the file
    :return: A list of integers read from the file
    """
    integers = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip any whitespace and convert the line to an integer
            integers.append(int(line.strip()))
    return integers

def read_floats_from_file(file_path):
    """
    Reads a file containing one integer on each line into an array.

    :param file_path: Path to the file
    :return: A list of integers read from the file
    """
    doubles = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip any whitespace and convert the line to an integer
            doubles.append(float(line.strip()))
    return doubles

# distances = dict()
# d_dist = dict()
# for x in l:
#     distances[x[0]] = dict()
#     d_dist[x[0]] = dict()
#     j = 0
#     while True:
#         file_path = folder_path + "/" + x[0] + "_b-unrank/distances_" + str(j) + ".csv"
#         if file_exists(file_path):
#             ll = read_integers_from_file(file_path)
#             for z in range(len(ll)):
#                 if ll[z] in d_dist[x[0]]:
#                     d_dist[x[0]][ll[z]].append((j,z))
#                 else:
#                     d_dist[x[0]][ll[z]] = [(j,z)]
#                 distances[x[0]][(j,z)] = ll[z]
#                 #distances[x[0]][(z,j)] = ll[z]
#         else:
#             # we have reached the number of nodes
#             break
#         j += 1


# In[90]:


#launch simulations on er query time on average
pair_dist = "average"
nb_queries_per_pair = 5000
nb_pairs = 100
import subprocess
algos = ["b-unrank", "ob-unrank", "i-unrank", "alias-unrank"]
d = { i[0]: {alg:[]   for alg in algos}  for i in l }
for x in l:
    file_path = folder_path + "/" + x[0]  + ".edge"
    print("filepath", file_path)
    #g = nx.read_edgelist(file_path, create_using=nx.Graph, data=False)
    ll = random_pairs(nb_pairs, info, x)
    if ll == -1:
        print("problem pair sampling")
        break
    #print(ll)
    for e in ll:
        print("current", e)
        for alg in algos:
            print("current alg", alg)
            subprocess.run(["./main", file_path ,x[1], alg, str(e[0][0]), str(e[0][1]), str(nb_queries_per_pair), "c"])
            file =  x[0] + "_" + alg + "/queries_operations_"+ str(nb_queries_per_pair) + ".txt"
            d[x[0]][alg].append(read_integer_from_file(folder_path + "/" + file)/(nb_queries_per_pair * e[1] ))
            


# In[78]:


# #launch simulations on er query time on long distance
# # last third of distances
# pair_dist = "long"
# max_dist = { i: max(d_dist[x[0]+ "_"+repr(x[1])])  for x in l  }


# nb_queries_per_pair = 500000
# nb_pairs = 30
# import subprocess
# algos = ["b-unrank", "ob-unrank", "i-unrank", "alias-unrank"]
# d = { i: {alg:[]   for alg in algos}  for i in l }

# for x in l:
#     i = x[1]
#     file_path = folder_path + "/" + x[0] + "_" + str(nb_nodes) + "_" + repr(i) + ".edges"
#     g = nx.read_edgelist(file_path, create_using=nx.Graph, data=False)
#     V = list(g.nodes())
#     pairs = []
#     if pair_dist == "long":
#         for zz in range(int(max_dist[i]*2/3), max_dist[i]):
#             pairs += d_dist[i][zz]
#     if pair_dist == "medium":
#         for zz in range(int(max_dist[i]*1/3), max_dist[i]*2/3):
#             pairs += d_dist[i][zz]
#     if pair_dist == "short":
#         for zz in range(int(max_dist[i]), max_dist[i]*1/3):
#             pairs += d_dist[i][zz]
#     ll = random_pairs_exact(nb_pairs, pairs)
#     if ll == -1:
#         print("problem pair sampling")
#         break
#     for e in ll:
#         for alg in algos:
#             subprocess.run(["./main", file_path ,"u", alg, str(e[0]), str(e[1]), str(nb_queries_per_pair), "c"])
#             file = "er_" + str(nb_nodes) + "_" + str(i) + "_" + alg + "/queries_operations_"+ str(nb_queries_per_pair) + ".txt"
#             d[x][alg].append(read_integer_from_file(folder_path + "/" + file))
            


# In[79]:


import matplotlib.pyplot as plt
import numpy as np

def plot_bar_chart(data, xlabel, ylabel, legend, filename, dim1 = 8, dim2 = 5, bar_w = 0.15):
    """
    Plots a bar chart where the x-axis is represented by the keys of the dictionary.
    Each key in the dictionary maps to another dictionary with 4 keys representing bars.
    The values are tuples (mean, standard deviation).

    :param data: Dictionary containing the data to plot
    """

    categories = list(data.keys())
    subcategories = list(next(iter(data.values())).keys())
    
    # Number of groups and bars per group
    n_groups = len(categories)
    n_bars = len(subcategories)
    
    fig, ax = plt.subplots(figsize=(dim1, dim2))
    
    # Create figure and axis
    #fig, ax = plt.subplots()

    # Bar width
    bar_width = bar_w

    # X locations for the groups
    index = np.arange(n_groups)
    
    # Iterate through each subcategory and plot the bars
    for i, subcategory in enumerate(subcategories):
        means = [data[category][subcategory][0] for category in categories]
        std_devs = [data[category][subcategory][1] for category in categories]
        ax.bar(index + i * bar_width, means, bar_width, yerr=std_devs, label=subcategory)

    # Add labels, title, and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(legend)
    ax.set_xticks(index + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(categories)
    ax.legend()


    # Show plot
    plt.rcParams.update({'font.size': 13})
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()

# # Example usage:
# data = {
#     'A': {'bar1': (5, 1), 'bar2': (6, 1.5), 'bar3': (7, 2), 'bar4': (8, 2.5)},
#     'B': {'bar1': (4, 1.2), 'bar2': (5, 1.3), 'bar3': (6, 1.8), 'bar4': (7, 2.1)},
#     'C': {'bar1': (3, 0.8), 'bar2': (4, 1.1), 'bar3': (5, 1.6), 'bar4': (6, 2.0)}
# }

# data = {i:  { e: calculate_mean_and_std(d[i][e])   for e in algos }  for i in list_p}

# plot_bar_chart(data)


# In[80]:


# def name(i):
#     if i[0] != "gr":
#         return i[0]+"_"+str(i[1])
#     else:
#         return str(i[1][0])+"_"+str(i[1][1])


# In[186]:


data = {i[0][:3]:  { e: calculate_mean_and_std(d[i[0]][e])   for e in algos }  for i in l}
save_dic(data,"queries_real")
plot_bar_chart(data, "real-world datasets", "#operations", "average #operations on queries", "queries_real.pdf")


# In[36]:


# data = {i:  { e: calculate_mean_and_std(d[i][e])   for e in algos }  for i in l if i[0]!="gr"}
# plot_bar_chart(data, "synth data", "#operations", "average #operations on queries", "queries2.pdf")


# In[94]:


#launch bars on pre-computations
import subprocess
algos = ["b-unrank", "ob-unrank", "i-unrank", "alias-unrank"]
d_pre = { i[0]: {alg:[]   for alg in algos}  for i in l }
for x in l:
    for alg in algos:
        file =  x[0] + "_" + alg + "/pre_time.csv"
        ll = read_integers_from_file(folder_path+"/"+file)
        d_pre[x[0]][alg] = ll


# In[187]:


data = {i[0][:3]:  { e: calculate_mean_and_std(d_pre[i[0]][e])   for e in algos }  for i in l}
save_dic(data,"pre_comp_real")

plot_bar_chart(data, "real-world datasets", "#time in ms", "average pre-computation time", "pre_comp_real.pdf")


# In[160]:


# data = {i[0]+"_"+repr(i[1]):  { e: calculate_mean_and_std(d_pre[i][e])   for e in algos }  for i in l if i[0]!="gr"}
# plot_bar_chart(data, "er, p = (xlog(n)/n)", "#time in ms", "average pre-computation time", "pre_comp2.pdf")


# In[153]:


def random_pairs_wasser(n, g, max_tries = 1000):
    V = g.nodes()
    res = []
    nb = 0
    tr = 0
    while nb <n and tr < max_tries:
        lV = list(V)
        pair = random.sample(lV, k = 2)
        if nx.has_path(g,pair[0],pair[1]):
            res.append(pair)
            nb += 1
        tr+=1
    if tr == max_tries:
        return -1
    return res


def construct_dag(l):
    g = nx.DiGraph()
    for e in l:
        for i in range(1,len(e)):
            if (e[i-1],e[i]) not in g.edges:
                g.add_edge(e[i-1],e[i])
    return g
# def connected_gnp(n,p, max_tries = 10, directed = False):
#     i = 0
#     while i < max_tries:
#         g = nx.fast_gnp_random_graph(N, p, directed=directed)
#         if nx.is_connected(g):
#             return g
#         i += 1
#     return None

def random_pairs_more_sh(g, V, max_tries = 10):
    i = 0
    while i < max_tries:
        lV = list(V)
        pair = random.sample(lV, k = 2)
        l = list(map(lambda x:tuple(x), nx.all_shortest_paths(g, source=pair[0], target=pair[1])))
        if len(l) > 1:
            return l, pair
        i += 1
    return None


from math import prod
def prob_URW(dag, w):
#     print("w", w)
#     for i in range(1,len(w)):
#         print("(", w[i],dag.in_degree(w[i]),")", end = " ")
    return prod( 1/dag.in_degree(w[i])  for i in range(1,len(w)))

def dist_URW(dag, l):
    return list(map( lambda x : prob_URW(dag, x), l ))

def random_weights(g,s,t):
    h = g.copy()
    n = len(g.nodes())
    for e in h.edges():
        h[e[0]][e[1]]['weight'] = 1 + random.uniform(-1/n,1/n)
    return nx.dijkstra_path(h, s, t, weight='weight')

def stat_random_weights(g,s,t,l, nb = 100):
    d = dict()
    for _ in range(nb):
        e = random_weights(g,s,t)
        te = tuple(e)
        if te in d:
            d[te] += 1
        else:
            d[te] = 1
    res = []
    for e in l:
        if e in d:
            res.append(d[e]/nb)
        else:
            res.append(0)
    return res
    


# In[154]:


import networkx as nx
def read_graph_edges(s):
    edges = set()
    with open(s,'r') as f:
        for line in f:
            x = line.split(" ")
            edges.add((x[0],x[1]))
    return list(edges)


# In[171]:


from scipy.stats import wasserstein_distance
#launch simulations on biased algos
pair_dist = "average"
nb_pairs = 20
import subprocess
algos = ["random_weights", "URW"]
d_was = { i[0]: {alg:[]   for alg in algos}  for i in l }
for x in l:
    print(x, info[x[0]])
    file_path = folder_path + "/" + x[0]  + ".edge"
    if x[1] == "u":
        g = nx.read_edgelist(file_path, create_using=nx.Graph, data=False)
    else:
        g = nx.read_edgelist(file_path, create_using=nx.DiGraph, data=False)
        
    ll = random_pairs(nb_pairs, info, x, more_than_one = 0)
    if ll == -1:
        print("problem pair sampling")
        break
    for e in ll:
        print(e)
        for alg in algos: 
            file =  x[0]+  "_" + "b-unrank" + "/"+ str(e[0][0]) + ".edges"
            #dag = nx.read_edgelist(folder_path + "/" + file, create_using=nx.DiGraph, data=False)
            edges = read_graph_edges(folder_path+"/"+file)
            dag = nx.from_edgelist(edges, create_using=nx.DiGraph())
            sl = list(map(lambda x:tuple(x), nx.all_shortest_paths(g, source=str(e[0][0]), target=str(e[0][1]))))
            print("nb shortest", len(sl))
            if alg == "random_weights":
                res = stat_random_weights(g,str(e[0][0]),str(e[0][1]),sl, nb = len(sl)*10)
            else:
                res = dist_URW(dag, sl )
            res_unif = [ 1/len(sl) for e in sl ]
            wr = wasserstein_distance(res_unif, res)
            print(alg, wr)
            d_was[x[0]][alg].append(wr)
            


# In[ ]:


data = {i[0][:3]:  { e: calculate_mean_and_std(d_was[i[0]][e])   for e in algos }  for i in l}
save_dic(data,"bias_real")

plot_bar_chart(data, "", "wasserstein distance", "wasserstein distance from uniform", "biase_real.pdf", dim1 = 5, dim2 = 5, bar_w = 0.25)


# In[184]:


#too long to run for now

# from scipy.stats import wasserstein_distance
# #launch simulations on biased algos
# pair_dist = "average"
# import subprocess
# algos = ["random_weights", "URW"]
# d_was_1 = { i[0]: {alg:[]   for alg in algos}  for i in l }
# for x in reversed(l):
#     print(x, info[x[0]])
#     file_path = folder_path + "/" + x[0]  + ".edge"
#     if x[1] == "u":
#         g = nx.read_edgelist(file_path, create_using=nx.Graph, data=False)
#     else:
#         g = nx.read_edgelist(file_path, create_using=nx.DiGraph, data=False)
        
#     ll = random_pairs(nb_pairs, info, x, more_than_one = 1)
#     if ll == -1:
#         print("problem pair sampling")
#         break
#     for e in ll:
#         print(e)
#         for alg in algos: 
#             file =  x[0]+  "_" + "b-unrank" + "/"+ str(e[0][0]) + ".edges"
#             #dag = nx.read_edgelist(folder_path + "/" + file, create_using=nx.DiGraph, data=False)
#             edges = read_graph_edges(folder_path+"/"+file)
#             dag = nx.from_edgelist(edges, create_using=nx.DiGraph())
#             sl = list(map(lambda x:tuple(x), nx.all_shortest_paths(g, source=str(e[0][0]), target=str(e[0][1]))))
#             print("nb shortest", len(sl))
#             if alg == "random_weights":
#                 res = stat_random_weights(g,str(e[0][0]),str(e[0][1]),sl, nb = len(sl)*10)
#             else:
#                 res = dist_URW(dag, sl )
#             res_unif = [ 1/len(sl) for e in sl ]
#             wr = wasserstein_distance(res_unif, res)
#             print(alg, wr)
#             d_was_1[x[0]][alg].append(wr)
            


# In[185]:


# data = {i[0]:  { e: calculate_mean_and_std(d_was_1[i[0]][e])   for e in algos }  for i in l}
# plot_bar_chart(data, "", "wasserstein distance", "wasserstein distance from uniform", "biase_more_one_real.pdf", dim1 = 5, dim2 = 5, bar_w = 0.25)


# In[177]:


def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


# In[182]:


memory = dict()
algos = ["b-unrank", "ob-unrank", "i-unrank", "alias-unrank"]
#launch bars on pre-computations
import subprocess
algos = ["b-unrank", "ob-unrank", "i-unrank", "alias-unrank"]
memory = { i[0]: {alg:[]   for alg in algos}  for i in l }
for x in l:
    for alg in algos:
        file =  x[0] + "_" + alg
        subprocess.Popen("rm " + folder_path+"/"+file+"/"+"queries*", shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        memory[x[0]][alg] = [get_folder_size(folder_path+"/"+file)]


# In[183]:


data = {i[0][:3]:  { e: calculate_mean_and_std(memory[i[0]][e])   for e in algos }  for i in l}
save_dic(data,"memory_real")

plot_bar_chart(data, "", "size in bytes", "Memory required to store data", "memory_real.pdf", dim1 = 6, dim2 = 5, bar_w = 0.15)


# In[ ]:




