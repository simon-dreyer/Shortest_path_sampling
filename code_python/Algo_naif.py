## Import

import networkx as nx
import random as r
import matplotlib.pyplot as plt
import osmnx as ox



## Exemples


Petersen = nx.petersen_graph()
C6 = nx.cycle_graph(6)
Biparti34 = nx.complete_bipartite_graph(3,4)





def downloadVille(location,nom_fichier):
    base_filepath = "C:/Users/Dreyer Simon/Documents/Travail/5A/Stage de recherche/code_python/data/"
    G = ox.graph_from_place(location, network_type="drive")
    G = rename_nodes(G)

    filepath = base_filepath + nom_fichier + ".graphml"
    ox.save_graphml(G, filepath)


def loadVille(nom_fichier):
    base_filepath = "C:/Users/Dreyer Simon/Documents/Travail/5A/Stage de recherche/code_python/data/"
    filepath = base_filepath + nom_fichier + ".graphml"

    return ox.load_graphml(filepath)


def testPlace(location):
    G = ox.graph_from_place(location, network_type="drive")
    print(len(G))
    ox.plot_graph(G)



Piedmont = loadVille("piedmont")
Alice_Springs = loadVille("alice_springs")


## Preprocessing

def Preprocessing_naif(G):
    """ Renvoie :
    - une liste table_pcc de taille n x n avec en coordonnee (dep,arr) la liste des pcc (dep) -> (arr)
    - une liste liste_pcc_dep de taille n avec en coordonnee (dep) la liste des pcc (dep) -> (?)
    - une liste liste_pcc qui contient tous les pcc """
    n = len(G)

    table_pcc = [[[] for arr in range(n)] for dep in range(n)]
    liste_pcc_dep = [[] for dep in range(n)]
    liste_pcc = []

    for dep in G.nodes():
        for arr in G.nodes():
            try:
                pcc_dep_arr_iter = nx.all_shortest_paths(G,dep,arr)
                for path in pcc_dep_arr_iter:
                    table_pcc[dep][arr].append(path)
                    liste_pcc_dep[dep].append(path)
                    liste_pcc.append(path)
            except nx.exception.NetworkXNoPath:
                table_pcc[dep][arr] = []

    return (table_pcc, liste_pcc_dep, liste_pcc)


## Générateur


def Uniforme_PCC_dep_arr_naif(prepG, dep, arr):
    table_pcc, liste_pcc_dep, liste_pcc = prepG
    rang = r.randint(0, len(table_pcc[dep][arr])-1)
    return table_pcc[dep][arr][rang]


def Uniforme_PCC_dep_naif(prepG, dep):
    table_pcc, liste_pcc_dep, liste_pcc = prepG
    rang = r.randint(0, len(liste_pcc[dep])-1)
    return liste_pcc_dep[dep][rang]


def Uniforme_PCC_naif(prepG):
    table_pcc, liste_pcc_dep, liste_pcc = prepG
    rang = r.randint(0, len(liste_pcc)-1)
    return liste_pcc[rang]