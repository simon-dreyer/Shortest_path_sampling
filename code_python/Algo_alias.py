## Imports

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import random as r
from math import *
from numpy import linalg as LA
import os
import osmnx as ox
from collections import deque



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



Piedmont = loadVille("piedmont")




## Fonctions



def BFS_DAG(G,source):
    """Renvoie un tuple (predecesseur,distance) avec predecesseur le DAG associé au BFS à partir de la source sous la forme d'un dictionnaire noeud: liste de predecesseurs et distance le dictionnaire des distances au point source"""
    pred = {}
    dist = {}
    n,m = len(G.nodes()), len(G.edges())

    seen = [False for i in range(n)]
    fifo = deque([])

    seen[source] = True
    pred[source] = []
    fifo.append(source)

    while (len(fifo) > 0):
        current_node = fifo.popleft()
        dist[current_node] = 0 if current_node==source else dist[pred[current_node][0]] + 1

        for neighbor in G.neighbors(current_node):
            if not seen[neighbor]:
                seen[neighbor] = True
                fifo.append(neighbor)
                pred[neighbor] = [current_node]
            elif neighbor != source and dist[pred[neighbor][0]] == dist[current_node]:
                pred[neighbor].append(current_node)

    return pred,dist






def Table_PCC_In_Place(pred,dist,source,table):
    """Même fonction que Table_PCC mais en remplissant la table passée en argument (supposée de la bonne taille)"""

    inv_dist = inverse_dist(dist)                                               # un dictionnaire avec en inv_dist[d] = liste des sommets à distance d de la source

    table[source] = 1
    for d in range(1,len(inv_dist)) :
        for x in inv_dist[d]:
            somme = 0
            for pred_de_x in pred[x]:
                somme += table[pred_de_x]
            table[x] = somme



def Calcul_seuil_alias(elements, distrib):
    """ Elements = liste des valeurs et distrib(element) renvoie le poids de l'élément. Renvoie les listes seuil et alias pour la méthode de l'alias """
    alveoles = [distrib(elements[i]) for i in range(len(elements))]
    seuil = [1 for _ in range(len(elements))]
    alias = [elements[i] for i in range(len(elements))]

    taille_alveole = sum(alveoles)/len(elements)
    surcharge = set()
    souscharge = set()
    for i in range(len(elements)):
        if alveoles[i] > taille_alveole:
            surcharge.add(i)
        if alveoles[i] < taille_alveole:
            souscharge.add(i)

    while len(souscharge) > 0 and len(surcharge) > 0:
        # On donne une partie du poids d'une alveole en surcharge à une alveole en souscharge
        i = souscharge.pop()
        j = surcharge.pop()
        seuil[i] = alveoles[i]/taille_alveole
        alias[i] = elements[j]

        # On reclasse l'alvéole qui a donné une partie de sa distribution
        alveoles[j] = alveoles[j] - (taille_alveole - alveoles[i])
        if alveoles[j] > taille_alveole:
            surcharge.add(j)
        if alveoles[j] < taille_alveole:
            souscharge.add(j)

    return (seuil, alias)



def Ajoute_alias_dag_IN_PLACE(dag, table):
    """ A la fin de cette fonction : dag[v] = [ (w, t, alias) pour tout w predecesseur de v]  """
    for v in dag.keys():
        sigma_sv = table[v]
        pred_v = dag[v]
        if len(pred_v) > 0:
            seuil, alias = Calcul_seuil_alias(pred_v, lambda w : table[w])
            for i in range(len(pred_v)):
                w , t, al = pred_v[i], seuil[i], alias[i]
                pred_v[i] = (w,t,al)



def Tire_alias(liste):
    """ Prend une liste d'éléments de la forme (valeur, seuil , alias) et renvoie une valeur aléatoirement avec la méthode de l'alias. """
    i = r.randint(0, len(liste)-1)
    t = r.random()

    valeur, seuil, alias = liste[i]
    if t<= seuil:
        return valeur
    else:
        return alias



## Algo v3

def Preprocessing_Graph_alias(G):
    """ Renvoie un tuple (dags,table_departs_arrivees,table_departs,nb_chemins) avec
        - dags[i] = liste des dictionnaires de predecesseurs lors d'un Dijkstra partant de i avec seuil et alias en coordonnées 2 et 3
        - table_departs_arrivees[depart][j] = (seuil_j, alias_j)
        - table_departs[i] = (seuil_i, alias_i) """


    n = len(G)

    dags = []
    tables = np.zeros((n,n))

    # Un BFS par noeud + remplissage de la table des pcc + ajoute des alias sur les arêtes
    for source in range(n):
        pred,dist =  BFS_DAG(G, source)
        Table_PCC_In_Place(pred,dist,source,tables[source])
        Ajoute_alias_dag_IN_PLACE(pred,tables[source])
        dags.append(pred)


    # Tables et variables pour déterminer les départs et arrivées
    table_departs_arrivees = [[0 for _ in range(n)] for _ in range(n)]
    for depart in range(n):
        seuil, alias = Calcul_seuil_alias([i for i in range(n)], lambda v : tables[depart][v])
        for i in range(n):
            table_departs_arrivees[depart][i] = (i, seuil[i], alias[i])

    table_departs = [0 for _ in range(n)]
    poids_departs = np.sum(tables,axis = 1)
    seuil, alias = Calcul_seuil_alias([i for i in range(n)], lambda v : poids_departs[v])
    for i in range(n):
        table_departs[i] = (i, seuil[i], alias[i])


    return (dags,table_departs_arrivees,table_departs)





def Marche_Aleatoire_depart_arrivee(preprocessing, depart, arrivee):
    """Prend en argument le preprocessing ainsi qu'un noeud de départ et d'arrivée. Renvoie un plus court chemin entre départ et arrivée avec proba uniforme. """

    dags = preprocessing[0]
    dag_travail = dags[depart]

    if not (arrivee in dag_travail):
        raise Exception("depart et arrivee pas dans la même composante connexe")

    chemin = [arrivee]
    noeud_courant = arrivee

    # On reconstruit le chemin de l'arrivée au départ en parcourant les prédecesseurs de proche en proche
    while (noeud_courant != depart):
        liste_pred = dag_travail[noeud_courant]
        noeud_courant = Tire_alias(liste_pred)
        chemin.append(noeud_courant)

    chemin.reverse()
    return chemin




def Marche_Aleatoire_depart(preprocessing, depart):
    """Prend en argument le preprocessing, ainsi qu'un noeud de départ. Renvoie un plus court chemin partant de départ uniformément """

    table_departs_arrivees = preprocessing[1]
    arrivee = Tire_alias(table_departs_arrivees[depart])
    return Marche_Aleatoire_depart_arrivee(preprocessing,depart,arrivee)



def Marche_Aleatoire(preprocessing):
    """Prend en argument le preprocessing. Renvoie un plus court chemin uniformément. """

    table_departs = preprocessing[2]
    depart = Tire_alias(table_departs)
    return Marche_Aleatoire_depart(preprocessing,depart)




def Draw_Marche_Aleatoire(G,preprocessing):
    plt.clf()
    pos = nx.spring_layout(G)
    nx.draw_networkx(G,pos,node_color='k')

    path = Marche_Aleatoire(preprocessing)
    path_edges = list(zip(path,path[1:]))

    nx.draw_networkx_nodes(G,pos,nodelist=path,node_color='r')
    nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color='r',width=10)
    plt.axis('equal')
    plt.show()


def DrawOX_Marche_Aleatoire(G,preprocessing):
    plt.clf()
    path = Marche_Aleatoire(preprocessing)
    return ox.plot_graph_route(G, path, orig_dest_size=0, node_size=0)




## Utils



def inverse_dist(dist):
    """Prend le dictionnaire des distance et renvoie un dictionnaire où inv[d] = [liste des à distance d]"""
    inverse = {}
    for noeud,distance in dist.items():
        inverse.setdefault(distance, []).append(noeud)
    return inverse


def rename_nodes(G):
    return nx.convert_node_labels_to_integers(G)




