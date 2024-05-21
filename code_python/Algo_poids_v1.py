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



## Fonctions

def Dijkstra_dist(G,source):
    """Renvoie le dictionnaire des distances au point source"""
    dist = nx.single_source_dijkstra_path_length(G,source, weight = "length")
    return dist




def Selected_edges(G,dist,poids,epsilon):
    """ Renvoie la liste des arêtes qui peuvent être empruntées dans des ePCC. """
    selected = []
    for e in G.edges(keys = True):
        u,v = e[0],e[1]
        if (u in dist and v in dist and poids[e] <= (1 + epsilon)*(dist[v] - dist[u])):
            selected.append(e)
    return selected



def Table_ePCC_In_Place(G,dist,poids,tables,source, epsilon):
    """ Remplit une table avec en coord (source,arrivee) le nb d'epsilon PCC de source à arrivee. """
    nodes_by_increasing_dist = sorted(dist.keys(), key=lambda x:dist[x])

    tables[source][source] = 1

    for u in nodes_by_increasing_dist[1:]:
        for e in G.in_edges(u,keys = True):                                      # key = True pour traiter le cas où on a des multi-edges
            if (e[0] in dist and poids[e] <= (1 + epsilon)*(dist[e[1]] - dist[e[0]])):
                tables[source][u] += tables[source][e[0]]



def Construct_DAG(G,selected_edges,tables,source):
    """ Construit le DAG enraciné en source qui donne les ePredecesseur avec la somme partielle des ePCC en deuxième coordonnée. """
    pred = {}
    pred[source] = []
    for e in selected_edges:
        u, v = e[0], e[1]
        if v in pred:
            pred[v].append((u,tables[source][u] + pred[v][-1][1]))
        else :
            pred[v] = [(u,tables[source][u])]
    return pred






## Algo v3

def Preprocessing_Graph_poids_v1(G,epsilon = 0.1):
    """ Renvoie un tuple (dags,table_departs,nb_chemins) avec
        - dags[i] = liste des dictionnaires de predecesseurs avec source i avec la somme partielle en 2eme coordonnée
        dags[depart][i][1] = dags[depart][i-1][1] + nb de ePCC (depart)-->(dags[depart][i])
        - table_departs_arrivees[depart][j] = table_departs_arrivees[depart][j-1] + nb de ePCC (depart)-->(j)
        - table_departs[i] = table_departs[i-1] + nombre d'epsilon pcc (i)-->(?) """

    n = len(G)
    tables = np.zeros((n,n))
    dags = []

    poids = nx.get_edge_attributes(G,"length")

    for source in range(n):
        dist = Dijkstra_dist(G,source)
        Table_ePCC_In_Place(G,dist,poids,tables,source,epsilon)
        selected_edges = Selected_edges(G,dist,poids,epsilon)
        dag_pred = Construct_DAG(G,selected_edges,tables,source)
        dags.append(dag_pred)


    # Tables et variables pour déterminer les départs et arrivées
    table_departs_arrivees = np.copy(tables)
    for depart in range(n):
        for k in range(n-1):
            table_departs_arrivees[depart][k+1] += table_departs_arrivees[depart][k]

    table_departs = np.sum(tables,axis = 1)
    for k in range(n-1):
        table_departs[k+1] += table_departs[k]

    nb_chemins = int(table_departs[-1])

    return (dags,table_departs_arrivees,table_departs,nb_chemins)




def Unranking_ePCC_depart_arrivee_v1(preprocessing, depart, arrivee, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ et d'arrivée. Renvoie l'epsilon pcc entre départ et arrivée de rang 'rang' """

    dags = preprocessing[0]

    dag_travail = dags[depart]

    if not (arrivee in dag_travail):
        raise Exception("depart et arrivee pas dans la même composante connexe")

    if len(dag_travail[arrivee]) > 0:
        nb_pcc_depart_arrivee = dag_travail[arrivee][-1][1]
        rang_reduit = rang % nb_pcc_depart_arrivee

    chemin = [arrivee]
    noeud_courant = arrivee

    # On reconstruit le chemin de l'arrivée au départ en parcourant les prédecesseurs de proche en proche
    while (noeud_courant != depart):
        liste_pred = dag_travail[noeud_courant]
        indice_nv_noeud = recherche_dicho_par_coordonnee(liste_pred,rang_reduit,1)
        if indice_nv_noeud > 0:
            rang_reduit = rang_reduit - liste_pred[indice_nv_noeud - 1][1]

        noeud_courant = liste_pred[indice_nv_noeud][0]

        chemin.append(noeud_courant)

    chemin.reverse()
    return chemin




def Unranking_ePCC_depart_v1(preprocessing, depart, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ. Renvoie le plus court chemin partant de départ de rang 'rang' """

    table_departs_arrivees = preprocessing[1]
    rang_reduit = rang % int(table_departs_arrivees[depart][-1])

    # On trouve le noeud d'arrivée
    arrivee = recherche_dicho(table_departs_arrivees[depart],rang_reduit)

    return Unranking_ePCC_depart_arrivee_v1(preprocessing,depart,arrivee,rang_reduit)



def Unranking_ePCC_v1(preprocessing,rang):
    """Prend en argument le preprocessing de la fonction de preprocessing. Renvoie le plus court chemin de rang 'rang' """

    table_departs,nb_chemins = preprocessing[2],preprocessing[3]
    rang_reduit = rang % nb_chemins

    # On trouve le noeud de départ
    depart = recherche_dicho(table_departs,rang_reduit)

    return Unranking_ePCC_depart_v1(preprocessing,depart,rang_reduit)



def Uniforme_ePCC_v1(preprocessing):
    """Renvoie un plus court chemin de G avec probabilité uniforme sur tous les plus courts chemins"""
    nb_chemins = preprocessing[3]
    rang = r.randint(0,nb_chemins-1)

    return Unranking_ePCC_v1(preprocessing,rang)


## Dessin



def Draw_Uniforme_ePCC_v1(G,preprocessing):
    plt.clf()
    pos = nx.spring_layout(G)
    nx.draw_networkx(G,pos,node_color='k')

    path = Uniforme_ePCC_v1(preprocessing)
    path_edges = list(zip(path,path[1:]))

    nx.draw_networkx_nodes(G,pos,nodelist=path,node_color='r')
    nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color='r',width=10)
    plt.axis('equal')
    plt.show()


def DrawOX_Uniforme_ePCC_v1(G,preprocessing):
    plt.clf()
    path = Uniforme_ePCC_v1(preprocessing)
    return ox.plot_graph_route(G, path, orig_dest_size=0, node_size=0)




## Utils



def recherche_dicho(table,rang):
    """Renvoie l'indice i tel que table[i-1] <= rang < table[i]"""
    if (rang < table[0]):
        return 0

    a = 0
    b = len(table)-1

    while(b-a > 1):
        #On garde l'invariant table[a] <= rang < table[b]
        m = (a + b)//2
        if table[m]<=rang:
            a = m
        else:
            b = m
    return b


def recherche_dicho_par_coordonnee(table,rang,coordonnee):
    """Renvoie l'indice i tel que table[i-1][coordonnee] <= rang < table[i][coordonnee]"""
    if (rang < table[0][coordonnee]):
        return 0

    a = 0
    b = len(table)-1

    while(b-a > 1):
        #On garde l'invariant table[a][coordonnee] <= rang < table[b][coordonnee]
        m = (a + b)//2
        if table[m][coordonnee]<=rang:
            a = m
        else:
            b = m
    return b



def inverse_dist(dist):
    """Prend le dictionnaire des distance et renvoie un dictionnaire où inv[d] = [liste des à distance d]"""
    inverse = {}
    for noeud,distance in dist.items():
        inverse.setdefault(distance, []).append(noeud)
    return inverse


def rename_nodes(G):
    return nx.convert_node_labels_to_integers(G)


def mesure_chemin(G, chemin, weight = "length", multiGraph = True):
    poids = 0
    for i in range(1,len(chemin)):
        poids_edge = G[chemin[i-1]][chemin[i]][0][weight] if multiGraph else G[chemin[i-1]][chemin[i]][weight]
        poids += poids_edge
    return poids



