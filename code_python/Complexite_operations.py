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


# Petersen = nx.petersen_graph()
# C6 = nx.cycle_graph(6)
# Biparti34 = nx.complete_bipartite_graph(3,4)
# Piedmont = loadVille("piemont")





## Fonctions

def c_Dijkstra_DAG(G,source):
    """Renvoie un tuple (predecesseur,distance) avec predecesseur le DAG associé au Dijkstra à partir de la source sous la forme d'un dictionnaire noeud: liste de predecesseurs et distance le dictionnaire des distances au point source"""
    pred,dist = nx.dijkstra_predecessor_and_distance(G,source)
    n,m = len(G.nodes()), len(G.edges())
    compteur = floor((n+m)*log(n+1))
    return pred,dist,compteur


def c_BFS_DAG(G,source):
    """Renvoie un tuple (predecesseur,distance) avec predecesseur le DAG associé au BFS à partir de la source sous la forme d'un dictionnaire noeud: liste de predecesseurs et distance le dictionnaire des distances au point source"""
    pred = {}
    dist = {}
    n,m = len(G.nodes()), len(G.edges())

    seen = [False for i in range(n)]
    fifo = deque([])

    seen[source] = True
    pred[source] = []
    fifo.append(source)

    compteur = 1

    while (len(fifo) > 0):
        current_node = fifo.popleft()
        dist[current_node] = 0 if current_node==source else dist[pred[current_node][0]] + 1
        compteur += 1

        for neighbor in G.neighbors(current_node):
            compteur +=1
            if not seen[neighbor]:
                seen[neighbor] = True
                fifo.append(neighbor)
                pred[neighbor] = [current_node]
            elif neighbor != source and dist[pred[neighbor][0]] == dist[current_node]:
                pred[neighbor].append(current_node)

    return pred,dist,compteur


def c_Table_PCC(pred,dist,source):
    """Prend en argument le couple (DAG des predecesseurs,dictionnaire des distance) dans Dijsktra qui part de source. Renvoie un tableau avec en position i le nombre de plus court chemins qui partent de la source et arrivent en i"""
    n = len(pred)
    table = np.zeros(n)
    compteur = 0

    inv_dist,c = c_inverse_dist(dist)                                               # un dictionnaire avec en inv_dist[d] = liste des sommets à distance d de la source
    compteur += c

    table[source] = 1
    for d in range(1,len(inv_dist)) :
        for x in inv_dist[d]:
            somme = 0
            for pred_de_x in pred[x]:
                somme += table[pred_de_x]
                compteur += 1
            table[x] = somme
    return (table,compteur)


def c_Table_PCC_In_Place(pred,dist,source,table):
    """Même fonction que Table_PCC mais en remplissant la table passée en argument (supposée de la bonne taille)"""
    compteur = 0

    inv_dist,c = c_inverse_dist(dist)                                               # un dictionnaire avec en inv_dist[d] = liste des sommets à distance d de la source
    compteur += c


    table[source] = 1
    for d in range(1,len(inv_dist)) :
        for x in inv_dist[d]:
            somme = 0
            for pred_de_x in pred[x]:
                somme += table[pred_de_x]
                compteur +=1
            table[x] = somme
    return compteur



def c_Order_DAG_decreasing_chemins(table,dag):
    compteur = 0
    for k in dag.keys() :
        dag[k] = sorted(dag[k], key=lambda x : table[x], reverse = True)
        l = len(dag[k])
        compteur += floor(l * log(l + 1))                                       # Complexité théorique
    return compteur


def c_Distance_et_pcc_depuis_source(dist,table):
    """Renvoie un dictionnaire où dict[l] = liste des (sommet à distance l,nombre de pcc (source)-->(arrivee) ) """
    compteur = 0
    dico = {}

    for noeud,distance in dist.items():
        dico.setdefault(distance, []).append((noeud,table[noeud]))
        compteur += 1

    # On veut avoir des sommes partielles en 2ème coordonnée.
    for l in dico.keys():
        for i in range(1,len(dico[l])):
            arrivee,nombre_pcc_source_arrivee = dico[l][i]
            dico[l][i] = (arrivee, nombre_pcc_source_arrivee + dico[l][i-1][1])
            compteur += 1

    return (dico,compteur)



def c_Ajoute_somme_partielle_dag_IN_PLACE(pred,table):
    """ Après application de la fonction pred[k] = [ (pred de k, somme partielle sur les autres pred de k des pcc (source)--> (pred de k)), ... ] """
    compteur = 0
    for k in pred.keys():
        somme_partielle = 0
        for i in range(len(pred[k])):
            pred_de_k = pred[k][i]
            somme_partielle += table[pred_de_k]
            pred[k][i] = (pred_de_k, somme_partielle)
            compteur += 1

    return compteur


# Avec poids

def c_Dijkstra_dist(G,source):
    """Renvoie le dictionnaire des distances au point source"""
    n,m = len(G.nodes()), len(G.edges())
    compteur = 0

    dist = nx.single_source_dijkstra_path_length(G,source, weight = "length")
    compteur += floor((n+m)*log(n+1))
    return dist,compteur




def c_Selected_edges(G,dist,poids,epsilon):
    """ Renvoie la liste des arêtes qui peuvent être empruntées dans des ePCC. """
    selected = []
    compteur = 0

    for e in G.edges(keys = True):
        u,v = e[0],e[1]
        if (u in dist and v in dist and poids[e] <= (1 + epsilon)*(dist[v] - dist[u])):
            selected.append(e)
            compteur += 1
        compteur += 1
    return selected,compteur



def c_Table_ePCC_In_Place(G,dist,poids,tables,source, epsilon):
    """ Remplit une table avec en coord (source,arrivee) le nb d'epsilon PCC de source à arrivee. """
    compteur = 0
    n = len(G)

    nodes_by_increasing_dist = sorted(dist.keys(), key=lambda x:dist[x])
    compteur += floor(n*log(n))

    tables[source][source] = 1
    compteur += 1


    for u in nodes_by_increasing_dist[1:]:
        for e in G.in_edges(u,keys = True):                                      # key = True pour traiter le cas où on a des multi-edges
            if (e[0] in dist and poids[e] <= (1 + epsilon)*(dist[e[1]] - dist[e[0]])):
                tables[source][u] += tables[source][e[0]]
                compteur += 1
            compteur += 1
    return compteur



def c_Construct_DAG(G,selected_edges,tables,source):
    """ Construit le DAG enraciné en source qui donne les ePredecesseur avec la somme partielle des ePCC en deuxième coordonnée. """
    pred = {}
    pred[source] = []
    compteur = 0

    for e in selected_edges:
        u, v = e[0], e[1]
        if v in pred:
            pred[v].append((u,tables[source][u] + pred[v][-1][1]))
        else :
            pred[v] = [(u,tables[source][u])]
        compteur += 1
    return pred,compteur


def c_Calcul_seuil_alias(elements, distrib):
    """ Elements = liste des valeurs et distrib(element) renvoie le poids de l'élément. Renvoie les listes seuil et alias pour la méthode de l'alias """
    compteur = 0
    alveoles = [distrib(elements[i]) for i in range(len(elements))]
    compteur += len(elements)
    seuil = [1 for _ in range(len(elements))]
    alias = [elements[i] for i in range(len(elements))]

    taille_alveole = sum(alveoles)/len(elements)
    compteur += len(alveoles)

    surcharge = set()
    souscharge = set()
    for i in range(len(elements)):
        if alveoles[i] > taille_alveole:
            surcharge.add(i)
        if alveoles[i] < taille_alveole:
            souscharge.add(i)
        compteur += 1

    while len(souscharge) > 0 and len(surcharge) > 0:
        # On donne une partie du poids d'une alveole en surcharge à une alveole en souscharge
        i = souscharge.pop()
        j = surcharge.pop()
        seuil[i] = alveoles[i]/taille_alveole
        alias[i] = elements[j]
        compteur += 1

        # On reclasse l'alvéole qui a donné une partie de sa distribution
        alveoles[j] = alveoles[j] - (taille_alveole - alveoles[i])
        if alveoles[j] > taille_alveole:
            surcharge.add(j)
        if alveoles[j] < taille_alveole:
            souscharge.add(j)
        compteur += 1

    return (seuil, alias, compteur)



def c_Ajoute_alias_dag_IN_PLACE(dag, table):
    """ A la fin de cette fonction : dag[v] = [ (w, t, alias) pour tout w predecesseur de v]  """
    compteur = 0
    for v in dag.keys():
        sigma_sv = table[v]
        pred_v = dag[v]
        compteur += 1
        if len(pred_v) > 0:
            seuil, alias,c = c_Calcul_seuil_alias(pred_v, lambda w : table[w])
            compteur += c
            for i in range(len(pred_v)):
                w , t, al = pred_v[i], seuil[i], alias[i]
                pred_v[i] = (w,t,al)
                compteur += 1
    return compteur



def c_Tire_alias(liste):
    """ Prend une liste d'éléments de la forme (valeur, seuil , alias) et renvoie une valeur aléatoirement avec la méthode de l'alias. """

    compteur = 0
    i = r.randint(0, len(liste)-1)
    t = r.random()
    compteur += 2

    valeur, seuil, alias = liste[i]
    if t<= seuil:
        return (valeur, compteur)
    else:
        return (alias, compteur)



## Algo naif

def c_Preprocessing_Graph_naif(G):
    """ Renvoie un tuple (dags,tables,table_departs,nb_chemins) avec
        - dags[i] = liste des dictionnaires de predecesseurs lors d'un Dijkstra partant de i
        - tables[i][j] = table de dimension 2 qui donne le nombre de plus courts chemins partant de i et arrivant à j
        - table_departs[i] = nombre de plus court chemins partant du noeud i
        - nb_chemins = nombre total de plus courts chemins dans le graphe """

    compteur = 0
    n = len(G)

    dags = []
    tables = np.zeros((n,n))

    for source in range(n):
        pred,dist,c = c_Dijkstra_DAG(G,source)
        compteur += c

        dags.append(pred)
        compteur += c_Table_PCC_In_Place(pred,dist,source,tables[source])

    table_departs = np.sum(tables,axis = 1)                                     #nb_chemins[depart] = nb de PCC qui partent de depart
    compteur += n**2
    nb_chemins = int(np.sum(table_departs))
    compteur += n
    prep = (dags,tables,table_departs,nb_chemins)

    return (prep,compteur)


def c_Unranking_PCC_depart_arrivee_naif(preprocessing, depart, arrivee, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ et d'arrivée. Renvoie le plus court chemin entre départ et arrivée de rang 'rang' """
    compteur = 0
    dags,tables,table_departs,nb_chemins = preprocessing

    dag_travail = dags[depart]
    table_travail = tables[depart]

    if table_travail[arrivee] == 0:
        raise Exception("depart et arrivee pas dans la même composante connexe")

    rang_reduit = rang % tables[depart][arrivee]
    compteur +=1

    chemin = [arrivee]
    noeud_courant = arrivee

    # On reconstruit le chemin de l'arrivée au départ en parcourant les prédecesseurs de proche en proche
    while (noeud_courant != depart):
        i = 0
        pred_courant = dag_travail[noeud_courant][i]

        # On cherche le bon prédecesseur
        while (rang_reduit >= table_travail[pred_courant]):
            rang_reduit = rang_reduit - table_travail[pred_courant]
            i+=1
            pred_courant = dag_travail[noeud_courant][i]
            compteur += 1

        chemin.append(pred_courant)
        noeud_courant = pred_courant
        compteur +=1

    chemin.reverse()
    compteur += len(chemin)
    return (chemin,compteur)




def c_Unranking_PCC_depart_naif(preprocessing, depart, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ. Renvoie le plus court chemin partant de départ de rang 'rang' """
    compteur = 0
    dags,tables,table_departs,nb_chemins = preprocessing
    table_travail = tables[depart]

    rang_reduit = rang % table_departs[depart]
    compteur +=1


    # On trouve le noeud d'arrivée
    arrivee_courante = 0
    while (rang_reduit >= table_travail[arrivee_courante]):
        rang_reduit = rang_reduit - table_travail[arrivee_courante]
        arrivee_courante += 1
        compteur += 1

    chemin,c = c_Unranking_PCC_depart_arrivee_naif(preprocessing,depart,arrivee_courante,rang_reduit)
    compteur += c
    return (chemin,compteur)



def c_Unranking_PCC_naif(preprocessing,rang):
    """Prend en argument le preprocessing de la fonction de preprocessing. Renvoie le plus court chemin de rang 'rang' """
    compteur = 0
    dags,tables,table_departs,nb_chemins = preprocessing

    rang_reduit = rang % nb_chemins
    compteur += 1

    # On trouve le noeud de départ
    depart_courant = 0
    while (rang_reduit >= table_departs[depart_courant]):
        rang_reduit = rang_reduit - table_departs[depart_courant]
        depart_courant += 1
        compteur += 1

    chemin,c = c_Unranking_PCC_depart_naif(preprocessing,depart_courant,rang_reduit)
    compteur += c
    return (chemin,compteur)



def c_Uniforme_PCC_naif(preprocessing):
    """Renvoie un plus court chemin de G avec probabilité uniforme sur tous les plus courts chemins"""
    compteur = 0
    dags,tables,table_departs,nb_chemins = preprocessing
    rang = r.randint(0,nb_chemins-1)

    chemin,c = c_Unranking_PCC_naif(preprocessing,rang)
    compteur += c

    return (chemin,compteur)


## Algo ordonne
""" --> On ordonne les prédecesseurs par nombre de PCC décroissants dans les DAGs """


def c_Preprocessing_Graph_ordre(G):
    """ Renvoie un tuple (dags,tables,table_departs,nb_chemins) avec
        - dags[i] = liste des dictionnaires de predecesseurs lors d'un Dijkstra partant de i
        - tables[i][j] = table de dimension 2 qui donne le nombre de plus courts chemins partant de i et arrivant à j
        - table_departs[i] = nombre de plus court chemins partant du noeud i
        - nb_chemins = nombre total de plus courts chemins dans le graphe """

    compteur = 0
    n = len(G)

    dags = []
    tables = np.zeros((n,n))

    for source in range(n):
        pred,dist,c = c_Dijkstra_DAG(G,source)
        compteur += c

        dags.append(pred)
        compteur += c_Table_PCC_In_Place(pred,dist,source,tables[source])

    # On ordonne les predecesseurs par nombre de pcc incidents décroissant pour optimiser la génération
    for depart in range(len(dags)):
        compteur += c_Order_DAG_decreasing_chemins(tables[depart],dags[depart])

    table_departs = np.sum(tables,axis = 1)                                     #nb_chemins[depart] = nb de PCC qui partent de depart
    compteur += n**2
    nb_chemins = int(np.sum(table_departs))
    compteur += n
    prep = (dags,tables,table_departs,nb_chemins)

    return (prep,compteur)


def c_Unranking_PCC_depart_arrivee_ordre(preprocessing, depart, arrivee, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ et d'arrivée. Renvoie le plus court chemin entre départ et arrivée de rang 'rang' """
    compteur = 0
    dags,tables,table_departs,nb_chemins = preprocessing

    dag_travail = dags[depart]
    table_travail = tables[depart]

    if table_travail[arrivee] == 0:
        raise Exception("depart et arrivee pas dans la même composante connexe")

    rang_reduit = rang % tables[depart][arrivee]
    compteur +=1

    chemin = [arrivee]
    noeud_courant = arrivee

    # On reconstruit le chemin de l'arrivée au départ en parcourant les prédecesseurs de proche en proche
    while (noeud_courant != depart):
        i = 0
        pred_courant = dag_travail[noeud_courant][i]

        # On cherche le bon prédecesseur
        while (rang_reduit >= table_travail[pred_courant]):
            rang_reduit = rang_reduit - table_travail[pred_courant]
            i+=1
            pred_courant = dag_travail[noeud_courant][i]
            compteur += 1

        chemin.append(pred_courant)
        noeud_courant = pred_courant
        compteur +=1

    chemin.reverse()
    compteur += len(chemin)
    return (chemin,compteur)




def c_Unranking_PCC_depart_ordre(preprocessing, depart, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ. Renvoie le plus court chemin partant de départ de rang 'rang' """
    compteur = 0
    dags,tables,table_departs,nb_chemins = preprocessing
    table_travail = tables[depart]

    rang_reduit = rang % table_departs[depart]
    compteur +=1


    # On trouve le noeud d'arrivée
    arrivee_courante = 0
    while (rang_reduit >= table_travail[arrivee_courante]):
        rang_reduit = rang_reduit - table_travail[arrivee_courante]
        arrivee_courante += 1
        compteur += 1

    chemin,c = c_Unranking_PCC_depart_arrivee_ordre(preprocessing,depart,arrivee_courante,rang_reduit)
    compteur += c
    return (chemin,compteur)



def c_Unranking_PCC_ordre(preprocessing,rang):
    """Prend en argument le preprocessing de la fonction de preprocessing. Renvoie le plus court chemin de rang 'rang' """
    compteur = 0
    dags,tables,table_departs,nb_chemins = preprocessing

    rang_reduit = rang % nb_chemins
    compteur += 1

    # On trouve le noeud de départ
    depart_courant = 0
    while (rang_reduit >= table_departs[depart_courant]):
        rang_reduit = rang_reduit - table_departs[depart_courant]
        depart_courant += 1
        compteur += 1

    chemin,c = c_Unranking_PCC_depart_ordre(preprocessing,depart_courant,rang_reduit)
    compteur += c
    return (chemin,compteur)



def c_Uniforme_PCC_ordre(preprocessing):
    """Renvoie un plus court chemin de G avec probabilité uniforme sur tous les plus courts chemins"""
    compteur = 0
    dags,tables,table_departs,nb_chemins = preprocessing
    rang = r.randint(0,nb_chemins-1)

    chemin,c = c_Unranking_PCC_ordre(preprocessing,rang)
    compteur += c

    return (chemin,compteur)



## Algo v2
""" --> On ordonne les prédecesseurs par nombre de PCC décroissants dans les DAGs
    --> On fait une dichotomie pour déterminer les départ et arrivée s'ils ne sont pas imposés """


def c_Preprocessing_Graph_v2(G):
    """ Renvoie un tuple (dags,tables,table_departs,nb_chemins) avec
        - dags[i] = liste des dictionnaires de predecesseurs lors d'un Dijkstra partant de i
        - tables[i][j] = table de dimension 2 qui donne le nombre de plus courts chemins (i)-->(j)
        - table_departs_arrivees[depart][j] = table_departs_arrivees[depart][j-1] + nb de PCC (depart)-->(j)
        - table_departs[i] = table_departs[i-1] + nombre de plus court chemins (i)-->(?) """

    n = len(G)
    compteur = 0

    dags = []
    tables = np.zeros((n,n))

    # Un Dijkstra par noeud + remplissage de la table des pcc
    for source in range(n):
        pred,dist,c = c_Dijkstra_DAG(G,source)
        compteur += c

        dags.append(pred)
        compteur += c_Table_PCC_In_Place(pred,dist,source,tables[source])


    # On ordonne les predecesseurs par nombre de pcc incidents décroissant pour optimiser la génération
    for depart in range(len(dags)):
        compteur += c_Order_DAG_decreasing_chemins(tables[depart],dags[depart])


    # Tables et variables annexes pour optimisation
    table_departs_arrivees = np.copy(tables)
    compteur += n*n
    for depart in range(n):
        for k in range(n-1):
            table_departs_arrivees[depart][k+1] += table_departs_arrivees[depart][k]   #table_departs_arrivees[depart][k+1] = table_departs_arrivees[depart][k] + nb de PCC (depart)-->(k+1)
            compteur += 1

    table_departs = np.sum(tables,axis = 1)
    for k in range(n-1):
        table_departs[k+1] += table_departs[k]                                  #table_departs[k+1] = table_departs[k] + nb de PCC qui partent de k+1
        compteur += 1

    nb_chemins = int(table_departs[-1])
    prep = (dags,tables,table_departs_arrivees,table_departs,nb_chemins)

    return (prep,compteur)


def c_Unranking_PCC_depart_arrivee_v2(preprocessing, depart, arrivee, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ et d'arrivée. Renvoie le plus court chemin entre départ et arrivée de rang 'rang' """
    compteur = 0
    dags,tables = preprocessing[0],preprocessing[1]

    dag_travail = dags[depart]
    table_travail = tables[depart]

    if table_travail[arrivee] == 0:
        raise Exception("depart et arrivee pas dans la même composante connexe")

    rang_reduit = rang % tables[depart][arrivee]
    compteur +=1

    chemin = [arrivee]
    noeud_courant = arrivee
    compteur += 1

    # On reconstruit le chemin de l'arrivée au départ en parcourant les prédecesseurs de proche en proche
    while (noeud_courant != depart):
        i = 0
        pred_courant = dag_travail[noeud_courant][i]

        # On cherche le bon prédecesseur
        while (rang_reduit >= table_travail[pred_courant]):
            rang_reduit = rang_reduit - table_travail[pred_courant]
            i+=1
            pred_courant = dag_travail[noeud_courant][i]
            compteur +=1

        chemin.append(pred_courant)
        noeud_courant = pred_courant
        compteur +=1

    chemin.reverse()
    compteur += len(chemin)
    return (chemin,compteur)




def c_Unranking_PCC_depart_v2(preprocessing, depart, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ. Renvoie le plus court chemin partant de départ de rang 'rang' """
    compteur = 0
    table_departs_arrivees = preprocessing[2]
    rang_reduit = rang % int(table_departs_arrivees[depart][-1])
    compteur +=1

    # On trouve le noeud d'arrivée
    arrivee,c = c_recherche_dicho(table_departs_arrivees[depart],rang_reduit)
    compteur += c

    chemin,c = c_Unranking_PCC_depart_arrivee_v2(preprocessing,depart,arrivee,rang_reduit)
    compteur += c
    return (chemin,c)



def c_Unranking_PCC_v2(preprocessing,rang):
    """Prend en argument le preprocessing de la fonction de preprocessing. Renvoie le plus court chemin de rang 'rang' """
    compteur = 0
    table_departs,nb_chemins = preprocessing[3],preprocessing[4]
    rang_reduit = rang % nb_chemins
    compteur += 1

    # On trouve le noeud de départ
    depart,c = c_recherche_dicho(table_departs,rang_reduit)
    compteur += c
    chemin,c = c_Unranking_PCC_depart_v2(preprocessing,depart,rang_reduit)
    compteur += c

    return (chemin,compteur)



def c_Uniforme_PCC_v2(preprocessing):
    """Renvoie un plus court chemin de G avec probabilité uniforme sur tous les plus courts chemins"""
    nb_chemins = preprocessing[4]
    compteur = 0
    rang = r.randint(0,nb_chemins-1)
    compteur += 1

    chemin,c = c_Unranking_PCC_v2(preprocessing,rang)
    compteur += c

    return (chemin, compteur)



## Algo long
"""L'algo supporte des requêtes sur les longueurs des chemins demandés"""




def c_Preprocessing_Graph_long(G):
    """ Renvoie un tuple (dags,tables,depart_distance,dict_departs,nb_dist,nb_chemins) avec
        - dags[i] = liste des dictionnaires de predecesseurs lors d'un Dijkstra partant de i
        - tables[i][j] = table de dimension 2 qui donne le nombre de plus courts chemins (i)-->(j)
        - depart_distance[depart] = dictionnaire dont les clés sont [|0, excentricité(depart)|]
            depart_distance[depart][l] = liste des éléments (arrivée à distance l, somme partielle du nombre de PCC (depart)-->(arrivee) )
        - dict_departs[l] = liste des éléments (départ avec au moins un PCC de longueur l, nombre de PCC de longueur l de la forme (depart)-->(?) )
        - nb_pcc_par_longueur[l] = nombre de PCC de longueur <= l """

    compteur = 0
    n = len(G)

    dags = []
    tables = np.zeros((n,n))
    depart_distance = []
    dict_departs = {}
    nb_pcc_par_longueur = []
    nb_chemins = 0


    # Un Dijkstra par noeud + remplissage de la table des pcc + remplissage du dictionnaire des distances à (source)
    for source in range(n):
        pred,dist,c = c_Dijkstra_DAG(G,source)
        compteur += c

        dags.append(pred)
        c = c_Table_PCC_In_Place(pred,dist,source,tables[source])
        compteur += c

        distances_source,c = c_Distance_et_pcc_depuis_source(dist,tables[source])
        compteur += c

        depart_distance.append(distances_source)
        compteur += 1


    # On ordonne les predecesseurs par nombre de pcc incidents décroissant pour optimiser la génération
    for depart in range(len(dags)):
        c = c_Order_DAG_decreasing_chemins(tables[depart],dags[depart])
        compteur += c


    # On construit le dictionnaire dict_depart
    for depart in range(n):
        for l in range(len(depart_distance[depart])):
            derniere_arrivee,nb_pcc_distance_l = depart_distance[depart][l][-1]
            dict_departs.setdefault(l, []).append((depart,nb_pcc_distance_l))
            compteur += 1

    for l in dict_departs.keys():                                               # On veut avoir des sommes partielles en 2ème coordonnée.
        for i in range(1,len(dict_departs[l])):
            depart,pcc_depuis_depart = dict_departs[l][i]
            dict_departs[l][i] = (depart, pcc_depuis_depart + dict_departs[l][i-1][1])
            compteur += 1

    # Décompte des chemins
    for l in dict_departs.keys():
        nb_pcc_par_longueur.append(dict_departs[l][-1][1])
        compteur += 1
    for i in range(1,len(nb_pcc_par_longueur)):
        # On veut avoir des sommes partielles
        nb_pcc_par_longueur[i] += nb_pcc_par_longueur[i-1]
        compteur += 1

    nb_chemins = int(nb_pcc_par_longueur[-1])
    compteur += 1

    pred = (dags,tables,depart_distance,dict_departs,nb_pcc_par_longueur, nb_chemins)
    return (pred,compteur)


def c_Unranking_PCC_depart_arrivee_long(preprocessing, depart, arrivee, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ et d'arrivée. Renvoie le plus court chemin entre départ et arrivée de rang 'rang' """

    compteur = 0
    dags,tables = preprocessing[0],preprocessing[1]

    dag_travail = dags[depart]
    table_travail = tables[depart]

    if table_travail[arrivee] == 0:
        raise Exception("depart et arrivee pas dans la même composante connexe")

    rang_reduit = rang % tables[depart][arrivee]
    compteur += 1

    chemin = [arrivee]
    noeud_courant = arrivee
    compteur += 1

    # On reconstruit le chemin de l'arrivée au départ en parcourant les prédecesseurs de proche en proche
    while (noeud_courant != depart):
        i = 0
        pred_courant = dag_travail[noeud_courant][i]

        # On cherche le bon prédecesseur
        while (rang_reduit >= table_travail[pred_courant]):
            rang_reduit = rang_reduit - table_travail[pred_courant]
            i+=1
            pred_courant = dag_travail[noeud_courant][i]
            compteur += 1

        chemin.append(pred_courant)
        noeud_courant = pred_courant
        compteur += 1

    chemin.reverse()
    compteur += len(chemin)

    return (chemin,compteur)




def c_Unranking_PCC_depart_longueur_long(preprocessing, depart, longueur, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ. Renvoie le plus court chemin de longueur 'longueur' partant de départ de rang 'rang' """

    compteur = 0
    depart_distance = preprocessing[2]

    if len(depart_distance[depart]) <= longueur :
        raise Exception("Pas de plus courts chemins de cette longueur depuis ce départ.")

    liste_arrivees = depart_distance[depart][longueur]
    nb_pcc = int(liste_arrivees[-1][1])

    rang_reduit = rang % nb_pcc
    compteur += 1

    # On trouve le noeud d'arrivée
    indice_arrivee,c = c_recherche_dicho_par_coordonnee(liste_arrivees,rang_reduit,1)
    compteur += c

    arrivee = liste_arrivees[indice_arrivee][0]

    chemin,c = c_Unranking_PCC_depart_arrivee_long(preprocessing,depart,arrivee,rang_reduit)
    compteur += c

    return (chemin,compteur)



def c_Unranking_PCC_longueur_long(preprocessing,longueur,rang):
    """Prend en argument le preprocessing de la fonction de preprocessing. Renvoie le plus court chemin de longueur 'longueur' de rang 'rang' """

    compteur = 0
    nb_pcc_par_longueur = preprocessing[3]

    if len(nb_pcc_par_longueur) <= longueur :
        raise Exception("Pas de plus courts chemins de cette longueur.")

    liste_departs = nb_pcc_par_longueur[longueur]
    nb_pcc = int(liste_departs[-1][1])

    rang_reduit = rang % nb_pcc
    compteur += 1

    # On trouve le noeud de départ
    indice_depart,c = c_recherche_dicho_par_coordonnee(liste_departs,rang_reduit,1)
    compteur += c
    depart = liste_departs[indice_depart][0]

    chemin,c = c_Unranking_PCC_depart_longueur_long(preprocessing,depart,longueur,rang_reduit)

    compteur += c

    return (chemin,compteur)


def c_Unranking_PCC_long(preprocessing,rang):
    compteur = 0
    nb_pcc_par_longueur,nb_pcc = preprocessing[4],preprocessing[5]

    rang_reduit = rang % nb_pcc
    compteur += 1

    longueur,c = c_recherche_dicho(nb_pcc_par_longueur,rang_reduit)
    compteur += c

    chemin,c = c_Unranking_PCC_longueur_long(preprocessing,longueur,rang_reduit)
    compteur += c

    return (chemin,compteur)


def c_Uniforme_PCC_brut_long(preprocessing):
    """Renvoie un plus court chemin de G avec probabilité uniforme sur tous les plus courts chemins"""
    compteur = 0
    nb_chemins = preprocessing[5]
    rang = r.randint(0,nb_chemins-1)
    compteur += 1

    chemin,c = c_Unranking_PCC_long(preprocessing,rang)
    compteur += c

    return (chemin,compteur)



def c_Uniforme_PCC_long(preprocessing,l_min = 0,l_max = -1):
    """Renvoie un plus court chemin de G de longueur comprise entre l1 et l2 avec probabilité uniforme"""
    compteur = 0
    nb_pcc_par_longueur = preprocessing[4]
    longueur_plus_long_pcc = len(nb_pcc_par_longueur) - 1

    if (l_max<0 or l_max > longueur_plus_long_pcc):
        l_max = longueur_plus_long_pcc

    rang = r.randint(0,nb_pcc_par_longueur[l_max]-1) if l_min == 0 else r.randint(nb_pcc_par_longueur[l_min-1],nb_pcc_par_longueur[l_max]-1)
    compteur += 1

    longueur,c = c_recherche_dicho(nb_pcc_par_longueur,rang)
    compteur += c

    chemin,c =  c_Unranking_PCC_longueur_long(preprocessing, longueur,rang)
    compteur += c

    return (chemin, compteur)



## Algo long v2
""" --> On stocke des sommes partielles dans les dags de façon à pouvoir faire une dichotomie à toutes les étapes de construction du chemin """





def c_Preprocessing_Graph_long_v2(G):
    """ Renvoie un tuple (dags,depart_distance,dict_departs,nb_dist,nb_chemins) avec
        - dags[i] = liste des dictionnaires de predecesseurs lors d'un Dijkstra partant de i avec la somme partielle en 2eme coordonnée
        - depart_distance[depart] = dictionnaire dont les clés sont [|0, excentricité(depart)|]
            depart_distance[depart][l] = liste des éléments (arrivée à distance l, somme partielle du nombre de PCC (depart)-->(arrivee) )
        - dict_departs[l] = liste des éléments (départ avec au moins un PCC de longueur l, nombre de PCC de longueur l de la forme (depart)-->(?) )
        - nb_pcc_par_longueur[l] = nombre de PCC de longueur <= l """

    compteur = 0
    n = len(G)

    dags = []
    depart_distance = []
    dict_departs = {}
    nb_pcc_par_longueur = []
    nb_chemins = 0

    #tables[i][j] = table de dimension 2 qui donne le nombre de plus courts chemins (i)-->(j)
    tables = np.zeros((n,n))



    # Un Dijkstra par noeud + remplissage de la table des pcc + remplissage du dictionnaire des distances à (source)
    for source in range(n):
        pred,dist,c = c_Dijkstra_DAG(G,source)
        compteur += c

        dags.append(pred)
        compteur += 1

        c = c_Table_PCC_In_Place(pred,dist,source,tables[source])
        compteur += c

        c = c_Ajoute_somme_partielle_dag_IN_PLACE(pred,tables[source])                # On ajoute à chaque prédecesseurs, la somme partielle du nombre de pcc entre (source) et lui.
        compteur += c

        distances_source,c = c_Distance_et_pcc_depuis_source(dist,tables[source])
        compteur += c

        depart_distance.append(distances_source)
        compteur += 1


    # On construit le dictionnaire dict_depart
    for depart in range(n):
        for l in range(len(depart_distance[depart])):
            derniere_arrivee,nb_pcc_distance_l = depart_distance[depart][l][-1]
            dict_departs.setdefault(l, []).append((depart,nb_pcc_distance_l))
            compteur += 1

    for l in dict_departs.keys():                                               # On veut avoir des sommes partielles en 2ème coordonnée.
        for i in range(1,len(dict_departs[l])):
            depart,pcc_depuis_depart = dict_departs[l][i]
            dict_departs[l][i] = (depart, pcc_depuis_depart + dict_departs[l][i-1][1])
            compteur += 1

    # Décompte des chemins
    for l in dict_departs.keys():
        nb_pcc_par_longueur.append(dict_departs[l][-1][1])
        compteur += 1
    for i in range(1,len(nb_pcc_par_longueur)):
        # On veut avoir des sommes partielles
        nb_pcc_par_longueur[i] += nb_pcc_par_longueur[i-1]
        compteur += 1

    nb_chemins = int(nb_pcc_par_longueur[-1])
    compteur += 1

    prep = (dags,depart_distance,dict_departs,nb_pcc_par_longueur, nb_chemins)

    return (prep,compteur)


def c_Unranking_PCC_depart_arrivee_long_v2(preprocessing, depart, arrivee, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ et d'arrivée. Renvoie le plus court chemin entre départ et arrivée de rang 'rang' """

    compteur = 0
    dags = preprocessing[0]

    dag_travail = dags[depart]

    if not (arrivee in dag_travail):
        raise Exception("depart et arrivee pas dans la même composante connexe")

    if len(dag_travail[arrivee]) > 0:
        nb_pcc_depart_arrivee = dag_travail[arrivee][-1][1]
        rang_reduit = rang % nb_pcc_depart_arrivee
        compteur += 1

    chemin = [arrivee]
    noeud_courant = arrivee
    compteur += 1

    # On reconstruit le chemin de l'arrivée au départ en parcourant les prédecesseurs de proche en proche
    while (noeud_courant != depart):
        liste_pred = dag_travail[noeud_courant]
        indice_nv_noeud,c = c_recherche_dicho_par_coordonnee(liste_pred,rang_reduit,1)
        compteur += c

        if indice_nv_noeud > 0:
            rang_reduit = rang_reduit - liste_pred[indice_nv_noeud - 1][1]
            compteur += 1

        noeud_courant = liste_pred[indice_nv_noeud][0]

        chemin.append(noeud_courant)
        compteur += 1

    chemin.reverse()
    compteur += len(chemin)

    return (chemin,compteur)




def c_Unranking_PCC_depart_longueur_long_v2(preprocessing, depart, longueur, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ. Renvoie le plus court chemin de longueur 'longueur' partant de départ de rang 'rang' """

    compteur = 0
    depart_distance = preprocessing[1]

    if len(depart_distance[depart]) <= longueur :
        raise Exception("Pas de plus courts chemins de cette longueur depuis ce départ.")

    liste_arrivees = depart_distance[depart][longueur]
    nb_pcc = int(liste_arrivees[-1][1])

    rang_reduit = rang % nb_pcc
    compteur += 1

    # On trouve le noeud d'arrivée
    indice_arrivee,c = c_recherche_dicho_par_coordonnee(liste_arrivees,rang_reduit,1)
    compteur += c

    arrivee = liste_arrivees[indice_arrivee][0]

    chemin,c = c_Unranking_PCC_depart_arrivee_long_v2(preprocessing,depart,arrivee,rang_reduit)
    compteur += c

    return (chemin,compteur)



def c_Unranking_PCC_longueur_long_v2(preprocessing,longueur,rang):
    """Prend en argument le preprocessing de la fonction de preprocessing. Renvoie le plus court chemin de longueur 'longueur' de rang 'rang' """

    compteur = 0
    nb_pcc_par_longueur = preprocessing[2]

    if len(nb_pcc_par_longueur) <= longueur :
        raise Exception("Pas de plus courts chemins de cette longueur.")

    liste_departs = nb_pcc_par_longueur[longueur]
    nb_pcc = int(liste_departs[-1][1])

    rang_reduit = rang % nb_pcc
    compteur += 1

    # On trouve le noeud de départ
    indice_depart,c = c_recherche_dicho_par_coordonnee(liste_departs,rang_reduit,1)
    compteur += c

    depart = liste_departs[indice_depart][0]

    chemin,c = c_Unranking_PCC_depart_longueur_long_v2(preprocessing,depart,longueur,rang_reduit)
    compteur += c

    return (chemin,c)


def c_Unranking_PCC_long_v2(preprocessing,rang):
    compteur = 0
    nb_pcc_par_longueur,nb_pcc = preprocessing[3],preprocessing[4]

    rang_reduit = rang % nb_pcc
    compteur += 1

    longueur,c = c_recherche_dicho(nb_pcc_par_longueur,rang_reduit)
    compteur += c

    chemin,c = c_Unranking_PCC_longueur_long_v2(preprocessing,longueur,rang_reduit)
    compteur += c

    return (chemin,compteur)



def c_Uniforme_PCC_long_v2(preprocessing,l_min = 0,l_max = -1):
    """Renvoie un plus court chemin de G de longueur comprise entre l1 et l2 avec probabilité uniforme"""
    compteur = 0
    nb_pcc_par_longueur = preprocessing[3]
    longueur_plus_long_pcc = len(nb_pcc_par_longueur) - 1

    if (l_max<0 or l_max > longueur_plus_long_pcc):
        l_max = longueur_plus_long_pcc

    rang = r.randint(0,nb_pcc_par_longueur[l_max]-1) if l_min == 0 else r.randint(nb_pcc_par_longueur[l_min-1],nb_pcc_par_longueur[l_max]-1)
    compteur += 1

    longueur,c = c_recherche_dicho(nb_pcc_par_longueur,rang)
    compteur += c

    chemin,c =  c_Unranking_PCC_longueur_long_v2(preprocessing, longueur,rang)
    compteur += c

    return (chemin, compteur)




## Algo v3
""" --> On stocke des sommes partielles dans les dags de façon à pouvoir faire une dichotomie à toutes les étapes de construction du chemin """


def c_Preprocessing_Graph_v3(G):
    """ Renvoie un tuple (dags,table_departs,nb_chemins) avec
        - dags[i] = liste des dictionnaires de predecesseurs lors d'un Dijkstra partant de i avec la somme partielle en 2eme coordonnée
        dags[depart][i][1] = dags[depart][i-1][1] + nb de PCC (depart)-->(dags[depart][i])
        - table_departs_arrivees[depart][j] = table_departs_arrivees[depart][j-1] + nb de PCC (depart)-->(j)
        - table_departs[i] = table_departs[i-1] + nombre de plus court chemins (i)-->(?) """

    n = len(G)

    dags = []
    tables = np.zeros((n,n))
    compteur = 0

    # Un Dijkstra par noeud + remplissage de la table des pcc
    for source in range(n):
        pred,dist,c = c_BFS_DAG(G,source)
        compteur += c

        dags.append(pred)
        c = c_Table_PCC_In_Place(pred,dist,source,tables[source])
        compteur += c

        c = c_Ajoute_somme_partielle_dag_IN_PLACE(pred,tables[source])                # On ajoute à chaque prédecesseurs, la somme partielle du nombre de pcc entre (source) et lui.
        compteur += c

        compteur += 1

    # Tables et variables pour déterminer les départs et arrivées
    table_departs_arrivees = np.copy(tables)
    for depart in range(n):
        for k in range(n-1):
            table_departs_arrivees[depart][k+1] += table_departs_arrivees[depart][k]   #table_departs_arrivees[depart][k+1] = table_departs_arrivees[depart][k] + nb de PCC (depart)-->(k+1)
            compteur += 1

    table_departs = np.sum(tables,axis = 1)
    compteur += n*n

    for k in range(n-1):
        table_departs[k+1] += table_departs[k]                                  #table_departs[k+1] = table_departs[k] + nb de PCC qui partent de k+1
        compteur += 1

    nb_chemins = int(table_departs[-1])
    compteur += 1

    prep = (dags,table_departs_arrivees,table_departs,nb_chemins)
    return (prep,compteur)



def c_Unranking_PCC_depart_arrivee_v3(preprocessing, depart, arrivee, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ et d'arrivée. Renvoie le plus court chemin entre départ et arrivée de rang 'rang' """

    dags = preprocessing[0]
    compteur = 0

    dag_travail = dags[depart]

    if not (arrivee in dag_travail):
        raise Exception("depart et arrivee pas dans la même composante connexe")

    if len(dag_travail[arrivee]) > 0:
        nb_pcc_depart_arrivee = dag_travail[arrivee][-1][1]
        rang_reduit = rang % nb_pcc_depart_arrivee
        compteur += 1

    chemin = [arrivee]
    noeud_courant = arrivee
    compteur += 1

    # On reconstruit le chemin de l'arrivée au départ en parcourant les prédecesseurs de proche en proche
    while (noeud_courant != depart):
        liste_pred = dag_travail[noeud_courant]
        indice_nv_noeud,c = c_recherche_dicho_par_coordonnee(liste_pred,rang_reduit,1)
        compteur += c

        if indice_nv_noeud > 0:
            rang_reduit = rang_reduit - liste_pred[indice_nv_noeud - 1][1]
            compteur += 1

        noeud_courant = liste_pred[indice_nv_noeud][0]
        chemin.append(noeud_courant)
        compteur += 1

    chemin.reverse()
    compteur += len(chemin)
    return (chemin,compteur)




def c_Unranking_PCC_depart_v3(preprocessing, depart, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ. Renvoie le plus court chemin partant de départ de rang 'rang' """

    table_departs_arrivees = preprocessing[1]
    compteur = 0
    rang_reduit = rang % int(table_departs_arrivees[depart][-1])
    compteur += 1

    # On trouve le noeud d'arrivée
    arrivee,c = c_recherche_dicho(table_departs_arrivees[depart],rang_reduit)

    chemin,c = c_Unranking_PCC_depart_arrivee_v3(preprocessing,depart,arrivee,rang_reduit)
    compteur += c
    return (chemin,compteur)



def c_Unranking_PCC_v3(preprocessing,rang):
    """Prend en argument le preprocessing de la fonction de preprocessing. Renvoie le plus court chemin de rang 'rang' """

    table_departs,nb_chemins = preprocessing[2],preprocessing[3]
    compteur = 0

    rang_reduit = rang % nb_chemins
    compteur += 1

    # On trouve le noeud de départ
    depart,c = c_recherche_dicho(table_departs,rang_reduit)
    compteur += c

    chemin,c = c_Unranking_PCC_depart_v3(preprocessing,depart,rang_reduit)
    compteur += c

    return (chemin,compteur)



def c_Uniforme_PCC_v3(preprocessing):
    """Renvoie un plus court chemin de G avec probabilité uniforme sur tous les plus courts chemins"""
    nb_chemins = preprocessing[3]
    compteur = 0
    rang = r.randint(0,nb_chemins-1)
    compteur += 1

    chemin,c = c_Unranking_PCC_v3(preprocessing,rang)
    compteur += c

    return (chemin,compteur)



## Algo alias


def c_Preprocessing_Graph_alias(G):
    """ Renvoie un tuple (dags,table_departs_arrivees,table_departs,nb_chemins) avec
        - dags[i] = liste des dictionnaires de predecesseurs lors d'un Dijkstra partant de i avec seuil et alias en coordonnées 2 et 3
        - table_departs_arrivees[depart][j] = (seuil_j, alias_j)
        - table_departs[i] = (seuil_i, alias_i) """


    n = len(G)

    dags = []
    tables = np.zeros((n,n))
    compteur = 0

    # Un BFS par noeud + remplissage de la table des pcc + ajoute des alias sur les arêtes
    for source in range(n):
        pred,dist,c =  c_BFS_DAG(G, source)
        compteur += c

        c = c_Table_PCC_In_Place(pred,dist,source,tables[source])
        compteur += c

        c = c_Ajoute_alias_dag_IN_PLACE(pred,tables[source])
        compteur += c

        dags.append(pred)
        compteur += 1


    # Tables et variables pour déterminer les départs et arrivées
    table_departs_arrivees = [[0 for _ in range(n)] for _ in range(n)]
    for depart in range(n):
        seuil, alias, c = c_Calcul_seuil_alias([i for i in range(n)], lambda v : tables[depart][v])
        compteur += c

        for i in range(n):
            table_departs_arrivees[depart][i] = (i, seuil[i], alias[i])
            compteur += 1

    table_departs = [0 for _ in range(n)]
    poids_departs = np.sum(tables,axis = 1)
    compteur += n**2

    seuil, alias, c = c_Calcul_seuil_alias([i for i in range(n)], lambda v : poids_departs[v])
    compteur += c
    for i in range(n):
        table_departs[i] = (i, seuil[i], alias[i])
        compteur += 1

    prep = (dags,table_departs_arrivees,table_departs)
    return (prep, compteur)





def c_Marche_Aleatoire_depart_arrivee(preprocessing, depart, arrivee):
    """Prend en argument le preprocessing ainsi qu'un noeud de départ et d'arrivée. Renvoie un plus court chemin entre départ et arrivée avec proba uniforme. """

    dags = preprocessing[0]
    dag_travail = dags[depart]
    compteur = 0

    if not (arrivee in dag_travail):
        raise Exception("depart et arrivee pas dans la même composante connexe")

    chemin = [arrivee]
    noeud_courant = arrivee
    compteur += 1

    # On reconstruit le chemin de l'arrivée au départ en parcourant les prédecesseurs de proche en proche
    while (noeud_courant != depart):
        liste_pred = dag_travail[noeud_courant]
        noeud_courant, c = c_Tire_alias(liste_pred)
        compteur += c
        chemin.append(noeud_courant)
        compteur += 1

    chemin.reverse()
    compteur += len(chemin)
    return (chemin, compteur)




def c_Marche_Aleatoire_depart(preprocessing, depart):
    """Prend en argument le preprocessing, ainsi qu'un noeud de départ. Renvoie un plus court chemin partant de départ uniformément """

    table_departs_arrivees = preprocessing[1]
    compteur = 0
    arrivee, c = c_Tire_alias(table_departs_arrivees[depart])
    compteur += c
    chemin, c = c_Marche_Aleatoire_depart_arrivee(preprocessing,depart,arrivee)
    compteur += c
    return (chemin, compteur)



def c_Marche_Aleatoire(preprocessing):
    """Prend en argument le preprocessing. Renvoie un plus court chemin uniformément. """

    table_departs = preprocessing[2]
    compteur = 0
    depart,c = c_Tire_alias(table_departs)
    compteur += c
    chemin, c = c_Marche_Aleatoire_depart(preprocessing,depart)
    compteur += c
    return (chemin,compteur)




## Algo poids v1


def c_Preprocessing_Graph_poids_v1(G,epsilon = 0.5):
    """ Renvoie un tuple (dags,table_departs,nb_chemins) avec
        - dags[i] = liste des dictionnaires de predecesseurs avec source i avec la somme partielle en 2eme coordonnée
        dags[depart][i][1] = dags[depart][i-1][1] + nb de ePCC (depart)-->(dags[depart][i])
        - table_departs_arrivees[depart][j] = table_departs_arrivees[depart][j-1] + nb de ePCC (depart)-->(j)
        - table_departs[i] = table_departs[i-1] + nombre d'epsilon pcc (i)-->(?) """

    compteur = 0
    n,m = len(G.nodes()), len(G.edges())
    tables = np.zeros((n,n))
    dags = []

    poids = nx.get_edge_attributes(G,"length")
    compteur += m

    for source in range(n):
        dist,c = c_Dijkstra_dist(G,source)
        compteur += c

        c = c_Table_ePCC_In_Place(G,dist,poids,tables,source,epsilon)
        compteur += c

        selected_edges,c = c_Selected_edges(G,dist,poids,epsilon)
        compteur += c

        dag_pred,c = c_Construct_DAG(G,selected_edges,tables,source)
        compteur += c

        dags.append(dag_pred)
        compteur += 1


    # Tables et variables pour déterminer les départs et arrivées
    table_departs_arrivees = np.copy(tables)
    compteur += n*n

    for depart in range(n):
        for k in range(n-1):
            table_departs_arrivees[depart][k+1] += table_departs_arrivees[depart][k]
            compteur += 1

    table_departs = np.sum(tables,axis = 1)
    compteur += n*n

    for k in range(n-1):
        table_departs[k+1] += table_departs[k]
        compteur += 1

    nb_chemins = int(table_departs[-1])
    compteur += 1

    prep = (dags,table_departs_arrivees,table_departs,nb_chemins)
    return prep,compteur




def c_Unranking_ePCC_depart_arrivee_v1(preprocessing, depart, arrivee, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ et d'arrivée. Renvoie l'epsilon pcc entre départ et arrivée de rang 'rang' """

    dags = preprocessing[0]
    compteur = 0

    dag_travail = dags[depart]

    if not (arrivee in dag_travail):
        raise Exception("depart et arrivee pas dans la même composante connexe")

    if len(dag_travail[arrivee]) > 0:
        nb_pcc_depart_arrivee = dag_travail[arrivee][-1][1]
        rang_reduit = rang % nb_pcc_depart_arrivee
        compteur += 1

    chemin = [arrivee]
    noeud_courant = arrivee
    compteur += 1

    # On reconstruit le chemin de l'arrivée au départ en parcourant les prédecesseurs de proche en proche
    while (noeud_courant != depart):
        liste_pred = dag_travail[noeud_courant]
        indice_nv_noeud,c = c_recherche_dicho_par_coordonnee(liste_pred,rang_reduit,1)
        compteur += c

        if indice_nv_noeud > 0:
            rang_reduit = rang_reduit - liste_pred[indice_nv_noeud - 1][1]
            compteur += 1

        noeud_courant = liste_pred[indice_nv_noeud][0]
        chemin.append(noeud_courant)
        compteur += 1

    chemin.reverse()
    compteur += len(chemin)
    return (chemin,compteur)




def c_Unranking_ePCC_depart_v1(preprocessing, depart, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ. Renvoie le plus court chemin partant de départ de rang 'rang' """

    table_departs_arrivees = preprocessing[1]
    compteur = 0
    rang_reduit = rang % int(table_departs_arrivees[depart][-1])
    compteur += 1

    # On trouve le noeud d'arrivée
    arrivee,c = c_recherche_dicho(table_departs_arrivees[depart],rang_reduit)

    chemin,c = c_Unranking_ePCC_depart_arrivee_v1(preprocessing,depart,arrivee,rang_reduit)
    compteur += c
    return (chemin,compteur)



def c_Unranking_ePCC_v1(preprocessing,rang):
    """Prend en argument le preprocessing de la fonction de preprocessing. Renvoie le plus court chemin de rang 'rang' """

    table_departs,nb_chemins = preprocessing[2],preprocessing[3]
    compteur = 0

    rang_reduit = rang % nb_chemins
    compteur += 1

    # On trouve le noeud de départ
    depart,c = c_recherche_dicho(table_departs,rang_reduit)
    compteur += c

    chemin,c = c_Unranking_ePCC_depart_v1(preprocessing,depart,rang_reduit)
    compteur += c

    return (chemin,compteur)



def c_Uniforme_ePCC_v1(preprocessing):
    """Renvoie un plus court chemin de G avec probabilité uniforme sur tous les plus courts chemins"""
    nb_chemins = preprocessing[3]
    compteur = 0
    rang = r.randint(0,nb_chemins-1)
    compteur += 1

    chemin,c = c_Unranking_ePCC_v1(preprocessing,rang)
    compteur += c

    return (chemin,compteur)




## Algo_poids_long_v1



def c_Preprocessing_Graph_poids_long_v1(G,epsilon = 0.5):
    """ Renvoie un tuple (dags, liste_bornes_distance_asc, listes_depart_distance_asc, nb_chemins) avec
        - dags[i] = liste des dictionnaires de predecesseurs avec source i avec la somme partielle en 2eme coordonnée
        dags[depart][i][1] = dags[depart][i-1][1] + nb de ePCC (depart)-->(dags[depart][i])
        - liste_bornes_distance_asc = liste des (dep, arr, dist(dep,arr), nb ePCC (dep) -> (arr) en somme partielle) ordonnées par dist(dep,arr) croissant
        - listes_depart_distance_asc[depart] = liste des (arr, dist(depart,arr), nb ePCC (depart) -> (arr) en somme partielle) ordonnées par dist(depart,arr) croissant """

    compteur = 0

    n = len(G)
    tables = np.zeros((n,n))
    dags = []
    liste_bornes_distance_asc = []
    listes_depart_distance_asc = []

    poids = nx.get_edge_attributes(G,"length")
    compteur += len(G.edges())

    for source in range(n):
        dist,c = c_Dijkstra_dist(G,source)
        compteur += c

        c = c_Table_ePCC_In_Place(G,dist,poids,tables,source,epsilon)
        compteur += c

        selected_edges,c = c_Selected_edges(G,dist,poids,epsilon)
        compteur += c

        dag_pred,c = c_Construct_DAG(G,selected_edges,tables,source)
        compteur += c

        dags.append(dag_pred)
        compteur += 1

        liste_source = []
        for arrivee in dist:
            liste_source.append((arrivee, dist[arrivee], tables[source][arrivee]))
            liste_bornes_distance_asc.append((source, arrivee, dist[arrivee], tables[source][arrivee]))
            compteur += 1

        # On ordonne par distance croissante
        liste_source = sorted(liste_source, key= lambda x : x[1])
        compteur += len(liste_source)*log(len(liste_source)+1)

        # On met les sommes partielles
        nb_ePCC_courant = 0
        for i in range(len(liste_source)):
            arrivee, distance, nb_ePCC = liste_source[i]
            liste_source[i] = (arrivee, distance, nb_ePCC + nb_ePCC_courant)
            nb_ePCC_courant += nb_ePCC
            compteur += 1

        listes_depart_distance_asc.append(liste_source)
        compteur += 1


    # On ordonne par distance croissante
    liste_bornes_distance_asc = sorted(liste_bornes_distance_asc, key= lambda x : x[2])
    compteur += len(liste_bornes_distance_asc)*log(len(liste_bornes_distance_asc)+1)

    # On met les sommes partielles
    nb_ePCC_courant = 0
    for i in range(len(liste_bornes_distance_asc)):
        depart, arrivee, distance, nb_ePCC = liste_bornes_distance_asc[i]
        liste_bornes_distance_asc[i] = (depart, arrivee, distance, nb_ePCC + nb_ePCC_courant)
        nb_ePCC_courant += nb_ePCC
        compteur += 1

    nb_chemins = int(nb_ePCC_courant)
    compteur += 1

    pred = (dags, liste_bornes_distance_asc, listes_depart_distance_asc, nb_chemins)
    return (pred,compteur)




def c_Unranking_ePCC_depart_arrivee_long_v1(preprocessing, depart, arrivee, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ et d'arrivée. Renvoie l'epsilon pcc entre départ et arrivée de rang 'rang' """

    dags = preprocessing[0]
    compteur = 0

    dag_travail = dags[depart]

    if not (arrivee in dag_travail):
        raise Exception("depart et arrivee pas dans la même composante connexe")

    if len(dag_travail[arrivee]) > 0:
        nb_pcc_depart_arrivee = dag_travail[arrivee][-1][1]
        rang_reduit = rang % nb_pcc_depart_arrivee
        compteur += 1

    chemin = [arrivee]
    noeud_courant = arrivee
    compteur += 1

    # On reconstruit le chemin de l'arrivée au départ en parcourant les prédecesseurs de proche en proche
    while (noeud_courant != depart):
        liste_pred = dag_travail[noeud_courant]
        indice_nv_noeud,c = c_recherche_dicho_par_coordonnee_inf(liste_pred,rang_reduit,1)
        compteur += c

        if indice_nv_noeud > 0:
            rang_reduit = rang_reduit - liste_pred[indice_nv_noeud - 1][1]
            compteur += 1

        noeud_courant = liste_pred[indice_nv_noeud][0]
        chemin.append(noeud_courant)
        compteur += 1

    chemin.reverse()
    compteur += len(chemin)
    return (chemin,compteur)




def c_Unranking_ePCC_depart_long_v1(preprocessing, depart, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ. Renvoie le plus court chemin partant de départ de rang 'rang' """

    compteur = 0

    liste_arrivee_distance_asc = preprocessing[2][depart]
    rang_reduit = rang % int(liste_arrivee_distance_asc[-1][-1])
    compteur += 1

    # On trouve le noeud d'arrivée
    i,c = c_recherche_dicho_par_coordonnee_inf(liste_arrivee_distance_asc, rang_reduit, 2)
    compteur += c

    arrivee,distance,nb_ePCC = liste_arrivee_distance_asc[i]

    # On reduit le rang
    rang_reduit = rang_reduit if i==0 else rang_reduit - liste_arrivee_distance_asc[i-1][-1]
    compteur += 1

    chemin,c = c_Unranking_ePCC_depart_arrivee_long_v1(preprocessing,depart,arrivee,rang_reduit)
    compteur += c
    return (chemin, compteur)



def c_Unranking_ePCC_long_v1(preprocessing,rang):
    """Prend en argument le preprocessing de la fonction de preprocessing. Renvoie le plus court chemin de rang 'rang' """

    compteur = 0

    liste_bornes_distance_asc, nb_chemins = preprocessing[1],preprocessing[3]
    rang_reduit = rang % nb_chemins
    compteur += 1

    # On trouve les bornes
    i,c = c_recherche_dicho_par_coordonnee_inf(liste_bornes_distance_asc, rang_reduit, 3)
    compteur += c

    depart, arrivee, distance, nb_ePCC = liste_bornes_distance_asc[i]

    # On reduit le rang
    rang_reduit = rang_reduit if i==0 else rang_reduit - liste_bornes_distance_asc[i-1][-1]
    compteur += 1

    chemin,c = c_Unranking_ePCC_depart_arrivee_long_v1(preprocessing,depart,arrivee,rang_reduit)
    compteur += c

    return (chemin,compteur)



def c_Uniforme_ePCC_brut_long_v1(preprocessing):
    """ Renvoie un plus court chemin de G avec probabilité uniforme sur tous les plus courts chemins """

    compteur = 0

    nb_chemins = preprocessing[3]
    rang = r.randint(0,nb_chemins-1)
    compteur += 1

    chemin,c = c_Unranking_ePCC_long_v1(preprocessing,rang)
    compteur += c
    return (chemin,compteur)



def c_Uniforme_ePCC_depart_long_v1(preprocessing, depart, l_min = 0, l_max = -1):
    """ Renvoie un epsilon plus court chemin de G entre depart et un noeud à distance d € ]l1, l2] de depart avec probabilité uniforme """

    compteur = 0

    liste_arrivee_distance_asc = preprocessing[2][depart]
    longueur_plus_long_pcc = liste_arrivee_distance_asc[-1][1]

    if (l_max<0):
        l_max = longueur_plus_long_pcc

    compteur += 1

    # On regarde les coordonnées des rangs qui respectent les critères de longueur
    i_min,c = c_recherche_dicho_par_coordonnee_sup(liste_arrivee_distance_asc, l_min, 1) #liste_arrivee_distance_asc[i_min][1] <= l_min < liste_arrivee_distance_asc[i_min+1][1]
    compteur += c

    i_max,c = c_recherche_dicho_par_coordonnee_sup(liste_arrivee_distance_asc, l_max, 1) #liste_arrivee_distance_asc[i_max][1] <= l_max < liste_arrivee_distance_asc[i_max+1][1]
    compteur += c


    if (i_min == i_max):
        raise Exception("Pas de paires de noeuds à distance d avec l_min < d <= l_max")

    rang = r.randint(0,liste_arrivee_distance_asc[i_max][2]-1) if i_min == -1 else r.randint(liste_arrivee_distance_asc[i_min][2], liste_arrivee_distance_asc[i_max][2]-1)
    compteur += 1


    # On trouve les bornes associées à ce rang
    i,c = c_recherche_dicho_par_coordonnee_inf(liste_arrivee_distance_asc, rang, 2)      #liste_arrivee_distance_asc[i - 1][2] <= rang < liste_arrivee_distance_asc[i][2]
    compteur += c

    arrivee, distance, nb_ePCC = liste_arrivee_distance_asc[i]

    rang_reduit = rang - liste_arrivee_distance_asc[i - 1][-1] if i > 0 else rang
    compteur += 1

    chemin,c = c_Unranking_ePCC_depart_arrivee_long_v1(preprocessing, depart, arrivee, rang_reduit)
    compteur += c
    return (chemin, compteur)



def c_Uniforme_ePCC_long_v1(preprocessing,l_min = 0,l_max = -1):
    """ Renvoie un epsilon plus court chemin de G entre deux noeuds à distance d € ]l1, l2] avec probabilité uniforme """

    compteur = 0

    liste_bornes_distance_asc = preprocessing[1]
    longueur_plus_long_pcc = liste_bornes_distance_asc[-1][2]

    if (l_max<0):
        l_max = longueur_plus_long_pcc
    compteur += 1

    # On regarde les coordonnées des rangs qui respectent les critères de longueur
    i_min,c = c_recherche_dicho_par_coordonnee_sup(liste_bornes_distance_asc, l_min, 2) #liste_bornes_distance_asc[i_min][2] <= l_min < liste_bornes_distance_asc[i_min+1][2]
    compteur += c

    i_max,c = c_recherche_dicho_par_coordonnee_sup(liste_bornes_distance_asc, l_max, 2) #liste_bornes_distance_asc[i_max][2] <= l_max < liste_bornes_distance_asc[i_max+1][2]
    compteur += c

    rang = r.randint(0,liste_bornes_distance_asc[i_max][3]-1) if i_min == -1 else r.randint(liste_bornes_distance_asc[i_min][3], liste_bornes_distance_asc[i_max][3]-1)
    compteur += 1


    # On trouve les bornes associées à ce rang
    i,c = c_recherche_dicho_par_coordonnee_inf(liste_bornes_distance_asc, rang, 3)      #liste_bornes_distance_asc[i - 1][3] <= rang < liste_bornes_distance_asc[i][3]
    compteur += c

    depart, arrivee, distance, nb_ePCC = liste_bornes_distance_asc[i]

    rang_reduit = rang - liste_bornes_distance_asc[i - 1][-1] if i > 0 else rang
    compteur += 1

    chemin,c = c_Unranking_ePCC_depart_arrivee_long_v1(preprocessing, depart, arrivee, rang_reduit)
    compteur += c

    return (chemin,compteur)




## Simulation


def calcul_donnees(fonction_prep, fonction_gen, echantillon, nbRequetes = 10000, gen_normalise = True, attente = False):
    """Compte les opérations élémentaires lors du preprocessing et des requetes pour tout les graphes de l'échantillon"""
    donnees_prep = []
    donnees_gen = []
    nb_noeuds = []

    nb_attente = 1
    for G in echantillon:
        if attente:
            print("Graphe en cours : n°" + str(nb_attente))
            nb_attente += 1
        prep,compteur_prep = fonction_prep(G)

        compteur_gen = 0

        for i in range(nbRequetes):
            pcc,c = fonction_gen(prep)
            compteur_gen += c/len(pcc) if gen_normalise else c

        donnees_prep.append(compteur_prep)
        donnees_gen.append(compteur_gen)
        nb_noeuds.append(len(G))


    return (nb_noeuds,donnees_prep,donnees_gen)



def donnees_generateurs_seeded(liste_fonctions_unrank, liste_prep, nb_chemins, nbRequetes = 1000):
    """ Calcule le nombre d'opérations élémentaires pour toutes les fonctions unrank de la liste lorsqu'elles sont appelées sur le même chemin. """
    k = len(liste_fonctions_unrank)
    data = [[] for i in range(k)]
    for i in range(nbRequetes):
        rg = r.randint(0,nb_chemins-1)
        for j in range(k):
            data[j].append(liste_fonctions_unrank[j](liste_prep[j], rg)[1])
    return data



def echantillon_erdos_renyi(p = lambda n : 2*log(n)/n, nb_noeuds = [floor(x) for x in np.logspace(1,2.5,50)], poids = True, multiGraph = True):
    """Génère un échantillon de graphes aléatoires d'Erdos-Renyi."""

    echantillon = []

    for n in nb_noeuds:
        G = nx.erdos_renyi_graph(n,p(n),directed = True)
        if poids:
            for e in G.edges():
                u,v = e[0], e[1]
                G[u][v]["length"] = r.random()
        if multiGraph:
            G = nx.MultiDiGraph(G)

        echantillon.append(G)

    return echantillon



def echantillon_barabasi_albert(m = 3, nb_noeuds = [floor(x) for x in np.logspace(1,2.5,50)], poids = True, multiGraph = True):
    """Génère un échantillon de graphes aléatoires de Barabasi_Albert."""

    echantillon = []

    for n in nb_noeuds:
        G = nx.barabasi_albert_graph(n,m)
        if poids:
            for e in G.edges():
                u,v = e[0], e[1]
                G[u][v]["length"] = r.random()
        if multiGraph:
            G = nx.MultiDiGraph(G)

        echantillon.append(G)

    return echantillon



def echantillon_geometrique(densite = 1, nb_noeuds = [floor(x) for x in np.logspace(1,2.5,50)]):
    """Génère un échantillon de graphes aléatoires géométriques 2D. densité = n*4/3*pi*r^3"""

    echantillon = []

    for n in nb_noeuds:
        echantillon.append(nx.random_geometric_graph(n,pow(3*densite/(n*pi*4),1/3)))

    return echantillon


def echantillon_villes(noeuds_min = 0, noeuds_max = 50000):
    nom_fichier_ville = ["ales", "alice_springs", "alloue", "bergerac", "cabourg", "dieppe", "figeac", "la_courtine", "limoges", "nicosie", "orsay", "oulan_bator", "palaiseau", "piedmont", "reykjavik", "tours", "yellowknife"]
    echant = []
    for fichier in nom_fichier_ville:
        G = loadVille(fichier)
        if (len(G) >= noeuds_min and len(G) <= noeuds_max):
            echant.append(G)
    return sorted(echant, key=lambda x:len(x))




## Complexité du générateur

def complexite_generateur_donnees(prepG, fonction_gen, nbRequetes = 1000):
    tailles_chemins = []
    temps_generation = []
    for i in range(nbRequetes):
        path,compteur = fonction_gen(prepG)
        tailles_chemins.append(len(path))
        temps_generation.append(compteur)
    return (tailles_chemins,temps_generation)


def complexite_generateur_toutes_les_longueurs(prepG, fonction_gen, l_max, l_min = 0, nbRequetes_par_l = 5):
    """ Necessite un preprocessing et une fonction génératrice qui supporte les requêtes de longueur """
    longueurs = [l for l in range(l_min, l_max + 1)]
    temps_generation = []

    for l in longueurs:
        moy = 0
        for i in range(nbRequetes_par_l):
            path,compteur = fonction_gen(prepG, l_min = l, l_max = l)
            moy += compteur
        moy = moy/nbRequetes_par_l
        temps_generation.append(moy)
    return (longueurs, temps_generation)


def complexite_generateur_points(donnees):
    tailles_chemins,temps_generation = donnees

    plt.clf()
    plt.plot(tailles_chemins,temps_generation,"ob")
    plt.show()


def complexite_generateur_moyennes(donnees):
    tailles_chemins,temps_generation = donnees
    tailles = {}
    temps_total = {}

    for i in range(len(tailles_chemins)):
        l = tailles_chemins[i]
        tailles[l] = tailles.setdefault(l, 0) + 1
        temps_total[l] = temps_total.setdefault(l,0) + temps_generation[i]

    longueurs = []
    temps_moyen = []
    for k in tailles.keys():
        longueurs.append(k)
        temps_moyen.append(temps_total[k]/tailles[k])

    plt.clf()
    plt.plot(longueurs,temps_moyen,"ob")
    plt.show()





def complexite_generateur_comparaison(liste_donnees):
    plt.clf()

    for j in range(len(liste_donnees)):
        tailles_chemins,temps_generation = liste_donnees[j]
        tailles = {}
        temps_total = {}

        for i in range(len(tailles_chemins)):
            l = tailles_chemins[i]
            tailles[l] = tailles.setdefault(l, 0) + 1
            temps_total[l] = temps_total.setdefault(l,0) + temps_generation[i]

        longueurs = []
        temps_moyen = []
        for k in tailles.keys():
            longueurs.append(k)
            temps_moyen.append(temps_total[k]/tailles[k])


        plt.plot(longueurs,temps_moyen, "s", label = f"Generateur {j}")

    plt.legend()
    plt.show()







## Dessin

def dessin(abcisse, ordonnees_liste, legendes_liste, titre = "", nom_axe_x = "", nom_axe_y = "", logScale = True):
    """ Faire un beau dessin. """
    plt.clf()

    if logScale:
        plt.xscale("log") ; plt.yscale("log")

    for i in range(len(ordonnees_liste)):
        ordonnee = ordonnees_liste[i]
        legende = legendes_liste[i]
        plt.plot(abcisse,ordonnee, label = legende)

    plt.xlabel(nom_axe_x)
    plt.ylabel(nom_axe_y)
    plt.legend()
    plt.title(titre)
    plt.show()


def complexite_graphe(donnees,preprocessing = True, generateur = True, logScale = False):
    nb_noeuds,donnees_prep,donnees_gen = donnees

    plt.clf()
    if logScale:
        plt.xscale("log") ; plt.yscale("log")

    if preprocessing:
        plt.plot(nb_noeuds,donnees_prep, color = "green", label = "Preprocessing")

    if generateur:
        plt.plot(nb_noeuds,donnees_gen, color = "red", label = "Generateur")

    plt.legend()
    plt.show()


def complexite_comparaison_donnees(liste_donnees, preprocessing = True, generateur = True,logScale = True):
    plt.clf()
    if logScale:
        plt.xscale("log") ; plt.yscale("log")

    if preprocessing:
        for i in range(len(liste_donnees)):
            nb_noeuds,donnees_prep,donnees_gen = liste_donnees[i]
            plt.plot(nb_noeuds,donnees_prep, label = f"Preprocessing {i}")

    if generateur:
        for i in range(len(liste_donnees)):
            nb_noeuds,donnees_prep,donnees_gen = liste_donnees[i]
            plt.plot(nb_noeuds,donnees_gen, label = f"Generateur {i}")

    plt.legend()
    plt.show()



def complexite_graphe_comparaison(nb_noeuds,donnees,f,logScale = True):
    facteur_rescaling = donnees[-1]/f(nb_noeuds[-1])
    comparaison = [facteur_rescaling*f(n) for n in nb_noeuds]

    plt.clf()
    if logScale:
        plt.xscale("log") ; plt.yscale("log")


    plt.plot(nb_noeuds,comparaison, color = "blue", label = "f(n)")
    plt.plot(nb_noeuds,donnees, color = "red", label = "Donnees")
    plt.legend()
    plt.show()


def dessin_data_gen(data_gen):
    plt.clf()
    for i in range(len(data_gen)):
        plt.plot(data_gen[i], label = "Generateur " + str(i))
    plt.legend()
    plt.show()



## Autres mesures

def proportion_pred(dags):
    dico_pred = {}
    for dag in dags:
        for noeud in dag:
            nb_pred = len(dag[noeud])
            if nb_pred in dico_pred:
                dico_pred[nb_pred] += 1
            else :
                dico_pred[nb_pred] = 1
    abs = []
    ord = []
    for k in sorted(dico_pred.keys()):
        abs.append(k)
        ord.append(dico_pred[k])
    plt.clf()
    plt.plot(abs,ord)
    plt.show()


## Utils



def c_recherche_dicho(table,rang):
    """Renvoie l'indice i tel que table[i-1] <= rang < table[i]"""
    compteur = 0
    if (rang < table[0]):
        return (0,compteur)

    a = 0
    b = len(table)-1

    while(b-a > 1):
        #On garde l'invariant table[a] <= rang < table[b]
        compteur +=1
        m = (a + b)//2
        if table[m]<=rang:
            a = m
        else:
            b = m
    return (b,compteur)


def c_recherche_dicho_par_coordonnee(table,rang,coordonnee):
    """Renvoie l'indice i tel que table[i-1][coordonnee] <= rang < table[i][coordonnee]"""
    compteur = 0
    if (rang < table[0][coordonnee]):
        return (0,compteur)

    a = 0
    b = len(table)-1

    while(b-a > 1):
        #On garde l'invariant table[a][coordonnee] <= rang < table[b][coordonnee]
        compteur +=1
        m = (a + b)//2
        if table[m][coordonnee]<=rang:
            a = m
        else:
            b = m
    return (b,compteur)


def c_recherche_dicho_par_coordonnee_inf(table,rang,coordonnee):
    """Renvoie l'indice i tel que table[i-1][coordonnee] <= rang < table[i][coordonnee]"""
    compteur = 0
    if (rang < table[0][coordonnee]):
        return (0,compteur)

    a = 0
    b = len(table)-1

    while(b-a > 1):
        #On garde l'invariant table[a][coordonnee] <= rang < table[b][coordonnee]
        compteur +=1
        m = (a + b)//2
        if table[m][coordonnee]<=rang:
            a = m
        else:
            b = m
    return (b,compteur)


def c_recherche_dicho_par_coordonnee_sup(table,rang,coordonnee):
    """Renvoie l'indice i tel que table[i][coordonnee] <= rang < table[i+1][coordonnee]"""
    compteur = 0
    if (rang <= table[0][coordonnee]):
        return (-1,compteur)

    a = 0
    b = len(table)

    while(b-a > 1):
        #On garde l'invariant table[a][coordonnee] <= rang < table[b][coordonnee]
        compteur += 1
        m = (a + b)//2
        if table[m][coordonnee]<=rang:
            a = m
        else:
            b = m
    return (a, compteur)




def c_inverse_dist(dist):
    """Prend le dictionnaire des distance et renvoie un dictionnaire où inv[d] = [liste des à distance d]"""
    inverse = {}
    compteur = 0

    for noeud,distance in dist.items():
        inverse.setdefault(distance, []).append(noeud)
        compteur += 1
    return (inverse,compteur)


def rename_nodes(G):
    return nx.convert_node_labels_to_integers(G)




