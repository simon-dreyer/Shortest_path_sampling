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

def Dijkstra_DAG(G,source):
    """Renvoie un tuple (predecesseur,distance) avec predecesseur le DAG associé au Dijkstra à partir de la source sous la forme d'un dictionnaire noeud: liste de predecesseurs et distance le dictionnaire des distances au point source"""
    pred,dist = nx.dijkstra_predecessor_and_distance(G,source)
    return pred,dist


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




def Table_PCC(n,pred,dist,source):
    """Prend en argument le couple (DAG des predecesseurs,dictionnaire des distance) dans Dijsktra qui part de source. Renvoie un tableau avec en position i le nombre de plus court chemins qui partent de la source et arrivent en i"""
    table = np.zeros(n)

    inv_dist = inverse_dist(dist)                                               # un dictionnaire avec en inv_dist[d] = liste des sommets à distance d de la source

    table[source] = 1
    for d in range(1,len(inv_dist)) :
        for x in inv_dist[d]:
            somme = 0
            for pred_de_x in pred[x]:
                somme += table[pred_de_x]
            table[x] = somme
    return table


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


def Ajoute_somme_partielle_dag_IN_PLACE(pred,table):
    """ Après application de la fonction pred[k] = [ (pred de k, somme partielle sur les autres pred de k des pcc (source)--> (pred de k)), ... ] """
    for k in pred.keys():
        somme_partielle = 0
        for i in range(len(pred[k])):
            pred_de_k = pred[k][i]
            somme_partielle += table[pred_de_k]
            pred[k][i] = (pred_de_k, somme_partielle)


## Algo v3


def Init_Preprocessing(G):
    """ Renvoie un tuple (dags, dico_departs_arrivees, sources_vues) avec
        - dags = dico des DAG de predecesseurs
        dags[i] = DAG de predecesseurs lors d'un BFS partant de i avec la somme partielle en 2eme coordonnée
        dags[depart][i][1] = dags[depart][i-1][1] + nb de PCC (depart)-->(dags[depart][i])

        - dico_departs_arrivees = dico de tables de longueur n
        dico_departs_arrivees[depart][j] = table_departs_arrivees[depart][j-1] + nb de PCC (depart)-->(j)

        - sources_vues = liste de taille n qui indique si on a déjà traité la source ou non """

    dags = {}
    dico_departs_arrivees = {}
    sources_vues = [False for i in range(len(G.nodes()))]
    return (dags, dico_departs_arrivees, sources_vues)



def Update_Preprocessing_source(G, prepG, source, dijkstra = False):
    """ Fait le preprocessing pour une source "source". """

    n = len(G)
    dags, dico_departs_arrivees, sources_vues = prepG


    pred,dist = Dijkstra_DAG(G,source) if dijkstra else BFS_DAG(G, source)
    dags[source] = pred
    table_PCC = Table_PCC(n,pred,dist,source)
    Ajoute_somme_partielle_dag_IN_PLACE(pred,table_PCC)                         # On ajoute à chaque prédecesseurs, la somme partielle du nombre de pcc entre (source) et lui.


    # Table pour déterminer l'arrivée
    table_arrivees = np.copy(table_PCC)
    for k in range(n-1):
        table_arrivees[k+1] += table_arrivees[k]                        #table_departs_arrivees[depart][k+1] = table_departs_arrivees[depart][k] + nb de PCC (depart)-->(k+1)
    dico_departs_arrivees[source] = table_arrivees

    sources_vues[source] = True






def Unranking_PCC_depart_arrivee_dynamique(preprocessing, depart, arrivee, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ et d'arrivée. Renvoie le plus court chemin entre départ et arrivée de rang 'rang' """

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



def Uniforme_PCC_depart_arrivee_dynamique(G, prepG, depart, arrivee):
    """Renvoie un plus court chemin de G avec probabilité uniforme sur tous les plus courts chemins entre depart et arrivee"""
    sources_vues = prepG[2]
    if not sources_vues[depart]:
        Update_Preprocessing_source(G, prepG, depart)

    dico_departs_arrivees = prepG[1]
    nb_chemins = dico_departs_arrivees[depart][arrivee] if arrivee == 0 else dico_departs_arrivees[depart][arrivee] - dico_departs_arrivees[depart][arrivee-1]
    rang = r.randint(0,nb_chemins-1)

    return Unranking_PCC_depart_arrivee_dynamique(prepG, depart, arrivee ,rang)





def Unranking_PCC_depart_dynamique(preprocessing, depart, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ. Renvoie le plus court chemin partant de départ de rang 'rang' """

    dico_departs_arrivees = preprocessing[1]
    rang_reduit = rang % int(dico_departs_arrivees[depart][-1])

    # On trouve le noeud d'arrivée
    arrivee = recherche_dicho(dico_departs_arrivees[depart],rang_reduit)

    return Unranking_PCC_depart_arrivee_dynamique(preprocessing,depart,arrivee,rang_reduit)




def Uniforme_PCC_depart_dynamique(G, prepG, depart):
    """Renvoie un plus court chemin de G avec probabilité uniforme sur tous les plus courts chemins partant de depart"""
    sources_vues = prepG[2]
    if not sources_vues[depart]:
        Update_Preprocessing_source(G, prepG, depart)

    dico_departs_arrivees = prepG[1]
    nb_chemins = dico_departs_arrivees[depart][-1]
    rang = r.randint(0,nb_chemins-1)

    return Unranking_PCC_depart_dynamique(prepG, depart, rang)



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



## Traceroute

def traceroute_sources_destinations(G, k, m):
    """ Tire uniformement k départs et m arrivees et renvoie le graphe induit par un PCC tiré uniformément entre toutes les combinaisons source-destination possibles. """
    n = len(G.nodes())
    sources = r.sample(G.nodes(), k)
    destinations = r.sample(G.nodes(), m)
    prepG = Init_Preprocessing(G)

    sampled_graph = nx.Graph()
    for depart in sources:
        for arrivee in destinations:
            path = Uniforme_PCC_depart_arrivee_dynamique(G,prepG, depart, arrivee)
            ajoute_path(sampled_graph, path)

    return sampled_graph


def traceroute_sources_destinations_article(G, k, m):
    """ Tire uniformement k départs et m arrivees, on met un poids réel sur chaque arête et renvoie le graphe induit par les (uniques) PCC entre toutes les combinaisons source-destination possibles. """
    n = len(G.nodes())
    sources = r.sample(G.nodes(), k)
    destinations = r.sample(G.nodes(), m)

    G_1 = G.copy()
    ajoute_poids(G_1)

    sampled_graph = nx.Graph()
    for depart in sources:
        for arrivee in destinations:
            path = nx.shortest_path(G_1, source = depart, target = arrivee, weight = "weight")
            ajoute_path(sampled_graph, path)

    return sampled_graph


def ajoute_path(G, path):
    """ Ajoute les noeuds et arêtes de path au graphe G. """
    if len(path)==0:
        return G
    previous_node = path[0]
    G.add_node(previous_node)
    for current_node in path[1:]:
        G.add_node(current_node)
        G.add_edge(previous_node, current_node)
        previous_node = current_node


def OX_dessine_sampled_graph(G, sampled_G):
    G_1 = G.copy()
    for (u,v) in G.edges():
        if (u,v) not in sampled_G.edges():
            G_1.remove_edge(u,v)
    for n in G.nodes():
        if n not in sampled_G.nodes():
            G_1.remove_node(n)
    ox.plot_graph(G_1)


def ajoute_poids(G):
    """ Donne un poids 1 + epsilon à toutes les arêtes avec -1/N < epsilon < 1/N  """
    N = len(G.nodes())
    for (u,v) in G.edges():
        poids = 1 + 2*(r.random()- 0.5)/N
        G[u][v]["weight"] = poids
        G[v][u]["weight"] = poids



## Expérience


def echantillon_erdos_renyi(p = lambda n : 2*log(n)/n, nb_noeuds = [floor(x) for x in np.logspace(1,2.5,50)], poids = False, multiGraph = False):
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



def degre_distribution(G):
    """ Renvoie une liste telle que l[d] compte la proportion de noeuds de degré >= d """
    degres = [0 for i in range(len(G.nodes))]
    max_deg = 0
    for n,d in G.degree():
        degres[d] += 1
        if d > max_deg:
            max_deg = d

    N = len(G.nodes())
    distrib = [1.0]
    for i in range(max_deg+1):
        freq = distrib[i] - degres[i]/N
        distrib.append(freq)

    return distrib


def plot_degre_distrib(G):
    plt.clf()
    distrib = degre_distribution(G)
    plt.plot(distrib)

    plt.show()


def compare_degre_distrib(G, k, m):
    """ Affiche les distribution de degré reelle, avec notre algo, avec l'algo de l'article. """
    distrib_reelle = degre_distribution(G)

    tavant = time.time()
    sampled_G = traceroute_sources_destinations(G, k,m)
    tapres = time.time()
    print("Temps de calcul algo uniforme : " + str(tapres-tavant))
    distrib_algo = degre_distribution(sampled_G)

    tavant = time.time()
    sampled_G_article = traceroute_sources_destinations_article(G,k,m)
    tapres = time.time()
    print("Temps de calcul algo article : " + str(tapres-tavant))
    distrib_article = degre_distribution(sampled_G_article)

    plt.clf()
    plt.plot(distrib_reelle, label = "Réel")
    plt.plot(distrib_algo, label = "Avec PCC uniforme")
    plt.plot(distrib_article, label = "Article")
    plt.legend()
    plt.show()