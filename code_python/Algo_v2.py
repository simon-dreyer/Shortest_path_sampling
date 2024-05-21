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



Piedmont = loadVille("piemont")




## Fonctions

def Dijkstra_DAG(G,source):
    """Renvoie un tuple (predecesseur,distance) avec predecesseur le DAG associé au Dijkstra à partir de la source sous la forme d'un dictionnaire noeud: liste de predecesseurs et distance le dictionnaire des distances au point source"""
    pred,dist = nx.dijkstra_predecessor_and_distance(G,source)
    return pred,dist



def Table_PCC(pred,dist,source):
    """Prend en argument le couple (DAG des predecesseurs,dictionnaire des distance) dans Dijsktra qui part de source. Renvoie un tableau avec en position i le nombre de plus court chemins qui partent de la source et arrivent en i"""
    n = len(pred)
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


## Algo
""" --> On ordonne les prédecesseurs par nombre de PCC décroissants dans les DAGs
    --> On fait une dichotomie pour déterminer les départ et arrivée s'ils ne sont pas imposés """


def Preprocessing_Graph(G):
    """ Renvoie un tuple (dags,tables,table_departs,nb_chemins) avec
        - dags[i] = liste des dictionnaires de predecesseurs lors d'un Dijkstra partant de i
        - tables[i][j] = table de dimension 2 qui donne le nombre de plus courts chemins (i)-->(j)
        - table_departs_arrivees[depart][j] = table_departs_arrivees[depart][j-1] + nb de PCC (depart)-->(j)
        - table_departs[i] = table_departs[i-1] + nombre de plus court chemins (i)-->(?) """

    #G = rename_nodes(G)

    n = len(G)

    dags = []
    tables = np.zeros((n,n))

    # Un Dijkstra par noeud + remplissage de la table des pcc
    for source in range(n):
        pred,dist = Dijkstra_DAG(G,source)
        dags.append(pred)
        Table_PCC_In_Place(pred,dist,source,tables[source])


    # On ordonne les predecesseurs par nombre de pcc incidents décroissant pour optimiser la génération
    for depart in range(len(dags)):
        Order_DAG_decreasing_chemins(tables[depart],dags[depart])


    # Tables et variables annexes pour optimisation
    table_departs_arrivees = np.copy(tables)
    for depart in range(n):
        for k in range(n-1):
            table_departs_arrivees[depart][k+1] += table_departs_arrivees[depart][k]   #table_departs_arrivees[depart][k+1] = table_departs_arrivees[depart][k] + nb de PCC (depart)-->(k+1)

    table_departs = np.sum(tables,axis = 1)
    for k in range(n-1):
        table_departs[k+1] += table_departs[k]                                  #table_departs[k+1] = table_departs[k] + nb de PCC qui partent de k+1

    nb_chemins = int(table_departs[-1])


    return (dags,tables,table_departs_arrivees,table_departs,nb_chemins)


def Order_DAG_decreasing_degree(G,dag):
    for k in dag.keys() :
        dag[k] = sorted(dag[k], key=lambda x : nx.degree(G,x), reverse = True)


def Order_DAG_decreasing_chemins(table,dag):
    for k in dag.keys() :
        dag[k] = sorted(dag[k], key=lambda x : table[x], reverse = True)


def Unranking_PCC_depart_arrivee(preprocessing, depart, arrivee, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ et d'arrivée. Renvoie le plus court chemin entre départ et arrivée de rang 'rang' """

    dags,tables = preprocessing[0],preprocessing[1]

    dag_travail = dags[depart]
    table_travail = tables[depart]

    if table_travail[arrivee] == 0:
        raise Exception("depart et arrivee pas dans la même composante connexe")

    rang_reduit = rang % tables[depart][arrivee]

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

        chemin.append(pred_courant)
        noeud_courant = pred_courant

    chemin.reverse()
    return chemin




def Unranking_PCC_depart(preprocessing, depart, rang):
    """Prend en argument le preprocessing de la fonction de preprocessing, ainsi qu'un noeud de départ. Renvoie le plus court chemin partant de départ de rang 'rang' """

    table_departs_arrivees = preprocessing[2]
    rang_reduit = rang % int(table_departs_arrivees[depart][-1])

    # On trouve le noeud d'arrivée
    arrivee = recherche_dicho(table_departs_arrivees[depart],rang_reduit)

    return Unranking_PCC_depart_arrivee(preprocessing,depart,arrivee,rang_reduit)



def Unranking_PCC(preprocessing,rang):
    """Prend en argument le preprocessing de la fonction de preprocessing. Renvoie le plus court chemin de rang 'rang' """

    table_departs,nb_chemins = preprocessing[3],preprocessing[4]
    rang_reduit = rang % nb_chemins

    # On trouve le noeud de départ
    depart = recherche_dicho(table_departs,rang_reduit)

    return Unranking_PCC_depart(preprocessing,depart,rang_reduit)



def Uniforme_PCC(preprocessing):
    """Renvoie un plus court chemin de G avec probabilité uniforme sur tous les plus courts chemins"""
    nb_chemins = preprocessing[4]
    rang = r.randint(0,nb_chemins-1)

    return Unranking_PCC(preprocessing,rang)


def Draw_Uniforme_PCC(G,preprocessing):
    plt.clf()
    pos = nx.spring_layout(G)
    nx.draw_networkx(G,pos,node_color='k')

    path = Uniforme_PCC(preprocessing)
    path_edges = list(zip(path,path[1:]))

    nx.draw_networkx_nodes(G,pos,nodelist=path,node_color='r')
    nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color='r',width=10)
    plt.axis('equal')
    plt.show()


def DrawOX_Uniforme_PCC(G,preprocessing):
    plt.clf()
    path = Uniforme_PCC(preprocessing)
    fig,ax = ox.plot_graph_route(G, path, orig_dest_size=0, node_size=0)
    fig.show()






## Simulation


def complexite_temps_donnees_erdos_renyi(c = 1, nb_noeuds = [floor(x) for x in np.logspace(1,3,50)], tailleEchantillon = 3, nbRequetes = 5):
    """ Calcule les performances temporelles du preprocessing et des requetes pour des graphes aléatoires d'Erdos-Renyi."""

    donnees_prep = []
    donnees_gen = []

    for n in nb_noeuds:

        # Création de l'échantillon
        echantillon = []
        for i in range(tailleEchantillon):
            echantillon.append(nx.erdos_renyi_graph(n,c/n))

        # Mesure des performances de preprocessing et de Unranking
        moyenne_prep = 0
        moyenne_gen = 0

        for G in echantillon:
            avant = time.time(); prep = Preprocessing_Graph(G); apres = time.time()
            moyenne_prep += (apres - avant)/tailleEchantillon

            avant = time.time()
            for i in range(nbRequetes):
                pcc = Uniforme_PCC(prep)
            apres = time.time()
            moyenne_gen += (apres - avant)/tailleEchantillon

        donnees_prep.append(moyenne_prep)
        donnees_gen.append(moyenne_gen)
    return (nb_noeuds,donnees_prep,donnees_gen)



def complexite_temps_donnees_barabasi_albert(m = 3, nb_noeuds = [floor(x) for x in np.logspace(1,2.5,50)], tailleEchantillon = 3, nbRequetes = 5):
    """ Calcule les performances temporelles du preprocessing et des requetes pour des graphes aléatoires d'Erdos-Renyi."""

    donnees_prep = []
    donnees_gen = []

    for n in nb_noeuds:

        # Création de l'échantillon
        echantillon = []
        for i in range(tailleEchantillon):
            echantillon.append(nx.barabasi_albert_graph(n,m))

        # Mesure des performances de preprocessing et de Unranking
        moyenne_prep = 0
        moyenne_gen = 0

        for G in echantillon:
            avant = time.time(); prep = Preprocessing_Graph(G); apres = time.time()
            moyenne_prep += (apres - avant)/tailleEchantillon

            avant = time.time()
            for i in range(nbRequetes):
                pcc = Uniforme_PCC(prep)
            apres = time.time()
            moyenne_gen += (apres - avant)/tailleEchantillon

        donnees_prep.append(moyenne_prep)
        donnees_gen.append(moyenne_gen)
    return (nb_noeuds,donnees_prep,donnees_gen)




def complexite_temps_donnees_geometrique(densite = 1, nb_noeuds = [floor(x) for x in np.logspace(1,2.5,50)], tailleEchantillon = 3, nbRequetes = 5):
    """ Calcule les performances temporelles du preprocessing et des requetes pour des graphes aléatoires géométriques 2D. densité = n*4/3*pi*r^3"""

    donnees_prep = []
    donnees_gen = []

    for n in nb_noeuds:

        # Création de l'échantillon
        echantillon = []
        for i in range(tailleEchantillon):
            echantillon.append(nx.random_geometric_graph(n,pow(3*densite/(n*pi*4),1/3)))

        # Mesure des performances de preprocessing et de Unranking
        moyenne_prep = 0
        moyenne_gen = 0

        for G in echantillon:
            avant = time.time(); prep = Preprocessing_Graph(G); apres = time.time()
            moyenne_prep += (apres - avant)/tailleEchantillon

            avant = time.time()
            for i in range(nbRequetes):
                pcc = Uniforme_PCC(prep)
            apres = time.time()
            moyenne_gen += (apres - avant)/tailleEchantillon

        donnees_prep.append(moyenne_prep)
        donnees_gen.append(moyenne_gen)
    return (nb_noeuds,donnees_prep,donnees_gen)


# Dessin


def complexite_temps_graphe(donnees,preprocessing = True, generateur = True, logScale = False):
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


def complexite_temps_graphe_comparaison(nb_noeuds,donnees,f,logScale = True):
    facteur_rescaling = donnees[-1]/f(nb_noeuds[-1])
    comparaison = [facteur_rescaling*f(n) for n in nb_noeuds]

    plt.clf()
    if logScale:
        plt.xscale("log") ; plt.yscale("log")

    plt.plot(nb_noeuds,donnees, color = "red", label = "Donnees")
    plt.plot(nb_noeuds,comparaison, color = "blue", label = "f(n)")
    plt.legend()
    plt.show()



# Complexité du générateur

def complexite_generateur_donnees(prepG, nbRequetes = 1000):
    tailles_chemins = []
    temps_generation = []
    for i in range(nbRequetes):
        avant = time.time(); path = Uniforme_PCC(prepG); apres = time.time()
        tailles_chemins.append(len(path))
        temps_generation.append(apres-avant)
    return (tailles_chemins,temps_generation)


def complexite_generateur_points(donnees):
    tailles_chemins,temps_generation = donnees

    plt.clf()
    plt.plot(tailles_chemins,temps_generation,"ob")
    plt.show()







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


def Predecessors_to_Successors(pred):
    """A partir du dictionnaire des prédecesseurs, renvoie le dictionnaire des successeurs dans le DAG"""
    succ = {}

    for key in pred.keys():
        succ[key]=[]

    for a,liste_predecesseurs_a in pred.items():
        for x in liste_predecesseurs_a :
            succ[x].append(a)

    return succ



def inverse_dist(dist):
    """Prend le dictionnaire des distance et renvoie un dictionnaire où inv[d] = [liste des à distance d]"""
    inverse = {}
    for noeud,distance in dist.items():
        inverse.setdefault(distance, []).append(noeud)
    return inverse


def rename_nodes(G):
    return nx.convert_node_labels_to_integers(G)




