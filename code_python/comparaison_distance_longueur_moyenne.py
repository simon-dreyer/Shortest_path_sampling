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




## Preprocessing avec longueur

def Dijkstra_DAG(G,source):
    """Renvoie un tuple (predecesseur,distance) avec predecesseur le DAG associé au Dijkstra à partir de la source sous la forme d'un dictionnaire noeud: liste de predecesseurs et distance le dictionnaire des distances au point source"""
    pred,dist = nx.dijkstra_predecessor_and_distance(G,source)
    return pred,dist



def inverse_dist(dist):
    """Prend le dictionnaire des distance et renvoie un dictionnaire où inv[d] = [liste des à distance d]"""
    inverse = {}
    for noeud,distance in dist.items():
        inverse.setdefault(distance, []).append(noeud)
    return inverse



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



def Distance_et_pcc_depuis_source(dist,table):
    """ Renvoie un dictionnaire où dict[l] = liste des (sommet à distance l,nombre de pcc (source)-->(arrivee) ) """
    dico = {}
    for noeud,distance in dist.items():
        dico.setdefault(distance, []).append((noeud,table[noeud]))

    # On veut avoir des sommes partielles en 2ème coordonnée.
    for l in dico.keys():
        for i in range(1,len(dico[l])):
            arrivee,nombre_pcc_source_arrivee = dico[l][i]
            dico[l][i] = (arrivee, nombre_pcc_source_arrivee + dico[l][i-1][1])

    return dico



def Ajoute_somme_partielle_dag_IN_PLACE(pred,table):
    """ Après application de la fonction pred[k] = [ (pred de k, somme partielle sur les autres pred de k des pcc (source)--> (pred de k)), ... ] """
    for k in pred.keys():
        somme_partielle = 0
        for i in range(len(pred[k])):
            pred_de_k = pred[k][i]
            somme_partielle += table[pred_de_k]
            pred[k][i] = (pred_de_k, somme_partielle)



def Preprocessing_Graph_long_v2(G):
    """ Renvoie un tuple (dags,depart_distance,dict_departs,nb_dist,nb_chemins) avec
        - dags[i] = liste des dictionnaires de predecesseurs lors d'un Dijkstra partant de i avec la somme partielle en 2eme coordonnée
        - depart_distance[depart] = dictionnaire dont les clés sont [|0, excentricité(depart)|]
            depart_distance[depart][l] = liste des éléments (arrivée à distance l, somme partielle du nombre de PCC (depart)-->(arrivee) )
        - dict_departs[l] = liste des éléments (départ avec au moins un PCC de longueur l, nombre de PCC de longueur l de la forme (depart)-->(?) )
        - nb_pcc_par_longueur[l] = nombre de PCC de longueur <= l """

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
        pred,dist = Dijkstra_DAG(G,source)
        dags.append(pred)
        Table_PCC_In_Place(pred,dist,source,tables[source])
        Ajoute_somme_partielle_dag_IN_PLACE(pred,tables[source])                # On ajoute à chaque prédecesseurs, la somme partielle du nombre de pcc entre (source) et lui.
        distances_source = Distance_et_pcc_depuis_source(dist,tables[source])
        depart_distance.append(distances_source)


    # On construit le dictionnaire dict_depart
    for depart in range(n):
        for l in range(len(depart_distance[depart])):
            derniere_arrivee,nb_pcc_distance_l = depart_distance[depart][l][-1]
            dict_departs.setdefault(l, []).append((depart,nb_pcc_distance_l))

    for l in dict_departs.keys():                                               # On veut avoir des sommes partielles en 2ème coordonnée.
        for i in range(1,len(dict_departs[l])):
            depart,pcc_depuis_depart = dict_departs[l][i]
            dict_departs[l][i] = (depart, pcc_depuis_depart + dict_departs[l][i-1][1])

    # Décompte des chemins
    for l in dict_departs.keys():
        nb_pcc_par_longueur.append(dict_departs[l][-1][1])
    for i in range(1,len(nb_pcc_par_longueur)):
        # On veut avoir des sommes partielles
        nb_pcc_par_longueur[i] += nb_pcc_par_longueur[i-1]

    nb_chemins = int(nb_pcc_par_longueur[-1])


    return (dags,depart_distance,dict_departs,nb_pcc_par_longueur, nb_chemins)






## Echantillon


def echantillon_erdos_renyi(p = lambda n : 2*log(n)/n, nb_noeuds = [floor(x) for x in np.logspace(1,2.5,50)]):
    """Génère un échantillon de graphes aléatoires d'Erdos-Renyi."""

    echantillon = []

    for n in nb_noeuds:
        echantillon.append(nx.erdos_renyi_graph(n,p(n)))

    return echantillon



def echantillon_erdos_renyi_p(p_min, p_max, n = 100, nb_points = 30):
    """Génère un échantillon de graphes aléatoires d'Erdos-Renyi à nombre de noeuds constant. Renvoie également la liste des proba utilisées"""

    proba = [x for x in np.linspace(max(0, p_min), min(1,p_max),nb_points)]
    echantillon = []

    for p in proba:
        echantillon.append(nx.erdos_renyi_graph(n,p))

    return echantillon,proba



def echantillon_barabasi_albert(m = 3, nb_noeuds = [floor(x) for x in np.logspace(1,2.5,50)]):
    """Génère un échantillon de graphes aléatoires de Barabasi_Albert."""

    echantillon = []

    for n in nb_noeuds:
        echantillon.append(nx.barabasi_albert_graph(n,m))

    return echantillon



def echantillon_grille_carree(nb_noeuds_cote = [i for i in range(1,21)]):
    """Génère un échantillon de graphes n x n."""

    echantillon = []

    for n in nb_noeuds_cote:
        G = nx.grid_2d_graph(n,n)
        G = nx.convert_node_labels_to_integers(G)
        echantillon.append(G)

    return echantillon



def echantillon_grille(dimensions = [(i,i,i) for i in range(1,10)]):
    """Génère un échantillon de graphes n x n."""

    echantillon = []

    for dim in dimensions:
        G = nx.grid_graph(dim)
        G = nx.convert_node_labels_to_integers(G)
        echantillon.append(G)

    return echantillon


def shrek_graph(n,m):
    """2 fils de tailles m (oreilles) colles sur un clique de taille n (tête)"""
    K = nx.complete_graph(n)
    L = nx.path_graph(m)
    R = nx.path_graph(m)
    LR = nx.union(L,R,rename = ("L", "R"))
    G = nx.union(K,LR)

    for i in K.nodes():
        G.add_edge(i,"L0")
        G.add_edge(i,"R0")

    G = nx.convert_node_labels_to_integers(G)
    return G


def echantillon_shrek(fm, n_list = [floor(x) for x in np.linspace(10,300,50)]):
    echant = []
    for n in n_list:
        m = floor(fm(n))
        echant.append(shrek_graph(n,m))
    return echant



## Acquisition


def calcul_longueur_et_distance_moyenne(echantillon):
    """Renvoie le tripplet (nb_noeuds,distances_moyennes,longueurs_moyennes)"""
    longueurs_moyennes = []
    distances_moyennes = []
    nb_noeuds = []

    chargement = 0

    for G in echantillon:
        #print(chargement)
        n = len(G)
        nb_noeuds.append(n)


        prep = Preprocessing_Graph_long_v2(G)

        depart_distance = prep[1]
        moyenne_dist = 0
        nb_paires = 0
        for depart in range(n):
            for l in depart_distance[depart].keys():
                moyenne_dist += l*len(depart_distance[depart][l])
                nb_paires += len(depart_distance[depart][l])

        distances_moyennes.append(moyenne_dist/nb_paires)



        nb_pcc_par_longueur, nb_chemins = prep[3],prep[4]
        moyenne_long = 0
        for l in range(1,len(nb_pcc_par_longueur)):
            moyenne_long += (nb_pcc_par_longueur[l]-nb_pcc_par_longueur[l-1])*l/nb_chemins      # Car somme partielles

        longueurs_moyennes.append(moyenne_long)

        chargement += 1


    return (nb_noeuds,distances_moyennes,longueurs_moyennes)



def calcul_rapport_lmoy_sur_dmoy(donnees):
    """Renvoie la liste des rapport longueur_moyenne/distance_moyenne."""
    nb_noeuds,distances_moyennes,longueurs_moyennes = donnees
    rapports = []
    for i in range(len(nb_noeuds)):
        rapports.append(longueurs_moyennes[i]/distances_moyennes[i])

    return rapports




def nombre_PCC(echantillon):
    nb_noeuds = []
    nb_PCC = []
    for G in echantillon:
        n = len(G)
        nb_noeuds.append(n)

        prep = Preprocessing_Graph_long_v2(G)
        nb_chemins = prep[4]
        nb_PCC.append(nb_chemins)

    return (nb_noeuds,nb_PCC)


def degre_moyen(G):
    n = len(G)
    return sum([G.degree[node] for node in G.nodes()])/n




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



def compare_dist_long(donnees, abcisse = None, logScale = False, valeur_theorique = None):
    nb_noeuds,distances_moyennes,longueurs_moyennes = donnees

    variable_x = nb_noeuds if (abcisse == None) else abcisse

    plt.clf()
    if logScale:
        plt.xscale("log") ; plt.yscale("log")

    plt.plot(variable_x,distances_moyennes, color = "green", label = "Distance moyenne entre deux noeuds")

    plt.plot(variable_x,longueurs_moyennes, color = "red", label = "Longueur moyenne d'un pcc")

    if valeur_theorique != None:
        plt.plot(variable_x,valeur_theorique, color = "orange", label = "Valeur_theorique")

    plt.legend()
    plt.show()



def rapport_long_sur_dist(donnees, abcisse = None, logScale = False):
    nb_noeuds,distances_moyennes,longueurs_moyennes = donnees
    variable_x = nb_noeuds if (abcisse == None) else abcisse

    rapports = []
    for i in range(len(nb_noeuds)):
        rapports.append(longueurs_moyennes[i]/distances_moyennes[i])

    plt.clf()
    if logScale:
        plt.xscale("log") ; plt.yscale("log")

    plt.plot(variable_x,rapports, color = "blue", label = "rapport (longueur moyenne pcc) / (distance moyenne)")

    plt.legend()
    plt.show()



def graphe_nb_PCC (donnees, abcisse = None, logScale = False):
    nb_noeuds,nb_PCC = donnees

    variable_x = nb_noeuds if (abcisse == None) else abcisse

    plt.clf()
    if logScale:
        plt.xscale("log") ; plt.yscale("log")

    plt.plot(variable_x, nb_PCC, color = "orange", label = "Nombre de plus courts chemins")

    plt.legend()
    plt.show()

