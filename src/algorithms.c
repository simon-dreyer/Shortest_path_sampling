#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "algorithms.h"

void print_graph(Graph graph){
  // Print the edges
  printf("Edges of the graph:\n");
  for (uli i = 0; i < graph.edge_count; i++) {
    if(graph.is_nb_dag){
      printf("%lu -> %lu : %lu\n", graph.edges[i].src, graph.edges[i].dest, graph.edges[i].nb);
    }
    else{
      printf("%lu -> %lu\n", graph.edges[i].src, graph.edges[i].dest);
    }
  }
  uli nb_nodes = count_nodes(graph);
  printf("Number of nodes : %lu number of edges : %lu\n", nb_nodes, graph.edge_count);
}

void write_graph(const char *filename, Graph graph) {
  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    perror("Error opening file for writing");
    return;
  }
  for (uli i = 0; i < graph.edge_count; i++) {
    fprintf(file, "%lu %lu\n", graph.edges[i].src, graph.edges[i].dest);
  }

  fclose(file);
}

// Function to read graph from file
Graph read_graph(const char *filename) {
    FILE *file;
    Edge *edges = NULL;
    int edge_capacity = 10; // Initial capacity for edges array
    int edge_count = 0;
    Graph graph = {NULL, 0};

    // Allocate initial memory for edges
    edges = (Edge *)malloc(edge_capacity * sizeof(Edge));
    if (edges == NULL) {
        perror("Error allocating memory");
        return graph;
    }

    // Open the file
    file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        free(edges);
        return graph;
    }

    // Read the file and store edges

    // I ignore addiotnal values after the source and destination. Its made on purpose
    // So that if we want to account for weighted graphs it is possible
    char line[256];
    while (fgets(line, sizeof(line), file)) {
      if (sscanf(line, "%lu %lu", &edges[edge_count].src, &edges[edge_count].dest) == 2) {
        edge_count++;

        // If the current capacity is reached, double the array size
        if (edge_count >= edge_capacity) {
          edge_capacity *= 2;
          edges = (Edge *)realloc(edges, edge_capacity * sizeof(Edge));
          if (edges == NULL) {
            perror("Error reallocating memory");
            fclose(file);
            return graph;
          }
        }
      }
    }


    // Close the file
    fclose(file);

    graph.edges = edges;
    graph.edge_count = edge_count;
    return graph;
}



// Function to create a new hash set
HashSet* create_hash_set(int size) {
    HashSet *hash_set = (HashSet *)malloc(sizeof(HashSet));
    hash_set->table = (Node **)malloc(sizeof(Node *) * size);
    hash_set->size = size;
    for (int i = 0; i < size; i++) {
        hash_set->table[i] = NULL;
    }
    return hash_set;
}

// Hash function
int hash(int key, int size) {
    return abs(key) % size;
}

// Function to check if a key exists in the hash set
int contains(HashSet *hash_set, int key) {
    int hash_index = hash(key, hash_set->size);
    Node *entry = hash_set->table[hash_index];
    while (entry != NULL) {
        if (entry->key == key) {
            return 1;
        }
        entry = entry->next;
    }
    return 0;
}

// Function to add a key to the hash set
void add(HashSet *hash_set, int key) {
    if (!contains(hash_set, key)) {
        int hash_index = hash(key, hash_set->size);
        Node *new_node = (Node *)malloc(sizeof(Node));
        new_node->key = key;
        new_node->next = hash_set->table[hash_index];
        hash_set->table[hash_index] = new_node;
    }
}

// Function to free the hash set
void free_hash_set(HashSet *hash_set) {
    for (int i = 0; i < hash_set->size; i++) {
        Node *entry = hash_set->table[i];
        while (entry != NULL) {
            Node *prev = entry;
            entry = entry->next;
            free(prev);
        }
    }
    free(hash_set->table);
    free(hash_set);
}

uli count_nodes(Graph graph) {
  HashSet *hash_set = create_hash_set(2 * graph.edge_count);
  int node_count = 0;

  for (uli i = 0; i < graph.edge_count; i++) {
    uli src = graph.edges[i].src;
    uli dest = graph.edges[i].dest;

    if (!contains(hash_set, src)) {
      add(hash_set, src);
      node_count++;
    }
    if (!contains(hash_set, dest)) {
      add(hash_set, dest);
      node_count++;
    }
  }

  free_hash_set(hash_set);
  return node_count;
}

void create_adjacency_list(Graph graph, uli* adj_list, uli*  node_count, uli nb_nodes) {
  for (uli i = 0; i < graph.edge_count; i++) {
    int src = graph.edges[i].src;
    int dest = graph.edges[i].dest;
    *((adj_list+src*nb_nodes) + node_count[src]) = dest;
    node_count[src] ++;
    //adj_list[src][node_count[src]++] = dest;
    //adj_list[dest][node_count[dest]++] = src; // If the graph is undirected
  }
}

BFS_ret bfs(int start_node, uli* adj_list, uli*  node_count, uli nb_nodes) {
  uli visited[nb_nodes];
  uli* distances;
  uli* nb_paths;
  distances = malloc(nb_nodes* sizeof(uli));
  nb_paths = malloc(nb_nodes* sizeof(uli));
  for(uli i = 0;i < nb_nodes;i++){
    distances[i] = ULONG_MAX;
    nb_paths[i] = 0;
  }
  memset(visited, 0, sizeof visited);
  uli queue[nb_nodes];
  uli front = 0, rear = 0;
  BFS_ret res;


  // values for the dag
  Edge *edges = NULL;
  int edge_capacity = 10; // Initial capacity for edges array
  int edge_count = 0;
  Graph graph = {NULL, 0};
  graph.is_nb_dag = 1;
  graph.is_weighted = 0;
  // Allocate initial memory for edges
  edges = (Edge *)malloc(edge_capacity * sizeof(Edge));
  if (edges == NULL) {
    perror("Error allocating memory");
    return res;
  }

  visited[start_node] = 1;
  distances[start_node] = 0;
  nb_paths[start_node] = 1;
  queue[rear++] = start_node;

  printf("BFS starting from node %d:\n", start_node);
  while (front < rear) {
    uli current_node = queue[front++];
    printf("%lu ", current_node);

    for (uli i = 0; i < node_count[current_node]; i++) {
      uli neighbor = *((adj_list + current_node*nb_nodes) + i);
      if (!visited[neighbor]) {
        visited[neighbor] = 1;
        queue[rear++] = neighbor;
        distances[neighbor] = distances[current_node] + 1;
      }
      if (distances[current_node] + 1 == distances[neighbor]) {
        // add edges to the dag
        edges[edge_count].src = current_node;
        edges[edge_count].dest = neighbor;
        edges[edge_count].nb = nb_paths[current_node];
        edge_count++;
        nb_paths[neighbor] = nb_paths[neighbor] + nb_paths[current_node];
        // If the current capacity is reached, double the array size
        if (edge_count >= edge_capacity) {
          edge_capacity *= 2;
          edges = (Edge *)realloc(edges, edge_capacity * sizeof(Edge));
          if (edges == NULL) {
            perror("Error reallocating memory");
            return res;
          }
        } 
      }
    }
  }
  printf("\n");
  graph.edges = edges;
  graph.edge_count = edge_count;
  res.g = graph;
  res.dist = distances;
  res.paths = nb_paths;
  return res;
}

//send dag values to this function
void dag_to_partial_sum(Graph *g, uli nb_nodes)
{
  uli visited[nb_nodes];
  uli last[nb_nodes];
  memset(visited, 0, sizeof visited);
  memset(last, 0, sizeof last);

  for(uli i = 0; i < g->edge_count; i++){
    if(visited[g->edges[i].dest]){
      g->edges[i].nb = g->edges[i].nb + last[g->edges[i].dest];
      last[g->edges[i].dest] = g->edges[i].nb;
    }
    else{
      last[g->edges[i].dest] = g->edges[i].nb;
      visited[g->edges[i].dest] = 1;
    }
  }
}
