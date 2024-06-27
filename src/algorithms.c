#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "algorithms.h"

/////////////////////////////////////////// Read, write, print graps ///////////////////////////////

void print_graph(Graph* graph, int full_info){
  // Print the edges
  if (full_info){
    printf("Edges of the graph:\n");
    for (uli i = 0; i < graph->edge_count; i++) {
      if(graph->is_nb_dag){
        printf("%lu -> %lu : %lu\n", graph->edges[i].src, graph->edges[i].dest, graph->edges[i].nb);
      }
      else{
        printf("%lu -> %lu\n", graph->edges[i].src, graph->edges[i].dest);
      }
    } 
  }
  uli nb_nodes = count_nodes(graph);
  printf("Number of nodes : %lu number of edges : %lu\n", nb_nodes, graph->edge_count);
}

void write_graph(const char *filename, Graph* graph) {
  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    perror("Error opening file for writing");
    return;
  }
  for (uli i = 0; i < graph->edge_count; i++) {
    fprintf(file, "%lu %lu %lu\n", graph->edges[i].src, graph->edges[i].dest, graph->edges[i].nb);
  }

  fclose(file);
}

// Function to read graph from file
Graph read_graph(const char *filename, int is_weighted) {
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
    int nb_correct = (is_weighted == 1) ? 3 : 2;
    char line[256];
    int nb_read;
    while (fgets(line, sizeof(line), file)) {
      if(! is_weighted)
        nb_read = sscanf(line, "%lu %lu", &edges[edge_count].src, &edges[edge_count].dest);
      else
        nb_read = sscanf(line, "%lu %lu %lu", &edges[edge_count].src, &edges[edge_count].dest, &edges[edge_count].nb);
      if (nb_read == nb_correct) {
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


//////////////////////////////////////////////   Other Functions Hash Set and List /////////////////////////////////


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
        if (entry->data == key) {
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
        new_node->data = key;
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


Node* createNode(uli data) {

    Node* newNode = (Node*)malloc(sizeof(Node));
    if (newNode == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    newNode->data = data;
    newNode->next = NULL;
    return newNode;
}

// Function to initialize a new list
void initList(List* list) {
    list->head = NULL;
    list->tail = NULL;
}

// Function to add a node to the end of the list
void addNode(List* list, uli data) {
    Node* newNode = createNode(data);
    if (list->head == NULL) {
      list->head = newNode;
      list->tail = newNode;
    } else {
      list->tail->next = newNode;
      list->tail = newNode;
    }
}

// Function to print the list
void printList(Node* head) {
    Node* temp = head;
    while (temp != NULL) {
        printf("%lu -> ", temp->data);
        temp = temp->next;
    }
    printf("NULL\n");
}

// Function to free the list
void freeList(Node* head) {
    Node* temp;
    while (head != NULL) {
        temp = head;
        head = head->next;
        free(temp);
    }
}

// Function to reverse the list
void reverseList(List* list) {
    Node* prev = NULL;
    Node* curr = list->head;
    Node* next = NULL;
    list->tail = list->head;

    while (curr != NULL) {
        next = curr->next;  // Store the next node
        curr->next = prev;  // Reverse the current node's pointer
        prev = curr;        // Move the prev and curr pointers one step forward
        curr = next;
    }

    list->head = prev;  // Update the head pointer to the new first node
}

// Function to write the list to a file
void writeListToFile(List* head, uli nb, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
        return;
    }
    for(uli i = 0; i < nb; i ++){
      if(head[i].head != NULL){
        Node* temp = head[i].head;
        while (temp->next != NULL) {
          fprintf(file, "%lu,", temp->data);
          temp = temp->next;
        }
        fprintf(file, "%lu\n", temp->data); 
      }
    }

    fclose(file);
}
/////////////////////////////////////: End HashSet and List ////////////////////////////

uli count_nodes(Graph* graph) {
  HashSet *hash_set = create_hash_set(2 * graph->edge_count);
  int node_count = 0;

  for (uli i = 0; i < graph->edge_count; i++) {
    uli src = graph->edges[i].src;
    uli dest = graph->edges[i].dest;

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

void create_adjacency_list(Graph* graph, Couple_adj** adj_list, uli*  node_count, uli nb_nodes, char* directed, int is_reversed) {
  //Couple_adj* res = (Couple_adj*) malloc(graph->edge_count * sizeof(Couple_adj));
  if(!is_reversed){
    for (uli i = 0; i < graph->edge_count; i++) {
      uli src = graph->edges[i].src;
      uli dest = graph->edges[i].dest;
      //res[i].v = dest;
      //res[i].nb = graph->edges[i].nb;
      adj_list[src][node_count[src]].v = dest;
      adj_list[src][node_count[src]].nb = graph->edges[i].nb;
      //(*((adj_list+src*nb_nodes) + node_count[src])).v = dest;
      //(*((adj_list+src*nb_nodes) + node_count[src])).nb = graph->edges[i].nb;
      node_count[src] ++;
      if (strcmp(directed,"u") == 0)
        {
          //res[i].v = src;
          //res[i].nb = graph->edges[i].nb;
          adj_list[dest][node_count[dest]].v = src;
          adj_list[dest][node_count[dest]].nb = graph->edges[i].nb;
          /* (*((adj_list+dest*nb_nodes) + node_count[dest])).v = src; */
          /* (*((adj_list+dest*nb_nodes) + node_count[dest])).nb = graph->edges[i].nb; */
          node_count[dest] ++;
        }
    } 
  }
  else{
    for (uli i = 0; i < graph->edge_count; i++) {
      uli dest = graph->edges[i].src;
      uli src = graph->edges[i].dest;
      //res[i].v = dest;
      //res[i].nb = graph->edges[i].nb;
      adj_list[src][node_count[src]].v = dest;
      adj_list[src][node_count[src]].nb = graph->edges[i].nb;
      /* (*((adj_list+src*nb_nodes) + node_count[src])).v = dest; */
      /* (*((adj_list+src*nb_nodes) + node_count[src])).nb = graph->edges[i].nb; */
      node_count[src] ++;
      if (strcmp(directed,"u") == 0)
        {
          //res[i].v = src;
          //res[i].nb = graph->edges[i].nb;
          adj_list[dest][node_count[dest]].v = src;
          adj_list[dest][node_count[dest]].nb = graph->edges[i].nb;
          /* (*((adj_list+dest*nb_nodes) + node_count[dest])).v = src; */
          /* (*((adj_list+dest*nb_nodes) + node_count[dest])).nb = graph->edges[i].nb; */
          node_count[dest] ++;
        }
    }
  }
}

BFS_ret bfs(int start_node, Couple_adj** adj_list, uli*  node_count, uli nb_nodes) {
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
    exit(-1);
  }

  visited[start_node] = 1;
  distances[start_node] = 0;
  nb_paths[start_node] = 1;
  queue[rear++] = start_node;

  //printf("BFS starting from node %d:\n", start_node);
  while (front < rear) {
    uli current_node = queue[front++];
    //printf("%lu ", current_node);
    Couple_adj z;
    for (uli i = 0; i < node_count[current_node]; i++) {
      z = adj_list[current_node][i];
      //*((adj_list + current_node*nb_nodes) + i);
      uli neighbor = z.v;
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
            exit(-1);
          }
        } 
      }
    }
  }
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


////////////////////////////////////////// Sorting part ///////////////////////////////////////

int compareElem(const void *a, const void *b) {
  Couple_pred *A = (Couple_pred *)a;
  Couple_pred *B = (Couple_pred *)b;
  return (A->r < B->r) - (A->r > B->r);
}

Edge * optimal_bunrank_order(Graph* graph, Couple_pred* nb_paths_from_s, uli nb_nodes, Couple_adj** adj_list, uli*  node_count){
  // il est possible de faire mieux en pratique probablement en ordonnant directement les aretes entrantes d'un noeud
  qsort(nb_paths_from_s, nb_nodes, sizeof(Couple_pred) ,compareElem);
  /* for(uli jj=0; jj<nb_nodes;jj++) */
  /*   printf("v %lu nb %lu edge count %lu node count : %lu\n",nb_paths_from_s[jj].v, nb_paths_from_s[jj].r, graph->edge_count, node_count[jj]); */
  Edge * new_edges = (Edge *) malloc(graph->edge_count * sizeof(Edge));
  if (new_edges == NULL)
    {
      printf("optimal_bunrank_order: malloc problem allocation\n");
      exit(-1);
    }
  uli ii = 0;
  for(uli i = 0; i < nb_nodes; i++){
    for(uli j = 0; j < node_count[nb_paths_from_s[i].v]; j++){
      new_edges[ii].src = nb_paths_from_s[i].v;
      new_edges[ii].dest = adj_list[nb_paths_from_s[i].v][j].v;
      new_edges[ii].nb = nb_paths_from_s[i].r;
      new_edges[ii].weight = 0.0;
      ii++;
    }
  }
  /* puts("hello2"); */
  /* free(graph->edges); */
  /* puts("hello3"); */
  /* graph->edges = new_edges; */
  /* puts("hello4"); */
  return new_edges;
}
