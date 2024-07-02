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

      printf("%lu -> %lu : %lu %lu %lg\n", graph->edges[i].src, graph->edges[i].dest, graph->edges[i].nb, graph->edges[i].alias, graph->edges[i].prob);
    }
  }
  printf("Number of nodes : %lu number of edges : %lu\n", graph->nb_nodes, graph->edge_count);
}

//here we could spare space in b-unrank and only write edges of the dag without weights
void write_graph(const char *filename, Graph* graph, int is_alias) {
  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    perror("Error opening file for writing");
    return;
  }
  if(is_alias){
    for (uli i = 0; i < graph->edge_count; i++) {
      fprintf(file, "%lu %lu %lu %lg\n", graph->edges[i].src, graph->edges[i].dest, graph->edges[i].alias, graph->edges[i].prob);
    }
  }
  else{
    for (uli i = 0; i < graph->edge_count; i++) {
      fprintf(file, "%lu %lu %lu\n", graph->edges[i].src, graph->edges[i].dest, graph->edges[i].nb);
    }
  }

  fclose(file);
}

// Function to read graph from file, if is_weighted is true then is_alias should be false and vice-versa
Graph read_graph(const char *filename, int is_nb_dag, int is_alias) {
    FILE *file;
    Edge *edges = NULL;
    uli edge_capacity = 10; // Initial capacity for edges array
    uli edge_count = 0;
    Graph graph = {NULL, 0,0,0,0};
    if(is_nb_dag){
      graph.is_nb_dag = 1;
    }
    if(is_alias){
      graph.is_alias = 1;
    }
    // Allocate initial memory for edges
    edges = (Edge *)malloc(edge_capacity * sizeof(Edge));
    for(uli i = 0;i< edge_capacity;i++){
      edges[i].src = 0;
      edges[i].dest = 0;
      edges[i].nb = 0;
      edges[i].alias = 0;
      edges[i].prob = 0.0;
    }
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
    int nb_correct = 0;
    if(is_nb_dag){
      nb_correct = 3;
    }
    else if(is_alias){
      nb_correct = 4;
    }
    else{
      nb_correct = 2;
    }

    //int nb_correct = (is_weighted == 1) ? 3 : 2;
    char line[256];
    int nb_read;
    while (fgets(line, sizeof(line), file)) {
      if(is_nb_dag){
        nb_read = sscanf(line, "%lu %lu %lu", &edges[edge_count].src, &edges[edge_count].dest, &edges[edge_count].nb);
      }
      else if(is_alias){
        nb_read = sscanf(line, "%lu %lu %lu %lg", &edges[edge_count].src, &edges[edge_count].dest, &edges[edge_count].alias, &edges[edge_count].prob);
      }
      else{
        nb_read = sscanf(line, "%lu %lu", &edges[edge_count].src, &edges[edge_count].dest); 
      }

      if (nb_read == nb_correct) {
        edge_count++;

        // If the current capacity is reached, double the array size
        if (edge_count >= edge_capacity) {
          edge_capacity *= 2;
          edges = (Edge *)realloc(edges, edge_capacity * sizeof(Edge));
          for(uli i = edge_capacity/2;i< edge_capacity;i++){
            edges[i].src = 0;
            edges[i].dest = 0;
            edges[i].nb = 0;
            edges[i].prob = 0.0;
            edges[i].alias = 0;
          }
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
    graph.nb_nodes = count_nodes(edges, edge_count);
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
void writeResults(List* head, uli nb, const char* filename, const char* timename, const char* operationname, double time, uli nb_operations, char* time_or_operations) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
        return;
    }
    FILE* file2 = fopen(timename, "w");
    if (file2 == NULL) {
      fprintf(stderr, "Error opening file %s for writing\n", timename);
      return;
    }
    FILE* file3 = fopen(operationname, "w");
    if (file3 == NULL) {
      fprintf(stderr, "Error opening file %s for writing\n", operationname);
      return;
    }
    if(strcmp(time_or_operations,"t") == 0){
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
    }
    fprintf(file2, "%f", time);
    fprintf(file3, "%lu", nb_operations);

    fclose(file);
    fclose(file2);
}

Dictionary* createDictionary(uli size) {
    Dictionary* dict = (Dictionary*)malloc(sizeof(Dictionary));
    dict->size = size;
    dict->table = (Node_dic**)malloc(size * sizeof(Node_dic*));
    for (uli i = 0; i < size; ++i) {
        dict->table[i] = NULL;
    }
    return dict;
}

// Hash function to map keys to indices
uli hashFunction(uli key, uli size) {
    return key % size;
}

// Function to insert a key-value pair into the dictionary
void insert(Dictionary* dict, uli key, uli value) {
    uli index = hashFunction(key, dict->size);
    Node_dic* newNode = (Node_dic*)malloc(sizeof(Node_dic));
    newNode->key = key;
    newNode->value = value;
    newNode->next = dict->table[index];
    dict->table[index] = newNode;
}

// Function to find a value by key in the dictionary
bool find(Dictionary* dict, uli key, uli* value) {
    uli index = hashFunction(key, dict->size);
    Node_dic* current = dict->table[index];
    while (current != NULL) {
        if (current->key == key) {
            *value = current->value;
            return true;
        }
        current = current->next;
    }
    return false;
}

// Function to free the memory used by the dictionary
void freeDictionary(Dictionary* dict) {
    for (uli i = 0; i < dict->size; ++i) {
        Node_dic* current = dict->table[i];
        while (current != NULL) {
            Node_dic* temp = current;
            current = current->next;
            free(temp);
        }
    }
    free(dict->table);
    free(dict);
}

void printDictionary(Dictionary* dict) {
    for (uli i = 0; i < dict->size; ++i) {
        Node_dic* current = dict->table[i];
        while (current != NULL) {
            printf("Key: %lu, Value: %lu\n", current->key, current->value);
            current = current->next;
        }
    }
}


/////////////////////////////////////: End HashSet and List ////////////////////////////

HashSet* nodes(Graph* graph) {
  HashSet *hash_set = create_hash_set(2 * graph->edge_count);

  for (uli i = 0; i < graph->edge_count; i++) {
    uli src = graph->edges[i].src;
    uli dest = graph->edges[i].dest;

    if (!contains(hash_set, src)) {
      add(hash_set, src);
    }
    if (!contains(hash_set, dest)) {
      add(hash_set, dest);
    }
  }

  //free_hash_set(hash_set);
  return hash_set;
}

uli count_nodes(Edge* edges, uli edge_count) {
  HashSet *hash_set = create_hash_set(2 * edge_count);
  int node_count = 0;

  for (uli i = 0; i < edge_count; i++) {
    uli src = edges[i].src;
    uli dest = edges[i].dest;

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

void fill_in_node_count(Graph* graph, uli*  node_count, Dictionary* id_rev, char* directed, int is_reversed) {
  //printf("node count directed %s, %d is_reversed\n",directed,is_reversed);
  uli value = 0;
  if(!is_reversed){
    for (uli i = 0; i < graph->edge_count; i++) {
      uli src = graph->edges[i].src;
      uli dest = graph->edges[i].dest;
      find(id_rev, src, &value);
      node_count[value] ++;
      if (strcmp(directed,"u") == 0)
        {
          find(id_rev, dest, &value);
          node_count[value] ++;
        }
    } 
  }
  else{
    for (uli i = 0; i < graph->edge_count; i++) {
      uli dest = graph->edges[i].src;
      uli src = graph->edges[i].dest;
      find(id_rev, src, &value);
      node_count[value] ++;
      if (strcmp(directed,"u") == 0)
        {
          find(id_rev, dest, &value);
          node_count[value] ++;
        }
    }
  }
}


void fill_in_adjacency_list(Graph* graph, Couple_adj** adj_list, uli*  node_count, Dictionary* id_rev, char* directed, int is_reversed, double** prob, uli** alias, int is_alias_read) {
  uli dsrc, ddest;
  if(!is_reversed){
    for (uli i = 0; i < graph->edge_count; i++) {
      uli src = graph->edges[i].src;
      uli dest = graph->edges[i].dest;
      find(id_rev, src, &dsrc);
      find(id_rev, dest, &ddest);
      adj_list[dsrc][node_count[dsrc]].v = ddest;
      adj_list[dsrc][node_count[dsrc]].nb = graph->edges[i].nb;
      if(is_alias_read){
        alias[dsrc][node_count[dsrc]] = graph->edges[i].alias;
        prob[dsrc][node_count[dsrc]] = graph->edges[i].prob;
      }

      node_count[dsrc] ++;
      
      if (strcmp(directed,"u") == 0)
        {
          
          adj_list[ddest][node_count[ddest]].v = dsrc;
          adj_list[ddest][node_count[ddest]].nb = graph->edges[i].nb;
          if(is_alias_read){
            alias[ddest][node_count[ddest]] = graph->edges[i].alias;
            prob[ddest][node_count[ddest]] = graph->edges[i].prob;
          }
          node_count[ddest] ++;
        }
    } 
  }
  else{
    for (uli i = 0; i < graph->edge_count; i++) {
      uli dest = graph->edges[i].src;
      uli src = graph->edges[i].dest;
      find(id_rev, src, &dsrc);
      find(id_rev, dest, &ddest);

      adj_list[dsrc][node_count[dsrc]].v = ddest;
      adj_list[dsrc][node_count[dsrc]].nb = graph->edges[i].nb;
      if(is_alias_read){
        alias[dsrc][node_count[dsrc]] = graph->edges[i].alias;
        prob[dsrc][node_count[dsrc]] = graph->edges[i].prob;
      }
      node_count[dsrc] ++;
      if (strcmp(directed,"u") == 0)
        {
          adj_list[ddest][node_count[ddest]].v = dsrc;
          adj_list[ddest][node_count[ddest]].nb = graph->edges[i].nb;
          if(is_alias_read){
            alias[ddest][node_count[ddest]] = graph->edges[i].alias;
            prob[ddest][node_count[ddest]] = graph->edges[i].prob;
          }
          node_count[ddest] ++;
        }
    }
  }
}

void fill_in_create_prob_alias(Graph* graph, Couple_adj** list_adj, uli*  node_count, double*** new_prob, uli*** alias){

  (*new_prob) = (double**) malloc(graph->nb_nodes*sizeof(double*));
  (*alias) = (uli**) malloc(graph->nb_nodes*sizeof(uli*));
  if ( !(*new_prob) || !(*alias) ) {
    return exit(-1);
  }
  for(uli i = 0;i<graph->nb_nodes;i++){
    (*new_prob)[i] = NULL;
    (*alias)[i] = NULL;
  }


  for(uli i = 0;i<graph->nb_nodes;i++){
    //printf("node_count i %lu\n",node_count[i]);
    if(node_count[i] > 0){
      (*new_prob)[i] = (double*) malloc(node_count[i]*sizeof(double));
      (*alias)[i] = (uli*) malloc(node_count[i]*sizeof(uli));
    }
  }
  for(uli i = 0;i<graph->nb_nodes;i++){
    if(node_count[i] > 0){
    double * prob = (double*) malloc(node_count[i]*sizeof(graph->nb_nodes));
    uli somme = 0;
    for(uli j=0;j<node_count[i];j++){
      (*alias)[i][j] = 0;
      (*new_prob)[i][j] = 0.0;

      somme = somme + list_adj[i][j].nb;
    }
    //printf("somme %lu from %lu\n", somme,i);
    for(uli j=0;j<node_count[i];j++){
      //printf("                 nb %lu somme %lu prob %lg\n", list_adj[i][j].nb,somme, ((double) list_adj[i][j].nb)/((double) somme));
      prob[j] = ((double) list_adj[i][j].nb)/((double) somme);
    }
    for(uli j=0;j<node_count[i];j++){
      //printf(" i : %lu  j=%lu->  : prob %lg, alias %lu, newprob %lg \n", i,j,prob[j], (*alias)[i][j], (*new_prob)[i][j]);
    }
    create_alias_tables(prob, node_count[i] , (*alias)[i], (*new_prob)[i]);
    free(prob);
    }
    else{
      (*new_prob)[i] = NULL;
      (*alias)[i] = NULL;
    }

  }
}


Graph_rep create_adjacency_list(Graph* g, char* directed, int is_reversed, int is_alias_create, int is_alias_read){

  Graph_rep A;

  // creation of ids and ids rev
  HashSet* h = nodes(g);
  uli* nodes_id = (uli*) malloc(g->nb_nodes*sizeof(uli));
  //printf("create adj nb_nodes %lu \n", g->nb_nodes);
  Dictionary* nodes_id_rev = createDictionary(g->nb_nodes);
  uli zz = 0;
  for (uli i = 0; i < h->size; ++i) {
  Node* current = h->table[i];
  while (current != NULL) {
      //printf("%lu\n", current->data);
      nodes_id[zz] = current->data;
      current = current->next;
      zz++;
    }
  }
  for(uli i = 0;i < g->nb_nodes; i++){
    insert(nodes_id_rev ,nodes_id[i],i);
  }
  A.ids = nodes_id;
  A.id_rev = nodes_id_rev;
  //end creation of ids and id rev


  // fill in node count
  Couple_adj** adj_list = NULL;
  double ** prob = NULL;
  uli ** alias = NULL;

  adj_list = (Couple_adj**) malloc(g->nb_nodes * sizeof(Couple_adj*));

  if(is_alias_read){
    prob = (double**) malloc(g->nb_nodes*sizeof(double*));
    alias = (uli**) malloc(g->nb_nodes*sizeof(uli*));
  }
  // for(uli i =0; i < g->nb_nodes;i++){
  //   adj_list[i] = NULL;
  // }

  //printf("start mallocs\n");
  uli* node_count = NULL;
  node_count = (uli*) malloc(g->nb_nodes*sizeof(uli));
  memset(node_count, 0, g-> nb_nodes*sizeof(uli));
  fill_in_node_count(g, node_count, nodes_id_rev, directed, is_reversed);
  //printf("end fill in node_count\n");
  // end fill in node count
  // for(uli j = 0; j< g->nb_nodes;j++)
  //   printf("during creation node_count[%lu] = %lu\n",j,node_count[j]);

  for(uli z = 0; z < g->nb_nodes; z++){
    if(node_count[z] > 0){
      adj_list[z] = (Couple_adj*) malloc(node_count[z] * sizeof(Couple_adj));
      if(is_alias_read){
        prob[z] = (double*) malloc(node_count[z] * sizeof(double));
        alias[z] = (uli*) malloc(node_count[z] * sizeof(uli));
      }
    }
    else{
      adj_list[z] = NULL;
      if(is_alias_read){
        prob[z] = NULL;
        alias[z] = NULL;
      } 
    }
  }

  for(uli z = 0; z < g->nb_nodes; z++){
    for(uli y = 0; y < node_count[z]; y++){
      adj_list[z][y].v = 0;
      adj_list[z][y].nb = 0;
      if(is_alias_read){
        prob[z][y] = 0.0;
        alias[z][y] = 0;
      }
    }
  }
  //printf("init phase adj list finished\n");
  memset(node_count, 0, g->nb_nodes*sizeof(uli));
  fill_in_adjacency_list(g, adj_list, node_count, nodes_id_rev, directed, is_reversed, prob, alias, is_alias_read);
  //printf("fill adj list finished\n");
  // fill in prob and alias
  if(is_alias_create){
    /* prob = (double**) malloc(g->nb_nodes * sizeof(double*)); */
    /* alias = (uli**) malloc(g->nb_nodes * sizeof(uli*)); */
    fill_in_create_prob_alias(g, adj_list, node_count, &prob, &alias);
  }


  // for(uli j = 0; j< g->nb_nodes;j++)
  // printf("end creation node_count[%lu] = %lu\n",j,node_count[j]);
  A.node_count = node_count;
  A.adj_list = adj_list;
  A.nb_nodes = g->nb_nodes;
  A.alias = alias;
  A.prob = prob;

  free_hash_set(h);
  return A;
}

void add_alias_prob_to_graph(Graph * g, Graph_rep* a){
  // print_graph_rep(a);
  // printf("nb nodes here : %lu \n", g->nb_nodes);
  uli* current_node =(uli*) malloc(g->nb_nodes*sizeof(uli));
  memset(current_node, 0, g->nb_nodes*sizeof(uli));
  for(uli i = 0;i < g->edge_count;i++){
    // flip edges of g
    uli tmp = g->edges[i].src;
    g->edges[i].src = g->edges[i].dest;
    g->edges[i].dest = tmp;
  }
  for(uli i = 0;i < g->edge_count;i++){
    // flip edges of g
    /* uli tmp = g->edges[i].src; */
    /* g->edges[i].src = g->edges[i].dest; */
    /* g->edges[i].dest = tmp; */
    uli value1;
    find(a->id_rev, g->edges[i].src, &value1);
    // printf("current src %lu\n",value1);
    g->edges[i].alias = a->alias[value1][current_node[value1]];
    g->edges[i].prob = a->prob[value1][current_node[value1]];
    current_node[value1]++;
    //find(a->id_rev, g->edge[i].dest, &value2);
  }
  free(current_node);
}

void free_graph_rep(Graph_rep* g){
  for(uli i=0; i < g->nb_nodes;i++){
    free(g->adj_list[i]);
    if(g->prob != NULL){
      free(g->prob[i]);
      free(g->alias[i]);
    }
  }
  free(g->node_count);
  free(g->ids);
  freeDictionary(g->id_rev);
  free(g->adj_list);
  free(g->prob);
  free(g->alias);
}


void print_graph_rep(Graph_rep* g){
  printf("Graph Rep*************\n nb nodes %lu\n",g->nb_nodes);
  for(uli i = 0;i < g->nb_nodes;i++){
    printf("node_count[%lu] = %lu\n",i,g->node_count[i]);
  }
  for(uli i = 0;i < g->nb_nodes;i++){
    printf("ids[%lu] = %lu\n",i,g->ids[i]);
  }
  printDictionary(g->id_rev);

  for(uli i = 0;i < g->nb_nodes;i++){
    for(uli j = 0;j < g->node_count[i];j++){
      printf("adj_list[%lu][%lu] = v %lu, r %lu \n",i,j,g->adj_list[i][j].v, g->adj_list[i][j].nb);
    }
  }
  if(g->prob != NULL){
    for(uli i = 0;i < g->nb_nodes;i++){
      for(uli j = 0;j < g->node_count[i];j++){
        printf("prob[%lu][%lu] = %lg \n",i,j,g->prob[i][j]);
      }
    }

    for(uli i = 0;i < g->nb_nodes;i++){
      for(uli j = 0;j < g->node_count[i];j++){
        printf("alias[%lu][%lu] = %lu \n",i,j,g->alias[i][j]);
      }
    }
  }
}


BFS_ret bfs(int start_node, Graph_rep* A) {
  uli visited[A->nb_nodes];
  uli* distances;
  uli* nb_paths;
  distances = malloc(A->nb_nodes* sizeof(uli));
  nb_paths = malloc(A->nb_nodes* sizeof(uli));
  for(uli i = 0;i < A->nb_nodes;i++){
    distances[i] = ULONG_MAX;
    nb_paths[i] = 0;
  }
  memset(visited, 0, sizeof visited);
  uli queue[A->nb_nodes];
  uli front = 0, rear = 0;
  BFS_ret res;


  // values for the dag
  Edge *edges = NULL;
  int edge_capacity = 10; // Initial capacity for edges array
  int edge_count = 0;
  Graph graph = {NULL, 0, 0, 0, 0};
  graph.is_nb_dag = 1;
  graph.is_alias = 0;
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
    for (uli i = 0; i < A->node_count[current_node]; i++) {
      z = A->adj_list[current_node][i];
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
        edges[edge_count].prob = 0.0;
        edges[edge_count].alias = 0;
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
  graph.nb_nodes = count_nodes(edges, edge_count);
  graph.is_nb_dag = 1;
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

Edge * optimal_bunrank_order(uli edge_count, Couple_pred* nb_paths_from_s, Graph_rep* A){
  // il est possible de faire mieux en pratique probablement en ordonnant directement les aretes entrantes d'un noeud
  qsort(nb_paths_from_s, A->nb_nodes, sizeof(Couple_pred) ,compareElem);
  /* for(uli jj=0; jj<nb_nodes;jj++) */
  /*   printf("v %lu nb %lu edge count %lu node count : %lu\n",nb_paths_from_s[jj].v, nb_paths_from_s[jj].r, graph->edge_count, node_count[jj]); */
  Edge * new_edges = (Edge *) malloc(edge_count * sizeof(Edge));
  if (new_edges == NULL)
    {
      printf("optimal_bunrank_order: malloc problem allocation\n");
      exit(-1);
    }
  uli ii = 0;
  for(uli i = 0; i < A->nb_nodes; i++){
    for(uli j = 0; j < A->node_count[nb_paths_from_s[i].v]; j++){
      new_edges[ii].src = nb_paths_from_s[i].v;
      new_edges[ii].dest = A->adj_list[nb_paths_from_s[i].v][j].v;
      new_edges[ii].nb = nb_paths_from_s[i].r;
      ii++;
    }
  }
  return new_edges;
}
//////////////////////// Alias Sampling //////////////////////////////////////////

void create_alias_tables(double* probabilities, uli n, uli* alias, double* prob) {

    uli* small = (uli*)malloc(n * sizeof(uli));
    uli* large = (uli*)malloc(n * sizeof(uli));
    
    double* scaled_prob = (double*)malloc(n * sizeof(double));
    memcpy(scaled_prob, probabilities, n * sizeof(double));
    
    // Scale the probabilities
    for (uli i = 0; i < n; ++i) {
        scaled_prob[i] *= n;
    }
    
    uli small_count = 0;
    uli large_count = 0;
    
    // Populate small and large lists
    for (uli i = 0; i < n; ++i) {
        if (scaled_prob[i] < 1.0) {
            small[small_count++] = i;
        } else {
            large[large_count++] = i;
        }
    }
    // Construct the alias and prob tables
    while (small_count > 0 && large_count > 0) {
        uli s = small[--small_count];
        uli l = large[--large_count];
        prob[s] = scaled_prob[s];
        alias[s] = l;
        scaled_prob[l] = (scaled_prob[l] + scaled_prob[s]) - 1.0;
        if (scaled_prob[l] < 1.0) {
            small[small_count++] = l;
        } else {
            large[large_count++] = l;
        }
    }
    // Remaining probabilities
    while (large_count > 0) {
        uli l = large[--large_count];
        prob[l] = 1.0;
    }
    while (small_count > 0) {
        uli s = small[--small_count];
        prob[s] = 1.0;
    }
    // Free temporary arrays
    free(small);
    free(large);
    free(scaled_prob);
}

// Function to generate a sample
uli sample(uli* alias, double* prob, uli n, gsl_rng * R) {
  uli column = gsl_rng_uniform_int(R, n);
    double p = (double)rand() / RAND_MAX;
    if (p < prob[column]) {
        return column;
    } else {
        return alias[column];
    }
}

Pred_op sample_op(uli* alias, double* prob, uli n, gsl_rng * R) {
  Pred_op x;
  x.op = 0;

  uli column = gsl_rng_uniform_int(R, n);
  x.op ++;

  double p = (double)rand() / RAND_MAX;
  x.op += 3;

  x.op ++;
  if (p < prob[column]) {
    x.v = column;
    return x;
  } else {
    x.v = alias[column];
    return x;
  }
}

///////////////////////// Unranking functions ////////////////////////////

// not used and not up to date
// uli dicho_label(uli v, Couple_adj** adj_list, uli* nb_count, uli nb_nodes, uli* nb_paths_from_s, uli r){
//   uli i = 0;
//   uli j = nb_count[v]-1;
//   // printf("dicho i %lu j %lu\n",i,j);
//   while(j - i + 1 > 1){
//     uli x = (i+j - 1)/2;
//     // printf("x %lu\n", x);
//     Couple_adj y = adj_list[v][x]; // *((adj_list+v*nb_nodes + x));
//     //uli w = x.v;
//     //uli nb = x.nb;
//     if(y.nb > r){
//       j = x;
//     }
//     else{
//       i = x + 1;
//     }
//   }
//   return j;

// }

Couple_pred find_pred_opti(uli v, Graph_rep* g, uli* nb_paths_from_s, uli r){
  //uli i = dicho_label(v, adj_list, nb_count, nb_nodes, nb_paths_from_s,r);
  // avoid function call dicho_label
  uli i = 0;
  uli j = g->node_count[v]-1;
  // printf("dicho i %lu j %lu\n",i,j);
  while(j - i + 1 > 1){
    uli x = (i+j - 1)/2;
    // printf("x %lu\n", x);
    Couple_adj y = g->adj_list[v][x]; // *((adj_list+v*nb_nodes + x));
    //uli w = x.v;
    //uli nb = x.nb;
    if(y.nb > r){
      j = x;
    }
    else{
      i = x + 1;
    }
  }
  i = j;
  // end function call replacement



  Couple_adj y = g->adj_list[v][i];  // *((adj_list+v*nb_nodes + i));
  uli rp = r;
  uli w = y.v;
  if (i > 0){
    Couple_adj z = g->adj_list[v][i-1]; // *((adj_list+v*nb_nodes + (i-1)));
    //uli wp = z.v;
    rp = r - z.nb;
  }
  Couple_pred res;
  // printf("pred found %lu, %lu \n", w, rp);
  res.v = w;
  res.r = rp;
  return res;
}

Couple_pred find_pred(uli v, Graph_rep* g, uli* nb_paths_from_s, uli r){
  //printf("find pred start : v %lu r %lu\n",v,r);
  uli i = 0;
  Couple_adj z = g->adj_list[v][0]; // *((adj_list+v*nb_nodes));
  uli w = z.v;
  //printf("--------- val %lu \n", nb_paths_from_s[w]);
  uli rp = r - nb_paths_from_s[w];
  // printf("ici just first pred %lu %lu \n", w, rp);
  while(rp < r){
    i = i + 1;
    //printf("look for pred iteration %lu : %lu %lu\n", i, w,rp);
    z = g->adj_list[v][i];  // *((adj_list+v*nb_nodes) + i);
    w = z.v;
    rp = rp - nb_paths_from_s[w];
    //printf("--------- val %lu \n", nb_paths_from_s[w]);
  }
  rp = rp + nb_paths_from_s[w];
  Couple_pred y;
  //printf("pred found %lu, %lu \n", w, rp);
  y.v = w;
  y.r = rp;
  return y;
}

List build_rank_b(Graph_rep* g, uli* nb_paths_from_s, uli s, uli t, uli rank, char* which){
  uli source_node, target_node;
  find(g->id_rev, s, &source_node);
  find(g->id_rev, t, &target_node);
  // uli source_node = g->id_rev[s];
  // uli target_node = g->id_rev[t];

  List path;
  initList(&path);

  uli r = rank;
  uli v = target_node;
  Couple_pred x;
  while(v != source_node){
    //printf("new iteration while current v %lu\n",v);
    addNode(&path, g->ids[v]);
    if (strcmp(which,"i-unrank") == 0){
      x = find_pred_opti(v, g, nb_paths_from_s, r); 
    }
    else{
      x = find_pred(v, g, nb_paths_from_s, r);
    }
    r = x.r;
    v = x.v;
    //  printf("couple result node %lu new rank %lu\n",v,r);
  }
  addNode(&path, g->ids[v]);
  return path;
}

/////////////////// End Unranking Functions ///////////////////////////////////

uli rand_pred_alias(uli v, Graph_rep* g, gsl_rng * R){
  uli i = sample(g->alias[v], g->prob[v], g->node_count[v], R);
  return g->adj_list[v][i].v;
}

Pred_op rand_pred_alias_op(uli v, Graph_rep* g, gsl_rng * R){
  Pred_op x;
  x.op = 0;

  Pred_op zz = sample_op(g->alias[v], g->prob[v], g->node_count[v], R);
  uli i = zz.v;
  x.op += zz.op;
  x.v = g->adj_list[v][i].v;
  return x;
}


uli rand_pred(uli v, Graph_rep* g, uli* nb_paths_from_s, gsl_rng * R){
  uli r = gsl_rng_uniform_int(R, nb_paths_from_s[v]);
  //printf("find pred start : v %lu r %lu\n",v,r);
  uli i = 0;
  Couple_adj z = g->adj_list[v][0]; // *((adj_list+v*nb_nodes));
  uli w = z.v;
  //printf("--------- val %lu \n", nb_paths_from_s[w]);
  uli rp = r - nb_paths_from_s[w];
  // printf("ici just first pred %lu %lu \n", w, rp);
  while(rp < r){
    i = i + 1;
    //printf("look for pred iteration %lu : %lu %lu\n", i, w,rp);
    z = g->adj_list[v][i];  // *((adj_list+v*nb_nodes) + i);
    w = z.v;
    rp = rp - nb_paths_from_s[w];
    //printf("--------- val %lu \n", nb_paths_from_s[w]);
  }
  return w;
}

Pred_op rand_pred_op(uli v, Graph_rep* g, uli* nb_paths_from_s, gsl_rng * R){
  Pred_op x;
  x.op = 0;

  uli r = gsl_rng_uniform_int(R, nb_paths_from_s[v]);
  x.op += 2;

  uli i = 0;
  Couple_adj z = g->adj_list[v][0]; // *((adj_list+v*nb_nodes));
  uli w = z.v;
  uli rp = r - nb_paths_from_s[w];
  x.op += 5;

  while(rp < r){
    i = i + 1;
    x.op += 3;

    z = g->adj_list[v][i];  // *((adj_list+v*nb_nodes) + i);
    w = z.v;
    rp = rp - nb_paths_from_s[w];
    x.op += 4;

  }
  x.v = w;
  return x;
}

uli rand_pred_opti(uli v, Graph_rep* g, uli* nb_paths_from_s, gsl_rng * R){
  //uli i = dicho_label(v, adj_list, nb_count, nb_nodes, nb_paths_from_s,r);
  // avoid function call dicho_label
  uli r = gsl_rng_uniform_int(R, nb_paths_from_s[v]);
  uli i = 0;
  uli j = g->node_count[v]-1;
  // printf("dicho i %lu j %lu\n",i,j);
  while(j - i + 1 > 1){
    uli x = (i+j - 1)/2;
    // printf("x %lu\n", x);
    Couple_adj y = g->adj_list[v][x]; // *((adj_list+v*nb_nodes + x));
    //uli w = x.v;
    //uli nb = x.nb;
    if(y.nb > r){
      j = x;
    }
    else{
      i = x + 1;
    }
  }
  i = j;
  // end function call replacement



  Couple_adj y = g->adj_list[v][i];  // *((adj_list+v*nb_nodes + i));
  uli w = y.v;
  if (i > 0){
    Couple_adj z = g->adj_list[v][i-1]; // *((adj_list+v*nb_nodes + (i-1)));
    w = z.v;
  }
  return w;
}

Pred_op rand_pred_opti_op(uli v, Graph_rep* g, uli* nb_paths_from_s, gsl_rng * R){
  Pred_op xx;
  xx.op = 0;

  uli r = gsl_rng_uniform_int(R, nb_paths_from_s[v]);
  uli i = 0;
  uli j = g->node_count[v]-1;
  xx.op = xx.op + 4;

  while(j - i + 1 > 1){
    uli x = (i+j - 1)/2;
    xx.op = xx.op + 7;

    Couple_adj y = g->adj_list[v][x]; // *((adj_list+v*nb_nodes + x));
    xx.op++;

    if(y.nb > r){
      j = x;
      xx.op = xx.op + 1;
    }
    else{
      i = x + 1;
      xx.op = xx.op + 2;
    }
    xx.op = xx.op + 1;
  }
  i = j;

  Couple_adj y = g->adj_list[v][i];  // *((adj_list+v*nb_nodes + i));
  uli w = y.v;
  xx.op ++;

  if (i > 0){
    Couple_adj z = g->adj_list[v][i-1]; // *((adj_list+v*nb_nodes + (i-1)));
    w = z.v;
    xx.op += 2;
  }
  xx.v = w;
  return xx;
}

List BRW(Graph_rep* g, uli* nb_paths_from_s, uli s, uli t, char* which, gsl_rng * R){
  uli source_node, target_node;
  find(g->id_rev, s, &source_node);
  find(g->id_rev, t, &target_node);
  // uli source_node = g->id_rev[s];
  // uli target_node = g->id_rev[t];

  List path;
  initList(&path);

  uli v = target_node;
  while(v != source_node){
    //printf("new iteration while current v %lu\n",v);
    addNode(&path, g->ids[v]);
    if (strcmp(which,"i-unrank") == 0){
      v = rand_pred_opti(v, g, nb_paths_from_s, R); 
    }
    else if (strcmp(which,"alias-unrank") == 0){
      v = rand_pred_alias(v, g, R); 
    }
    else{
      v = rand_pred(v, g, nb_paths_from_s, R);
    }
    //  printf("couple result node %lu \n",v);
  }
  addNode(&path, g->ids[v]);
  return path;
}

uli BRW_op(Graph_rep* g, uli* nb_paths_from_s, uli s, uli t, char* which, gsl_rng * R){
  uli source_node, target_node;
  Count_op res;
  res.op = 0;

  //List path;
  find(g->id_rev, s, &source_node);
  find(g->id_rev, t, &target_node);
  res.op += 2;
  // uli source_node = g->id_rev[s];
  // uli target_node = g->id_rev[t];


  //initList(&res.path);

  Pred_op x;
  x.op = 0;

  uli v = target_node;
  res.op ++;

  while(v != source_node){
    res.op ++;
    //printf("new iteration while current v %lu\n",v);
    //addNode(&path, g->ids[v]);
    res.op += 4;

    if (strcmp(which,"i-unrank") == 0){
      x = rand_pred_opti_op(v, g, nb_paths_from_s, R);
      v = x.v;
      res.op += x.op;
    }
    else if (strcmp(which,"alias-unrank") == 0){
      x = rand_pred_alias_op(v, g, R);
      v = x.v;
      res.op += x.op;
    }
    else{
      x = rand_pred_op(v, g, nb_paths_from_s, R);
      v = x.v;
      res.op += x.op;
    }
    //  printf("couple result node %lu \n",v);
  }
  //addNode(&path, g->ids[v]);
  res.op += 4;
  return res.op;
}
