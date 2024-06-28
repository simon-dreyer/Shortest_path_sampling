// Struct to represent an edge
#include <stdbool.h>

typedef unsigned long int uli;
typedef double wei;

typedef struct Edge{
  uli src;
  uli dest;
  wei weight;
  uli nb;
} Edge;

typedef struct Graph{
  Edge *edges;
  uli edge_count;
  uli nb_nodes;
  int is_weighted;
  int is_nb_dag;
} Graph;


typedef struct Node {
  uli data;
  struct Node* next;
} Node;

// Define the list structure
typedef struct Lis{
  Node* head;
  Node* tail;
} List;

typedef struct HashSet{
  Node **table;
  uli size;
} HashSet;

typedef struct BFS_ret {
  Graph g;
  uli* dist;
  uli* paths;
} BFS_ret;

typedef struct Couple_pred {
  uli v;
  uli r;
} Couple_pred;

typedef struct Couple_adj {
  uli v;
  uli nb;
} Couple_adj;

typedef struct Node_dic {
    uli key;
    uli value;
    struct Node_dic* next;
} Node_dic;

typedef struct Dictionary {
    Node_dic **table;
    uli size;
} Dictionary;

typedef struct Graph_rep{
  uli* ids;
  Dictionary* id_rev;
  uli* node_count;
  Couple_adj** adj_list;
  uli nb_nodes;
} Graph_rep;


Graph read_graph(const char *filename, int is_weighted);
uli count_nodes(Edge* edges, uli nb);
Graph_rep create_adjacency_list(Graph* g, char* directed, int is_reversed);
void free_graph_rep(Graph_rep* g);
void print_graph_rep(Graph_rep* g);
BFS_ret bfs(int start_node, Graph_rep* g);
void print_graph(Graph* graph, int full_info);
void write_graph(const char *filename, Graph* graph);
void dag_to_partial_sum(Graph *g, uli nb_nodes);
Edge* optimal_bunrank_order(uli edge_count, Couple_pred* nb_paths_from_s, Graph_rep* A);
void addNode(List* li, uli data);
void printList(Node* head);
void freeList(Node* head);
void reverseList(List* head);
void writeListToFile(List* head, uli nb_elements, const char* filename);
void initList(List* list);

bool find(Dictionary* dict, uli key, uli* value);