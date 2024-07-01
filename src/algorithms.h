// Struct to represent an edge
#include <stdbool.h>
#include <gsl/gsl_rng.h>

typedef unsigned long int uli;
typedef double wei;

typedef struct Edge{
  uli src;
  uli dest;
  uli nb;
  uli alias;
  double prob;
} Edge;

typedef struct Graph{
  Edge *edges;
  uli edge_count;
  uli nb_nodes;
  int is_nb_dag;
  int is_alias;
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
  double prob;
  uli alias;
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
  uli** alias;
  double** prob;
  Couple_adj** adj_list;
  uli nb_nodes;
} Graph_rep;

typedef struct Count_op{
  List path;
  uli op;
} Count_op;

typedef struct Pred_op{
  uli v;
  uli op;
} Pred_op;


Graph read_graph(const char *filename, int is_weighted, int is_alias);
uli count_nodes(Edge* edges, uli nb);
Graph_rep create_adjacency_list(Graph* g, char* directed, int is_reversed, int is_alias_create, int is_alias_read);
void free_graph_rep(Graph_rep* g);
void print_graph_rep(Graph_rep* g);
BFS_ret bfs(int start_node, Graph_rep* g);
void print_graph(Graph* graph, int full_info);
void write_graph(const char *filename, Graph* graph, int is_alias);
void dag_to_partial_sum(Graph *g, uli nb_nodes);
Edge* optimal_bunrank_order(uli edge_count, Couple_pred* nb_paths_from_s, Graph_rep* A);
void addNode(List* li, uli data);
void printList(Node* head);
void freeList(Node* head);
void reverseList(List* head);
void writeResults(List* head, uli nb, const char* filename, const char* timename, const char* operationname, double time, uli nb_operations, char* time_or_operations);
void initList(List* list);
List BRW(Graph_rep* g, uli* nb_paths_from_s, uli s, uli t, char* which, gsl_rng * R);
uli BRW_op(Graph_rep* g, uli* nb_paths_from_s, uli s, uli t, char* which, gsl_rng * R);

bool find(Dictionary* dict, uli key, uli* value);
void add_alias_prob_to_graph(Graph * g, Graph_rep* a);
void create_alias_tables(double* probabilities, uli n, uli* alias, double* prob);
