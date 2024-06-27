// Struct to represent an edge

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


Graph read_graph(const char *filename, int is_weighted);
uli count_nodes(Graph* graph);
void create_adjacency_list(Graph* graph, Couple_adj** adj_list, uli*  node_count, uli nb_nodes, char* directed, int is_reversed);
BFS_ret bfs(int start_node, Couple_adj** adj_list, uli*  node_count, uli nb_nodes);
void print_graph(Graph* graph, int full_info);
void write_graph(const char *filename, Graph* graph);
void dag_to_partial_sum(Graph *g, uli nb_nodes);
Edge* optimal_bunrank_order(Graph* graph, Couple_pred* nb_paths_from_s, uli nb_nodes, Couple_adj** adj_list, uli*  node_count);
void addNode(List* li, uli data);
void printList(Node* head);
void freeList(Node* head);
void reverseList(List* head);
void writeListToFile(List* head, uli nb_elements, const char* filename);
void initList(List* list);
