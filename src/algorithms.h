// Struct to represent an edge

// Struct to represent an edge

typedef unsigned long int uli;

typedef struct {
  uli src;
  uli dest;
} Edge;

typedef struct {
  Edge *edges;
  uli edge_count;
} Graph;

typedef struct Node {
  uli key;
  struct Node *next;
} Node;

typedef struct {
  Node **table;
  uli size;
} HashSet;

typedef struct BFS_ret {
  Graph g;
  uli* dist;
} BFS_ret;


Graph read_graph(const char *filename);
uli count_nodes(Graph graph);
void create_adjacency_list(Graph graph, uli* adj_list, uli*  node_count, uli nb_nodes);
BFS_ret bfs(int start_node, uli* adj_list, uli*  node_count, uli nb_nodes);
void print_graph(Graph graph);
void write_graph(const char *filename, Graph graph);
