#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "algorithms.h"

char* get_first_part(const char *str) {
  // Make a copy of the input string since strtok modifies the original string
  char *str_copy = strdup(str);
  if (str_copy == NULL) {
    perror("Memory allocation failed");
    return NULL;
  }

  // Use strtok to split the string by dot character
  char *token = strtok(str_copy, ".");
  if (token == NULL) {
    // No dot found in the string
    return NULL;
  }

  // Return the first part of the string
  return token;
}


int main(int argc, char *argv[]) {
  char filename[100];
  Graph  graph;
  BFS_ret  res;

  // Read the graph from the file, file names in argv[1]
  graph = read_graph(argv[1]);
  if (graph.edges == NULL) {
    return EXIT_FAILURE;
  }

  // Print the graph
  print_graph(graph);

  uli nb_nodes = count_nodes(graph);
  uli adj_list[nb_nodes][nb_nodes];
  memset(adj_list, 0, sizeof adj_list);
  uli node_count[nb_nodes];
  memset(node_count, 0, sizeof node_count);
  create_adjacency_list(graph, (uli *)adj_list, node_count, nb_nodes);

  // Perform BFS from all nodes and store dags in a directory with the filename before dot
  char *first_part = get_first_part(argv[1]);
  struct stat st = {0};

  // create the directory if does not exist
  if (stat(first_part, &st) == -1) {
    mkdir(first_part, 0700);
  }
  char dist_name[100];
  strcpy(dist_name, first_part);  // Copy str1 into result
  strcat(dist_name, "/");   // Add a space to result
  strcat(dist_name,"distances.csv");  // Concatenate str2 to result
  FILE* ptr = fopen(dist_name,"w");
  uli tmp = 0;
  int n = snprintf(NULL, 0, "%lu", tmp);
  char buf[n+1];
  char * endPtr;
  for(uli start_node = 0; start_node < nb_nodes; start_node ++){
    res = bfs(start_node, (uli *)adj_list, node_count, nb_nodes);

    // Print the dag
    print_graph(res.g);

    // Print distances
    for(uli i = 0; i < nb_nodes; i++){
      printf("dist %lu -> %lu = %lu\n", start_node, i, res.dist[i]);
      memset(buf, '\0', sizeof(buf));
      n = snprintf(NULL, 0, "%lu", res.dist[i]);
      int c = snprintf(buf, n+1, "%lu", res.dist[i]);
      if (i == nb_nodes - 1){
        fprintf(ptr, "%s\n", buf);
      }
      else{
        fprintf(ptr, "%s ", buf);
      }
    }
    // string of node id in buf
    memset(buf, '\0', sizeof(buf));
    n = snprintf(NULL, 0, "%lu", start_node);
    int c = snprintf(buf, n+1, "%lu", start_node);



    char result[100];
    strcpy(result, first_part);  // Copy str1 into result
    strcat(result, "/");   // Add a space to result
    strcat(result,buf);  // Concatenate str2 to result
    strcat(result,".edges");  // Concatenate str3 to result
    write_graph(result, res.g);

  }
  // If want to start from specific node
  //uli start_node = strtoul(argv[2],&endPtr,10);

  // Free allocated memory
  free(graph.edges);

  return 0;
}
