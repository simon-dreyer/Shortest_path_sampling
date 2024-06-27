#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <gsl/gsl_rng.h>
#include "algorithms.h"

// In total the graph can not contain more than 4294967295 shortest paths since we use unsigned long int


gsl_rng * R;  /* global generator */

char* string_from_uli(uli x){
  uli tmp = 0;
  int n = snprintf(NULL, 0, "%lu", tmp);
  char* buf =(char*) malloc((n+1)*sizeof(char));
  memset(buf, '\0', (n+1)*sizeof(char));
  n = snprintf(NULL, 0, "%lu", x);
  snprintf(buf, n+1, "%lu", x);
  return buf;
}

char* get_first_part(const char *str, const char * which) {
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
  printf("Graph name : %s\n", token);
  char* result = NULL;
  result = (char*) malloc(100*sizeof(char));
  result[0]='\0';
  const char dash[] = "_";
  strcat(result, token);
  strcat(result, dash);
  strcat(result, which);
  // Return the first part of the string
  free(str_copy);
  return result;
}

uli dicho_label(uli v, Couple_adj** adj_list, uli* nb_count, uli nb_nodes, uli* nb_paths_from_s, uli r){
  uli i = 0;
  uli j = nb_count[v]-1;
  // printf("dicho i %lu j %lu\n",i,j);
  while(j - i + 1 > 1){
    uli x = (i+j - 1)/2;
    // printf("x %lu\n", x);
    Couple_adj y = adj_list[v][x]; // *((adj_list+v*nb_nodes + x));
    //uli w = x.v;
    //uli nb = x.nb;
    if(y.nb > r){
      j = x;
    }
    else{
      i = x + 1;
    }
  }
  return j;

}

Couple_pred find_pred_opti(uli v, Couple_adj** adj_list, uli* nb_count, uli nb_nodes, uli* nb_paths_from_s, uli r){
  //uli i = dicho_label(v, adj_list, nb_count, nb_nodes, nb_paths_from_s,r);
  // avoid function call dicho_label
  uli i = 0;
  uli j = nb_count[v]-1;
  // printf("dicho i %lu j %lu\n",i,j);
  while(j - i + 1 > 1){
    uli x = (i+j - 1)/2;
    // printf("x %lu\n", x);
    Couple_adj y = adj_list[v][x]; // *((adj_list+v*nb_nodes + x));
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







  Couple_adj y = adj_list[v][i];  // *((adj_list+v*nb_nodes + i));
  uli rp = r;
  uli w = y.v;
  if (i > 0){
    Couple_adj z = adj_list[v][i-1]; // *((adj_list+v*nb_nodes + (i-1)));
    //uli wp = z.v;
    rp = r - z.nb;
  }
  Couple_pred res;
  // printf("pred found %lu, %lu \n", w, rp);
  res.v = w;
  res.r = rp;
  return res;
}

Couple_pred find_pred(uli v, Couple_adj** adj_list, uli nb_nodes, uli* nb_paths_from_s, uli r){
  //printf("find pred start : v %lu r %lu\n",v,r);
  uli i = 0;
  Couple_adj z = adj_list[v][0]; // *((adj_list+v*nb_nodes));
  uli w = z.v;
  //printf("--------- val %lu \n", nb_paths_from_s[w]);
  uli rp = r - nb_paths_from_s[w];
  // printf("ici just first pred %lu %lu \n", w, rp);
  while(rp < r){
    i = i + 1;
    //printf("look for pred iteration %lu : %lu %lu\n", i, w,rp);
    z = adj_list[v][i];  // *((adj_list+v*nb_nodes) + i);
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

List build_rank_b(Couple_adj** adj_list, uli* node_count, uli nb_nodes, uli* nb_paths_from_s, uli source_node, uli target_node, uli rank, char* which){
  List path;
  initList(&path);

  uli r = rank;
  uli v = target_node;
  Couple_pred x;
  while(v != source_node){
    //printf("new iteration while current v %lu\n",v);
    addNode(&path, v);
    if (strcmp(which,"i-unrank") == 0){
      x = find_pred_opti(v, adj_list, node_count, nb_nodes, nb_paths_from_s, r); 
    }
    else{
      x = find_pred(v, adj_list, nb_nodes, nb_paths_from_s, r);
    }
    r = x.r;
    v = x.v;
    //  printf("couple result node %lu new rank %lu\n",v,r);
  }
  addNode(&path, v);
  return path;
}

void queries(Graph* graph, uli source_node, uli target_node, uli nb_queries, char* first_part, char* which){
  printf("start queries\n");
  //uli tmp;
  //int n = snprintf(NULL, 0, "%lu", tmp);
  //char string_node[n+1];
  //char string_queries[n+1];
  /* memset(string_node, '\0', sizeof(string_node)); */
  /* memset(string_queries, '\0', sizeof(string_queries)); */
  /* n = snprintf(NULL, 0, "%lu", source_node); */
  /* int c = snprintf(string_node, n+1, "%lu", source_node); */
  char nbpath_name[100];
  strcpy(nbpath_name, first_part);  // Copy str1 into result
  strcat(nbpath_name, "/nb_paths_");
  strcat(nbpath_name, string_from_uli(source_node));
  strcat(nbpath_name, ".csv");

  char dag_name[100];
  strcpy(dag_name, first_part);  // Copy str1 into result
  strcat(dag_name, "/");
  strcat(dag_name, string_from_uli(source_node));
  strcat(dag_name, ".edges");

  // printf("nbpath_name : %s \n", nbpath_name);
  char line[256];
  uli nb_nodes = count_nodes(graph);
  uli* nb_paths_from_s = (uli*)malloc((nb_nodes+1) * sizeof(uli));
  if (nb_paths_from_s == NULL) {
    perror("Error allocating memory");
  }

  FILE *file = fopen(nbpath_name, "r");
  if (file == NULL) {
    perror("Error opening file");
    free(nb_paths_from_s);
    exit(-1);
  }

  // Read nb paths from s to all others
  uli node = 0;
  while (fgets(line, sizeof(line), file)) {
      if(sscanf(line, "%lu", &nb_paths_from_s[node]) != 1){
        printf("Problem reading values of nb_paths \n");
        exit(-1);
      }
      //printf("nb paths from %lu\n", nb_paths_from_s[node]);
      node += 1;
  }
  // printf("finished reading vals, val target %lu \n", nb_paths_from_s[target_node]);

  Graph dag = read_graph(dag_name, 1);
  //printf("DAG is:\n");
  //print_graph(&dag,1);
  uli nb_nodes_dag = count_nodes(&dag);
  // printf("nb nodes dag: %lu \n", nb_nodes_dag);
  Couple_adj** adj_list;
  adj_list = (Couple_adj**) malloc(nb_nodes_dag * sizeof(Couple_adj*));
  for(uli i = 0; i < nb_nodes_dag; i++){
    adj_list[i] = (Couple_adj*) malloc(nb_nodes_dag * sizeof(Couple_adj));
  }

  for(uli i = 0; i < nb_nodes_dag; i++){
    for(uli j = 0; j < nb_nodes_dag; j++){
      adj_list[i][j].v = 0;
      adj_list[i][j].nb = 0;
    }
  }
  uli* node_count;
  node_count = (uli*) malloc(nb_nodes_dag*sizeof(uli));
  memset(node_count, 0, nb_nodes_dag*sizeof(uli));
  create_adjacency_list(&dag, adj_list, node_count, nb_nodes_dag, "d", 1);

  /* for(uli i = 0; i < nb_nodes_dag; i++){ */
  /*   for(uli j = 0; j < node_count[i]; j++){ */
  /*     printf("in DAG %lu -> %lu\n",i,adj_list[i][j].v); */
  /*   } */
  /* } */
  // Sample Rank

  List* paths = (List*) malloc(nb_queries * sizeof(List));
  struct timeval t1, t2;
  double elapsedTime;
  // start timer
  gettimeofday(&t1, NULL);

  for(uli que = 0; que < nb_queries; que++){
    uli rank = gsl_rng_uniform_int(R, nb_paths_from_s[target_node]);
    //printf("******************** query %lu rank sampled : %lu from a total of %lu \n", que, rank, nb_paths_from_s[target_node]);

    paths[que] = build_rank_b(adj_list, node_count, nb_nodes_dag, nb_paths_from_s, source_node, target_node, rank, which);
    //printf("sampled path : \n");
    //reverseList(&paths[que]);
    //printList(paths[que].head);
  }
  // stop timer
  gettimeofday(&t2, NULL);
  // compute and print the elapsed time in millisec
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
  printf("%f ms.\n", elapsedTime);

  char res[100];
  /* n = snprintf(NULL, 0, "%lu", nb_queries); */
  /* c = snprintf(string_queries, n+1, "%lu", nb_queries); */
  strcpy(res, first_part);  // Copy str1 into result
  strcat(res, "/");
  strcat(res, "queries_");
  strcat(res, string_from_uli(nb_queries));
  strcat(res, ".txt");
  writeListToFile(paths, nb_queries, res);

  for(uli que = 0; que < nb_queries; que++){
    freeList(paths[que].head);
  }

  for(uli i = 0; i < nb_nodes; i++){
    free(adj_list[i]);
  }
  free(adj_list);
  //printf("end preprocessing bef \n");
  free(node_count);

  printf("end queries\n");
}


void preprocessing(Graph* graph, char* directed, char* which, char* first_part){

  struct timeval t1, t2;
  double elapsedTime;
  // start timer
  gettimeofday(&t1, NULL);

  BFS_ret  res;
  uli nb_nodes = count_nodes(graph);
  char* x;
  //Couple_adj adj_list[nb_nodes][nb_nodes];
  Couple_adj** adj_list;
  adj_list = (Couple_adj**) malloc(nb_nodes * sizeof(Couple_adj*));
  for(uli i = 0; i < nb_nodes; i++){
    adj_list[i] = (Couple_adj*) malloc(nb_nodes * sizeof(Couple_adj));
  }

  for(uli i = 0; i < nb_nodes; i++){
    for(uli j = 0; j < nb_nodes; j++){
      adj_list[i][j].v = 0;
      adj_list[i][j].nb = 0;
    }
  }
  uli* node_count;
  node_count = (uli*) malloc(nb_nodes*sizeof(uli));
  memset(node_count, 0, nb_nodes*sizeof(node_count));
  create_adjacency_list(graph, adj_list, node_count, nb_nodes, directed, 0);

  char dist_name[100];
  char nb_path_name[100];
  char tot_path_name[100];
  char time_name[100];
  /* int n = snprintf(NULL, 0, "%lu", tmp); */
  /* char buf[n+1]; */
  /* char string_node[n+1]; */
  uli total = 0;
  strcpy(tot_path_name, first_part);  // Copy str1 into result
  strcat(tot_path_name, "/");   // Add slash
  strcat(tot_path_name,"total_paths.csv");  // Concatenate str2 to result

  strcpy(time_name, first_part);  // Copy str1 into result
  strcat(time_name, "/");   // Add slash
  strcat(time_name,"pre_time.csv");  // Concatenate str2 to result
  //printf("tot file : %s\n",tot_path_name);
  FILE* ptr3 = fopen(tot_path_name,"w");
  FILE* ptr4 = fopen(time_name,"w");
  FILE* ptr = NULL;
  FILE* ptr2 = NULL;

  // Perform BFS from all nodes and store dags in a directory with the filename before dot
  for(uli start_node = 0; start_node < nb_nodes; start_node ++){
    //printf("%lu ", start_node);
    res = bfs(start_node, adj_list, node_count, nb_nodes);
    if(start_node%1000 == 0){
      printf("cur %lu ,", start_node/1000);
    }


    if(strcmp(which,"i-unrank") == 0)
      {
        dag_to_partial_sum(&res.g, nb_nodes);
      }
    if(strcmp(which,"ob-unrank") == 0)
      {
        //adj list for dag
        uli nb_nodes_dag = count_nodes(&res.g);
        Couple_adj** adj_list_dag;
        adj_list_dag = (Couple_adj**) malloc(nb_nodes_dag * sizeof(Couple_adj*));
        for(uli z = 0; z < nb_nodes_dag; z++){
          adj_list_dag[z] = (Couple_adj*) malloc(nb_nodes_dag * sizeof(Couple_adj));
        }

        for(uli z = 0; z < nb_nodes_dag; z++){
          for(uli y = 0; y < nb_nodes_dag; y++){
            adj_list_dag[z][y].v = 0;
            adj_list_dag[z][y].nb = 0;
          }
        }
        uli* node_count_dag;
        node_count_dag = (uli*) malloc(nb_nodes_dag*sizeof(uli));
        memset(node_count_dag, 0, nb_nodes_dag*sizeof(node_count_dag));
        //printf("directed : %s\n", directed);
        create_adjacency_list(&res.g, adj_list_dag, node_count_dag, nb_nodes_dag, "d", 0);
        //end adj list for dag

        Couple_pred* ordered_array = (Couple_pred*) malloc(sizeof(Couple_pred)*nb_nodes_dag);
        for(uli ii=0;ii<nb_nodes_dag;ii++){
          ordered_array[ii].v = ii;
          ordered_array[ii].r = res.paths[ii];
        }

        Edge* new_edges = optimal_bunrank_order(&res.g,  ordered_array, nb_nodes, adj_list_dag, node_count_dag);
        free(res.g.edges);
        res.g.edges = new_edges;
        free(ordered_array);
        free(node_count_dag);
        free(adj_list_dag);
      }
    // Print the dag
    //print_graph(&res.g,0);

    /* memset(string_node, '\0', sizeof(string_node)); */
    /* n = snprintf(NULL, 0, "%lu", start_node); */
    /* int c = snprintf(string_node, n+1, "%lu", start_node); */

    // stop timer
    gettimeofday(&t2, NULL);
    // compute and print the elapsed time in millisec
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms


    strcpy(dist_name, first_part);  // Copy str1 into result
    strcat(dist_name, "/");   // Add slash
    strcat(dist_name,"distances_");  // Concatenate str2 to result
    x = string_from_uli(start_node);
    strcat(dist_name,x);
    free(x);
    strcat(dist_name,".csv");
    //printf("dist file: %s\n",dist_name);
    ptr = fopen(dist_name,"w");

    strcpy(nb_path_name, first_part);  // Copy str1 into result
    strcat(nb_path_name, "/");   // Add slash
    strcat(nb_path_name,"nb_paths_");  // Concatenate str2 to result
    x = string_from_uli(start_node);
    strcat(nb_path_name,x);  // Concatenate str2 to result
    free(x);
    strcat(nb_path_name,".csv");  // Concatenate str2 to result
    //printf("nb paths file: %s\n",nb_path_name);
    ptr2 = fopen(nb_path_name,"w");

    // Print distances
    for(uli i = 0; i < nb_nodes; i++){
      //printf("dist %lu -> %lu = %lu\n", start_node, i, res.dist[i]);
      /* memset(buf, '\0', sizeof(buf)); */
      /* n = snprintf(NULL, 0, "%lu", res.dist[i]); */
      /* int c = snprintf(buf, n+1, "%lu", res.dist[i]); */
      x = string_from_uli(res.dist[i]);
      fprintf(ptr, "%s\n", x);
      free(x);
      /* if (i == nb_nodes - 1){ */
      /*   fprintf(ptr, "%s\n", buf); */
      /* } */
      /* else{ */
      /*   fprintf(ptr, "%s ", buf); */
      /* } */
    }

    // Print nb_paths
    uli total_nb_paths = 0;
    for(uli i = 0; i < nb_nodes; i++){
      total_nb_paths += res.paths[i];
      //printf(" nb_paths [%lu] = %lu\n", i, res.paths[i]);

      /* memset(buf, '\0', sizeof(buf)); */
      /* n = snprintf(NULL, 0, "%lu", res.paths[i]); */
      /* int c = snprintf(buf, n+1, "%lu", res.paths[i]); */
      x = string_from_uli(res.paths[i]);
      fprintf(ptr2, "%s\n", x);
      free(x);
      /* if (i == nb_nodes - 1){ */
      /*   fprintf(ptr2, "%s\n", buf); */
      /* } */
      /* else{ */
      /*   fprintf(ptr2, "%s ", buf); */
      /* } */
    }

    total += total_nb_paths;
    // Print total paths from start_node
    /* memset(buf, '\0', sizeof(buf)); */
    /* n = snprintf(NULL, 0, "%lu", total_nb_paths); */
    /* c = snprintf(buf, n+1, "%lu", total_nb_paths); */
    x = string_from_uli(total_nb_paths);
    fprintf(ptr3, "%s\n", x);
    free(x);

    fprintf(ptr4, "%f\n", elapsedTime);




    // string of node id in buf
    /* memset(buf, '\0', sizeof(buf)); */
    /* n = snprintf(NULL, 0, "%lu", start_node); */
    /* c = snprintf(buf, n+1, "%lu", start_node); */



    char result[100];
    strcpy(result, first_part);  // Copy str1 into result
    strcat(result, "/");   // Add a space to result
    x = string_from_uli(start_node);
    strcat(result,x);  // Concatenate str2 to result
    free(x);
    strcat(result,".edges");  // Concatenate str3 to result
    //printf("edge file: %s\n",result);
    write_graph(result, &res.g);

    fclose(ptr);
    fclose(ptr2);

    free(res.g.edges);
    free(res.dist);
    free(res.paths);
    //printf("end loop\n");
  }

  // Print total paths
  /* memset(buf, '\0', sizeof(buf)); */
  /* n = snprintf(NULL, 0, "%lu", total); */
  /* int c = snprintf(buf, n+1, "%lu", total); */
  x = string_from_uli(total);
  fprintf(ptr3, "%s ", x);
  free(x);
  fclose(ptr3);

  printf("end writing after loop bfs\n");
  // If want to start from specific node
  //uli start_node = strtoul(argv[2],&endPtr,10);
  printf("nb nodes when free %lu\n", nb_nodes);
  for(uli i = 0; i < nb_nodes; i++){
    free(adj_list[i]);
  }
  free(adj_list);
  printf("end preprocessing bef \n");
  free(node_count);

  printf("end preprocessing\n");
}





int main(int argc, char *argv[]) {

  /* List list; */
  /* initList(&list); */
  /* for(int i = 0;i< 10000;i++) */
  /*   addNode(&list, 100); */

  // char filename[100];
  Graph  graph;

  // Read the graph from the file, file names in argv[1]
  char* directed = argv[2];
  char* which = argv[3];
  //char* source = argv[4];
  //char* target = argv[5];

  if(! (strcmp(which, "b-unrank") == 0 || strcmp(which, "ob-unrank") == 0 || strcmp(which, "i-unrank") == 0) ){
    printf("problem with algorithm for preprocessing not implemented \n");
    return EXIT_FAILURE;
  }

  graph = read_graph(argv[1], 0);
  if (graph.edges == NULL) {
    return EXIT_FAILURE;
  }

  char *first_part = NULL;
  first_part = get_first_part(argv[1], argv[3]);

  // Print the graph
  print_graph(&graph, 0);


  // PREPROCESSING PHASE
  struct stat st = {0};
  // create the directory if does not exist
  if (stat(first_part, &st) == -1) {
    printf("preprocessing.....\n");
    mkdir(first_part, 0700);
    preprocessing(&graph, directed, which, first_part);
    printf("preprocessing finished fine \n");
  }
  else{
    printf("preprocessing for this type is already present\n");
  }

  // GENERATION PHASE
  const gsl_rng_type * T;
  gsl_rng_env_setup();

  struct timeval tv; // Seed generation based on time
  gettimeofday(&tv,0);
  //unsigned long mySeed = tv.tv_sec + tv.tv_usec;
  unsigned long mySeed2 = 0;
  T = gsl_rng_default; // Generator setup
  R = gsl_rng_alloc (T);
  gsl_rng_set(R, mySeed2);


  uli source_node = strtoul(argv[4],NULL,10);
  uli target_node = strtoul(argv[5],NULL,10);
  uli nb_que = strtoul(argv[6],NULL,10);

  queries(&graph, source_node, target_node, nb_que, first_part, which);
  

  // Free allocated memory
  free(graph.edges);
  gsl_rng_free(R);

  free(first_part);

  return 0;
}
