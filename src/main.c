#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include "algorithms.h"

// In total the graph can not contain more than 4294967295 shortest paths since we use unsigned long int


//gsl_rng * R;  /* global generator */



char* string_from_uli(uli x){
  char tmp2[100];
  int n = snprintf(tmp2, 100, "%lu", x);
  //printf("string from %d and x %lu \n", n, x);
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



void queries(Graph* graph, uli source_node, uli target_node, uli nb_queries, char* first_part, char* which, char* time_or_operations, char* directed){
  printf("start queries\n");
  //Graph_rep origin = create_adjacency_list(graph, directed, 0, 0, 0);

  char* x;
  char nbpath_name[100];
  strcpy(nbpath_name, first_part);  // Copy str1 into result
  strcat(nbpath_name, "/nb_paths_");
  x = string_from_uli(source_node);
  strcat(nbpath_name, x);
  free(x);
  strcat(nbpath_name, ".csv");

  char dag_name[100];
  strcpy(dag_name, first_part);  // Copy str1 into result
  strcat(dag_name, "/");
  x = string_from_uli(source_node);
  strcat(dag_name, x);
  free(x);
  strcat(dag_name, ".edges");
  Graph dag;
  if(strcmp(which,"alias") == 0){
    dag = read_graph(dag_name, 0, 1);
  }
  else{
    dag = read_graph(dag_name, 1, 0);
  }


  // printf("nbpath_name : %s \n", nbpath_name);
  char line[256];
  uli* nb_paths_from_s = NULL;
  nb_paths_from_s = (uli*)malloc((graph->nb_nodes+1) * sizeof(uli));
  if (nb_paths_from_s == NULL) {
    perror("Error allocating memory");
  }
  memset(nb_paths_from_s, 0, graph->nb_nodes*sizeof(uli));

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
      //printf("nb %lu %lu,", node, nb_paths_from_s[node]);
      node += 1;
  }

  Graph_rep rdag;
  if(strcmp(which,"alias") == 0){
    rdag = create_adjacency_list(&dag, "d", 0, 0, 1);
  }
  else{
   rdag = create_adjacency_list(&dag, "d", 1, 0, 0); 
  }
  /* print_graph_rep(&rdag); */
  /* puts("fin"); */

  List* paths = (List*) malloc(nb_queries * sizeof(List));
  if(paths == NULL){
    printf("paths problem allocation \n");
    exit(-1);
  }
  for(uli iii=0; iii < nb_queries;iii++){
    paths[iii].head = NULL;
    paths[iii].tail = NULL;

  }

  /* Graph_rep org = create_adjacency_list(graph, "d", 0, 0, 0); */
  /* print_graph_rep(&org); */

  struct timeval t1, t2;
  double elapsedTime;
  uli nb_operations = 0;
  uli nb_operations_tmp = 0;
  // start timer
  gettimeofday(&t1, NULL);

  printf("******************************\n");

  if(strcmp(time_or_operations,"t") == 0){
    printf("count time not operations\n");
    for(uli que = 0; que < nb_queries; que++){
      paths[que] = BRW(&rdag, nb_paths_from_s, source_node, target_node, which);
      //unrank uncomment bellow
      /* uli rank = gsl_rng_uniform_int(R, nb_paths_from_s[target_node]); */
      /* //printf("******************** query %lu rank sampled : %lu from a total of %lu \n", que, rank, nb_paths_from_s[target_node]); */

      /* paths[que] = build_rank_b(&rdag, nb_paths_from_s, source_node, target_node, rank, which); */
      //printf("sampled path : \n");
      //reverseList(&paths[que]);
      //printList(paths[que].head);
    } 
  }
  else if(strcmp(time_or_operations,"c") == 0){
    printf("count operations not time\n");
    //printf("nb paths n %lu\n", n);
    //char tmpc2;
    //scanf("%c", &tmpc2);
    for(uli que = 0; que < nb_queries; que++){
      nb_operations_tmp = BRW_op(&rdag, nb_paths_from_s, source_node, target_node, which);
    nb_operations += nb_operations_tmp;
    }
  }
  else{
    printf("option for complexity not implemented\n");
    exit(-1);
  }
  // stop timer
  gettimeofday(&t2, NULL);
  // compute and print the elapsed time in millisec
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
  char time_res[100];
  char operation_res[100];
  strcpy(time_res, first_part);  // Copy str1 into result
  strcat(time_res, "/");
  strcat(time_res, "queries_time_");
  x = string_from_uli(nb_queries);
  strcat(time_res, x);
  free(x);
  strcpy(operation_res, first_part);  // Copy str1 into result
  strcat(operation_res, "/");
  strcat(operation_res, "queries_operations_");
  x = string_from_uli(nb_queries);
  strcat(operation_res, x);
  free(x);
  strcat(time_res, ".txt");
  strcat(operation_res, ".txt");
  printf("%f ms.\n", elapsedTime);
  printf("%lu operations.\n", nb_operations);

  char res[100];
  /* n = snprintf(NULL, 0, "%lu", nb_queries); */
  /* c = snprintf(string_queries, n+1, "%lu", nb_queries); */
  strcpy(res, first_part);  // Copy str1 into result
  strcat(res, "/");
  strcat(res, "queries_");
  x = string_from_uli(nb_queries);
  strcat(res, x);
  free(x);
  strcat(res, ".txt");
  writeResults(paths, nb_queries, res, time_res, operation_res, elapsedTime, nb_operations, time_or_operations);

  for(uli que = 0; que < nb_queries; que++){
    freeList(paths[que].head);
  }

 free_graph_rep(&rdag);

  free(paths);
  free(nb_paths_from_s);
  free(dag.edges);

  fclose(file);

  //free_graph_rep(&origin);
  printf("end queries\n");
}


void preprocessing(Graph* graph, char* directed, char* which, char* first_part){


  BFS_ret  res;
  char* x;

  Graph_rep A = create_adjacency_list(graph, directed, 0, 0, 0);
  //puts("graphe principale print repr");
  //print_graph_rep(&A);

  char dist_name[100];
  char nb_path_name[100];
  char tot_path_name[100];
  char time_name[100];
  uli total = 0;
  strcpy(tot_path_name, first_part);  // Copy str1 into result
  strcat(tot_path_name, "/");   // Add slash
  strcat(tot_path_name,"total_paths.csv");  // Concatenate str2 to result

  strcpy(time_name, first_part);  // Copy str1 into result
  strcat(time_name, "/");   // Add slash
  strcat(time_name,"pre_time");  // Concatenate str2 to result
  //strcpy(time_name, first_part);  // Copy str1 into result
  strcat(time_name,".csv");  // Concatenate str2 to result
  //printf("tot file : %s\n",tot_path_name);
  FILE* ptr3 = fopen(tot_path_name,"w");
  FILE* ptr4 = fopen(time_name,"w");
  FILE* ptr = NULL;
  FILE* ptr2 = NULL;
  
  // Perform BFS from all nodes and store dags in a directory with the filename before dot
  for(uli start_node = 0; start_node < graph->nb_nodes; start_node ++){

    struct timeval t1, t2;
    long elapsedTime;
    // start timer
    gettimeofday(&t1, NULL);

    // printf("start node : %lu \n", start_node);
    // print_graph_rep(&A);

    res = bfs(start_node, &A);
    if(start_node%1000 == 0){
      printf("cur %lu ,", start_node/1000);
    }

    //printf("START NODE %lu\n",start_node);
    //print_graph(&res.g,1);
    //puts("end pring graph");


    if(strcmp(which,"binary") == 0)
      {
        dag_to_partial_sum(&res.g, graph->nb_nodes);
      }
    if(strcmp(which,"ordered") == 0)
      {
        uli tmp;
        Graph_rep rdag = create_adjacency_list(&res.g, "d", 0, 0, 0);
        Couple_pred* ordered_array = (Couple_pred*) malloc(sizeof(Couple_pred)*rdag.nb_nodes);
        //puts("");
        for(uli ii=0;ii<rdag.nb_nodes;ii++){
          ordered_array[ii].v = rdag.ids[ii];
          tmp = rdag.ids[ii];
          //find(rdag.id_rev, rdag.ids[ii], &tmp);
          ordered_array[ii].r = res.paths[tmp];
        }
        //char tmpc;

        Edge* new_edges = optimal_bunrank_order(res.g.edge_count,  ordered_array, &rdag);
        free(res.g.edges);
        res.g.edges = new_edges;
        free(ordered_array);
        free_graph_rep(&rdag);
      }
    if(strcmp(which,"alias") == 0){
      // the last 1 is for create aliases since they do not exist yet
      //printf("create adj for alias\n");
      res.g.is_alias = 1;
      // printf("***************");
      // print_graph(&res.g,1);
      Graph_rep rdag = create_adjacency_list(&res.g, "d", 1, 1, 0);
      //printf("end create adj for alias\n");
      //print_graph_rep(&rdag);
      //then we need to modify &res.g to add aliases and prob to it, and then in queries we call create_adj with the last param at 0
      // printf("//////*************///////");
      // print_graph(&res.g,1);
      add_alias_prob_to_graph(&res.g, &rdag);
      free_graph_rep(&rdag);
      // printf("////////////////////");
      // print_graph(&res.g,1);

    }

    // stop timer
    gettimeofday(&t2, NULL);
    // compute and print the elapsed time in millisec
    elapsedTime = (t2.tv_sec-t1.tv_sec)*1000000 + t2.tv_usec-t1.tv_usec;

    //elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    //printf("elapsedTime %ld\n", elapsedTime);
    //elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms


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
    for(uli i = 0; i < graph->nb_nodes; i++){
      x = string_from_uli(res.dist[i]);
      fprintf(ptr, "%s\n", x);
      free(x);
      
    }

    // Print nb_paths
    uli total_nb_paths = 0;
    for(uli i = 0; i < graph->nb_nodes; i++){
      total_nb_paths += res.paths[i];
      
      x = string_from_uli(res.paths[i]);
      fprintf(ptr2, "%s\n", x);
      free(x);
      
    }

    total += total_nb_paths;
    // Print total paths from start_node
    
    x = string_from_uli(total_nb_paths);
    fprintf(ptr3, "%s\n", x);
    free(x);

    fprintf(ptr4, "%ld\n", elapsedTime);




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
    //print_nb_paths(res.paths, res.g.nb_nodes);
    //char tmpc;
    //print_graph(&res.g,1);
    //scanf("%c", &tmpc);
    if(strcmp(which,"alias") == 0){
      write_graph(result, &res.g,1);
    }
    else{
      write_graph(result, &res.g, 0);
    }

    fclose(ptr);
    fclose(ptr2);

    free(res.g.edges);
    free(res.dist);
    free(res.paths);
    //printf("end loop\n");
  }

  // Print total paths
 
  x = string_from_uli(total);
  fprintf(ptr3, "%s ", x);
  free(x);
  fclose(ptr3);
  fclose(ptr4);

  printf("end writing after loop bfs\n");
  // If want to start from specific node
  //uli start_node = strtoul(argv[2],&endPtr,10);
  free_graph_rep(&A);

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

  if(! (strcmp(which, "linear") == 0 || strcmp(which, "ordered") == 0 || strcmp(which, "binary") == 0 || strcmp(which, "alias") == 0) ){
    printf("problem with algorithm for preprocessing not implemented \n");
    return EXIT_FAILURE;
  }

  graph = read_graph(argv[1], 0, 0);
  if (graph.edges == NULL) {
    return EXIT_FAILURE;
  }

  char *first_part = NULL;
  first_part = get_first_part(argv[1], argv[3]);

  // Print the graph
  //print_graph(&graph, 0);


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
  /* const gsl_rng_type * T; */
  /* gsl_rng_env_setup(); */

  struct timeval tv; // Seed generation based on time
  gettimeofday(&tv,0);
  //unsigned long mySeed = tv.tv_sec + tv.tv_usec;
  //unsigned long mySeed2 = 0;
  //T = gsl_rng_default; // Generator setup
  //R = gsl_rng_alloc (gsl_rng_taus2);
  //gsl_rng_set(R, mySeed2);

  srand((unsigned) time(NULL));



  uli source_node = strtoul(argv[4],NULL,10);
  uli target_node = strtoul(argv[5],NULL,10);
  uli nb_que = strtoul(argv[6],NULL,10);

  char* time_or_operations = argv[7];

  queries(&graph, source_node, target_node, nb_que, first_part, which, time_or_operations, directed);

  // Free allocated memory
  free(graph.edges);
  //gsl_rng_free(R);

  free(first_part);

  return 0;
}
