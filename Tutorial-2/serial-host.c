/*
 * This code contains the logic and idea behind the SSSP algorithm with
 * host-only code. This is written to help and model the gpu implementation of
 * the algorithm
 */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INF 1e9

/*
 * Enumeration for types of error
 */
enum Error {
  SUCCESS,
  CLI_ERROR,
  FILE_NOT_FOUND,
  GRAPH_ERROR,
  READ_ERROR,
};

struct Data {
  int dest; // destination vertex
  int wgt;  // weight of the edge
};

// Structure denoting edge between vertices x and y
struct Edge {
  int x; // vertex x
  int y; // vertex y
  int c; // cost
};

/*
 * Graph is stored in CSR Format
 */
struct Graph {
  int size;               // no of vertices
  int edges;              // no of edges
  struct Edge *edge_list; // edge list representation
  int *dist;              // distance array
  int *degree;            // degree of vertices
  struct Data *csr;       // csr representation of graph
  int *offset;            // offset for csr representation
};

static void initializeGraph(struct Graph *graph) {
  graph->size = 0;
  graph->edges = 0;
  graph->edge_list = NULL;
  graph->dist = NULL;
  graph->degree = NULL;
  graph->csr = NULL;
  graph->offset = NULL;
}

static void deleteGraph(struct Graph *graph) {
  if (graph->edge_list != NULL)
    free(graph->edge_list);
  if (graph->dist != NULL)
    free(graph->dist);
  if (graph->degree != NULL)
    free(graph->degree);
  if (graph->csr != NULL)
    free(graph->csr);
  if (graph->offset != NULL)
    free(graph->offset);
}

/*
 * Pretty printing error messages
 */
static void printError(enum Error err, char *error_msg) {
  switch (err) {
  case CLI_ERROR:
    printf("[CLI-ERROR] ");
    break;
  case FILE_NOT_FOUND:
    printf("[FILE-NOT-FOUND] ");
    break;
  case READ_ERROR:
    printf("[READ-ERROR] ");
    break;
  case GRAPH_ERROR:
    printf("[GRAPH-ERROR] ");
    break;
  default:
    printf("[UNKNOWN-ERROR] ");
    break;
  }

  printf("%s\n", error_msg);
}

/*
 * Read the graph and store it in CSR Format
 */
static enum Error readGraph(struct Graph *graph, FILE *graph_file) {
  // parse no of vertex
  if (fscanf(graph_file, "%d", &graph->size) != 1) {
    printError(READ_ERROR, "Couldn't parse size of the graph (no of vertices)");
    return READ_ERROR;
  }

  // parse no of edges
  if (fscanf(graph_file, "%d", &(graph->edges)) != 1) {
    printError(READ_ERROR, "Couldn't parse size of the graph (no of edges)");
    return READ_ERROR;
  }

  // edge list for the graph
  graph->edge_list = (struct Edge *)malloc(sizeof(struct Edge) * graph->edges);

  for (int i = 0; i < graph->edges; i++) {
    struct Edge edge;

    if (fscanf(graph_file, "%d %d %d", &(edge.x), &(edge.y), &(edge.c)) != 3) {
      printError(READ_ERROR, "Couldn't parse edge");
      return READ_ERROR;
    }

    if (edge.x >= graph->size || edge.y >= graph->size || edge.x < 0 ||
        edge.y < 0) {
      printError(READ_ERROR, "Edge vertices are out of range");
      return READ_ERROR;
    }

    // store the edge on the edge list
    graph->edge_list[i] = edge;
  }

  // validate vertices and edges count
  if (graph->size <= 0 || graph->edges <= 0) {
    printError(GRAPH_ERROR, "Empty/Invalid graph");
    return READ_ERROR;
  }

  // directed graph
  graph->csr = (struct Data *)malloc(sizeof(struct Data) * graph->edges);
  graph->offset = (int *)malloc(sizeof(int) * graph->size);
  graph->degree = (int *)calloc(graph->size, sizeof(int));
  graph->dist = (int *)malloc(sizeof(int) * graph->size);

  // count out degree of each vertex
  for (int i = 0; i < graph->edges; i++)
    graph->degree[graph->edge_list[i].x]++;

  int acc = 0; // accumulated offset
  // calculate offset for CSR representation
  for (int i = 0; i < graph->size; i++) {
    graph->offset[i] = acc;
    acc += graph->degree[i];
  }

  // current offsets to write data on graph structure
  int *current = (int *)malloc(sizeof(int) * graph->size);
  memcpy(current, graph->offset, sizeof(int) * graph->size);

  // write destination and weight of edge in CSR format
  for (int i = 0; i < graph->edges; i++) {
    int src = graph->edge_list[i].x;
    int dest = graph->edge_list[i].y;
    int wgt = graph->edge_list[i].c;

    struct Data fdata = {dest, wgt}; // forward edge

    graph->csr[current[src]++] = fdata;
  }

  // set computed distance to be INFINITY macro (denoting no path) for all
  // vertices
  for (int i = 0; i < graph->size; i++)
    graph->dist[i] = INF;

  // free up temporary memory
  free(current);

  return SUCCESS;
}

/*
 * Helper Function to find the SSSP from source vertex on graph
 * which is called in a loop to find the distance in an iterative manner
 */
static int HSSSP(struct Graph *graph, int vertex) {
  // if vertex is not reached yet by source, then terminate expansion
  if (graph->dist[vertex] == INF)
    return 0;

  int changed = 0; // counter for the no of changes
  int start = graph->offset[vertex];
  int end =
      (vertex == graph->size - 1) ? graph->edges : graph->offset[vertex + 1];

  for (int i = start; i < end; i++) {
    struct Data data = graph->csr[i];
    int candid = graph->dist[vertex] + data.wgt;

    if (graph->dist[data.dest] > candid) {
      graph->dist[data.dest] = candid;
      changed++;
    }
  }

  return changed;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printError(CLI_ERROR, "Usage: serial-host <filename> <source-vertex>");
    return CLI_ERROR;
  }

  // source vertex
  int src = atoi(argv[2]);

  printf("Reading from %s...\n", argv[1]);

  FILE *graph_file = fopen(argv[1], "r");
  // if file is not found
  if (graph_file == NULL) {
    printError(FILE_NOT_FOUND, "Please select a valid file from disk");
    return FILE_NOT_FOUND;
  }

  struct Graph graph;
  initializeGraph(&graph);

  // if reading failed
  if (readGraph(&graph, graph_file) == READ_ERROR) {
    printError(READ_ERROR, "Given file doesn't match the syntax, please format "
                           "the graph as mentioned in graphs/README.md");
    deleteGraph(&graph);
    return READ_ERROR;
  }

  if (src < 0 || src >= graph.size) {
    printError(CLI_ERROR, "Source vertex out of range for given graph");
    deleteGraph(&graph);
    return CLI_ERROR;
  }

  printf("Serially Computing Single Source Shortest Path on host (SSSP) from "
         "vertex [%d]\n",
         src);

  // set distance from src to src as zero as
  graph.dist[src] = 0;

  // calculate the distance by SSSP algorithm
  for (int i = 0; i < (graph.size - 1); i++) {
    int changed = 0;
    for (int v = 0; v < graph.size; v++)
      changed += HSSSP(&graph, v);
    // early break out of loop
    if (changed == 0)
      break;
  }

  // print result
  for (int i = 0; i < graph.size; i++) {
    if (graph.dist[i] != INF)
      printf("Destination: %d | Distance: %d\n", i, graph.dist[i]);
    else
      printf("Destination: %d | Distance: %s\n", i,
             "INFINITY [There exists no path]");
  }

  // free up graph structures
  deleteGraph(&graph);

  // close the file
  fclose(graph_file);

  return SUCCESS;
}
