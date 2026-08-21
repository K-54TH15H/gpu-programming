/*
 * This contains the CUDA Implementation which is the spritual successor
 * to the serial-host.c implementation to utilize gpu's computing capability
 * to increase the throughput for faster results
 */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INF 1000000000
__device__ int d_changed; // used for counting changed in each kernel call

/*
 * Enumeration for types of error
 */
enum Error {
  SUCCESS,
  CLI_ERROR,
  FILE_NOT_FOUND,
  GRAPH_ERROR,
  READ_ERROR,
  CUDA_ERROR
};

/*
 * Enumeration for type of object
 */
enum Mode { HOST, STAGED, DEVICE };

/*
 * Type of element stored in the CSR format of the graph
 */
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
  enum Mode mode;         // mode in which the graph is stored
};

/*
 * Initialize the graph data structure
 */
static void initializeGraph(struct Graph *graph) {
  graph->size = 0;
  graph->edges = 0;
  graph->edge_list = NULL;
  graph->dist = NULL;
  graph->degree = NULL;
  graph->csr = NULL;
  graph->offset = NULL;
  graph->mode = HOST;
}

/*
 * Pretty printing error messages
 */
static void printError(enum Error err, const char *error_msg) {
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
  case CUDA_ERROR:
    printf("[CUDA-ERROR]: ");
    break;
  default:
    printf("[UNKNOWN-ERROR] ");
    break;
  }

  printf("%s\n", error_msg);
}

/*
 * Helper function to clean up allocated resources on both host and device
 */
static void cleanup(struct Graph *h_graph, struct Graph *s_graph,
                    struct Graph *d_graph) {
  // host memory
  if (h_graph != NULL) {
    free(h_graph->edge_list);
    free(h_graph->dist);
    free(h_graph->csr);
    free(h_graph->offset);
    free(h_graph->degree);
  }

  // free only staged fields
  if (s_graph != NULL) {
    cudaFree(s_graph->dist);
    cudaFree(s_graph->csr);
    cudaFree(s_graph->offset);
  }
  // free the struct allocated on device
  if (d_graph != NULL)
    cudaFree(d_graph);
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
__global__ static void HSSSP(struct Graph *graph) {
  int vertex = (blockIdx.x * blockDim.x) + threadIdx.x;

  // bounds checking
  if (vertex >= graph->size)
    return;

  // if vertex is not reached yet, then terminate expansion
  if (graph->dist[vertex] == INF)
    return;

  int start = graph->offset[vertex];
  int end =
      (vertex == graph->size - 1) ? graph->edges : graph->offset[vertex + 1];

  for (int i = start; i < end; i++) {
    struct Data data = graph->csr[i];
    int candid = graph->dist[vertex] + data.wgt;

    // store the minimum distance
    int old = atomicMin(&graph->dist[data.dest], candid);

    if (candid < old)
      atomicAdd(&d_changed, 1);
  }
}

/*
 * Stages a host graph with required device fields
 */
static enum Error stageGraph(struct Graph *s_graph, struct Graph *h_graph) {
  s_graph->size = h_graph->size;   // required for device
  s_graph->edges = h_graph->edges; // required for devic
  s_graph->edge_list = NULL;       // not required for device
  s_graph->degree = NULL;          // not required for device
  s_graph->mode = STAGED;          // set mode to staged

  // allocate on device
  if (cudaMalloc(&s_graph->csr, sizeof(struct Data) * h_graph->edges) !=
      cudaSuccess)
    return CUDA_ERROR;
  if (cudaMalloc(&s_graph->dist, sizeof(int) * h_graph->size) != cudaSuccess)
    return CUDA_ERROR;
  if (cudaMalloc(&s_graph->offset, sizeof(int) * h_graph->size) != cudaSuccess)
    return CUDA_ERROR;

  // copy to device from host
  if (cudaMemcpy(s_graph->csr, h_graph->csr,
                 sizeof(struct Data) * h_graph->edges,
                 cudaMemcpyHostToDevice) != cudaSuccess)
    return CUDA_ERROR;
  if (cudaMemcpy(s_graph->dist, h_graph->dist, sizeof(int) * h_graph->size,
                 cudaMemcpyHostToDevice) != cudaSuccess)
    return CUDA_ERROR;
  if (cudaMemcpy(s_graph->offset, h_graph->offset, sizeof(int) * h_graph->size,
                 cudaMemcpyHostToDevice) != cudaSuccess)
    return CUDA_ERROR;

  return SUCCESS;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printError(CLI_ERROR, "Usage: parallel-device <filename> <source-vertex>");
    return CLI_ERROR;
  }

  cudaError_t err;

  // source vertex
  int src = atoi(argv[2]);

  printf("Reading from %s...\n", argv[1]);

  FILE *graph_file = fopen(argv[1], "r");
  // if file is not found
  if (graph_file == NULL) {
    printError(FILE_NOT_FOUND, "Please select a valid file from disk");
    return FILE_NOT_FOUND;
  }

  struct Graph h_graph;  // host graph
  struct Graph s_graph;  // staged host graph
  struct Graph *d_graph; // device graph

  initializeGraph(&h_graph);
  initializeGraph(&s_graph);

  // if reading failed
  if (readGraph(&h_graph, graph_file) == READ_ERROR) {
    printError(READ_ERROR, "Given file doesn't match the syntax, please format "
                           "the graph as mentioned in graphs/README.md");
    cleanup(&h_graph, NULL, NULL);
    return READ_ERROR;
  }

  if (src < 0 || src >= h_graph.size) {
    printError(CLI_ERROR, "Source vertex out of range for given graph");
    cleanup(&h_graph, NULL, NULL);
    return CLI_ERROR;
  }

  // set distance from src to src as zero as it's trivial
  h_graph.dist[src] = 0;

  // stage the graph for device memcpy
  if (stageGraph(&s_graph, &h_graph) != SUCCESS) {
    err = cudaGetLastError();
    printError(CUDA_ERROR, cudaGetErrorString(err));
    cleanup(&h_graph, &s_graph, NULL);
    return CUDA_ERROR;
  }

  // allocate device graph
  if (cudaMalloc(&d_graph, sizeof(struct Graph)) != cudaSuccess) {
    err = cudaGetLastError();
    printError(CUDA_ERROR, cudaGetErrorString(err));
    cleanup(&h_graph, &s_graph, NULL);
    return CUDA_ERROR;
  }

  s_graph.mode = DEVICE; // set mode to device before copying

  if (cudaMemcpy(d_graph, &s_graph, sizeof(struct Graph),
                 cudaMemcpyHostToDevice) != cudaSuccess) {

    s_graph.mode = STAGED; // revert back to staged mode

    err = cudaGetLastError();
    printError(CUDA_ERROR, cudaGetErrorString(err));
    cleanup(&h_graph, &s_graph, d_graph);
    return CUDA_ERROR;
  }

  s_graph.mode = STAGED; // revert back to staged mode

  printf(
      "Parallely Computing Single Source Shortest Path on device (SSSP) from "
      "vertex [%d]\n",
      src);

  int num_threads = 64; // 2 warps per block
  int num_blocks = ((h_graph.size + num_threads - 1) /
                    num_threads); // ceil of the no of blocks required
  int h_changed;                  // flag for early exit

  // calculate the distance by SSSP algorithm
  for (int i = 0; i < (h_graph.size - 1); i++) {
    // initialize counter to 0
    const int zero = 0;
    err = cudaMemcpyToSymbol(d_changed, &zero, sizeof(int), 0,
                             cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
      printError(CUDA_ERROR, cudaGetErrorString(err));
      cleanup(&h_graph, &s_graph, d_graph);
      return CUDA_ERROR;
    }

    HSSSP<<<num_blocks, num_threads>>>(d_graph);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printError(CUDA_ERROR, cudaGetErrorString(err));
      cleanup(&h_graph, &s_graph, d_graph);
      return CUDA_ERROR;
    }

    // results of i+1-th iteration depends upon i-th iteration
    err = cudaDeviceSynchronize();
    // synchronization failure
    if (err != cudaSuccess) {
      printError(CUDA_ERROR, cudaGetErrorString(err));
      cleanup(&h_graph, &s_graph, d_graph);
      return CUDA_ERROR;
    }

    err = cudaMemcpyFromSymbol(&h_changed, d_changed, sizeof(int), 0,
                               cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
      printError(CUDA_ERROR, cudaGetErrorString(err));
      cleanup(&h_graph, &s_graph, d_graph);
      return CUDA_ERROR;
    }

    // early exit
    if (h_changed == 0)
      break;
  }

  // copy results from device to host
  if (cudaMemcpy(h_graph.dist, s_graph.dist, sizeof(int) * h_graph.size,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    err = cudaGetLastError();
    printError(CUDA_ERROR, cudaGetErrorString(err));
    cleanup(&h_graph, &s_graph, d_graph);

    return CUDA_ERROR;
  }

  // print result
  for (int i = 0; i < h_graph.size; i++) {
    if (h_graph.dist[i] != INF)
      printf("Destination: %d | Distance: %d\n", i, h_graph.dist[i]);
    else
      printf("Destination: %d | Distance: %s\n", i,
             "INFINITY [There exists no path]");
  }

  // free up graph structures
  cleanup(&h_graph, &s_graph, d_graph);

  // close the file
  fclose(graph_file);

  return SUCCESS;
}
