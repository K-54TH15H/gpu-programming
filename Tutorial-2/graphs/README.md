# Graphs
Graphs are stored in `.gph` files in the given samples which is followed as just a convention.
The graph parser assumes the given graph is a directed graph. If you would like to use a undirected graph
then input both the forward and backward edge of same weight to simulate a undirected graph.

## Files
`sample-graph-0.gph` - Contains a simple directed graph following the syntax specified below 

## Syntax
File-name: `<graph-file-name>.gph`
```
<no-of-vertices>
<no-of-edges>
<end-point-1-1> <end-point-1-2> <weight-1>
<end-point-2-1> <end-point-2-2> <weight-2>
...
...
...
...
<end-point-n-1> <end-point-n-2> <weight-n>
