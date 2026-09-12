# Tutorial - 3
Implemented BFS algorithm for graphs on both host and device in CSR format. Graphs can be read from hard-disk to host-ram and stored in CSR format.

## Files
`serial-bfs.c` - Implemented a serialized version of BFS algorithm for host


`parallel-bfs.c` - Implemented a parallelized version of BFS algorithm for device. 

## Usage
Compile the files with respective tools and run the executable

```bash
./parallel-bfs <graph-file> <source-vertex>
```

```bash
./serial-bfs <graph-file> <source-vertex>
```

## Algorithm
>[!NOTE]
> The algorithm finds the distance based on the level and not based on the weights of edges
> level is defined as the no of edges taken to reach the destination vertex from the source vertex given.

## Graphs
>[!WARNING]
>The algorithm must only be called on graphs with no negative cycles.

Sample graphs are stored in `graphs/`. Syntax for writing a `.gph` file is given by
`graphs/README.md`
