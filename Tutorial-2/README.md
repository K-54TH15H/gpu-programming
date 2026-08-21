# Tutorial - 2
Implemented SSSP algorithm for graphs on both host and device in CSR format. Graphs can be read from hard-disk to host-ram and stored in CSR format.

## Files
`serial-host.c` - Implemented a serialized version of SSSP algorithm for host


`parallel-device.c` - Implemented a parallelized version of SSSP algorithm for device. 

## Usage
Compile the files with respective tools and run the executable

```bash
./parallel-device <graph-file> <source-vertex>
```

```bash
./serial-host <graph-file> <source-vertex>
```

## Graphs
>[!WARNING]
>The algorithm must only be called on graphs with no negative cycles.

Sample graphs are stored in `graphs/`. Syntax for writing a `.gph` file is given by
`graphs/README.md`
