# Matrix Multiplication with profiling

All the development took place in bhavani-hpc cluster for this profiling.
Short summary on the profiling results obtained from `stats-summary.txt`

`submit.sh` is the script submitted to run this analysis.

>[!NOTE]
> Compilation was done manually, the profiling raw data files are uploaded for reference.
> The summary stats are generated and written to `profiling-results.txt`.

>[!WARNING]
> Before diving into the summary, we will keep in mind the cost to initializing the matrices
> takes up the most as the program randomly generates the matrices with `curandStates`.
> This initialization cost is around 85% of the total cost spent for the program, which is a price
> we are willing to pay to compare between the naive and tiled matrix multiplication on any two general matrices
> where each cell contains integers in the range [0, 50]

## Naive Matrix Multiplication
This algorithm computes each element by accessing the device global memory.
This naive matrix-multiplication takes up 16,95,746 ns for computation

## Tiled Matrix Multiplication
This algorithm loads necessary tiles and performs necessary operations for each element.
The tiled matrix multiplication takes up 12,10,274 ns for computation

## Observation made by profiling
This implementation change speeds up our algorithm by x1.40.
Speedup is most likely to increase on increase in `MATRIX_DIM`
as cache misses occur frequently with naive implementation which
is avoided in the implementation of tiling as it loads into shared memory.
