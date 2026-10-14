# Matrix Multiplication with profiling

All the development took place in bhavani-hpc cluster for this profiling.
Short summary on the profiling results obtained from `stats-summary.txt`

>[!WARNING]
> Before diving into the summary, we will keep in mind the cost to initializing the matrices
> takes up the most as the program randomly generates the matrices with `curandStates`.
> This initialization cost is 99.9% of the total cost spent for the program, which is a price
> we are willing to pay to compare between the naive and tiled matrix multiplication on any two general matrices

## Naive Matrix Multiplication - TILE 32
This algorithm computes each element by accessing the device global memory.
This naive matrix-multiplication takes up 10,52,130 ns for computation

## Tiled Matrix Multiplication - TILE 32
This algorithm loads necessary tiles and performs necessary operations for each element.
The tiled matrix multiplication takes up 8,38,210 ns for computation

## Observation made by profiling
This implementation change speeds up our algorithm by x1.25.
Speedup is most likely to increase on increase in `MATRIX_DIM`
as cache misses occur frequently with naive implementation which
is avoided in the implementation of tiling as it loads into shared memory.
