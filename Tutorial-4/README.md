# TENSOR CORE - MATRIX MULTIPLICATION
This implements the matrix-multiplication between two square matrices of dimensions - (64x64)

We will leverage the performance benefit provided by the tensor cores, 
this is implemented by a warp of 32 threads for each tile of the matrix.
 
The result produced by the computation kernel is verified by host generated solution for each cell.

> [!NOTE]
> A tolerance of `1e-1` is introduced as floating point arithmetic in the device architecture
> is different from host architecture. 
