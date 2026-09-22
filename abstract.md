----
# Reformulating High-Order Finite Elements for Modern Accelerators

Michał Wichrowski, Heidelberg University

## Multigrid, All the Way Down: Patch Smoothers

Vertex-patch smoothers make geometric multigrid robust with respect to the polynomial degree, but local patch solves
often dominate the cost. Classical fast diagonalization relies on separability, restricting it to Cartesian patches
and simple coefficients. I replace it with a nested, matrix-free p-multigrid method on each patch. One local V-cycle
costs $\mathcal O(p^{d+1})$ operations, the same as a single operator application, and requires only $\mathcal O(p^d)$
memory. Because no separability is required, the same smoother applies to distorted meshes, variable coefficients and
the Stokes problem. All computations remain patch-local and cache-resident, making data locality the key to efficiency.

## GPU-First Finite Elements

GPUs reward a different representation: storing each field persistently cell-wise, without ever forming an assembled
global vector, makes memory access coalesced by construction. A primal–dual pairing identity allows the Krylov iteration
to operate directly on unassembled data while producing exactly the same iterates as the assembled method. Direct
stiffness summation then needs neither index lists nor atomics, and hanging-node constraints are absorbed into the
multigrid transfers. In a Triton implementation, I compare sum factorization, fast diagonalization, dense element
matrices and even–odd symmetry splits, including mappings of the resulting tensor contractions to GPU tensor cores. The
fastest kernels reach about 85% of the device's peak memory bandwidth, and the Laplace operator uses up to 51% of the
A100's tensor-core compute capacity, outperforming libCEED BP1 by factors of 1.6-4.4. For the biharmonic problem, the complete multigrid solve runs about
ten times faster than a recent GPU tensor-product multigrid solver on the same device.

## Going general: unfitted methods

Finally, I turn to complex geometries. CutFEM retains a Cartesian background mesh, and hence its
tensor-product structure, while the physical boundary may cut the cells arbitrarily. The ghost penalty can then be
factorized into one-dimensional matrices and evaluated in $\mathcal O(p^{d+1})$ operations. For finite-strain elasticity,
a single augmented energy functional, combined with automatic differentiation, generates both residuals and tangents.
For vertex-patch multigrid, two-level convergence bounds independent of the mesh size and of how the geometry
cuts the mesh are provable. I will close with numerical evidence on p-robustness of the method.
