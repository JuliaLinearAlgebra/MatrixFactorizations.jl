# MatrixFactorizations.jl

[![CI](https://github.com/JuliaLinearAlgebra/MatrixFactorizations.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaLinearAlgebra/MatrixFactorizations.jl/actions/workflows/ci.yml)

[![codecov](https://codecov.io/gh/JuliaLinearAlgebra/MatrixFactorizations.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaLinearAlgebra/MatrixFactorizations.jl)

A Julia package to contain non-standard matrix factorizations. At the moment it
implements the QL, RQ, and UL factorizations, a combined Cholesky factorization with inverse,
and polar decompositions.
In the future it may include other factorizations such
as the LQ factorization.

## QL Factorization

The QL Factorization  is analogous to the QR factorization but with a lower-triangular matrix:
```julia
julia> using MatrixFactorizations

julia> A = randn(5,5);

julia> ql(A)
MatrixFactorizations.QL{Float64,Array{Float64,2}}
Q factor:
5×5 MatrixFactorizations.QLPackedQ{Float64,Array{Float64,2}}:
  0.155574  -0.112555   -0.686362   -0.701064   0.0233276
  0.363037  -0.0583277  -0.0329428   0.152682   0.916736
 -0.670742   0.294355   -0.547867    0.347157   0.206843
 -0.61094   -0.566041    0.324432   -0.353128   0.276396
  0.144425  -0.759527   -0.349868    0.489877  -0.199681
L factor:
5×5 Array{Float64,2}:
 -1.75734     0.0         0.0        0.0       0.0   
 -0.58336    -2.08515     0.0        0.0       0.0   
  0.0213518   0.0722477  -2.01325    0.0       0.0   
  0.930364   -0.46796    -0.322493  -2.9091    0.0   
  1.08725     0.746217    0.549688  -1.10194  -2.0581

julia> b = randn(5); ql(A) \ b ≈ A \ b
true
```

## Jordan canonical form

Every square matrix has a Jordan canonical form. Although Jordan blocks are unstable
with respect to perturbations, the Jordan canonical form of a rational matrix with
rational eigenvalues can be found exactly:
```julia

julia> A = [0 0 1 7 -1; -5 -6 -6 -35 5; 1 1 -7 7 -1; 0 0 0 -9 0; 2 1 -5 -42 -3];

julia> λ = [-9, -7, -7, -1, -1//1];

julia> V, J = jordan(A, λ)
Jordan{Rational{Int64}, Matrix{Rational{Int64}}, Matrix{Rational{Int64}}}
Generalized eigenvectors:
5×5 Matrix{Rational{Int64}}:
 0  0   0  1   0
 0  1   0  0  -1
 0  1  -1  0   0
 1  0   0  0   0
 7  1  -1  1  -1
Jordan normal form:
5×5 Matrix{Rational{Int64}}:
 -9   0   0   0   0
  0  -7   1   0   0
  0   0  -7   0   0
  0   0   0  -1   1
  0   0   0   0  -1

julia> A == V*J/V
true
```
