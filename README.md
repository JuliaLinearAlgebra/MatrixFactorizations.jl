# MatrixFactorizations.jl

[![Build Status](https://travis-ci.org/JuliaMatrices/MatrixFactorizations.jl.svg?branch=master)](https://travis-ci.org/JuliaMatrices/MatrixFactorizations.jl) 
[![Build status](https://ci.appveyor.com/api/projects/status/xc5mk54w38u891n7?svg=true)](https://ci.appveyor.com/project/dlfivefifty/matrixfactorizations-jl)

[![codecov](https://codecov.io/gh/JuliaMatrices/MatrixFactorizations.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaMatrices/MatrixFactorizations.jl)

A Julia package to contain non-standard matrix factorizations. At the moment it 
implements the QL factorization, though in the future may include other factorizations such 
as the UL, LQ, and RQ factorizations.

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
