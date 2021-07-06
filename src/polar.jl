# This file is based on the PolarFact package by weijianzhang
# License is MIT

export polar, PolarDecomposition

"""
    polar(A; algorithm::Symbol, maxiter, tol, verbose)

Compute the polar decomposition of the matrix `A`.

Returns a [`PolarDecomposition`](@ref).

Arguments:

`algorithm` (default: `:newton`) may be one of:
- ``:newton``: scaled Newton's method
- ``:qdwh``: the QR-based Dynamically weighted Halley (QDWH) method
- ``:halley``: Halley's method
- ``:schulz``: the Newton Schulz method
- ``:hybrid``: a hybrid Newton method
- ``:svd``: the SVD method

`maxiter` defaults to 100; `tol` defaults to `∛eps(t)`; `verbose` defaults to `false`.
"""
function polar end


"""
    PolarDecomposition <: Factorization

Matrix factorization type of the polar decomposition of a matrix ``A``.
This is the return type of [`polar(_)`](@ref), the corresponding matrix factorization function.

Every ``m-by-n`` matrix ``A`` has a polar decomposition

```math
A = U H
```

where the ``m-by-n`` matrix ``U`` has orthonormal columns if ``m>n``
or orthonormal rows if ``m<n`` and the ``n-by-n`` matrix ``H`` is
Hermitian positive semidefinite. For a square matrix ``A``, ``H`` is
unique. If in addition, ``A`` is nonsingular, then ``H`` is positive
definite and ``U`` is unique.

Iterating the decomposition produces the components ``U`` and ``H``.

"""
struct PolarDecomposition{T,M<:AbstractArray{T},Mh<:AbstractArray{T}} <: Factorization{T}
    U::M
    H::Mh
    niters::Integer
    converged::Bool

    function PolarDecomposition{T,M,Mh}(U, H,
                    niter::Integer,
                    converged::Bool) where {T,M<:AbstractArray{T},Mh<:AbstractArray{T}}
        require_one_based_indexing(U, H)
        size(U, 2) == size(H, 1) ||
               throw(DimensionMismatch("Inner dimension of U and H mismatch."))
        new{T,M,Mh}(U, H, niter, converged)
    end
end
function PolarDecomposition{T}(U::AbstractArray{T}, H::AbstractArray{T}, niter=1, converged=true) where {T}
    PolarDecomposition{T,typeof(U),typeof(H)}(U,H,niter,converged)
end

Base.iterate(P::PolarDecomposition) = (P.U, Val(:U))
Base.iterate(P::PolarDecomposition, ::Val{:U}) = (P.H, Val(:done))
Base.iterate(P::PolarDecomposition, ::Val{:done}) = nothing

module Polar
using LinearAlgebra

using ..MatrixFactorizations: PolarDecomposition

using Printf
# common types or functions


# the objective type

struct Objective{T}
    absolute::T  # absolute error
    relative::T  # relative error

    function Objective(absolute::T, relative::T) where {T <: Real}
        absolute >= 0 || error("absolute must be non-negative.")
        relative >= 0 || error("relative must be non-negative.")

        new{T}(absolute, relative)
    end
end


function evaluate_objv(preU::Matrix{T}, U::Matrix{T}) where {T}
    rel = opnorm(preU - U, Inf) / opnorm(preU, Inf)
    abs = opnorm(I - U'*U, Inf)
    return Objective(rel, abs)
end


abstract type PolarUpdater end

abstract type PolarAlg end

# common algorithm skeleton for iterative updating methods

function common_iter!(updater::PolarUpdater,
                         X::Matrix{T},
                         U::Matrix{T},
                         H::Matrix{T},
                         maxiter::Int,
                         verbose::Bool,
                         tol) where {T}

    preU = Array{T}(undef, size(X))
    copyto!(U, X)
    converged = false
    t = 0
    if verbose
        @printf("%-5s    %-13s    %-13s\n", "Iter.", "Rel. err.",  "Obj.")
    end

    while !converged && t < maxiter
        t += 1
        copyto!(preU, U)
        update_U!(updater, U)

        # determine convergence
        diff = norm(preU - U)
        if diff < tol
            converged = true
        end

        # display infomation
        if verbose
            objv = evaluate_objv(preU, U)
            @printf("%5d    %13.6e    %13.6e\n",
                    t, objv.absolute, objv.relative)
        end
    end

    # compute H
    mul!(H, U', X)
    H = (1/T(2)) * (H + H')
    return PolarDecomposition{T}(U, H, t, converged)
end

# Scaling iterative algorithm
function common_iter_scal!(updater::PolarUpdater,
                              X::Matrix{T},
                              U::Matrix{T},
                              H::Matrix{T},
                              maxiter::Int,
                              verbose::Bool,
                              tol) where {T}

    preU = Array{T}(undef, size(X))
    copyto!(U, X)
    converged = false
    t = 0
    if verbose
        @printf("%-5s    %-13s    %-13s\n", "Iter.", "Rel. err.",  "Obj.")
    end

    while !converged && t < maxiter
        t += 1
        copyto!(preU, U)
        update_U!(updater, U)

        # determine convergence
        diff = norm(preU - U)
        if diff < tol
            converged = true
        end

        # determine scaling
        reldiff = diff/norm(U) # relative error
        if updater.scale && (reldiff < updater.scale_tol)
            updater.scale = false
        end

        # display infomation
        if verbose
            objv = evaluate_objv(preU, U)
            @printf("%5d    %13.6e    %13.6e\n",
                    t, objv.absolute, objv.relative)
        end
    end

    # compute H
    mul!(H, U', X)
    H = (1/T(2)) * (H + H')
    return PolarDecomposition{T}(U, H, t, converged)
end



# Hybrid iteration algorithm
function common_iter_hybr!(updater1::PolarUpdater,
                              updater2::PolarUpdater,
                              X::Matrix{T},
                              U::Matrix{T},
                              H::Matrix{T},
                              maxiter::Int,
                              verbose::Bool,
                              tol,
                              theta) where {T} # theta is the switch parameter

    preU = Array{T}(undef, size(X))
    copyto!(U, X)
    converged = false
    switched = false
    t = 0
    if verbose
        @printf("%-5s    %-13s    %-13s\n", "Iter.", "Rel. err.",  "Obj.")
    end

    while !converged && t < maxiter
        t += 1
        copyto!(preU, U)

        if switched
            update_U!(updater2, U)
        else
            obj = opnorm(I - U'*U, 1)
            if obj > theta # theta is the switch parameter
                update_U!(updater1, U)
            else
                switched = true
                update_U!(updater2, U)
            end
        end

        # determine convergence
        diff = norm(preU - U)
        if diff < tol
            converged = true
        end

        # display infomation
        if verbose
            objv = evaluate_objv(preU, U)
            @printf("%5d    %13.6e    %13.6e\n",
                    t, objv.absolute, objv.relative)
        end
    end

    # compute H
    mul!(H, U', X)
    H = (1/T(2)) * (H + H')
    return PolarDecomposition{T}(U, H, t, converged)
end

"""
    Polar.NewtonAlg{T} <: Polar.PolarAlg

Newton's method for polar decomposition

Reference:
Nicholas J. Higham, Computing the Polar Decomposition ---with Applications,
SIAM J. Sci. Statist. Comput. Vol. 7, Num 4 (1986) pp. 1160-1174.
"""
mutable struct NewtonAlg{T,RT<:Real} <: PolarAlg
    maxiter::Int        # maximum number of iterations.
    scale::Bool         # whether to scale Newton iteration.
    verbose::Bool       # whether to show procedural information
    tol::RT        # tolerance for convergence
    scale_tol::RT  # tolerance for acceleration scaling

    function NewtonAlg{T}( ;maxiter::Integer=100,
                       verbose::Bool=false,
                       scale::Bool=true,
                       tol::RT=cbrt(eps(real(T))),
                       scale_tol::Real=eps(real(T))^(1/4)) where {T,RT<:Real}
        maxiter > 1 || error("maxiter must be greater than 1.")
        tol > 0 || error("tol must be positive.")
        scale_tol > 0 || error("scale_tol must be positive.")

        new{T,RT}(maxiter,
            scale,
            verbose,
            tol,
            RT(scale_tol))
    end
end


function solve!(alg::NewtonAlg{T},
                   X::Matrix{T}, U::Matrix{T}, H::Matrix{T}) where {T}
    common_iter_scal!(NewtonUpdater(alg.scale, alg.scale_tol), X, U, H, alg.maxiter, alg.verbose, alg.tol)
end

mutable struct NewtonUpdater{T} <: PolarUpdater
    scale::Bool
    scale_tol::T
end

function update_U!(upd::NewtonUpdater, U::Matrix{T}) where {T}
    scale = upd.scale
    Uinv = Array{T}(undef, size(U))
    Uinvt = Array{T}(undef, size(U))
    copyto!(Uinv, inv(U))

    # 1, Inf-norm scaling
    if scale
        g = (opnorm(Uinv,1) * opnorm(Uinv, Inf) / (opnorm(U,1) * opnorm(U, Inf)) )^(1/4)
    else
        g = one(T)
    end
    adjoint!(Uinvt, Uinv)
    copyto!(U, (g * U + Uinvt /g) / convert(T, 2))

end



"""
     Polar.NewtonSchulzAlg{T} <: Polar.PolarAlg

 Newton-Schulz algorithm for polar decomposition

 This method can only apply to matrix ``A`` such that ``norm(A) < sqrt(3)``.

Reference:
[3] Günther Schulz, Iterative Berechnung der reziproken Matrix, Z. Angew.
Math. Mech.,13:57-59, (1933) pp. 114, 181.
"""
mutable struct NewtonSchulzAlg{T,RT} <: PolarAlg
    maxiter::Int     # maximum number of iterations.
    verbose::Bool    # whether to show procedural information
    tol::RT     # convergence tolerance.

    function NewtonSchulzAlg{T}(; maxiter::Integer=100,
                             verbose::Bool=false,
                             tol::RT=cbrt(eps(real(T)))) where {T,RT<:Real}
        maxiter > 1 || error("maxiter must be greater than 1.")
        tol > 0 || error("tol must be positive.")

        new{T,RT}(Int(maxiter),
            verbose,
            tol)
    end
end

function solve!(alg::NewtonSchulzAlg,
                   X::Matrix{T}, U::Matrix{T}, H::Matrix{T}) where {T}

    # Newton Schulz converge quadratically if norm(X) < sqrt(3)

    # Revisor's note: this test makes the implementation pointless,
    # since opnorm does an SVD.
    # opnorm(X) < convert(real(T), sqrt(3)) || throw(ArgumentError("The norm of the input matrix must be smaller than sqrt(3)."))

    common_iter!(NewtonSchulzUpdater(), X, U, H, alg.maxiter, alg.verbose, alg.tol)
end

struct NewtonSchulzUpdater <: PolarUpdater end

function update_U!(upd::NewtonSchulzUpdater, U::Matrix{T}) where {T}
    UtU = Array{T}(undef, size(U))
    mul!(UtU, adjoint(U), U)
    copyto!(U, 0.5*U*(3*I - UtU))
end

#
# Compute Polar Decomposition via SVD
#

struct SVDAlg <: PolarAlg end

function solve!(alg::SVDAlg,
                X::Matrix{T}, U::Matrix{T}, H::Matrix{T}) where {T}

    F = svd(X, full = false)
    PQt = Array{T}(undef, size(U))
    mul!(PQt, F.U, adjoint(F.V))
    copyto!(U, PQt)
    copyto!(H, F.V * Diagonal(F.S) * F.Vt)
    H = (1/T(2)) * (H + H')

    return PolarDecomposition{T}(U, H, 1, true)
end

"""
    Polar.HalleyAlg{T} <: Polar.PolarAlg

Halley's method for the polar decomposition

Reference:
Y. Nakatsukasa, Z. Bai and F. Gygi, Optimizing Halley's iteration
for computing the matrix polar decomposition, SIAM, J. Mat. Anal.
Vol. 31, Num 5 (2010) pp. 2700-2720

"""
mutable struct HalleyAlg{T,RT} <: PolarAlg
    maxiter::Int
    verbose::Bool
    tol::RT

    function HalleyAlg{T}( ;maxiter::Integer=100,
                       verbose::Bool=false,
                       tol::RT = cbrt(eps(T))) where {T,RT<:Real}
        maxiter > 1 || error("maxiter must be greater than 1.")
        tol > 0 || error("tol must be positive.")

        new{T,RT}(maxiter,
            verbose,
            tol)
    end
end

function solve!(alg::HalleyAlg{T},
                X::Matrix{T}, U::Matrix{T}, H::Matrix{T}) where {T}
    common_iter!(HalleyUpdater(), X, U, H, alg.maxiter, alg.verbose, alg.tol)
end

struct HalleyUpdater <: PolarUpdater end


function update_U!(upd::HalleyUpdater, U::Matrix{T}) where {T}
    UtU = Array{T}(undef, size(U))
    mul!(UtU, adjoint(U), U)
    copyto!(U, U * (3*I + UtU)* inv(I + 3*UtU))
end

"""
    Polar.QDWHAlg{T} <: Polar.PolarAlg

QR-based Dynamically Weighted Halley (QDWH) algorithm for polar decomposition

Reference: Optimizing Halley's iteration for computing the matrix
           polar decomposition, Yuji Nakatsukasa, Zhaojun Bai and
           Francois Gygi, SIAM, J. Mat. Anal. Vol. 31, Num 5 (2010)
           pp. 2700-2720

Limitations: 1. the QDWH method should support `m > n` matrix itself in the future.
             2. the computing of `alpha` and `L` in `solve!` can be improved by using
                norm and condition number estimate, which were
                not available in Julia when written.
"""
mutable struct QDWHAlg{T,RT} <: PolarAlg
    maxiter::Int
    verbose::Bool
    piv::Bool       # whether to pivot
    tol::RT

    function QDWHAlg{T}( ;maxiter::Integer=100,
                     verbose::Bool=false,
                     piv::Bool=true,
                     tol::RT=cbrt(eps(T))) where {T,RT<:Real}
        maxiter > 1 || error("maxiter must be greater than 1.")
        tol > 0 || error("tol must be positive.")

        new{T,RT}(maxiter,
            verbose,
            piv,
            tol)
    end
end


function solve!(alg::QDWHAlg,
                X::Matrix{T}, U::Matrix{T}, H::Matrix{T}) where {T}
    # alpha is an estimate of the largest singular value of the
    # original matrix
    X_temp = Array{T}(undef, size(X))
    copyto!(X_temp, X)

    n = size(X_temp, 1)
    alpha = norm(X_temp)
    for i in length(X_temp)
        X_temp[i] /= alpha # form X0
    end

    # L is a lower bound for the smallest singular value of X0
    smin_est = opnorm(X_temp, 1)/cond(X_temp, 1)
    L  = smin_est/convert(T, sqrt(n))

    common_iter!(QDWHUpdater(alg.piv, L), X, U, H, alg.maxiter, alg.verbose, alg.tol)
end

mutable struct QDWHUpdater{T} <: PolarUpdater
    piv::Bool   # whether to pivot QR factorization
    L::T  # a lower bound for the smallest singluar value of each update matrix U
end


if VERSION < v"1.7-"
    ColumnNorm() = Val(true)
end

function update_U!(upd::QDWHUpdater, U::Matrix{T}) where {T}
    piv = upd.piv
    L = upd.L
    m, n = size(U)
    B = Array{T}(undef, m+n, n)
    Q1 = Array{T}(undef, n, n)
    Q2 = Array{T}(undef, n, n)
    # Compute paramters L, a, b, c
    L2 = L^2
    dd = try
        (4 * (1 - L2)/L2^2)^(1/3)
        catch
        (complex(4 * (1 -L2)/L2^2, 0))^(1/3)
    end
    sqd = sqrt(1+dd)
    a = sqd + 0.5 * sqrt(8 - 4 * dd + 8 * (2 - L2)/(L2 * sqd))
    a = real(a)
    b = (a - 1)^2 / 4
    c = a + b - 1

    # update L
    upd.L = L * (a + b * L2)/(1 + c * L2)

    copyto!(B, [sqrt(c)*U; Matrix(one(T)*I,n,n)])
    if piv
        F = qr(B, ColumnNorm())
    else
        F = qr(B)
    end
    copyto!(Q1, Matrix(F.Q)[1:m, :])
    copyto!(Q2, Matrix(F.Q)[m+1:end, :])
    copyto!(U, b / c * U + (a - b / c) / sqrt(c) * Q1 * Q2')

end

"""
    Polar.NewtonHybridAlg{T} <: Polar.PolarAlg

Hybrid Newton and Newton-Schulz algorithm for polar decomposition

Reference:
Nicholas J. Higham and Robert S. Schreiber, Fast Polar Decomposition
of an arbitrary matrix, SIAM, J. Sci. Statist. Comput. Vol. 11, No. 4
(1990) pp. 648-655.
"""
mutable struct NewtonHybridAlg{T,RT<:Real} <: PolarAlg
    maxiter::Int    # maximum number of iterations.
    verbose::Bool   # whether to show procedural information.
    tol::RT    # convergence tolerance
    theta::RT  # switch parameter

    function NewtonHybridAlg{T}( ; maxiter::Integer=100,
                             verbose::Bool=false,
                             tol::RT=cbrt(eps(real(T))),
                             theta::RT=convert(real(T), 0.6)) where {T,RT<:Real}
        maxiter > 1 || error("maxiter must  be greater than 1.")
        tol > 0 || error("tol must be positive.")
        theta > 0 || error("theta must be positive.")

        new{T,RT}(maxiter,
            verbose,
            tol,
            theta)
    end
end

function solve!(alg::NewtonHybridAlg,
                X::Matrix{T}, U::Matrix{T}, H::Matrix{T}) where {T}
    common_iter_hybr!(NewtonUpdater(true, eps(real(T))^(1/4)), NewtonSchulzUpdater(), X, U, H,
                      alg.maxiter, alg.verbose, alg.tol, alg.theta)
end

isiterative(alg::PolarAlg) = true
isiterative(alg::SVDAlg) = false

end # submodule Polar


function polar(A::AbstractMatrix{T};
                   alg::Symbol=:newton,
                   maxiter::Integer=100,
                   tol::Real = cbrt(eps(real(T))),
                   verbose::Bool=false) where {T}

    # choose algorithm
    algorithm =
       alg == :newton ? Polar.NewtonAlg{T}(; maxiter=maxiter, tol=tol, verbose=verbose) :
       alg == :qdwh ?  Polar.QDWHAlg{T}(; maxiter=maxiter, tol=tol, verbose=verbose) :
       alg == :halley ? Polar.HalleyAlg{T}(; maxiter=maxiter, tol=tol, verbose=verbose) :
       alg == :svd ? Polar.SVDAlg() :
       alg == :schulz ? Polar.NewtonSchulzAlg{T}(; maxiter=maxiter, tol=tol, verbose=verbose) :
       alg == :hybrid ? Polar.NewtonHybridAlg{T}(; maxiter=maxiter, tol=tol, verbose=verbose) :
       error("Invalid algorithm.")
    polar(A, algorithm)
end

"""
    polar(::AbstractMatrix, ::Polar.PolarAlg)
"""
function polar(A::AbstractMatrix{T}, algorithm::Polar.PolarAlg) where {T}
    # Initialization: if m > n, do QR factorization
    m, n = size(A)
    mm = m
    if m > n
        if Polar.isiterative(algorithm)
            m = n
            F = qr(A)
            A = F.R
        end
    elseif m < n
        throw(ArgumentError("The row dimension of the input matrix must be
              greater or equal to column dimension."))
    end
    U = Array{T}(undef, m, n)
    H = Array{T}(undef, n, n)
    # solve for polar factors
    r = Polar.solve!(algorithm, A, U, H)
    if mm > m
        return PolarDecomposition{T}(F.Q * r.U, r.H, r.niters, r.converged)
    end
    return r
end
