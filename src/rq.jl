export RQ, rq, rq!

"""
    RQ <: Factorization

An RQ matrix factorization stored in a packed format, typically obtained from
[`rq`](@ref). If ``A`` is an `m`×`n` matrix, then

```math
A = [0 R] Q
```

where ``Q`` is an orthogonal/unitary matrix and ``R`` is upper triangular (trapezoidal
if `m < n`).
The matrix ``Q`` is stored as a sequence of Householder reflectors ``v_i``
and coefficients ``\\tau_i`` where:

```math
Q = \\prod_{i=1}^{\\min(m,n)} (I - \\tau_i v_i v_i^T).
```

Iterating the decomposition produces the components `R` and `Q`.

The object has two fields:

* `factors` is an `m`×`n` matrix.

  - The upper triangular part contains the elements of ``R``, that is `R =
    triu(F.factors)` for a `QR` object `F`.

  - The subdiagonal part contains the reflectors ``v_i`` stored in a packed format where
    ``v_i`` is the ``i``th column of the matrix `V = I + tril(F.factors, -1)`.

* `τ` is a vector  of length `min(m,n)` containing the coefficients ``\tau_i``.

"""
struct RQ{T,S<:AbstractMatrix{T}} <: Factorization{T}
    factors::S
    τ::Vector{T}

    function RQ{T,S}(factors, τ) where {T,S<:AbstractMatrix{T}}
        require_one_based_indexing(factors)
        new{T,S}(factors, τ)
    end
end
RQ(factors::AbstractMatrix{T}, τ::Vector{T}) where {T} = RQ{T,typeof(factors)}(factors, τ)
function RQ{T}(factors::AbstractMatrix, τ::AbstractVector) where {T}
    RQ(convert(AbstractMatrix{T}, factors), convert(Vector{T}, τ))
end

# iteration for destructuring into components
Base.iterate(S::RQ) = (S.R, Val(:Q))
Base.iterate(S::RQ, ::Val{:Q}) = (S.Q, Val(:done))
Base.iterate(S::RQ, ::Val{:done}) = nothing

Base.size(F::RQ, dim::Integer) = size(getfield(F, :factors), dim)
Base.size(F::RQ) = size(getfield(F, :factors))

function Base.getproperty(F::RQ, d::Symbol)
    m, n = size(F)
    if d === :R
        if m <= n
            return triu!(getfield(F, :factors)[1:m, n-m+1:n])
        else
            return triu(getfield(F, :factors),min(0,n-m))
        end
    elseif d === :Q
        return RQPackedQ(getfield(F, :factors), F.τ)
    else
        getfield(F, d)
    end
end

"""
    RQPackedQ <: AbstractMatrix

The orthogonal/unitary ``Q`` matrix of an RQ factorization stored in [`RQ`](@ref) format.
"""
struct RQPackedQ{T,S<:AbstractMatrix{T}} <: AbstractQ{T}
    factors::S
    τ::Vector{T}

    function RQPackedQ{T,S}(factors, τ) where {T,S<:AbstractMatrix{T}}
        require_one_based_indexing(factors)
        new{T,S}(factors, τ)
    end
end

# Conversions
RQ{T}(A::RQ) where {T} = RQ(convert(AbstractMatrix{T}, A.factors), convert(AbstractVector{T}, A.τ))
Factorization{T}(A::RQ{T}) where {T} = A
Factorization{T}(A::RQ) where {T} = RQ{T}(A)
AbstractMatrix(F::RQ) = F.R * F.Q
AbstractArray(F::RQ) = AbstractMatrix(F)
Matrix(F::RQ) = Array(AbstractArray(F))
Array(F::RQ) = Matrix(F)

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::RQ)
    summary(io, F); println(io)
    println(io, "R factor:")
    show(io, mime, F.R)
    println(io, "\nQ factor:")
    show(io, mime, F.Q)
end

RQPackedQ(factors::AbstractMatrix{T}, τ::Vector{T}) where {T} = RQPackedQ{T,typeof(factors)}(factors, τ)
function RQPackedQ{T}(factors::AbstractMatrix, τ::AbstractVector) where {T}
    RQPackedQ(convert(AbstractMatrix{T}, factors), convert(Vector{T}, τ))
end
RQPackedQ{T}(Q::RQPackedQ) where {T} = RQPackedQ(convert(AbstractMatrix{T}, Q.factors), convert(Vector{T}, Q.τ))
AbstractMatrix{T}(Q::RQPackedQ{T}) where {T} = Q
AbstractMatrix{T}(Q::RQPackedQ) where {T} = RQPackedQ{T}(Q)
# this makes the rectangular version, alas.
# function Matrix{T}(Q::RQPackedQ{T}) where {T<:BlasFloat}
#     m,n = size(Q.factors)
#     i0= m > n ? (m-n+1) : 1
#     LAPACK.orgrq!(Q.factors[i0:m,1:n], Q.τ)
# end

Matrix{T}(Q::RQPackedQ{S}) where {T,S} = Matrix{T}(lmul!(Q, Matrix{S}(I, size(Q,2), size(Q,2))))
Base.size(Q::RQPackedQ, dim::Integer) = size(getfield(Q, :factors), dim == 1 ? 2 : dim)
Base.size(Q::RQPackedQ) = size(Q, 1), size(Q, 2)

function lmul!(A::RQPackedQ{T,S}, B::StridedVecOrMat{T}) where {T<:BlasFloat, S<:StridedMatrix}
    m,n = size(A.factors)
    if m > n
        return LAPACK.ormrq!('L','N',view(A.factors, m-n+1:m, 1:n),A.τ,B)
    else
        return LAPACK.ormrq!('L','N',A.factors,A.τ,B)
    end
end
function lmul!(A::RQPackedQ, B::AbstractVecOrMat)
    require_one_based_indexing(B)
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if nA != mB
        throw(DimensionMismatch("matrix A has dimensions ($nA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    mm = min(mA,nA)
    @inbounds begin
        for k = mm:-1:1
            krow = mA-mm+k
            kcol = nA-mm+k
            for j = 1:nB
                vBj = B[kcol,j]
                for i = 1:kcol-1
                    # vBj += conj(Afactors[i,k])*B[i,j]
                    vBj += (Afactors[krow,i])*B[i,j]
                end
                vBj = conj(A.τ[k])*vBj
                B[kcol,j] -= vBj
                for i = 1:kcol-1
                    # B[i,j] -= Afactors[i,k]*vBj
                    B[i,j] -= conj(Afactors[krow,i])*vBj
                end
            end
        end
    end
    B
end

function (*)(B::StridedMatrix, Q::RQPackedQ)
    TBQ = promote_type(eltype(Q), eltype(B))
    Qnew = convert(AbstractMatrix{TBQ}, Q)
    if size(Q.factors, 2) == size(B, 2)
        Bnew = copy_oftype(B, TBQ)
    elseif size(Q.factors, 1) == size(B, 2)
        # People will expect this, misguided though it seems.
        Bnew = hcat(zeros(TBQ, size(B,1), size(Q.factors, 2) - size(B,2)),B)
    else
        throw(DimensionMismatch("first dimension of matrix must have size either $(size(Q.factors, 1)) or $(size(Q.factors, 2))"))
    end
    rmul!(Bnew, Qnew)
end

function lmul!(adjA::Adjoint{<:Any,<:RQPackedQ{T,S}}, B::StridedVecOrMat{T}
               ) where {T<:BlasReal,S<:StridedMatrix}
    A = adjA.parent
    m,n = size(A.factors)
    if m > n
        return LAPACK.ormrq!('L','T',view(A.factors, m-n+1:m, 1:n),A.τ,B)
    else
        return LAPACK.ormrq!('L','T',A.factors,A.τ,B)
    end
end
function lmul!(adjA::Adjoint{<:Any,<:RQPackedQ{T,S}}, B::StridedVecOrMat{T}
               ) where {T<:BlasComplex,S<:StridedMatrix}
    A = adjA.parent
    m,n = size(A.factors)
    if m > n
        return LAPACK.ormrq!('L','C',view(A.factors, m-n+1:m, 1:n),A.τ,B)
    else
        return LAPACK.ormrq!('L','C',A.factors,A.τ,B)
    end
end
function lmul!(adjA::Adjoint{<:Any,<:RQPackedQ}, B::AbstractVecOrMat)
    require_one_based_indexing(B)
    A = adjA.parent
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if nA != mB
        throw(DimensionMismatch("matrix A has dimensions ($nA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    mm = min(mA,nA)
    @inbounds begin
        for k = 1:mm
            krow = mA-mm+k
            kcol = nA-mm+k
            for j = 1:nB
                vBj = B[kcol,j]
                for i = 1:kcol-1
                    # vBj += conj(Afactors[i,k])*B[i,j]
                    vBj += (Afactors[krow,i])*B[i,j]
                end
                vBj = A.τ[k]*vBj
                B[kcol,j] -= vBj
                for i = 1:kcol-1
                    # B[i,j] -= Afactors[i,k]*vBj
                    B[i,j] -= conj(Afactors)[krow,i]*vBj
                end
            end
        end
    end
    B
end
function rmul!(A::StridedVecOrMat{T}, B::RQPackedQ{T,S}
               ) where {T<:BlasFloat,S<:StridedMatrix}
    m,n = size(B.factors)
    if m > n
        return LAPACK.ormrq!('R','N',view(B.factors, m-n+1:m, 1:n),B.τ,A)
    else
        return LAPACK.ormrq!('R','N',B.factors,B.τ,A)
    end
end
function rmul!(A::StridedMatrix,Q::RQPackedQ)
    mQ, nQ = size(Q.factors)
    mA, nA = size(A,1), size(A,2)
    if nA != nQ
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but matrix Q has dimensions ($nQ, $nQ)"))
    end
    Qfactors = Q.factors
    mm = min(mQ, nQ)
    @inbounds begin
        for k = 1:min(mQ,nQ)
            krow = mQ-mm+k
            kcol = nQ-mm+k
            for i = 1:mA
                vAi = A[i,kcol]
                for j = 1:kcol-1
                    # vAi += A[i,j]*Qfactors[j,k]
                    vAi += A[i,j]*conj(Qfactors[krow,j])
                end
                vAi = vAi*conj(Q.τ[k])
                A[i,kcol] -= vAi
                for j = 1:kcol-1
                    # A[i,j] -= vAi*conj(Qfactors[j,k])
                    A[i,j] -= vAi*(Qfactors[krow,j])
                end
            end
        end
    end
    A
end
function rmul!(A::StridedVecOrMat{T}, adjB::Adjoint{<:Any,<:RQPackedQ{T}}
               ) where {T<:BlasReal}
    B = adjB.parent
    m,n = size(B.factors)
    if m > n
        return LAPACK.ormrq!('R','T',view(B.factors, m-n+1:m, 1:n),B.τ,A)
    else
        return LAPACK.ormrq!('R','T',B.factors,B.τ,A)
    end
end
function rmul!(A::StridedVecOrMat{T}, adjB::Adjoint{<:Any,<:RQPackedQ{T}}
               ) where {T<:BlasComplex}
    B = adjB.parent
    m,n = size(B.factors)
    if m > n
        return LAPACK.ormrq!('R','C',view(B.factors, m-n+1:m, 1:n),B.τ,A)
    else
        return LAPACK.ormrq!('R','C',B.factors,B.τ,A)
    end
end
function rmul!(A::StridedMatrix, adjQ::Adjoint{<:Any,<:RQPackedQ})
    Q = adjQ.parent
    mQ, nQ = size(Q.factors)
    mA, nA = size(A,1), size(A,2)
    if nA != nQ
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but matrix Q has dimensions ($nQ, $nQ)"))
    end
    Qfactors = Q.factors
    mm = min(mQ, nQ)
    @inbounds begin
        for k = min(mQ,nQ):-1:1
            krow = mQ-mm+k
            kcol = nQ-mm+k
            for i = 1:mA
                vAi = A[i,kcol]
                for j = 1:kcol-1
                    # vAi += A[i,j]*Qfactors[j,k]
                    vAi += A[i,j]*conj(Qfactors[krow,j])
                end
                vAi = vAi*Q.τ[k]
                A[i,kcol] -= vAi
                for j = 1:kcol-1
                    # A[i,j] -= vAi*conj(Qfactors[j,k])
                    A[i,j] -= vAi*(Qfactors[krow,j])
                end
            end
        end
    end
    A
end

ldiv!(A::RQ{T}, B::StridedMatrix{S}) where {T,S} = lmul!(adjoint(A.Q), ldiv!(UpperTriangular(A.R), B))
ldiv!(A::RQ{T}, B::StridedVector{S}) where {T,S} = lmul!(adjoint(A.Q), ldiv!(UpperTriangular(A.R), B))

# apply shifted reflector from right
@inline function reflectorYlppa!(x::AbstractVector, τ::Number, A::StridedMatrix)
    require_one_based_indexing(x)
    m, n = size(A)
    if length(x) != n
        throw(DimensionMismatch("reflector has length $(length(x)), which must match the second dimension of matrix A, $n"))
    end
    @inbounds begin
        for j = 1:m
            # dot
            Ajv = A[j, n]
            for i = 1:n-1
                Ajv += A[j, i] * x[i+1]
            end

            Ajv = τ * Ajv

            # ger
            A[j, n] -= Ajv
            for i = 1:n-1
                A[j, i] -= Ajv * x[i+1]'
            end
        end
    end
    return A
end

function _rqfactUnblocked!(A::AbstractMatrix{T}) where {T}
    require_one_based_indexing(A)
    m, n = size(A)
    τ = zeros(T, min(m,n))
    v = zeros(T, max(m,n))
    mm = min(m,n)
    for k = mm:-1:1
        krow = m-mm+k
        kcol = n-mm+k
        v[1] = A[krow,kcol]'
        for i = 2:kcol
            v[i] = A[krow,i-1]'
        end
        x = view(v, 1:kcol)
        if kcol==1 && T<:Real
            τk = zero(T)
        else
            τk = reflector!(x)
        end
        τ[k] = τk
        reflectorYlppa!(x, τk, view(A, 1:krow-1, 1:kcol))
        A[krow,kcol] = v[1]'
        for i = 2:kcol
            A[krow,i-1] = v[i]'
        end
    end
    RQ(A, τ)
end

"""
    rq(A) -> F

Compute the RQ factorization of the matrix `A`: an orthogonal/unitary matrix `Q` and
and upper triangular or trapezoidal matrix `R` such that
```math
A = [0 R] Q
```
The returned object `F` stores the factorization in a packed format [`RQ`](@ref).
Unlike the `QR` factorization this does not handle least squares problems.
If `A` is `m`×`n` then `Q` is `n`×`n` and `R` is `m`×`min(m,n)`.
"""
function rq end

"""
    rq!(A) -> F

The same as [`rq`](@ref), but saves space by overwriting the input `A`.
"""
rq!(A::StridedMatrix{<:BlasFloat}) = RQ(LAPACK.gerqf!(A)...)
rq!(A::StridedMatrix) = _rqfactUnblocked!(A)

rq(x::Number) = rq(fill(x,1,1))
function rq(v::AbstractVector)
    require_one_based_indexing(v)
    rq(reshape(v, (length(v), 1)))
end

# There is a local version of this:
# using LinearAlgebra: _qreltype

function rq(A::AbstractMatrix{T}) where {T}
    require_one_based_indexing(A)
    AA = similar(A, LinearAlgebra._qreltype(T), size(A))
    copyto!(AA, A)
    return rq!(AA)
end
