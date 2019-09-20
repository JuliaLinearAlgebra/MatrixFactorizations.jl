# This file is based on a part of Julia. License is MIT: https://julialang.org/license

# It has modified the definition of QR to support general τ

# QR and Hessenberg Factorizations
"""
    QR <: Factorization

A QR matrix factorization stored in a packed format, typically obtained from
[`qr`](@ref). If ``A`` is an `m`×`n` matrix, then

```math
A = Q R
```

where ``Q`` is an orthogonal/unitary matrix and ``R`` is upper triangular.
The matrix ``Q`` is stored as a sequence of Householder reflectors ``v_i``
and coefficients ``\\tau_i`` where:

```math
Q = \\prod_{i=1}^{\\min(m,n)} (I - \\tau_i v_i v_i^T).
```

Iterating the decomposition produces the components `Q` and `R`.

The object has two fields:

* `factors` is an `m`×`n` matrix.

  - The upper triangular part contains the elements of ``R``, that is `R =
    triu(F.factors)` for a `QR` object `F`.

  - The subdiagonal part contains the reflectors ``v_i`` stored in a packed format where
    ``v_i`` is the ``i``th column of the matrix `V = I + tril(F.factors, -1)`.

* `τ` is a vector  of length `min(m,n)` containing the coefficients ``\tau_i``.

"""
struct QR{T,S<:AbstractMatrix{T},Tau<:AbstractVector{T}} <: Factorization{T}
    factors::S
    τ::Tau

    function QR{T,S,Tau}(factors, τ) where {T,S<:AbstractMatrix{T},Tau<:AbstractVector{T}}
        require_one_based_indexing(factors)
        new{T,S,Tau}(factors, τ)
    end
end
QR(factors::AbstractMatrix{T}, τ::AbstractVector{T}) where {T} = QR{T,typeof(factors), typeof(τ)}(factors, τ)
QR{T}(factors::AbstractMatrix, τ::AbstractVector) where {T} =
    QR(convert(AbstractMatrix{T}, factors), convert(AbstractVector{T}, τ))

QR(F::LinearAlgebra.QR) = QR(F.factors, F.τ)

# iteration for destructuring into components
Base.iterate(S::QR) = (S.Q, Val(:R))
Base.iterate(S::QR, ::Val{:R}) = (S.R, Val(:done))
Base.iterate(S::QR, ::Val{:done}) = nothing


function generic_qrfactUnblocked!(A::AbstractMatrix{T}) where {T}
    require_one_based_indexing(A)
    m, n = size(A)
    τ = zeros(T, min(m,n))
    for k = 1:min(m - 1 + !(T<:Real), n)
        x = view(A, k:m, k)
        τk = reflector!(x)
        τ[k] = τk
        reflectorApply!(x, τk, view(A, k:m, k + 1:n))
    end
    QR(A, τ)
end

qrfactUnblocked!(A::AbstractMatrix) = generic_qrfactUnblocked!(A)
qrfactUnblocked!(A::StridedMatrix{T}) where T<:BlasFloat = QR(LAPACK.geqrf!(A)...)

qrunblocked!(A::AbstractMatrix, ::Val{false}) = qrfactUnblocked!(A)
qrunblocked!(A::AbstractMatrix) = qrunblocked!(A, Val(false))

_qreltype(::Type{T}) where T = typeof(zero(T)/sqrt(abs2(one(T))))


function qrunblocked(A::AbstractMatrix{T}, arg) where T
    require_one_based_indexing(A)
    AA = similar(A, _qleltype(T), size(A))
    copyto!(AA, A)
    return qrunblocked!(AA, arg)
end
function qrunblocked(A::AbstractMatrix{T}) where T
    require_one_based_indexing(A)
    AA = similar(A, _qleltype(T), size(A))
    copyto!(AA, A)
    return qrunblocked!(AA)
end
qrunblocked(x::Number) = qrunblocked(fill(x,1,1))
function qrunblocked(v::AbstractVector)
    require_one_based_indexing(v)
    qrunblocked(reshape(v, (length(v), 1)))
end

# Conversions
QR{T}(A::QR) where {T} = QR(convert(AbstractMatrix{T}, A.factors), convert(AbstractVector{T}, A.τ))
Factorization{T}(A::QR{T}) where {T} = A
Factorization{T}(A::QR) where {T} = QR{T}(A)
AbstractMatrix(F::QR) = F.Q * F.R
AbstractArray(F::QR) = AbstractMatrix(F)
Matrix(F::QR) = Array(AbstractArray(F))
Array(F::QR) = Matrix(F)

function show(io::IO, mime::MIME{Symbol("text/plain")}, F::QR)
    summary(io, F); println(io)
    println(io, "Q factor:")
    show(io, mime, F.Q)
    println(io, "\nR factor:")
    show(io, mime, F.R)
end

@inline function getR(F::QR, _) 
    m, n = size(F)
    triu!(getfield(F, :factors)[1:min(m,n), 1:n])
end
@inline getQ(F::QR, _) = QRPackedQ(getfield(F, :factors), F.τ)

getR(F::QR) = getR(F, axes(F.factors))
getQ(F::QR) = getQ(F, axes(F.factors))

function getproperty(F::QR, d::Symbol)
    if d == :R
        return getR(F)
    elseif d == :Q
        return getQ(F)
    else
        getfield(F, d)
    end
end

Base.propertynames(F::QR, private::Bool=false) =
    (:R, :Q, (private ? fieldnames(typeof(F)) : ())...)

"""
    QRPackedQ <: AbstractMatrix

The orthogonal/unitary ``Q`` matrix of a QR factorization stored in [`QR`](@ref).
"""
struct QRPackedQ{T,S<:AbstractMatrix{T},Tau<:AbstractVector{T}} <: AbstractQ{T}
    factors::S
    τ::Tau

    function QRPackedQ{T,S,Tau}(factors, τ) where {T,S<:AbstractMatrix{T},Tau<:AbstractVector{T}}
        require_one_based_indexing(factors)
        new{T,S,Tau}(factors, τ)
    end
end
QRPackedQ(factors::AbstractMatrix{T}, τ::AbstractVector{T}) where {T} = QRPackedQ{T,typeof(factors),typeof(τ)}(factors, τ)
function QRPackedQ{T}(factors::AbstractMatrix, τ::AbstractVector) where {T}
    QRPackedQ(convert(AbstractMatrix{T}, factors), convert(AbstractVector{T}, τ))
end

QRPackedQ{T}(Q::QRPackedQ) where {T} = QRPackedQ(convert(AbstractMatrix{T}, Q.factors), convert(AbstractVector{T}, Q.τ))
QRPackedQ(Q::LinearAlgebra.QRPackedQ) = QRPackedQ(Q.factors, Q.τ)

AbstractMatrix{T}(Q::QRPackedQ{T}) where {T} = Q
AbstractMatrix{T}(Q::QRPackedQ) where {T} = QRPackedQ{T}(Q)


size(F::QR, dim::Integer) = size(getfield(F, :factors), dim)
size(F::QR) = size(getfield(F, :factors))


## Multiplication by Q
### QB
lmul!(A::QRPackedQ{T,S}, B::StridedVecOrMat{T}) where {T<:BlasFloat, S<:StridedMatrix} =
    LAPACK.ormqr!('L','N',A.factors,A.τ,B)

function lmul!(A::QRPackedQ, B::AbstractVecOrMat)
    require_one_based_indexing(B)
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    @inbounds begin
        for k = min(mA,nA):-1:1
            for j = 1:nB
                vBj = B[k,j]
                for i = k+1:mB
                    vBj += conj(Afactors[i,k])*B[i,j]
                end
                vBj = A.τ[k]*vBj
                B[k,j] -= vBj
                for i = k+1:mB
                    B[i,j] -= Afactors[i,k]*vBj
                end
            end
        end
    end
    B
end

### QcB
lmul!(adjA::Adjoint{<:Any,<:QRPackedQ{T,S,Vector{T}}}, B::StridedVecOrMat{T}) where {T<:BlasReal,S<:StridedMatrix} =
    (A = adjA.parent; LAPACK.ormqr!('L','T',A.factors,A.τ,B))
lmul!(adjA::Adjoint{<:Any,<:QRPackedQ{T,S,Vector{T}}}, B::StridedVecOrMat{T}) where {T<:BlasComplex,S<:StridedMatrix} =
    (A = adjA.parent; LAPACK.ormqr!('L','C',A.factors,A.τ,B))
function lmul!(adjA::Adjoint{<:Any,<:QRPackedQ}, B::AbstractVecOrMat)
    require_one_based_indexing(B)
    A = adjA.parent
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    @inbounds begin
        for k = 1:min(mA,nA)
            for j = 1:nB
                vBj = B[k,j]
                for i = k+1:mB
                    vBj += conj(Afactors[i,k])*B[i,j]
                end
                vBj = conj(A.τ[k])*vBj
                B[k,j] -= vBj
                for i = k+1:mB
                    B[i,j] -= Afactors[i,k]*vBj
                end
            end
        end
    end
    B
end

## AQ
rmul!(A::StridedVecOrMat{T}, B::QRPackedQ{T,S,Vector{T}}) where {T<:BlasFloat,S<:StridedMatrix} =
    LAPACK.ormqr!('R', 'N', B.factors, B.τ, A)
function rmul!(A::AbstractMatrix,Q::QRPackedQ)
    mQ, nQ = size(Q.factors)
    mA, nA = size(A,1), size(A,2)
    if nA != mQ
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but matrix Q has dimensions ($mQ, $nQ)"))
    end
    Qfactors = Q.factors
    @inbounds begin
        for k = 1:min(mQ,nQ)
            for i = 1:mA
                vAi = A[i,k]
                for j = k+1:mQ
                    vAi += A[i,j]*Qfactors[j,k]
                end
                vAi = vAi*Q.τ[k]
                A[i,k] -= vAi
                for j = k+1:nA
                    A[i,j] -= vAi*conj(Qfactors[j,k])
                end
            end
        end
    end
    A
end

### AQc
rmul!(A::StridedVecOrMat{T}, adjB::Adjoint{<:Any,<:QRPackedQ{T,S,Vector{T}}}) where {T<:BlasReal,S<:StridedMatrix} =
    (B = adjB.parent; LAPACK.ormqr!('R','T',B.factors,B.τ,A))
rmul!(A::StridedVecOrMat{T}, adjB::Adjoint{<:Any,<:QRPackedQ{T,S,Vector{T}}}) where {T<:BlasComplex,S<:StridedMatrix} =
    (B = adjB.parent; LAPACK.ormqr!('R','C',B.factors,B.τ,A))
function rmul!(A::AbstractMatrix, adjQ::Adjoint{<:Any,<:QRPackedQ})
    Q = adjQ.parent
    mQ, nQ = size(Q.factors)
    mA, nA = size(A,1), size(A,2)
    if nA != mQ
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but matrix Q has dimensions ($mQ, $nQ)"))
    end
    Qfactors = Q.factors
    @inbounds begin
        for k = min(mQ,nQ):-1:1
            for i = 1:mA
                vAi = A[i,k]
                for j = k+1:mQ
                    vAi += A[i,j]*Qfactors[j,k]
                end
                vAi = vAi*conj(Q.τ[k])
                A[i,k] -= vAi
                for j = k+1:nA
                    A[i,j] -= vAi*conj(Qfactors[j,k])
                end
            end
        end
    end
    A
end

# Julia implementation similar to xgelsy
function ldiv!(A::QR{T}, B::AbstractMatrix{T}) where T
    m, n = size(A)
    minmn = min(m,n)
    mB, nB = size(B)
    lmul!(adjoint(A.Q), view(B, 1:m, :))
    R = A.R
    @inbounds begin
        if n > m # minimum norm solution
            τ = zeros(T,m)
            for k = m:-1:1 # Trapezoid to triangular by elementary operation
                x = view(R, k, [k; m + 1:n])
                τk = reflector!(x)
                τ[k] = conj(τk)
                for i = 1:k - 1
                    vRi = R[i,k]
                    for j = m + 1:n
                        vRi += R[i,j]*x[j - m + 1]'
                    end
                    vRi *= τk
                    R[i,k] -= vRi
                    for j = m + 1:n
                        R[i,j] -= vRi*x[j - m + 1]
                    end
                end
            end
        end
        LinearAlgebra.ldiv!(UpperTriangular(view(R, :, 1:minmn)), view(B, 1:minmn, :))
        if n > m # Apply elementary transformation to solution
            B[m + 1:mB,1:nB] .= zero(T)
            for j = 1:nB
                for k = 1:m
                    vBj = B[k,j]
                    for i = m + 1:n
                        vBj += B[i,j]*R[k,i]'
                    end
                    vBj *= τ[k]
                    B[k,j] -= vBj
                    for i = m + 1:n
                        B[i,j] -= R[k,i]*vBj
                    end
                end
            end
        end
    end
    return B
end
ldiv!(A::QR, B::AbstractVector) = ldiv!(A, reshape(B, length(B), 1))[:]


function (\)(A::QR{TA}, B::AbstractVecOrMat{TB}) where {TA,TB}
    require_one_based_indexing(B)
    S = promote_type(TA,TB)
    m, n = size(A)
    m == size(B,1) || throw(DimensionMismatch("Both inputs should have the same number of rows"))

    AA = Factorization{S}(A)

    X = _zeros(S, B, n)
    X[1:size(B, 1), :] = B
    ldiv!(AA, X)
    return _cut_B(X, 1:n)
end

function (\)(A::QR{T}, BIn::VecOrMat{Complex{T}}) where T<:BlasReal
    require_one_based_indexing(BIn)
    m, n = size(A)
    m == size(BIn, 1) || throw(DimensionMismatch("left hand side has $m rows, but right hand side has $(size(BIn,1)) rows"))

# |z1|z3|  reinterpret  |x1|x2|x3|x4|  transpose  |x1|y1|  reshape  |x1|y1|x3|y3|
# |z2|z4|      ->       |y1|y2|y3|y4|     ->      |x2|y2|     ->    |x2|y2|x4|y4|
#                                                 |x3|y3|
#                                                 |x4|y4|
    B = reshape(copy(transpose(reinterpret(T, reshape(BIn, (1, length(BIn)))))), size(BIn, 1), 2*size(BIn, 2))

    X = _zeros(T, B, n)
    X[1:size(B, 1), :] = B

    ldiv!(A, X)

# |z1|z3|  reinterpret  |x1|x2|x3|x4|  transpose  |x1|y1|  reshape  |x1|y1|x3|y3|
# |z2|z4|      <-       |y1|y2|y3|y4|     <-      |x2|y2|     <-    |x2|y2|x4|y4|
#                                                 |x3|y3|
#                                                 |x4|y4|
    XX = reshape(collect(reinterpret(Complex{T}, copy(transpose(reshape(X, div(length(X), 2), 2))))), _ret_size(A, BIn))
    return _cut_B(XX, 1:n)
end
