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


function _qrfactUnblocked!(_, _, A::AbstractMatrix{T}, τ::AbstractVector) where {T}
    require_one_based_indexing(A)
    m, n = size(A)
    for k = 1:min(m - 1 + !(T<:Real), n)
        x = view(A, k:m, k)
        τk = reflector!(x)
        τ[k] = τk
        reflectorApply!(x, τk, view(A, k:m, k + 1:n))
    end
    QR(A, τ)
end

_qrfactUnblocked!(::AbstractColumnMajor, ::AbstractStridedLayout, A::AbstractMatrix{T}, τ::AbstractVector{T}) where T<:BlasFloat = QR(LAPACK.geqrf!(A,τ)...)
qrfactUnblocked!(A::AbstractMatrix, τ::AbstractVector) = _qrfactUnblocked!(MemoryLayout(A), MemoryLayout(τ), A, τ)
qrfactUnblocked!(A::AbstractMatrix{T}) where T = qrfactUnblocked!(A, zeros(T, min(size(A)...)))

qrunblocked!(A::AbstractMatrix, ::Val{false}) = qrfactUnblocked!(A)
qrunblocked!(A::AbstractMatrix) = qrunblocked!(A, Val(false))

_qreltype(::Type{T}) where T = typeof(zero(T)/sqrt(abs2(one(T))))


function qrunblocked(A::AbstractMatrix{T}, arg) where T
    require_one_based_indexing(A)
    AA = similar(A, _qreltype(T), size(A))
    copyto!(AA, A)
    return qrunblocked!(AA, arg)
end
function qrunblocked(A::AbstractMatrix{T}) where T
    require_one_based_indexing(A)
    AA = similar(A, _qreltype(T), size(A))
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

getL(F::QR, ::Tuple{AbstractVector,AbstractVector}) = getL(F, size(F))
getQ(F::QR, ::Tuple{AbstractVector,AbstractVector}) = getQ(F, size(F))


getR(F::QR) = getR(F, size(F.factors))
getQ(F::QR) = getQ(F, size(F.factors))

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

ldiv!(F::QR, B::AbstractVecOrMat) = ArrayLayouts.ldiv!(F, B)
ldiv!(F::QR, B::LayoutVector) = ArrayLayouts.ldiv!(F, B)
ldiv!(F::QR, B::LayoutMatrix) = ArrayLayouts.ldiv!(F, B)


"""
    QRPackedQ <: AbstractMatrix

The orthogonal/unitary ``Q`` matrix of a QR factorization stored in [`QR`](@ref).
"""
struct QRPackedQ{T,S<:AbstractMatrix{T},Tau<:AbstractVector{T}} <: LayoutQ{T}
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

AbstractQ{T}(Q::QRPackedQ{T}) where {T} = Q
AbstractQ{T}(Q::QRPackedQ) where {T} = QRPackedQ{T}(Q)
convert(::Type{AbstractQ{T}}, Q::QRPackedQ) where {T} = QRPackedQ{T}(Q)

Matrix{T}(Q::QRPackedQ{S}) where {T,S} =
    convert(Matrix{T}, lmul!(Q, Matrix{S}(I, size(Q, 1), min(size(Q.factors)...))))
Matrix(Q::QRPackedQ{S}) where {S} = Matrix{S}(Q)

if VERSION < v"1.10-"
    AbstractMatrix{T}(Q::QRPackedQ{T}) where {T} = Q
    AbstractMatrix{T}(Q::QRPackedQ) where {T} = QRPackedQ{T}(Q)
    convert(::Type{AbstractMatrix{T}}, Q::QRPackedQ) where {T} = QRPackedQ{T}(Q)
else
    AbstractMatrix{T}(Q::QRPackedQ) where {T} = Matrix{T}(Q)
end

size(F::QR, dim::Integer) = size(getfield(F, :factors), dim)
size(F::QR) = size(getfield(F, :factors))


## Multiplication by Q
### QB
if isdefined(LinearAlgebra, :AdjointQ) # VERSION >= v"1.10-"
    function (*)(Q::QRPackedQ, b::AbstractVector)
        T = promote_type(eltype(Q), eltype(b))
        if size(Q.factors, 1) == length(b)
            bnew = LinearAlgebra.copy_similar(b, T)
        elseif size(Q.factors, 2) == length(b)
            bnew = [b; zeros(T, size(Q.factors, 1) - length(b))]
        else
            throw(DimensionMismatch("vector must have length either $(size(Q.factors, 1)) or $(size(Q.factors, 2))"))
        end
        lmul!(convert(AbstractQ{T}, Q), bnew)
    end
    function (*)(Q::QRPackedQ, B::AbstractMatrix)
        T = promote_type(eltype(Q), eltype(B))
        if size(Q.factors, 1) == size(B, 1)
            Bnew = LinearAlgebra.copy_similar(B, T)
        elseif size(Q.factors, 2) == size(B, 1)
            Bnew = [B; zeros(T, size(Q.factors, 1) - size(B,1), size(B, 2))]
        else
            throw(DimensionMismatch("first dimension of matrix must have size either $(size(Q.factors, 1)) or $(size(Q.factors, 2))"))
        end
        lmul!(convert(AbstractQ{T}, Q), Bnew)
    end
    # function (*)(A::AbstractMatrix, adjQ::LinearAlgebra.AdjointQ{<:Any,<:QRPackedQ})
    #     Q = adjQ.Q
    #     T = promote_type(eltype(A), eltype(adjQ))
    #     adjQQ = convert(AbstractQ{T}, adjQ)
    #     if size(A,2) == size(Q.factors, 1)
    #         AA = LinearAlgebra.copy_similar(A, T)
    #         return rmul!(AA, adjQQ)
    #     elseif size(A,2) == size(Q.factors,2)
    #         return rmul!([A zeros(T, size(A, 1), size(Q.factors, 1) - size(Q.factors, 2))], adjQQ)
    #     else
    #         throw(DimensionMismatch("matrix A has dimensions $(size(A)) but Q-matrix B has dimensions $(size(adjQ))"))
    #     end
    # end
    # (*)(u::LinearAlgebra.AdjointAbsVec, Q::LinearAlgebra.AdjointQ{<:Any,<:QRPackedQ}) = (Q'u')'
end

MemoryLayout(::Type{<:QRPackedQ{<:Any,S,T}}) where {S,T} =
    QRPackedQLayout{typeof(MemoryLayout(S)),typeof(MemoryLayout(T))}()

MemoryLayout(::Type{<:QR{<:Any,S,T}}) where {S,T} =
    QRPackedLayout{typeof(MemoryLayout(S)),typeof(MemoryLayout(T))}()

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
