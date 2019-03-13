# This file is a modified from Julia's LinearAlgebra/qr.jl. License is MIT: https://julialang.org/license

# QL and Hessenberg Factorizations
"""
    QL <: Factorization

A QL matrix factorization stored in a packed format, typically obtained from
[`ql`](@ref). If ``A`` is an `m`×`n` matrix, then

```math
A = Q L
```

where ``Q`` is an orthogonal/unitary matrix and ``L`` is lower triangular.
The matrix ``Q`` is stored as a sequence of Householder reflectors ``v_i``
and coefficients ``\\tau_i`` where:

```math
Q = \\prod_{i=1}^{\\min(m,n)} (I - \\tau_i v_i v_i^T).
```

Iterating the decomposition produces the components `Q` and `L`.

The object has two fields:

* `factors` is an `m`×`n` matrix.

  - The lower triangular part contains the elements of ``L``, that is `L =
    tril(F.factors)` for a `QL` object `F`.

  - The subdiagonal part contains the reflectors ``v_i`` stored in a packed format where
    ``v_i`` is the ``i``th column of the matrix `V = I + tril(F.factors, -1)`.

* `τ` is a vector  of length `min(m,n)` containing the coefficients ``\tau_i``.

"""
struct QL{T,S<:AbstractMatrix{T},Tau<:AbstractVector{T}} <: Factorization{T}
    factors::S
    τ::Tau

    function QL{T,S,Tau}(factors, τ) where {T,S<:AbstractMatrix{T},Tau<:AbstractVector{T}}
        @assert !has_offset_axes(factors)
        new{T,S,Tau}(factors, τ)
    end
end
QL(factors::AbstractMatrix{T}, τ::AbstractVector{T}) where {T} = QL{T,typeof(factors),typeof(τ)}(factors, τ)
function QL{T}(factors::AbstractMatrix, τ::AbstractVector) where {T}
    QL(convert(AbstractMatrix{T}, factors), convert(AbstractVector{T}, τ))
end

# iteration for destructuring into components
Base.iterate(S::QL) = (S.Q, Val(:L))
Base.iterate(S::QL, ::Val{:L}) = (S.L, Val(:done))
Base.iterate(S::QL, ::Val{:done}) = nothing

function qlfactUnblocked!(A::AbstractMatrix{T}) where {T}
    @assert !has_offset_axes(A)
    m, n = size(A)
    τ = zeros(T, min(m,n))
    for k = min(m,n):-1:(1 + (T<:Real))
        ν = k+n-min(m,n)
        x = view(A, k:-1:1, ν)
        τk = reflector!(x)
        τ[k] = τk
        reflectorApply!(x, τk, view(A, k:-1:1, 1:ν-1))
    end
    QL(A, τ)
end


# Generic fallbacks

"""
    ql!(A, pivot=Val(false))

`ql!` is the same as [`ql`](@ref) when `A` is a subtype of
`StridedMatrix`, but saves space by overwriting the input `A`, instead of creating a copy.
An [`InexactError`](@ref) exception is thrown if the factorization produces a number not
representable by the element type of `A`, e.g. for integer types.

# Examples
```jldoctest
julia> a = [1. 2.; 3. 4.]
2×2 Array{Float64,2}:
 1.0  2.0
 3.0  4.0

julia> ql!(a)
LinearAlgebra.QLCompactWY{Float64,Array{Float64,2}}
Q factor:
2×2 LinearAlgebra.QLCompactWYQ{Float64,Array{Float64,2}}:
 -0.316228  -0.948683
 -0.948683   0.316228
 L factor:
2×2 Array{Float64,2}:
 -3.16228  -4.42719
  0.0      -0.632456

julia> a = [1 2; 3 4]
2×2 Array{Int64,2}:
 1  2
 3  4

julia> ql!(a)
ERROR: InexactError: Int64(-3.1622776601683795)
Stacktrace:
[...]
```
"""
ql!(A::AbstractMatrix, ::Val{false}) = qlfactUnblocked!(A)
ql!(A::AbstractMatrix) = ql!(A, Val(false))

_qleltype(::Type{T}) where T = typeof(zero(T)/sqrt(abs2(one(T))))

"""
    ql(A, pivot=Val(false)) -> F

Compute the QL factorization of the matrix `A`: an orthogonal (or unitary if `A` is
complex-valued) matrix `Q`, and a lower triangular matrix `L` such that

```math
A = Q L
```

The returned object `F` stores the factorization in a packed format:

 - `F` is a [`QL`](@ref) object.

The individual components of the decomposition `F` can be retrieved via property accessors:

 - `F.Q`: the orthogonal/unitary matrix `Q`
 - `F.L`: the lower triangular matrix `L`

Iterating the decomposition produces the components `Q`, `L`, and if extant `p`.

The following functions are available for the `QL` objects: [`inv`](@ref), [`size`](@ref),
and [`\\`](@ref). When `A` is rectangular, `\\` will return a least squares
solution and if the solution is not unique, the one with smallest norm is returned. When
`A` is not full rank, factorization with (column) pivoting is required to obtain a minimum
norm solution.

Multiplication with respect to either full/square or non-full/square `Q` is allowed, i.e. both `F.Q*F.L`
and `F.Q*A` are supported. A `Q` matrix can be converted into a regular matrix with
[`Matrix`](@ref).  This operation returns the "thin" Q factor, i.e., if `A` is `m`×`n` with `m>=n`, then
`Matrix(F.Q)` yields an `m`×`n` matrix with orthonormal columns.  To retrieve the "full" Q factor, an
`m`×`m` orthogonal matrix, use `F.Q*Matrix(I,m,m)`.  If `m<=n`, then `Matrix(F.Q)` yields an `m`×`m`
orthogonal matrix.

# Examples
```jldoctest
julia> A = [3.0 -6.0; 4.0 -8.0; 0.0 1.0]
3×2 Array{Float64,2}:
 3.0  -6.0
 4.0  -8.0
 0.0   1.0

julia> F = ql(A)
LinearAlgebra.QLCompactWY{Float64,Array{Float64,2}}
Q factor:
3×3 LinearAlgebra.QLCompactWYQ{Float64,Array{Float64,2}}:
 -0.6   0.0   0.8
 -0.8   0.0  -0.6
  0.0  -1.0   0.0
L factor:
2×2 Array{Float64,2}:
 -5.0  10.0
  0.0  -1.0

julia> F.Q * F.L == A
true
```
"""
function ql(A::AbstractMatrix{T}, arg) where T
    @assert !has_offset_axes(A)
    AA = similar(A, _qleltype(T), size(A))
    copyto!(AA, A)
    return ql!(AA, arg)
end
function ql(A::AbstractMatrix{T}) where T
    @assert !has_offset_axes(A)
    AA = similar(A, _qleltype(T), size(A))
    copyto!(AA, A)
    return ql!(AA)
end
ql(x::Number) = ql(fill(x,1,1))
function ql(v::AbstractVector)
    @assert !has_offset_axes(v)
    ql(reshape(v, (length(v), 1)))
end

# Conversions
QL{T}(A::QL) where {T} = QL(convert(AbstractMatrix{T}, A.factors), convert(AbstractVector{T}, A.τ))
Factorization{T}(A::QL{T}) where {T} = A
Factorization{T}(A::QL) where {T} = QL{T}(A)
AbstractMatrix(F::QL) = F.Q * F.L
AbstractArray(F::QL) = AbstractMatrix(F)
Matrix(F::QL) = Array(AbstractArray(F))
Array(F::QL) = Matrix(F)

function show(io::IO, mime::MIME{Symbol("text/plain")}, F::QL)
    summary(io, F); println(io)
    println(io, "Q factor:")
    show(io, mime, F.Q)
    println(io, "\nL factor:")
    show(io, mime, F.L)
end

function getproperty(F::QL, d::Symbol)
    m, n = size(F)
    if d == :L
        return tril!(copy(getfield(F, :factors)), max(n-m,0))
    elseif d == :Q
        return QLPackedQ(getfield(F, :factors), F.τ)
    else
        getfield(F, d)
    end
end

Base.propertynames(F::QL, private::Bool=false) =
    (:L, :Q, (private ? fieldnames(typeof(F)) : ())...)


"""
    QLPackedQ <: AbstractMatrix

The orthogonal/unitary ``Q`` matrix of a QL factorization stored in [`QL`](@ref).
"""
struct QLPackedQ{T,S<:AbstractMatrix{T},Tau<:AbstractVector{T}} <: AbstractQ{T}
    factors::S
    τ::Tau

    function QLPackedQ{T,S,Tau}(factors, τ) where {T,S<:AbstractMatrix{T},Tau<:AbstractVector{T}}
        @assert !has_offset_axes(factors)
        new{T,S,Tau}(factors, τ)
    end
end
QLPackedQ(factors::AbstractMatrix{T}, τ::AbstractVector{T}) where {T} = QLPackedQ{T,typeof(factors),typeof(τ)}(factors, τ)
function QLPackedQ{T}(factors::AbstractMatrix, τ::AbstractVector) where {T}
    QLPackedQ(convert(AbstractMatrix{T}, factors), convert(AbstractVector{T}, τ))
end

QLPackedQ{T}(Q::QLPackedQ) where {T} = QLPackedQ(convert(AbstractMatrix{T}, Q.factors), convert(AbstractVector{T}, Q.τ))
AbstractMatrix{T}(Q::QLPackedQ{T}) where {T} = Q
AbstractMatrix{T}(Q::QLPackedQ) where {T} = QLPackedQ{T}(Q)

size(Q::QLPackedQ, dim::Integer) = size(getfield(Q, :factors), dim == 2 ? 1 : dim)

size(F::QL, dim::Integer) = size(getfield(F, :factors), dim)
size(F::QL) = size(getfield(F, :factors))


## Multiplication by Q
### QB
function lmul!(A::QLPackedQ, B::AbstractVecOrMat)
    @assert !has_offset_axes(B)
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    @inbounds begin
        for k = 1:min(mA,nA)
            ν = k+nA-min(mA,nA)
            for j = 1:nB
                vBj = B[k,j]
                for i = 1:k-1
                    vBj += conj(Afactors[i,ν])*B[i,j]
                end
                vBj = A.τ[k]*vBj
                B[k,j] -= vBj
                for i = 1:k-1
                    B[i,j] -= Afactors[i,ν]*vBj
                end
            end
        end
    end
    B
end


function lmul!(adjA::Adjoint{<:Any,<:QLPackedQ}, B::AbstractVecOrMat)
    @assert !has_offset_axes(B)
    A = adjA.parent
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    @inbounds begin
        for k = min(mA,nA):-1:1
            ν = k+nA-min(mA,nA)
            for j = 1:nB
                vBj = B[k,j]
                for i = 1:k-1
                    vBj += conj(Afactors[i,ν])*B[i,j]
                end
                vBj = conj(A.τ[k])*vBj
                B[k,j] -= vBj
                for i = 1:k-1
                    B[i,j] -= Afactors[i,ν]*vBj
                end
            end
        end
    end
    B
end


### QBc/QcBc
function rmul!(A::AbstractMatrix,Q::QLPackedQ)
    mQ, nQ = size(Q.factors)
    mA, nA = size(A,1), size(A,2)
    if nA != mQ
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but matrix Q has dimensions ($mQ, $nQ)"))
    end
    Qfactors = Q.factors
    @inbounds begin
        for k = min(mQ,nQ):-1:1
            ν = k+nA-min(mA,nA)
            for i = 1:mA
                vAi = A[i,k]
                for j = 1:k-1
                    vAi += A[i,j]*Qfactors[j,ν]
                end
                vAi = vAi*Q.τ[k]
                A[i,k] -= vAi
                for j = 1:k-1
                    A[i,j] -= vAi*conj(Qfactors[j,ν])
                end
            end
        end
    end
    A
end

### AQc
function rmul!(A::StridedMatrix, adjQ::Adjoint{<:Any,<:QLPackedQ})
    Q = adjQ.parent
    mQ, nQ = size(Q.factors)
    mA, nA = size(A,1), size(A,2)
    if nA != mQ
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but matrix Q has dimensions ($mQ, $nQ)"))
    end
    Qfactors = Q.factors
    @inbounds begin
        for k = 1:min(mQ,nQ)
            ν = k+nA-min(mA,nA)
            for i = 1:mA
                vAi = A[i,k]
                for j = 1:k-1
                    vAi += A[i,j]*Qfactors[j,ν]
                end
                vAi = vAi*conj(Q.τ[k])
                A[i,k] -= vAi
                for j = 1:k-1
                    A[i,j] -= vAi*conj(Qfactors[j,ν])
                end
            end
        end
    end
    A
end

# Julia implementation similar to xgelsy
function ldiv!(A::QL{T}, B::AbstractMatrix{T}) where T
    m, n = size(A)
    minmn = min(m,n)
    mB, nB = size(B)
    lmul!(adjoint(A.Q), view(B, 1:m, :))
    L = A.L
    @inbounds begin
        if n > m # minimum norm solution
            τ = zeros(T,m)
            for k = m:-1:1 # Trapezoid to triangular by elementary operation
                x = view(L, k, [k; m + 1:n])
                τk = reflector!(x)
                τ[k] = conj(τk)
                for i = 1:k - 1
                    vLi = L[i,k]
                    for j = m + 1:n
                        vLi += L[i,j]*x[j - m + 1]'
                    end
                    vLi *= τk
                    L[i,k] -= vLi
                    for j = m + 1:n
                        L[i,j] -= vLi*x[j - m + 1]
                    end
                end
            end
        end
        LinearAlgebra.ldiv!(LowerTriangular(view(L, 1:minmn, :)), view(B, 1:minmn, :))
        if n > m # Apply elementary transformation to solution
            B[m + 1:mB,1:nB] .= zero(T)
            for j = 1:nB
                for k = 1:m
                    vBj = B[k,j]
                    for i = m + 1:n
                        vBj += B[i,j]*L[k,i]'
                    end
                    vBj *= τ[k]
                    B[k,j] -= vBj
                    for i = m + 1:n
                        B[i,j] -= L[k,i]*vBj
                    end
                end
            end
        end
    end
    return B
end
ldiv!(A::QL, B::AbstractVector) = ldiv!(A, reshape(B, length(B), 1))[:]


function (\)(A::QL{TA}, B::AbstractVecOrMat{TB}) where {TA,TB}
    @assert !has_offset_axes(B)
    S = promote_type(TA,TB)
    m, n = size(A)
    m == size(B,1) || throw(DimensionMismatch("left hand side has $m rows, but right hand side has $(size(B,1)) rows"))

    AA = Factorization{S}(A)

    X = _zeros(S, B, n)
    X[1:size(B, 1), :] = B
    ldiv!(AA, X)
    return _cut_B(X, 1:n)
end


function (\)(A::QL{T}, BIn::VecOrMat{Complex{T}}) where T<:BlasReal
    @assert !has_offset_axes(BIn)
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


# Should be in Base, StridedVector -> AbstractVector

function (*)(A::QLPackedQ, b::AbstractVector)
    TAb = promote_type(eltype(A), eltype(b))
    Anew = convert(AbstractMatrix{TAb}, A)
    if size(A.factors, 1) == length(b)
        bnew = copy_oftype(b, TAb)
    elseif size(A.factors, 2) == length(b)
        bnew = [b; zeros(TAb, size(A.factors, 1) - length(b))]
    else
        throw(DimensionMismatch("vector must have length either $(size(A.factors, 1)) or $(size(A.factors, 2))"))
    end
    lmul!(Anew, bnew)
end
function (*)(A::QLPackedQ, B::AbstractMatrix)
    TAB = promote_type(eltype(A), eltype(B))
    Anew = convert(AbstractMatrix{TAB}, A)
    if size(A.factors, 1) == size(B, 1)
        Bnew = copy_oftype(B, TAB)
    elseif size(A.factors, 2) == size(B, 1)
        Bnew = [B; zeros(TAB, size(A.factors, 1) - size(B,1), size(B, 2))]
    else
        throw(DimensionMismatch("first dimension of matrix must have size either $(size(A.factors, 1)) or $(size(A.factors, 2))"))
    end
    lmul!(Anew, Bnew)
end
