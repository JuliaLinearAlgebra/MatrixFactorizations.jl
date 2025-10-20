"""
    LULinv <: Factorization

Matrix factorization type of the `LUL⁻¹` factorization of a square matrix `A`. This
is the return type of [`lulinv`](@ref), the corresponding matrix factorization function.

The individual components of the factorization `F::LULinv` can be accessed via [`getproperty`](@ref):

| Component | Description                                 |
|:----------|:--------------------------------------------|
| `F.L`     | `L` (unit lower triangular) part of `LUL⁻¹` |
| `F.U`     | `U` (upper triangular) part of `LUL⁻¹`      |
| `F.p`     | (right) permutation `Vector`                |
| `F.P`     | (right) permutation `Matrix`                |

Iterating the factorization produces the components `F.L` and `F.U`.

# Examples
```jldoctest
julia> A = [4 3; 6 3]
2×2 Array{Int64,2}:
 4  3
 6  3

julia> F = lulinv(A)
LULinv{Float64, Matrix{Float64}}
L factor:
2×2 Matrix{Float64}:
  1.0      0.0
 -1.59067  1.0
U factor:
2×2 Matrix{Float64}:
 -0.772002  3.0
  0.0       7.772

julia> F.L * F.U / F.L ≈ A[F.p, F.p]
true

julia> l, u, p = lulinv(A); # destructuring via iteration

julia> l == F.L && u == F.U && p == F.p
true

julia> A = [-150 334 778; -89 195 464; 5 -10 -27]
3×3 Matrix{Int64}:
 -150  334  778
  -89  195  464
    5  -10  -27

julia> F = lulinv(A, [17, -2, 3//1]) # can input rational eigenvalues directly
LULinv{Rational{Int64}, Matrix{Rational{Int64}}}
L factor:
3×3 Matrix{Rational{Int64}}:
  1      0    0
 1//2    1    0
  0    -2//5  1
U factor:
3×3 Matrix{Rational{Int64}}:
 17  114//5  778
  0    -2     75
  0     0      3

julia> F.L * F.U / F.L == A[F.p, F.p]
true
```
"""
struct LULinv{T, S <: AbstractMatrix{T}, IPIV <: AbstractVector{<:Integer}} <: Factorization{T}
    factors::S
    ipiv::IPIV
    function LULinv{T, S, IPIV}(factors, ipiv) where {T, S <: AbstractMatrix{T}, IPIV <: AbstractVector{<:Integer}}
        require_one_based_indexing(factors)
        new{T, S, IPIV}(factors, ipiv)
    end
end


LULinv(factors::AbstractMatrix{T}, ipiv::AbstractVector{<:Integer}) where T = LULinv{T, typeof(factors), typeof(ipiv)}(factors, ipiv)
LULinv{T}(factors::AbstractMatrix, ipiv::AbstractVector{<:Integer}) where T = LULinv(convert(AbstractMatrix{T}, factors), ipiv)
LULinv{T}(F::LULinv) where T = LULinv{T}(F.factors, F.ipiv)

iterate(F::LULinv) = (F.L, Val(:U))
iterate(F::LULinv, ::Val{:U}) = (F.U, Val(:p))
iterate(F::LULinv, ::Val{:p}) = (F.p, Val(:done))
iterate(F::LULinv, ::Val{:done}) = nothing


function lulinvtype(T::Type)
    # In generic_ulfact!, the elements of the lower part of the matrix are
    # obtained using the division of two matrix elements. Hence their type can
    # be different (e.g. the division of two types with the same unit is a type
    # without unit).
    # The elements of the upper part are obtained by U - L * U / L
    # where U is an upper part element and L is a lower part element.
    # Therefore, the types LT, UT should be invariant under the map:
    # (LT, UT) -> begin
    #     L = oneunit(UT) / oneunit(UT)
    #     U = oneunit(UT) - L * oneunit(UT) / L
    #     typeof(L), typeof(U)
    # end
    # The following should handle most cases
    UT = typeof(oneunit(T) - (oneunit(T) / (oneunit(T) + zero(T)) * oneunit(T) * (oneunit(T) + zero(T)) / oneunit(T)))
    LT = typeof(oneunit(UT) / oneunit(UT))
    S = promote_type(T, LT, UT)
end


size(A::LULinv)    = size(getfield(A, :factors))
size(A::LULinv, i) = size(getfield(A, :factors), i)

function getL(F::LULinv{T}) where T
    n = size(F.factors, 1)
    L = tril!(getindex(getfield(F, :factors), 1:n, 1:n))
    for i in 1:n L[i, i] = one(T) end
    return L
end

function getU(F::LULinv)
    n = size(F.factors, 1)
    triu!(getindex(getfield(F, :factors), 1:n, 1:n))
end

function getproperty(F::LULinv{T, <: AbstractMatrix}, d::Symbol) where T
    n = size(F, 1)
    if d === :L
        return getL(F)
    elseif d === :U
        return getU(F)
    elseif d === :p
        return getfield(F, :ipiv)
    elseif d === :P
        return Matrix{T}(I, n, n)[:, F.p]
    else
        getfield(F, d)
    end
end

propertynames(F::LULinv, private::Bool=false) =
    (:L, :U, :p, :P, (private ? fieldnames(typeof(F)) : ())...)

function show(io::IO, mime::MIME{Symbol("text/plain")}, F::LULinv)
    summary(io, F); println(io)
    println(io, "L factor:")
    show(io, mime, F.L)
    println(io, "\nU factor:")
    show(io, mime, F.U)
end


lulinv(A::AbstractMatrix, pivot::Union{Val{false}, Val{true}}=Val(true); kwds...) = lulinv(A, eigvals(A), pivot; kwds...)
function lulinv(A::AbstractMatrix{T}, λ::AbstractVector{T}, pivot::Union{Val{false}, Val{true}}=Val(true); kwds...) where T
    S = lulinvtype(T)
    lulinv!(copy_oftype(A, S), copy_oftype(λ, S), pivot; kwds...)
end
function lulinv(A::AbstractMatrix{T1}, λ::AbstractVector{T2}, pivot::Union{Val{false}, Val{true}}=Val(true); kwds...) where {T1, T2}
    T = promote_type(T1, T2)
    S = lulinvtype(T)
    lulinv!(copy_oftype(A, S), copy_oftype(λ, S), pivot; kwds...)
end

function lulinv!(A::Matrix{T}, λ::Vector{T}, ::Val{Pivot} = Val(true); rtol::Real = size(A, 1)*eps(real(float(oneunit(T))))) where {T, Pivot}
    n = checksquare(A)
    n == length(λ) || throw(ArgumentError("Eigenvalue count does not match matrix dimensions."))
    v = zeros(T, n)
    ipiv = collect(1:n)
    for i in 1:n
        F = ul!(view(A, i:n, i:n) - λ[i]*I; check = false)
        nrm = norm(F.L)
        if Pivot
            ip = n+1-i
            while (ip > 1) && (norm(F.L[ip, ip]) > rtol*nrm)
                ip -= 1
            end
            fill!(v, zero(T))
            v[i] = one(T)
            idx = ip+1:n+1-i
            if !isempty(idx)
                v[idx.+(i-1)] .= -F.L[idx, ip]
                ldiv!(LowerTriangular(view(F.L, idx, idx)), view(v, idx.+(i-1)))
            end
            ip = ip+i-1
            if ip ≠ i
                tmp = ipiv[i]
                ipiv[i] = ipiv[ip]
                ipiv[ip] = tmp
                for j in 1:n
                    tmp = A[i, j]
                    A[i, j] = A[ip, j]
                    A[ip, j] = tmp
                end
                for j in 1:n
                    tmp = A[j, i]
                    A[j, i] = A[j, ip]
                    A[j, ip] = tmp
                end
            end
        else
            fill!(v, zero(T))
            v[i] = one(T)
            idx = 2:n+1-i
            if !isempty(idx)
                v[idx.+(i-1)] .= -F.L[idx, 1]
                ldiv!(LowerTriangular(view(F.L, idx, idx)), view(v, idx.+(i-1)))
            end
        end
        for k in 1:n
            for j in i+1:n
                A[k, i] += A[k, j]*v[j]
            end
        end
        for j in i:n
            for k in i+1:n
                A[k, j] -= A[i, j]*v[k]
            end
        end
        for k in i+1:n
            A[k, i] = v[k]
        end
    end
    return LULinv(A, ipiv)
end

function ldiv!(A::LULinv, B::AbstractVecOrMat)
    B .= B[invperm(A.p), :] # Todo: fix me
    L = UnitLowerTriangular(A.factors)
    lmul!(L, ldiv!(UpperTriangular(A.factors), ldiv!(L, B)))
    B .= B[A.p, :] # Todo: fix me
    return B
end

function rdiv!(A::AbstractVecOrMat, B::LULinv)
    A .= A[:, B.p] # Todo: fix me
    L = UnitLowerTriangular(B.factors)
    rdiv!(rdiv!(rmul!(A, L), UpperTriangular(B.factors)), L)
    A .= A[:, invperm(B.p)] # Todo: fix me
    return A
end

det(F::LULinv) = det(UpperTriangular(F.factors))
logabsdet(F::LULinv) = logabsdet(UpperTriangular(F.factors))
