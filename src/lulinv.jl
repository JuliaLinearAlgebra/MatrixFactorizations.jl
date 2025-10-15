"""
    LULinv <: Factorization

Matrix factorization type of the `LUL⁻¹` factorization of a square matrix `A`. This
is the return type of [`lulinv`](@ref), the corresponding matrix factorization function.

The individual components of the factorization `F::LULinv` can be accessed via [`getproperty`](@ref):

| Component | Description                              |
|:----------|:-----------------------------------------|
| `F.L`     | `L` (unit lower triangular) part of `UL` |
| `F.U`     | `U` (upper triangular) part of `UL`      |

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

julia> F.L * F.U / F.L ≈ A
true

julia> l, u = lulinv(A); # destructuring via iteration

julia> l == F.L && u == F.U
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

julia> F.L * F.U / F.L == A
true
```
"""
struct LULinv{T, S <: AbstractMatrix{T}} <: Factorization{T}
    factors::S
    function LULinv{T, S}(factors) where {T, S <: AbstractMatrix{T}}
        require_one_based_indexing(factors)
        new{T, S}(factors)
    end
end


LULinv(factors::AbstractMatrix{T}) where T = LULinv{T, typeof(factors)}(factors)
LULinv{T}(factors::AbstractMatrix) where T = LULinv(convert(AbstractMatrix{T}, factors))

iterate(F::LULinv) = (F.L, Val(:U))
iterate(F::LULinv, ::Val{:U}) = (F.U, Val(:done))
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
    if d === :L
        return getL(F)
    elseif d === :U
        return getU(F)
    else
        getfield(F, d)
    end
end

propertynames(F::LULinv, private::Bool=false) =
    (:L, :U, (private ? fieldnames(typeof(F)) : ())...)

function show(io::IO, mime::MIME{Symbol("text/plain")}, F::LULinv)
    summary(io, F); println(io)
    println(io, "L factor:")
    show(io, mime, F.L)
    println(io, "\nU factor:")
    show(io, mime, F.U)
end


lulinv(A::AbstractMatrix; kwds...) = lulinv(A, eigvals(A); kwds...)
function lulinv(A::AbstractMatrix{T}, λ::AbstractVector{T}; kwds...) where T
    S = lulinvtype(T)
    lulinv!(copy_oftype(A, S), copy_oftype(λ, S); kwds...)
end
function lulinv(A::AbstractMatrix{T1}, λ::AbstractVector{T2}; kwds...) where {T1, T2}
    T = promote_type(T1, T2)
    S = lulinvtype(T)
    lulinv!(copy_oftype(A, S), copy_oftype(λ, S); kwds...)
end

function lulinv!(A::Matrix{T}, λ::Vector{T}; rtol::Real = size(A, 1)*eps(real(float(oneunit(T))))) where T
    n = checksquare(A)
    n == length(λ) || throw(ArgumentError("Eigenvalue count does not match matrix dimensions."))
    v = zeros(T, n)
    for i in 1:n-1
        # We must find an eigenvector with nonzero "first" entry. A failed UL factorization of A-λI reveals this vector provided L₁₁ == 0.
        for j in 1:length(λ)
            F = ul!(view(A, i:n, i:n) - λ[j]*I; check=false)
            nrm = norm(F.L)
            if norm(F.L[1]) ≤ rtol*nrm # v[i] is a free parameter; we set it to 1.
                fill!(v, zero(T))
                v[i] = one(T)
                # Next, we must scan the remaining free parameters, set them to 0, so that we find a nonsingular lower-triangular linear system for the nontrivial remaining part of the eigenvector.
                idx = Int[]
                for k in 2:n+1-i
                    if norm(F.L[k, k]) > rtol*nrm
                        push!(idx, k)
                    end
                end
                v[idx.+(i-1)] .= -F.L[idx, 1]
                ldiv!(LowerTriangular(view(F.L, idx, idx)), view(v, idx.+(i-1)))
                deleteat!(λ, j)
                break
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
    return LULinv(A)
end

function ldiv!(F::LULinv, B::AbstractVecOrMat)
    L = UnitLowerTriangular(F.factors)
    return lmul!(L, ldiv!(UpperTriangular(F.factors), ldiv!(L, B)))
end

function rdiv!(B::AbstractVecOrMat, F::LULinv)
    L = UnitLowerTriangular(F.factors)
    return rdiv!(rdiv!(rmul!(B, L), UpperTriangular(F.factors)), L)
end

det(F::LULinv) = det(UpperTriangular(F.factors))
logabsdet(F::LULinv) = logabsdet(UpperTriangular(F.factors))
