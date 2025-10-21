"""
    Jordan <: Factorization

Jordan canonical form of a square matrix `A = VJV⁻¹`. This
is the return type of [`jordan`](@ref), the corresponding Jordan factorization function.

The individual components of the factorization `F::Jordan` can be accessed via [`getfield`](@ref):

| Component | Description                  |
|:----------|:-----------------------------|
| `F.V`     | `V` generalized eigenvectors |
| `F.J`     | `J` Jordan normal form       |

Iterating the factorization produces the components `F.V` and `F.J`.

# Examples
```jldoctest
julia> A = [5 4 2 1; 0 1 -1 -1; -1 -1 3 0; 1 1 -1 2//1]

julia> λ = [1, 2, 4, 4//1] # you will almost certainly need extremely accurate eigenvalues to proceed

julia> F = jordan(A, λ)
Jordan{Rational{Int64}, Matrix{Rational{Int64}}, Matrix{Rational{Int64}}}
Generalized eigenvectors:
4×4 Matrix{Rational{Int64}}:
  1   1  -1  -1
 -1  -1   0   0
  0   0   1   0
  0   1  -1   0
Jordan normal form:
4×4 Matrix{Rational{Int64}}:
 1  0  0  0
 0  2  0  0
 0  0  4  1
 0  0  0  4

julia> A*F.V == F.V*F.J
true

julia> V, J = jordan(A, λ); # destructuring via iteration

julia> V == F.V && J == F.J
true
```
"""
struct Jordan{T, R <: AbstractMatrix{T}, S <: AbstractMatrix{T}} <: Factorization{T}
    V::R
    J::S
end

Jordan{T}(V::AbstractMatrix, J::AbstractMatrix) where T = Jordan(convert(AbstractMatrix{T}, V), convert(AbstractMatrix{T}, J))
Jordan{T}(F::Jordan) where T = Jordan{T}(F.V, F.J)

iterate(F::Jordan) = (F.V, Val(:J))
iterate(F::Jordan, ::Val{:J}) = (F.J, Val(:done))
iterate(F::Jordan, ::Val{:done}) = nothing

function show(io::IO, mime::MIME{Symbol("text/plain")}, F::Jordan)
    summary(io, F); println(io)
    println(io, "Generalized eigenvectors:")
    show(io, mime, F.V)
    println(io, "\nJordan normal form:")
    show(io, mime, F.J)
end

# dangerous because Jordan blocks are unstable with respect to small perturbations
jordan(A::AbstractMatrix) = jordan(A, eigvals(A))

function jordan(A::AbstractMatrix{S}, λ::AbstractVector{T}) where {S, T}
    V = promote_type(S, T)
    jordan(convert(AbstractMatrix{V}, A), convert(AbstractVector{V}, λ))
end

function jordan(A::AbstractMatrix{T}, λ::AbstractVector{T}) where T
    PLEP, B = block_diagonalize(A, λ)
    F, J = block_diagonal_to_jordan(B)
    V = PLEP*F
    return Jordan(V, J)
end

function triangular_to_psychologically_block_diagonal(U::UpperTriangular{T, <: AbstractMatrix{T}}) where T
    n = checksquare(U)
    R = deepcopy(U)
    EACC = UpperTriangular(Matrix{T}(I, n, n))
    # note that this is sensitive to the order of conjugation.
    for j in 2:n
        for i in j-1:-1:1
            if R[i, i] != R[j, j]
                E = UpperTriangular(Matrix{T}(I, n, n))
                E[i, j] = R[i, j]/(R[j, j] - R[i, i])
                R = E\R*E
                EACC = EACC*E
            end
        end
    end
    return EACC, R
end

function psychologically_block_diagonal_to_block_diagonal(R::UpperTriangular{T, <: AbstractMatrix{T}}) where T
    p = sortperm(diag(R))
    n = length(p)
    P = Matrix{Int}(I, n, n)[:, p]
    B = R[p, p]
    P, B
end

function triangular_to_block_diagonal(U::UpperTriangular{T, <: AbstractMatrix{T}}) where T
    E, R = triangular_to_psychologically_block_diagonal(U)
    p = sortperm(diag(R))
    E[:, p], R[p, p]
end

function block_diagonalize(A::AbstractMatrix{T}, λ::AbstractVector{T}) where T
    F = lulinv(A, λ)
    PLEP, B = triangular_to_block_diagonal(UpperTriangular(F.factors))
    lmul!(UnitLowerTriangular(F.factors), PLEP)
    F.P*PLEP, B
end

function determine_block_sizes(B::AbstractMatrix{T}) where T
    n = checksquare(B)
    if n == 1
        m = [1]
        return m
    else
        m = Int[]
        i = 1
        while i < n
            t = 1
            while (i < n-1) && (B[i, i] == B[i+1, i+1])
                i += 1
                t += 1
            end
            if i == n-1
                if B[i, i] == B[i+1, i+1]
                    t += 1
                    push!(m, t)
                else
                    push!(m, t)
                    push!(m, 1)
                end
            else
                push!(m, t)
            end
            i += 1
        end
        return m
    end
end

function block_diagonal_to_jordan(B::Matrix{T}) where T
    n = checksquare(B)
    m = determine_block_sizes(B)
    cm = cumsum(m)
    pushfirst!(cm, 0)
    F = zeros(T, n, n)
    J = zeros(T, n, n)
    for i in 1:length(m)
        ir = cm[i]+1:cm[i+1]
        FB, JB = upper_triangular_block_to_jordan_blocks(B[ir, ir])
        F[ir, ir] .= FB
        J[ir, ir] .= JB
    end
    return F, J
end

# Contract: B_{i, j} = 0 for i > j. B_{i, i} = B_{j, j} for all i ≠ j.
function upper_triangular_block_to_jordan_blocks(B::Matrix{T}) where T
    n = checksquare(B)
    J = deepcopy(B)
    FACC = Matrix{T}(I, n, n)
    for j in 2:n
        # In column j, we shall introduce zeros in any row with a 1 in the {i, i+1} position.
        F = Matrix{T}(I, n, n)
        for i in 1:j-2
            if !iszero(J[i, i+1])
                F[i+1, j] = -J[i, j]
            end
        end
        J = F\J*F
        FACC = FACC*F
        while count(!iszero, J[1:j-1, j]) > 1
            # Next, we identify the first and next nonzeros in the last column. By the first step, they are across from the last row of a Jordan block.
            i1 = 1
            while i1 < j
                if !iszero(J[i1, j])
                    break
                else
                    i1 += 1
                end
            end
            i2 = i1+1
            while i2 < j
                if !iszero(J[i2, j])
                    break
                else
                    i2 += 1
                end
            end
            # With i1 and i2, we find the sizes of the corresponding Jordan blocks, s and t.
            i1s = i1
            while i1s > 1
                if iszero(J[i1s-1, i1s])
                    break
                else
                    i1s -= 1
                end
            end
            i2s = i2
            while i2s > 1
                if iszero(J[i2s-1, i2s])
                    break
                else
                    i2s -= 1
                end
            end
            i1r = i1s:i1
            i2r = i2s:i2
            s = length(i1r)
            t = length(i2r)
            # Eliminate one of the nonzeros or the other.
            if s ≤ t
                # eliminate α
                F = Matrix{T}(I, n, n)
                α = J[i1, j]
                β = J[i2, j]
                γ = α/β
                F[i1r, i2-s+1:i2] .= Matrix{T}(γ*I, s, s)
                J = F\J*F
                FACC = FACC*F
            else
                # eliminate β
                F = Matrix{T}(I, n, n)
                α = J[i1, j]
                β = J[i2, j]
                γ = β/α
                F[i2r, i1-t+1:i1] .= Matrix{T}(γ*I, t, t)
                J = F\J*F
                FACC = FACC*F
            end
        end
        # Next, we must permute to get the final nonzero to the bottom, if there is one at all.
        i1 = 1
        while i1 < j
            if !iszero(J[i1, j])
                break
            else
                i1 += 1
            end
        end
        if i1 < j-1
            # With i1, we find the size of the corresponding Jordan block, s.
            i1s = i1
            while i1s > 1
                if iszero(J[i1s-1, i1s])
                    break
                else
                    i1s -= 1
                end
            end
            i1r = i1s:i1
            s = length(i1r)
            p = [1:i1s-1; i1+1:j-1; i1r; j:n]
            #P = Matrix{T}(I, n, n)[:, p]
            #J = P'J*P
            #FACC = FACC*P
            J = J[p, p]
            FACC = FACC[:, p]
        end
        # Finally, we must scale off the nonzero entry to 1 to get it to conform to a Jordan block.
        if !iszero(J[j-1, j])
            F = Matrix{T}(I, n, n)
            F[j, j] = inv(J[j-1, j])
            J = F\J*F
            FACC = FACC*F
        end
    end
    return FACC, J
end
