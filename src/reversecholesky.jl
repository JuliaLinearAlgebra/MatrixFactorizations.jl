# This file was modified from
# a part of Julia. License is MIT: https://julialang.org/license

##########################
# Reverse Cholesky Factorization #
##########################

"""
    ReverseCholesky <: Factorization

Matrix factorization type of the reverse Cholesky factorization of a dense symmetric/Hermitian
positive definite matrix `A`. This is the return type of [`reversecholesky`](@ref),
the corresponding matrix factorization function.

The triangular reverse Cholesky factor can be obtained from the factorization `F::ReverseCholesky`
via `F.L` and `F.U`, where `A ≈ F.U * F.U' ≈ F.L' * F.L`.
```
"""
struct ReverseCholesky{T,S<:AbstractMatrix} <: Factorization{T}
    factors::S
    uplo::Char
    info::BlasInt

    function ReverseCholesky{T,S}(factors, uplo, info) where {T,S<:AbstractMatrix}
        require_one_based_indexing(factors)
        new(factors, uplo, info)
    end
end
ReverseCholesky(A::AbstractMatrix{T}, uplo::Symbol, info::Integer) where {T} =
    ReverseCholesky{T,typeof(A)}(A, char_uplo(uplo), info)
ReverseCholesky(A::AbstractMatrix{T}, uplo::AbstractChar, info::Integer) where {T} =
    ReverseCholesky{T,typeof(A)}(A, uplo, info)
ReverseCholesky(U::UpperTriangular{T}) where {T} = ReverseCholesky{T,typeof(U.data)}(U.data, 'U', 0)
ReverseCholesky(L::LowerTriangular{T}) where {T} = ReverseCholesky{T,typeof(L.data)}(L.data, 'L', 0)

# iteration for destructuring into components
Base.iterate(C::ReverseCholesky) = (C.U, Val(:U))
Base.iterate(C::ReverseCholesky, ::Val{:U}) = (C.L, Val(:done))
Base.iterate(C::ReverseCholesky, ::Val{:done}) = nothing

## Non BLAS/LAPACK element types (generic)
function _reverse_chol!(A::AbstractMatrix, ::Type{UpperTriangular})
    require_one_based_indexing(A)
    n = checksquare(A)
    realdiag = eltype(A) <: Complex
    @inbounds begin
        for k = n:-1:1
            cs = colsupport(A, k)
            rs = rowsupport(A, k)
            Akk = realdiag ? real(A[k,k]) : A[k,k]
            for j = (k+1:n) ∩ rs
                Akk -= realdiag ? abs2(A[k,j]) : A[k,j]'A[k,j]
            end
            A[k,k] = Akk
            Akk, info = _reverse_chol!(Akk, UpperTriangular)
            if info != 0
                return UpperTriangular(A), info
            end
            A[k,k] = Akk
            AkkInv = inv(Akk)
            for j = (k+1:n) ∩ rs
                @simd for i = (1:k-1) ∩ cs ∩ colsupport(A,j)
                    A[i,k] -= A[i,j]*A[k,j]'
                end
            end
            for i = (1:k-1) ∩ cs
                A[i,k] *= AkkInv'
            end
        end
    end
    return UpperTriangular(A), convert(BlasInt, 0)
end

function _reverse_chol!(A::AbstractMatrix, ::Type{LowerTriangular})
    require_one_based_indexing(A)
    n = checksquare(A)
    realdiag = eltype(A) <: Complex
    @inbounds begin
        for k = n:-1:1
            cs = colsupport(A, k)
            rs = rowsupport(A, k)
            Akk = realdiag ? real(A[k,k]) : A[k,k]
            for i = (k+1:n) ∩ cs
                Akk -= realdiag ? abs2(A[i,k]) : A[i,k]'A[i,k]
            end
            A[k,k] = Akk
            Akk, info = _reverse_chol!(Akk, LowerTriangular)
            if info != 0
                return LowerTriangular(A), info
            end
            A[k,k] = Akk
            AkkInv = inv(copy(Akk'))
            for j = (1:k-1) ∩ rs # colsupport == rowsupport
                for i = (k+1:n) ∩ cs ∩ colsupport(A,j)
                    A[k,j] -= A[i,k]'*A[i,j]
                end
                A[k,j] = AkkInv*A[k,j]
            end
        end
    end
    return UpperTriangular(A), convert(BlasInt, 0)
end

## Numbers
_reverse_chol!(x::Number, uplo) = LinearAlgebra._chol!(x, uplo)

## for StridedMatrices, check that matrix is symmetric/Hermitian

# cholesky!. Destructive methods for computing Cholesky factorization of real symmetric
# or Hermitian matrix
## No pivoting (default)
function reversecholesky!(A::RealHermSymComplexHerm, ::NoPivot = NoPivot(); check::Bool = true)
    C, info = _reverse_chol!(A.data, A.uplo == 'U' ? UpperTriangular : LowerTriangular)
    check && checkpositivedefinite(info)
    return ReverseCholesky(C.data, A.uplo, info)
end

### for AbstractMatrix, check that matrix is symmetric/Hermitian
"""
    reversecholesky!(A::AbstractMatrix, NoPivot(); check = true) -> ReverseCholesky

The same as [`reversecholesky`](@ref), but saves space by overwriting the input `A`,
instead of creating a copy. An [`InexactError`](@ref) exception is thrown if
the factorization produces a number not representable by the element type of
`A`, e.g. for integer types.
```
"""
function reversecholesky!(A::AbstractMatrix, ::NoPivot = NoPivot(); check::Bool = true)
    checksquare(A)
    if !ishermitian(A) # return with info = -1 if not Hermitian
        check && checkpositivedefinite(-1)
        return ReverseCholesky(A, 'U', convert(BlasInt, -1))
    else
        return reversecholesky!(Hermitian(A), NoPivot(); check = check)
    end
end

reversecholcopy(A) = cholcopy(A)
function reversecholcopy(A::SymTridiagonal)
    T = LinearAlgebra.choltype(A)
    Symmetric(Bidiagonal(AbstractVector{T}(A.dv), AbstractVector{T}(A.ev), :U))
end

function reversecholcopy(A::Symmetric{<:Any,<:Tridiagonal})
    T = LinearAlgebra.choltype(A)
    if A.uplo == 'U'
        Symmetric(Bidiagonal(AbstractVector{T}(parent(A).d), AbstractVector{T}(parent(A).du), :U))
    else
        Symmetric(Bidiagonal(AbstractVector{T}(parent(A).d), AbstractVector{T}(parent(A).dl), :L), :L)
    end
end

_copyifsametype(::Type{T}, A::AbstractMatrix{T}) where T = copy(A)
_copyifsametype(_, A) = A

function reversecholcopy(A::Symmetric{<:Any,<:Bidiagonal})
    T = LinearAlgebra.choltype(A)
    B = _copyifsametype(T, AbstractMatrix{T}(parent(A)))
    Symmetric{T,typeof(B)}(B, A.uplo)
end

# reversecholesky. Non-destructive methods for computing ReverseCholesky factorization of real symmetric
# or Hermitian matrix
## No pivoting (default)
"""
    reversecholesky(A, NoPivot(); check = true) -> ReverseCholesky

Compute the ReverseCholesky factorization of a dense symmetric positive definite matrix `A`
and return a [`ReverseCholesky`](@ref) factorization. The matrix `A` can either be a [`Symmetric`](@ref) or [`Hermitian`](@ref)
[`AbstractMatrix`](@ref) or a *perfectly* symmetric or Hermitian `AbstractMatrix`.

The triangular ReverseCholesky factor can be obtained from the factorization `F` via `F.L` and `F.U`,
where `A ≈ F.U * F.U' ≈ F.L' * F.L`.
"""
reversecholesky(A::AbstractMatrix, ::NoPivot=NoPivot(); check::Bool = true) =
    reversecholesky!(reversecholcopy(A); check)

function reversecholesky(A::AbstractMatrix{Float16}, ::NoPivot=NoPivot(); check::Bool = true)
    X = reversecholesky!(reversecholcopy(A); check = check)
    return ReverseCholesky{Float16}(X)
end


## Number
function reversecholesky(x::Number, uplo::Symbol=:U)
    C, info = _reverse_chol!(x, uplo)
    xf = fill(C, 1, 1)
    ReverseCholesky(xf, uplo, info)
end


function ReverseCholesky{T}(C::ReverseCholesky) where T
    Cnew = convert(AbstractMatrix{T}, C.factors)
    ReverseCholesky{T, typeof(Cnew)}(Cnew, C.uplo, C.info)
end
Factorization{T}(C::ReverseCholesky{T}) where {T} = C
Factorization{T}(C::ReverseCholesky) where {T} = ReverseCholesky{T}(C)

AbstractMatrix(C::ReverseCholesky) = C.uplo == 'U' ? C.U*C.U' : C.L'*C.L
AbstractArray(C::ReverseCholesky) = AbstractMatrix(C)
Matrix(C::ReverseCholesky) = Array(AbstractArray(C))
Array(C::ReverseCholesky) = Matrix(C)

copy(C::ReverseCholesky) = ReverseCholesky(copy(C.factors), C.uplo, C.info)


size(C::ReverseCholesky) = size(C.factors)
size(C::ReverseCholesky, d::Integer) = size(C.factors, d)

function getproperty(C::ReverseCholesky, d::Symbol)
    Cfactors = getfield(C, :factors)
    Cuplo    = getfield(C, :uplo)
    if d === :U
        return UpperTriangular(Cuplo === char_uplo(d) ? Cfactors : copy(Cfactors'))
    elseif d === :L
        return LowerTriangular(Cuplo === char_uplo(d) ? Cfactors : copy(Cfactors'))
    elseif d === :UL
        return (Cuplo === 'U' ? UpperTriangular(Cfactors) : LowerTriangular(Cfactors))
    else
        return getfield(C, d)
    end
end
Base.propertynames(F::ReverseCholesky, private::Bool=false) =
    (:U, :L, :UL, (private ? fieldnames(typeof(F)) : ())...)



issuccess(C::ReverseCholesky) = C.info == 0

adjoint(C::ReverseCholesky) = C

function show(io::IO, mime::MIME{Symbol("text/plain")}, C::ReverseCholesky)
    if issuccess(C)
        summary(io, C); println(io)
        println(io, "$(C.uplo) factor:")
        show(io, mime, C.UL)
    else
        print(io, "Failed factorization of type $(typeof(C))")
    end
end

function ldiv!(C::ReverseCholesky, B::AbstractVecOrMat)
    if C.uplo == 'L'
        return ldiv!(LowerTriangular(C.factors), ldiv!(adjoint(LowerTriangular(C.factors)), B))
    else
        return ldiv!(adjoint(UpperTriangular(C.factors)), ldiv!(UpperTriangular(C.factors), B))
    end
end

function rdiv!(B::AbstractMatrix, C::ReverseCholesky)
    if C.uplo == 'L'
        return rdiv!(rdiv!(B, LowerTriangular(C.factors)), adjoint(LowerTriangular(C.factors)))
    else
        return rdiv!(rdiv!(B, adjoint(UpperTriangular(C.factors))), UpperTriangular(C.factors))
    end
end

isposdef(C::ReverseCholesky) = C.info == 0

function det(C::ReverseCholesky)
    dd = one(real(eltype(C)))
    @inbounds for i in 1:size(C.factors,1)
        dd *= real(C.factors[i,i])^2
    end
    return dd
end

function logdet(C::ReverseCholesky)
    dd = zero(real(eltype(C)))
    @inbounds for i in 1:size(C.factors,1)
        dd += log(real(C.factors[i,i]))
    end
    dd + dd # instead of 2.0dd which can change the type
end



logabsdet(C::ReverseCholesky) = logdet(C), one(eltype(C)) # since C is p.s.d.


function getproperty(C::ReverseCholesky{<:Any,<:Diagonal}, d::Symbol)
    Cfactors = getfield(C, :factors)
    if d in (:U, :L, :UL)
        return Cfactors
    else
        return getfield(C, d)
    end
end

function getproperty(C::ReverseCholesky{<:Any, <:Bidiagonal}, d::Symbol)
    Cfactors = getfield(C, :factors)
    Cuplo    = getfield(C, :uplo)
    if d === :U && Cfactors.uplo === Cuplo === 'U'
        return Cfactors
    elseif d === :L && Cfactors.uplo === Cuplo === 'U'
        return Cfactors'
    elseif d === :U && Cfactors.uplo === Cuplo === 'L'
            return Cfactors'
        elseif d === :L && Cfactors.uplo === Cuplo === 'L'
            return Cfactors
    elseif d === :UL
        return Cfactors
    else
        return getfield(C, d)
    end
end

inv(C::ReverseCholesky{<:Any,<:Diagonal}) = Diagonal(map(inv∘abs2, C.factors.diag))