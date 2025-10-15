using MatrixFactorizations, LinearAlgebra, Random, ArrayLayouts, Test
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, rmul!, lmul!, require_one_based_indexing, checksquare
using MatrixFactorizations: QRCompactWYQLayout, AbstractQtype

if VERSION < v"1.12-"
    const FieldError = ErrorException
end

struct MyMatrix <: LayoutMatrix{Float64}
    A::Matrix{Float64}
end

Base.elsize(::Type{MyMatrix}) = sizeof(Float64)
Base.getindex(A::MyMatrix, k::Int, j::Int) = A.A[k,j]
Base.setindex!(A::MyMatrix, v, k::Int, j::Int) = setindex!(A.A, v, k, j)
Base.size(A::MyMatrix) = size(A.A)
Base.strides(A::MyMatrix) = strides(A.A)
Base.unsafe_convert(::Type{Ptr{T}}, A::MyMatrix) where T = Base.unsafe_convert(Ptr{T}, A.A)
ArrayLayouts.MemoryLayout(::Type{MyMatrix}) = DenseColumnMajor()

struct MyVector <: LayoutVector{Float64}
    A::Vector{Float64}
end

Base.elsize(::Type{MyVector}) = sizeof(Float64)
Base.getindex(A::MyVector, k::Int) = A.A[k]
Base.setindex!(A::MyVector, v, k::Int) = setindex!(A.A, v, k)
Base.size(A::MyVector) = size(A.A)
Base.strides(A::MyVector) = strides(A.A)
Base.unsafe_convert(::Type{Ptr{T}}, A::MyVector) where T = Base.unsafe_convert(Ptr{T}, A.A)
ArrayLayouts.MemoryLayout(::Type{MyVector}) = DenseColumnMajor()

include("test_ul.jl")
include("test_qr.jl")
include("test_ql.jl")
include("test_rq.jl")
include("test_polar.jl")
include("test_reversecholesky.jl")
include("test_lulinv.jl")

include("test_banded.jl")
