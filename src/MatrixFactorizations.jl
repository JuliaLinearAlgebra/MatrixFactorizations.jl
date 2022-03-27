module MatrixFactorizations
using Base, LinearAlgebra, ArrayLayouts
import Base: axes, axes1, getproperty, iterate, tail
import LinearAlgebra: BlasInt, BlasReal, BlasFloat, BlasComplex, axpy!,
                        copy_oftype, checksquare, adjoint, transpose, AdjOrTrans, HermOrSym
import LinearAlgebra.BLAS: libblas
import LinearAlgebra.LAPACK: liblapack, chkuplo, chktrans
import LinearAlgebra: cholesky, cholesky!, norm, diag, eigvals!, eigvals, eigen!, eigen,
            qr, axpy!, ldiv!, rdiv!, mul!, lu, lu!, ldlt, ldlt!, AbstractTriangular, inv,
            chkstride1, kron, lmul!, rmul!, factorize, StructuredMatrixStyle, det, logabsdet,
            AbstractQ, _zeros, _cut_B, _ret_size, require_one_based_indexing, checksquare,
            checknonsingular, ipiv2perm, copytri!, issuccess

import Base: getindex, setindex!, *, +, -, ==, <, <=, >,
                >=, /, ^, \, transpose, showerror, reindex, checkbounds, @propagate_inbounds

import Base: convert, size, view, unsafe_indices,
                first, last, size, length, unsafe_length, step,
                to_indices, to_index, show, fill!, promote_op,
                MultiplicativeInverses, OneTo, ReshapedArray,
                Array, Matrix, Vector, AbstractArray, AbstractMatrix, AbstractVector,
                               similar, copy, convert, promote_rule, rand,
                            IndexStyle, real, imag, Slice, pointer, unsafe_convert, copyto!

import ArrayLayouts: reflector!, reflectorApply!, materialize!, @_layoutlmul, @_layoutrmul, 
                     MemoryLayout, adjointlayout, AbstractQLayout, QRPackedQLayout,
                     QRCompactWYQLayout, AdjQRCompactWYQLayout, QRPackedLayout, AdjQRPackedQLayout



export ul, ul!, ql, ql!, qrunblocked, qrunblocked!, UL, QL, choleskyinv!, choleskyinv

abstract type LayoutQ{T} <: AbstractQ{T} end
@_layoutlmul LayoutQ
@_layoutlmul Adjoint{<:Any,<:LayoutQ}
@_layoutrmul LayoutQ
@_layoutrmul Adjoint{<:Any,<:LayoutQ}

axes(Q::LayoutQ, dim::Integer) = axes(getfield(Q, :factors), dim == 2 ? 1 : dim)
axes(Q::LayoutQ) = axes(Q, 1), axes(Q, 2)

size(Q::LayoutQ, dim::Integer) = size(getfield(Q, :factors), dim == 2 ? 1 : dim)
size(Q::LayoutQ) = size(Q, 1), size(Q, 2)

include("ul.jl")
include("qr.jl")
include("ql.jl")
include("rq.jl")
include("choleskyinv.jl")
include("polar.jl")

end #module
