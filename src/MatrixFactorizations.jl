module MatrixFactorizations
using Base, LinearAlgebra, ArrayLayouts
import Base: axes, axes1, getproperty, iterate, tail
import LinearAlgebra: BlasInt, BlasReal, BlasFloat, BlasComplex, axpy!,
                        copy_oftype, checksquare, adjoint, transpose, AdjOrTrans, HermOrSym
import LinearAlgebra.BLAS: libblas
import LinearAlgebra.LAPACK: liblapack, chkuplo, chktrans
import LinearAlgebra: cholesky, cholesky!, norm, diag, eigvals!, eigvals, eigen!, eigen,
            qr, axpy!, ldiv!, mul!, lu, lu!, ldlt, ldlt!, AbstractTriangular,
            chkstride1, kron, lmul!, rmul!, factorize, StructuredMatrixStyle, logabsdet,
            AbstractQ, _zeros, _cut_B, _ret_size, require_one_based_indexing, checksquare

import Base: getindex, setindex!, *, +, -, ==, <, <=, >,
                >=, /, ^, \, transpose, showerror, reindex, checkbounds, @propagate_inbounds

import Base: convert, size, view, unsafe_indices,
                first, last, size, length, unsafe_length, step,
                to_indices, to_index, show, fill!, promote_op,
                MultiplicativeInverses, OneTo, ReshapedArray,
                Array, Matrix, Vector, AbstractArray, AbstractMatrix, AbstractVector,
                               similar, copy, convert, promote_rule, rand,
                            IndexStyle, real, imag, Slice, pointer, unsafe_convert, copyto!

import ArrayLayouts: reflector!, reflectorApply!, materialize!, @layoutmatrix, @_layoutlmul, MemoryLayout, adjointlayout, AbstractQLayout



export ql, ql!, qrunblocked, qrunblocked!, QL, choleskyinv!, choleskyinv

abstract type LayoutQ{T} <: AbstractQ{T} end
@_layoutlmul LayoutQ
@_layoutlmul Adjoint{<:Any,<:LayoutQ}

include("qr.jl")
include("ql.jl")
include("choleskyinv.jl")

end #module
