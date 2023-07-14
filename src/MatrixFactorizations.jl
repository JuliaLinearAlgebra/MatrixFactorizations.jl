module MatrixFactorizations
using Base, LinearAlgebra, ArrayLayouts
import Base: axes, axes1, getproperty, iterate, tail
import LinearAlgebra: BlasInt, BlasReal, BlasFloat, BlasComplex, axpy!,
   copy_oftype, checksquare, adjoint, transpose, AdjOrTrans, HermOrSym,
   det, logdet, logabsdet, isposdef
import LinearAlgebra.LAPACK: chkuplo, chktrans
import LinearAlgebra: cholesky, cholesky!, norm, diag, eigvals!, eigvals, eigen!, eigen,
   qr, axpy!, ldiv!, rdiv!, mul!, lu, lu!, ldlt, ldlt!, AbstractTriangular, inv,
   chkstride1, kron, lmul!, rmul!, factorize, StructuredMatrixStyle, det, logabsdet,
   AbstractQ, _zeros, _cut_B, _ret_size, require_one_based_indexing, checksquare,
   checknonsingular, ipiv2perm, copytri!, issuccess, RealHermSymComplexHerm,
   cholcopy, checkpositivedefinite, char_uplo, copymutable_oftype

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
   QRCompactWYQLayout, AdjQRCompactWYQLayout, QRPackedLayout, AdjQRPackedQLayout,
   layout_getindex


export ul, ul!, ql, ql!, qrunblocked, qrunblocked!, UL, QL, reversecholesky, reversecholesky!, ReverseCholesky


const AdjointQtype = isdefined(LinearAlgebra, :AdjointQ) ? LinearAlgebra.AdjointQ : Adjoint
const AbstractQtype = AbstractQ <: AbstractMatrix ? AbstractMatrix : AbstractQ

const AdjointFact = isdefined(LinearAlgebra, :AdjointFactorization) ? LinearAlgebra.AdjointFactorization : Adjoint
const TransposeFact = isdefined(LinearAlgebra, :TransposeFactorization) ? LinearAlgebra.TransposeFactorization : Transpose

# The abstract type LayoutQ implicitly assumes that any subtype admits a field
# named factors. Based on this field, `size`, `axes` and context-dependent
# multiplication work. The same used to be the case before v1.9 with the even
# more generic LinearAlgebra.AbstractQ. Moreover, it is assumed that LayoutQ
# objects are flexible in size when multiplied from the left, or its adjoint
# from the right.
abstract type LayoutQ{T} <: AbstractQ{T} end
@_layoutlmul LayoutQ
@_layoutlmul AdjointQtype{<:Any,<:LayoutQ}
@_layoutrmul LayoutQ
@_layoutrmul AdjointQtype{<:Any,<:LayoutQ}

(*)(Q::LayoutQ, b::AbstractVector) = _mul(Q, b)
(*)(Q::LayoutQ, b::LayoutVector) = ArrayLayouts.mul(Q, b) # disambiguation w/ ArrayLayouts.jl
function _mul(Q::LayoutQ, b::AbstractVector)
   T = promote_type(eltype(Q), eltype(b))
   if size(Q.factors, 1) == length(b)
       bnew = copyto!(similar(b, T, size(b)), b)
   elseif size(Q.factors, 2) == length(b)
       bnew = [b; zeros(T, size(Q.factors, 1) - length(b))]
   else
       throw(DimensionMismatch("vector must have length either $(size(Q.factors, 1)) or $(size(Q.factors, 2))"))
   end
   lmul!(convert(AbstractQtype{T}, Q), bnew)
end
(*)(Q::LayoutQ, B::AbstractMatrix) = _mul(Q, B)
(*)(Q::LayoutQ, B::LayoutMatrix) = ArrayLayouts.mul(Q, B) # disambiguation w/ ArrayLayouts.jl
function _mul(Q::LayoutQ, B::AbstractMatrix)
   T = promote_type(eltype(Q), eltype(B))
   if size(Q.factors, 1) == size(B, 1)
       Bnew = copyto!(similar(B, T, size(B)), B)
   elseif size(Q.factors, 2) == size(B, 1)
       Bnew = [B; zeros(T, size(Q.factors, 1) - size(B,1), size(B, 2))]
   else
       throw(DimensionMismatch("first dimension of matrix must have size either $(size(Q.factors, 1)) or $(size(Q.factors, 2))"))
   end
   lmul!(convert(AbstractQtype{T}, Q), Bnew)
end
(*)(A::AbstractMatrix, adjQ::AdjointQtype{<:Any,<:LayoutQ}) = _mul(A, adjQ)
function _mul(A::AbstractMatrix, adjQ::AdjointQtype{<:Any,<:LayoutQ})
    Q = parent(adjQ)
    T = promote_type(eltype(A), eltype(adjQ))
    adjQQ = convert(AbstractQtype{T}, adjQ)
    if size(A,2) == size(Q.factors, 1)
        AA = copyto!(similar(A, T, size(A)), A)
        return rmul!(AA, adjQQ)
    elseif size(A,2) == size(Q.factors,2)
        return rmul!([A zeros(T, size(A, 1), size(Q.factors, 1) - size(Q.factors, 2))], adjQQ)
    else
        throw(DimensionMismatch("matrix A has dimensions $(size(A)) but Q-matrix B has dimensions $(size(adjQ))"))
    end
end
(*)(u::LinearAlgebra.AdjointAbsVec, Q::AdjointQtype{<:Any,<:LayoutQ}) = (Q'u')'

*(A::LayoutQ, B::AbstractTriangular) = mul(A, B)
*(A::AdjointQtype{<:Any,<:LayoutQ}, B::AbstractTriangular) = mul(A, B)
*(A::AbstractTriangular, B::LayoutQ) = mul(A, B)
*(A::AbstractTriangular, B::AdjointQtype{<:Any,<:LayoutQ}) = mul(A, B)

if VERSION < v"1.10-"
    (*)(Q::LayoutQ, b::StridedVector)  = _mul(Q, b)
    (*)(Q::LayoutQ, B::StridedMatrix)  = _mul(Q, B)
    (*)(Q::LayoutQ, B::Adjoint{<:Any,<:StridedVecOrMat}) = _mul(Q, B)
    (*)(A::StridedMatrix, adjQ::AdjointQtype{<:Any,<:LayoutQ})  = _mul(A, adjQ)
    (*)(A::Adjoint{<:Any,<:StridedMatrix}, adjQ::AdjointQtype{<:Any,<:LayoutQ}) = _mul(A, adjQ)

    Base.@propagate_inbounds getindex(Q::LayoutQ, i::Int, j::Int) = Q[:, j][i]
    function getindex(Q::LayoutQ, ::Colon, j::Int)
        y = zeros(eltype(Q), size(Q, 2))
        y[j] = 1
        lmul!(Q, y)
    end
    Base.@propagate_inbounds layout_getindex(A::LayoutQ, I::CartesianIndex) = A[to_indices(A, (I,))...]
    Base.@propagate_inbounds layout_getindex(A::LayoutQ, I::Int...) =
        Base.invoke(Base.getindex, Tuple{AbstractQ, typeof.(I)...}, A, I...)
    Base.@propagate_inbounds layout_getindex(A::LayoutQ, I::AbstractVector{Int}, J::AbstractVector{Int}) =
        hcat((A[:, j][I] for j in J)...)


    (*)(Q::LayoutQ, P::LayoutQ) = mul(Q, P)
    (*)(Q::LayoutQ, adjQ::Adjoint{<:Any,<:LayoutQ}) = mul(Q, adjQ)
    (*)(adjQ::Adjoint{<:Any,<:LayoutQ}, Q::LayoutQ) = mul(adjQ, Q)
    (*)(adjQ::Adjoint{<:Any,<:LayoutQ}, adjP::Adjoint{<:Any,<:LayoutQ}) = mul(adjQ, adjP)
end

axes(Q::LayoutQ, dim::Integer) = axes(getfield(Q, :factors), dim == 2 ? 1 : dim)
axes(Q::LayoutQ) = axes(Q, 1), axes(Q, 2)
copy(Q::LayoutQ) = Q
Base.@propagate_inbounds getindex(A::LayoutQ, I...) = layout_getindex(A, I...)
# by default, fall back to AbstractQ  methods
layout_getindex(A::LayoutQ, I...) =
    Base.invoke(Base.getindex, Tuple{AbstractQ, typeof.(I)...}, A, I...)

size(Q::LayoutQ, dim::Integer) = size(getfield(Q, :factors), dim == 2 ? 1 : dim)
size(Q::LayoutQ) = size(Q, 1), size(Q, 2)

include("ul.jl")
include("qr.jl")
include("ql.jl")
include("rq.jl")
include("polar.jl")
include("reversecholesky.jl")

end #module
