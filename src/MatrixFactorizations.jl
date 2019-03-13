module MatrixFactorizations
using Base, LinearAlgebra
import Base: axes, axes1, getproperty, iterate, tail
import LinearAlgebra: BlasInt, BlasReal, BlasFloat, BlasComplex, axpy!,
                        copy_oftype, checksquare, adjoint, transpose, AdjOrTrans, HermOrSym
import LinearAlgebra.BLAS: libblas
import LinearAlgebra.LAPACK: liblapack, chkuplo, chktrans
import LinearAlgebra: cholesky, cholesky!, norm, diag, eigvals!, eigvals, eigen!, eigen,
            qr, axpy!, ldiv!, mul!, lu, lu!, ldlt, ldlt!, AbstractTriangular, has_offset_axes,
            chkstride1, kron, lmul!, rmul!, factorize, StructuredMatrixStyle, logabsdet,
            QRPackedQ, AbstractQ, _zeros, _cut_B, _ret_size

import Base: getindex, setindex!, *, +, -, ==, <, <=, >,
                >=, /, ^, \, transpose, showerror, reindex, checkbounds, @propagate_inbounds

import Base: convert, size, view, unsafe_indices,
                first, last, size, length, unsafe_length, step,
                to_indices, to_index, show, fill!, promote_op,
                MultiplicativeInverses, OneTo, ReshapedArray,
                Array, Matrix, Vector, AbstractArray, AbstractMatrix, AbstractVector, 
                               similar, copy, convert, promote_rule, rand,
                            IndexStyle, real, imag, Slice, pointer, unsafe_convert, copyto!


export ql, ql!, QL                        

# Elementary reflection similar to LAPACK. The reflector is not Hermitian but
# ensures that tridiagonalization of Hermitian matrices become real. See lawn72
@inline function reflector!(x::AbstractVector)
    !has_offset_axes(x)
    n = length(x)
    n == 0 && return zero(eltype(x))
    @inbounds begin
        ξ1 = x[1]
        normu = abs2(ξ1)
        for i = 2:n
            normu += abs2(x[i])
        end
        if iszero(normu)
            return zero(ξ1/normu)
        end
        normu = sqrt(normu)
        ν = copysign(normu, real(ξ1))
        ξ1 += ν
        x[1] = -ν
        for i = 2:n
            x[i] /= ξ1
        end
    end
    ξ1/ν
end

# apply reflector from left
@inline function reflectorApply!(x::AbstractVector, τ::Number, A::AbstractVecOrMat)
    m,n = size(A,1),size(A,2)
    if length(x) != m
        throw(DimensionMismatch("reflector has length $(length(x)), which must match the first dimension of matrix A, $m"))
    end
    m == 0 && return A
    @inbounds begin
        for j = 1:n
            # dot
            vAj = A[1, j]
            for i = 2:m
                vAj += x[i]'*A[i, j]
            end

            vAj = conj(τ)*vAj

            # ger
            A[1, j] -= vAj
            for i = 2:m
                A[i, j] -= x[i]*vAj
            end
        end
    end
    return A
end

# Should be in Base, StridedVector -> AbstractVector

function (*)(A::AbstractQ, b::AbstractVector)
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
function (*)(A::AbstractQ, B::AbstractMatrix)
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

function *(Q::AbstractQ, adjB::Adjoint{<:Any,<:AbstractVecOrMat})
   B = adjB.parent
   TQB = promote_type(eltype(Q), eltype(B))
   Bc = similar(B, TQB, (size(B, 2), size(B, 1)))
   adjoint!(Bc, B)
   return lmul!(convert(AbstractMatrix{TQB}, Q), Bc)
end
function *(adjQ::Adjoint{<:Any,<:AbstractQ}, adjB::Adjoint{<:Any,<:AbstractVecOrMat})
   Q, B = adjQ.parent, adjB.parent
   TQB = promote_type(eltype(Q), eltype(B))
   Bc = similar(B, TQB, (size(B, 2), size(B, 1)))
   adjoint!(Bc, B)
   return lmul!(adjoint(convert(AbstractMatrix{TQB}, Q)), Bc)
end


include("ql.jl")

end #module
