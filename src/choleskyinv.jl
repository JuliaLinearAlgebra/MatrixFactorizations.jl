# choleskyinv.jl
# Cholesky factor and its inverse (complex-conjugate) transposed in one pass.
# For small matrices this is faster than computing the Cholesky factor
# using the `cholesky` function in LinearAlgebra and then invert it.


"""
Cholesky factorization and inverse Cholesky factorization, accessible
in fields `.c` ans `.ci`, respectively.
"""
struct CholeskyInv{T, F<:Factorization{T}} <: Factorization{T}
     c::F
     ci::F
end

"""
    choleskyinv(P::AbstractMatrix{T};
		check::Bool=true, tol::Real = √eps(T)) where T<:Union{Real, Complex}

 Compute the Cholesky factorization of a dense positive definite
 matrix P and return a `Choleskyinv` object, holding in field `.c`
 the Cholesky factorization and in field `ci` the inverse of the Cholesky
 factorization.

 The two factorizations are obtained in one pass and this is faster
 then calling Julia's [chelosky](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.cholesky)
 function and inverting the lower factor for small matrices.

 Input matrix `P` may be of type `Matrix` or `Hermitian`. Since only the
 lower triangle is used, `P` may also be a `LowerTriangular` view of a
 positive definite matrix.
 If `P` is real, it can also be of the `Symmetric` type.

 The algorithm is a *multiplicative Gaussian elimination*.

 **Notes:**
 The inverse Cholesky factor ``L^{-1}``, obtained as `.ci.L`,
 is an inverse square root (whitening matrix) of `P`, since
 ``L^{-1}PL^{-H}=I``. It therefore yields the inversion of ``P`` as
 ``P^{-1}=L^{-H}L^{-1}``. This is the fastest whitening matrix to be computed,
 however it yields poor numerical precision, especially for large matrices.

 The following relations holds:
 - ``L=PL^{-H}``,
 - ``L^{H}=L^{-1}P``,
 - ``L^{-H}=P^{-1}L``
 - ``L^{-1}=L^{H}P^{-1}``.

 We also have ``L^{H}L=L^{-1}P^{2}L^{-H}=UPU^H``, with ``U`` orthogonal
 (see below) and ``L^{-1}L^{-H}=L^{H}P^{-2}L=UP^{-1}U^H``.
 ``LL^{H}`` and ``L^{H}L`` are unitarily similar, that is,

 ``ULL^{H}U^H=L^{H}L``,

 where ``U=L^{-1}P^{1/2}``, with ``P^{1/2}=H`` the *principal* (unique symmetric)
 square root of ``P``. This is seen writing
 ``PP^{-1}=HHL^{-H}L^{-1}``; multiplying both sides on the left by ``L^{-1}``
 and on the right by ``L`` we obtain
 ``L^{-1}PP^{-1}L=L^{-1}HHL^{-H}=I=(L^{-1}H)(L^{-1}H)^H`` and since
 ``L^{-1}H`` is square it must be unitary.

 From these expressions we have
 - ``H=LU=U^HL^H,
 - ``L=HU^H``,
 - ``H^{-1}=U^HL^{-1}``
 - ``L^{-1}=UH^{-1}``.

 ``U`` is the *polar factor* of ``L^{H}``, *i.e.*, ``L^{H}=UH``,
 since ``LL^{H}=HU^HUH^H=H^2=P``.
 From ``L^{H}L=UCU^H`` we have ``L^{H}LU=UC=ULL^{H}`` and from
 ``U=L^{-1}H`` we have ``L=HU^H``.

 ## Examples
	using PosDefManifold, Test
	n = 40
	etol = 1e-9
	Y=randP(n)
	Yi=inv(Y)
	a=cholesky(Y)

	C=choleskyinv!(copy(Matrix(Y)))
	@test(norm(C.c.L*C.c.U-Y)/√n < etol)
	@test(norm(C.ci.U*C.ci.L-Yi)/√n < etol)

	# repeat the test for complex matrices
	Y=randP(ComplexF64, n)
	Yi=inv(Y)
	C=choleskyinv!(copy(Matrix(Y)))
	@test(norm(C.c.L*C.c.U-Y)/√n < etol)
	@test(norm(C.ci.U*C.ci.L-Yi)/√n < etol)

	# Benchmark
	using BenchmarkTools

	# computing the Cholesky factor and its inverse using LinearAlgebra
	function linearAlgebraWay(P)
		C=cholesky(P)
		Q=inv(C.L)
	end

	Y=randP(n)
	@benchmark(choleskyinv(Y))
	@benchmark(linearAlgebraWay(Y))

"""
choleskyinv(P::Union{Hermitian, Symmetric, Matrix, LowerTriangular};
	   		check::Bool = true,
	   		tol::Real = √eps(real(eltype(P)))) =
    choleskyinv!(copy(Matrix(P)); check=check, tol=tol)

"""
    choleskyinv!(P::AbstractMatrix{T};
		kind::Symbol = :LLt, tol::Real = √eps(T)) where T<:RealOrComplex
 The same thing as [`choleskyinv`](@ref), but destroys the input matrix.
"""
function choleskyinv!(P::Matrix{T};
			check::Bool = true,
			tol::Real = √eps(real(T))) where T<:Union{Real, Complex}
	LinearAlgebra.require_one_based_indexing(P)
	n = LinearAlgebra.checksquare(P)
	
	@inbounds for j=1:n-1
		check && abs2(P[j, j])<tol && throw(LinearAlgebra.PosDefException(1))
		for i=j+1:n
			θ = conj(P[i, j] / -P[j, j])
			for k=i:n P[k, i] += θ * P[k, j] end # update P and write D
			P[i, j] = conj(-θ) # write Cholesky factor
			for k=1:j-1 P[k, i] += θ * P[k, j] end # write inv Cholesky factor
			P[j, i] = θ # write inv Cholesky factor
		end
	end

	LT, ULT, UUT = LowerTriangular, UnitLowerTriangular, UnitUpperTriangular
	D=sqrt.(Diagonal(P))
	return CholeskyInv(Cholesky(LT(ULT(P)*D), :L, 0), Cholesky(LT(Matrix((UUT(P)*inv(D))')) , :L, 0 ))
end


function show(io::IO, ::MIME{Symbol("text/plain")}, C::CholeskyInv)
    println(io, "Cholesky (.c) and inverse Cholesky (.ci) factorizations")
	#show(io, C.c)
	#show(io, C.ci)
end
