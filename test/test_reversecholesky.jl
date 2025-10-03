# This file is modifed from a part of Julia. License is MIT: https://julialang.org/license

using Test, LinearAlgebra, Random, MatrixFactorizations
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, QRPivoted,
    PosDefException, RankDeficientException, chkfullrank
using ArrayLayouts: Fill

function unary_ops_tests(a, ca, tol; n=size(a, 1))
    @test inv(ca)*a ≈ Matrix(I, n, n)
    @test a*inv(ca) ≈ Matrix(I, n, n)
    @test abs((det(ca) - det(a))/det(ca)) <= tol # Ad hoc, but statistically verified, revisit
    @test logdet(ca) ≈ logdet(a)
    @test logdet(ca) ≈ log(det(ca))  # logdet is less likely to overflow
    logabsdet_ca = logabsdet(ca)
    logabsdet_a = logabsdet(a)
    @test logabsdet_ca[1] ≈ logabsdet_a[1]
    @test logabsdet_ca[2] ≈ logabsdet_a[2]
    @test isposdef(ca)
    @test_throws FieldError ca.Z
    @test size(ca) == size(a)
    @test Array(copy(ca)) ≈ a
end

function factor_recreation_tests(a_U, a_L)
    c_U = reversecholesky(a_U)
    c_L = reversecholesky(a_L)
    cl  = c_L.U
    ls = c_L.L
    @test Array(c_U) ≈ Array(c_L) ≈ a_U
    @test ls'*ls ≈ a_U
    @test triu(c_U.factors) ≈ c_U.U
    @test tril(c_L.factors) ≈ c_L.L
    @test istriu(cl)
    @test cl*cl' ≈ a_U
    @test cl*cl' ≈ a_L
end

@testset "ReverseCholesky" begin
    @testset "core functionality" begin
        n = 10

        # Split n into 2 parts for tests needing two matrices
        n1 = div(n, 2)
        n2 = 2*n1

        Random.seed!(12344)

        areal = randn(n,n)/2
        aimg  = randn(n,n)/2
        a2real = randn(n,n)/2
        a2img  = randn(n,n)/2
        breal = randn(n,2)/2
        bimg  = randn(n,2)/2

        @testset "basic sanity check" begin
            A = [4 2; 2 4]
            U = reversecholesky(A).U
            @test U*U' ≈ A
            @test cholesky(A).L[end:-1:1,end:-1:1] ≈ U

            A = Symmetric(areal) + 10I
            U = reversecholesky(A).U
            @test U*U' ≈ A

            A = Hermitian(areal + im*aimg) + 10I
            U = reversecholesky(A).U
            @test U*U' ≈ A

            A = Symmetric(areal, :L) + 10I
            U = reversecholesky(A).U
            @test U*U' ≈ A

            A = Hermitian(areal + im*aimg, :L) + 10I
            U = reversecholesky(A).U
            @test U*U' ≈ A
        end

        for eltya in (Float32, Float64, ComplexF32, ComplexF64, BigFloat, Int)
            a = eltya == Int ? rand(1:7, n, n) : convert(Matrix{eltya}, eltya <: Complex ? complex.(areal, aimg) : areal)
            a2 = eltya == Int ? rand(1:7, n, n) : convert(Matrix{eltya}, eltya <: Complex ? complex.(a2real, a2img) : a2real)

            ε = εa = eps(abs(float(one(eltya))))

            # Test of symmetric pos. def. strided matrix
            apd  = a'*a
            @inferred reversecholesky(apd)
            capd  = reversecholesky(apd)
            r     = capd.U
            κ     = cond(apd, 1) #condition number

            unary_ops_tests(apd, capd, ε*κ*n)
            if eltya != Int
                @test Factorization{eltya}(capd) === capd
                if eltya <: Real
                    @test Array(Factorization{complex(eltya)}(capd)) ≈ Array(factorize(complex(apd)))
                    @test eltype(Factorization{complex(eltya)}(capd)) == complex(eltya)
                end
            end
            @testset "throw for non-square input" begin
                A = rand(eltya, 2, 3)
                @test_throws DimensionMismatch reversecholesky(A)
                @test_throws DimensionMismatch reversecholesky!(A)
            end

            #Test error bound on reconstruction of matrix: LAWNS 14, Lemma 2.1

            #these tests were failing on 64-bit linux when inside the inner loop
            #for eltya = ComplexF32 and eltyb = Int. The E[i,j] had NaN32 elements
            #but only with Random.seed!(1234321) set before the loops.
            E = abs.(apd - r*r')
            for i=1:n, j=1:n
                @test E[i,j] <= (n+1)ε/(1-(n+1)ε)*real(sqrt(apd[i,i]*apd[j,j]))
            end
            E = abs.(apd - Matrix(capd))
            for i=1:n, j=1:n
                @test E[i,j] <= (n+1)ε/(1-(n+1)ε)*real(sqrt(apd[i,i]*apd[j,j]))
            end
            @test LinearAlgebra.issuccess(capd)
            @inferred(logdet(capd))

            apos = apd[1,1]
            @test all(x -> x ≈ √apos, cholesky(apos).factors)

            # Test cholesky with Symmetric/Hermitian upper/lower
            apds  = Symmetric(apd)
            apdsL = Symmetric(apd, :L)
            apdh  = Hermitian(apd)
            apdhL = Hermitian(apd, :L)
            if eltya <: Real
                capds = reversecholesky(apds)
                unary_ops_tests(apds, capds, ε*κ*n)
                if eltya <: BlasReal
                    capds = cholesky!(copy(apds))
                    unary_ops_tests(apds, capds, ε*κ*n)
                end
                ulstring = sprint((t, s) -> show(t, "text/plain", s), capds.UL)
                @test sprint((t, s) -> show(t, "text/plain", s), capds) == "$(typeof(capds))\nU factor:\n$ulstring"
            else
                capdh = reversecholesky(apdh)
                unary_ops_tests(apdh, capdh, ε*κ*n)
                capdh = reversecholesky!(copy(apdh))
                unary_ops_tests(apdh, capdh, ε*κ*n)
                capdh = reversecholesky!(copy(apd))
                unary_ops_tests(apd, capdh, ε*κ*n)
                ulstring = sprint((t, s) -> show(t, "text/plain", s), capdh.UL)
                @test sprint((t, s) -> show(t, "text/plain", s), capdh) == "$(typeof(capdh))\nU factor:\n$ulstring"
            end

            # test reversecholesky of 2x2 Strang matrix
            S = SymTridiagonal{eltya}([2, 2], [-1])
            for uplo in (:U, :L)
                @test Matrix(@inferred reversecholesky(Hermitian(S, uplo))) ≈ S
                if eltya <: Real
                    @test Matrix(@inferred reversecholesky(Symmetric(S, uplo))) ≈ S
                end
            end
            @test Matrix(reversecholesky(S).U) ≈ [sqrt(eltya(3)) -1; 0 2] / sqrt(eltya(2))
            @test Matrix(reversecholesky(S)) ≈ S

            # test extraction of factor and re-creating original matrix
            if eltya <: Real
                factor_recreation_tests(apds, apdsL)
            else
                factor_recreation_tests(apdh, apdhL)
            end



            for eltyb in (Float32, Float64, ComplexF32, ComplexF64, Int)
                b = eltyb == Int ? rand(1:5, n, 2) : convert(Matrix{eltyb}, eltyb <: Complex ? complex.(breal, bimg) : breal)
                εb = eps(abs(float(one(eltyb))))
                ε = max(εa,εb)

                for b in (b, view(b, 1:n, 1)) # Array and SubArray

                    # Test error bound on linear solver: LAWNS 14, Theorem 2.1
                    # This is a surprisingly loose bound
                    x = capd\b
                    @test norm(x-apd\b,1)/norm(x,1) <= (3n^2 + n + n^3*ε)*ε/(1-(n+1)*ε)*κ
                    @test norm(apd*x-b,1)/norm(b,1) <= (3n^2 + n + n^3*ε)*ε/(1-(n+1)*ε)*κ

                    @test norm(a*(capd\(a'*b)) - b,1)/norm(b,1) <= ε*κ*n # Ad hoc, revisit

                    if eltya != BigFloat && eltyb != BigFloat
                        lapd = reversecholesky(apdhL)
                        @test norm(apd * (lapd\b) - b)/norm(b) <= ε*κ*n
                        @test norm(apd * (lapd\b[1:n]) - b[1:n])/norm(b[1:n]) <= ε*κ*n
                    end
                end
            end

            for eltyb in (Float64, ComplexF64)
                Breal = convert(Matrix{BigFloat}, randn(n,n)/2)
                Bimg  = convert(Matrix{BigFloat}, randn(n,n)/2)
                B = (eltya <: Complex || eltyb <: Complex) ? complex.(Breal, Bimg) : Breal
                εb = eps(abs(float(one(eltyb))))
                ε = max(εa,εb)

                for B in (B, view(B, 1:n, 1:n)) # Array and SubArray

                    # Test error bound on linear solver: LAWNS 14, Theorem 2.1
                    # This is a surprisingly loose bound
                    BB = copy(B)
                    ldiv!(capd, BB)
                    @test norm(apd \ B - BB, 1) / norm(BB, 1) <= (3n^2 + n + n^3*ε)*ε/(1-(n+1)*ε)*κ
                    @test norm(apd * BB - B, 1) / norm(B, 1) <= (3n^2 + n + n^3*ε)*ε/(1-(n+1)*ε)*κ
                end
            end

            @testset "solve with generic Cholesky" begin
                Breal = convert(Matrix{BigFloat}, randn(n,n)/2)
                Bimg  = convert(Matrix{BigFloat}, randn(n,n)/2)
                B = eltya <: Complex ? complex.(Breal, Bimg) : Breal
                εb = eps(abs(float(one(eltype(B)))))
                ε = max(εa,εb)

                for B in (B, view(B, 1:n, 1:n)) # Array and SubArray

                    # Test error bound on linear solver: LAWNS 14, Theorem 2.1
                    # This is a surprisingly loose bound
                    cpapd = reversecholesky(eltya <: Complex ? apdh : apds)
                    BB = copy(B)
                    rdiv!(BB, cpapd)
                    @test norm(B / apd - BB, 1) / norm(BB, 1) <= (3n^2 + n + n^3*ε)*ε/(1-(n+1)*ε)*κ
                    @test norm(BB * apd - B, 1) / norm(B, 1) <= (3n^2 + n + n^3*ε)*ε/(1-(n+1)*ε)*κ
                    cpapd = reversecholesky(eltya <: Complex ? apdhL : apdsL)
                    BB = copy(B)
                    rdiv!(BB, cpapd)
                    @test norm(B / apd - BB, 1) / norm(BB, 1) <= (3n^2 + n + n^3*ε)*ε/(1-(n+1)*ε)*κ
                    @test norm(BB * apd - B, 1) / norm(B, 1) <= (3n^2 + n + n^3*ε)*ε/(1-(n+1)*ε)*κ
                end
            end
        end
    end

    @testset "behavior for non-positive definite matrices" for T in (Float64, ComplexF64)
        A = T[1 2; 2 1]
        B = T[1 2; 0 1]
        # check = (true|false)
        for M in (A, Hermitian(A), B)
            @test_throws PosDefException reversecholesky(M)
            @test_throws PosDefException reversecholesky!(copy(M))
            @test_throws PosDefException reversecholesky(M; check = true)
            @test_throws PosDefException reversecholesky!(copy(M); check = true)
            @test !LinearAlgebra.issuccess(reversecholesky(M; check = false))
            @test !LinearAlgebra.issuccess(reversecholesky!(copy(M); check = false))
        end
        str = sprint((io, x) -> show(io, "text/plain", x), reversecholesky(A; check = false))
    end

    @testset "Cholesky factor of Matrix with non-commutative elements, here 2x2-matrices" begin
        X = Matrix{Float64}[0.1*rand(2,2) for i in 1:3, j = 1:3]
        L = Matrix(MatrixFactorizations._reverse_chol!(X*X', LowerTriangular)[1])
        U = Matrix(MatrixFactorizations._reverse_chol!(X*X', UpperTriangular)[1])
        XX = Matrix(X*X')

        @test_broken sum(sum(norm, L'*L - XX)) < eps()
        @test_broken sum(sum(norm, U*U' - XX)) < eps()
    end

    @testset "Non-strided Cholesky solves" begin
        B = randn(5, 5)
        v = rand(5)
        @test reversecholesky(Diagonal(v)) \ B ≈ Diagonal(v) \ B
        @test B / reversecholesky(Diagonal(v)) ≈ B / Diagonal(v)
        @test inv(reversecholesky(Diagonal(v)))::Diagonal ≈ Diagonal(1 ./ v)
    end


    @testset "cholesky Diagonal" begin
        # real
        d = abs.(randn(3)) .+ 0.1
        D = Diagonal(d)
        CD = reversecholesky(D)
        CM = reversecholesky(Matrix(D))
        @test CD isa ReverseCholesky{Float64}
        @test CD.U ≈ Diagonal(.√d) ≈ CM.U
        @test D ≈ CD.L * CD.U
        @test CD.info == 0

        F = reversecholesky(Hermitian(I(3)))
        @test F isa ReverseCholesky{Float64,<:Diagonal}
        @test Matrix(F) ≈ I(3)

        # real, failing
        @test_throws PosDefException reversecholesky(Diagonal([1.0, -2.0]))
        Dnpd = reversecholesky(Diagonal([1.0, -2.0]); check = false)
        @test Dnpd.info == 1

        # complex
        D = complex(D)
        CD = reversecholesky(Hermitian(D))
        CM = reversecholesky(Matrix(Hermitian(D)))
        @test CD isa ReverseCholesky{ComplexF64,<:Diagonal}
        @test CD.U ≈ Diagonal(.√d) ≈ CM.U
        @test D ≈ CD.L * CD.U
        @test CD.info == 0

        # complex, failing
        D[2, 2] = 0.0 + 0im
        @test_throws PosDefException reversecholesky(D)        

        # InexactError for Int
        @test_throws InexactError reversecholesky!(Diagonal([2, 1]))
    end

    @testset "Cholesky for AbstractMatrix" begin
        S = SymTridiagonal(fill(2.0, 4), ones(3))
        C = reversecholesky(S)
        @test C.U * C.L ≈ S
    end

    @testset "constructor with non-BlasInt arguments" begin
        x = rand(5,5)
        chol = reversecholesky(x'x)

        factors, uplo, info = chol.factors, chol.uplo, chol.info

        @test ReverseCholesky(factors, uplo, Int32(info)) == chol
        @test ReverseCholesky(factors, uplo, Int64(info)) == chol
    end

    @testset "issue #37356, diagonal elements of hermitian generic matrix" begin
        B = Hermitian(hcat([one(BigFloat) + im]))
        @test Matrix(reversecholesky(B)) ≈ B
        C = Hermitian(hcat([one(BigFloat) + im]), :L)
        @test Matrix(reversecholesky(C)) ≈ C
    end

    @testset "constructing a ReverseCholesky factor from a triangular matrix" begin
        A = [1.0 2.0; 3.0 4.0]
        let
            U = UpperTriangular(A)
            C = ReverseCholesky(U)
            @test C isa ReverseCholesky{Float64}
            @test C.U == U
            @test C.L == U'
        end
        let
            L = LowerTriangular(A)
            C = ReverseCholesky(L)
            @test C isa ReverseCholesky{Float64}
            @test C.L == L
            @test C.U == L'
        end
    end

    @testset "adjoint of ReverseCholesky" begin
        A = randn(5, 5)
        A = A'A
        F = reversecholesky(A)
        b = ones(size(A, 1))
        @test F\b == F'\b
    end

    @testset "Float16" begin
        A = Float16[4. 12. -16.; 12. 37. -43.; -16. -43. 98.]
        B = reversecholesky(A)
        B32 = reversecholesky(Float32.(A))
        @test B isa ReverseCholesky{Float16, Matrix{Float16}}
        @test B.U isa UpperTriangular{Float16, Matrix{Float16}}
        @test B.L isa LowerTriangular{Float16, Matrix{Float16}}
        @test B.UL isa UpperTriangular{Float16, Matrix{Float16}}
        @test B.U ≈ B32.U
        @test B.L ≈ B32.L
        @test B.UL ≈ B32.UL
        @test Matrix(B) ≈ A
    end

    @testset "large Sparse" begin
        n = 1_000_000
        A = SymTridiagonal(fill(4,n), fill(1,n-1))
        R = reversecholesky(A)
        @test R isa ReverseCholesky{Float64, <:Bidiagonal}
        @test R.U isa Bidiagonal
        @test R.L isa Bidiagonal
        @test R.UL isa Bidiagonal
        # Bidiagonal multiplication not supported
        @test R.U*(R.U' * [1; zeros(n-1)]) ≈ A[:,1]
        @test R.L'*(R.L * [1; zeros(n-1)]) ≈ A[:,1]

        A = Tridiagonal(fill(1.0,n-1), fill(4.0,n), fill(1/2,n-1))
        R = reversecholesky(Symmetric(A))
        @test R.U*(R.U' * [1; zeros(n-1)]) ≈ A[1,:]
        @test R.L'*(R.L * [1; zeros(n-1)]) ≈ A[1,:]
        R = reversecholesky(Symmetric(A,:L))
        @test R.U*(R.U' * [1; zeros(n-1)]) ≈ A[:,1]
        @test R.L'*(R.L * [1; zeros(n-1)]) ≈ A[:,1]

        A = Bidiagonal(fill(4.0,n), fill(1.0,n-1), :U)
        R = reversecholesky(Symmetric(A))
        @test R.U*(R.U' * [1; zeros(n-1)]) ≈ A[1,:]
        @test R.L'*(R.L * [1; zeros(n-1)]) ≈ A[1,:]

        A = Bidiagonal(fill(4.0,n), fill(1.0,n-1), :L)
        R = reversecholesky(Symmetric(A,:L))
        @test R.U * (R.U' * [1; zeros(n-1)]) ≈ A[:,1]
        @test R.L'*(R.L * [1; zeros(n-1)]) ≈ A[:,1]

        A = Bidiagonal(fill(4,n), fill(1,n-1), :L)
        R = reversecholesky(Symmetric(A,:L))
        @test R.U * (R.U' * [1; zeros(n-1)]) ≈ A[:,1]
        @test R.L'*(R.L * [1; zeros(n-1)]) ≈ A[:,1]
    end

    @testset "coverage" begin
        A = [4 2; 2 4]
        R = reversecholesky(A)
        @test ReverseCholesky(R.factors, R.uplo, R.info) == R
        U,L = R
        @test U*U' ≈ A
        @test reversecholesky(5).U[1,1] ≈ sqrt(5)
        @test propertynames(R) == (:U, :L, :UL)
    end

    @testset "Lazy reversecholesky" begin
        A = SymTridiagonal(Fill(2,10), Fill(1,9))
        @test reversecholesky(A) == reversecholesky(SymTridiagonal(fill(2.0,10), fill(1.0,9))) == 
                reversecholesky(Symmetric(Bidiagonal(Fill(2,10), Fill(1,9), :U))) ==
                reversecholesky(Symmetric(Tridiagonal(Fill(3,9), Fill(2,10), Fill(1,9))))
    end
end