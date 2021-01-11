using MatrixFactorizations, LinearAlgebra, Random, ArrayLayouts, Test
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, rmul!, lmul!, require_one_based_indexing, checksquare

n = 10

# Split n into 2 parts for tests needing two matrices
n1 = div(n, 2)
n2 = 2*n1

Random.seed!(1234321)

areal = randn(n,n)/2
aimg  = randn(n,n)/2
a2real = randn(n,n)/2
a2img  = randn(n,n)/2
breal = randn(n,2)/2
bimg  = randn(n,2)/2

using MatrixFactorizations: RQPackedQ

using MatrixFactorizations: rq, rq!
const Our=MatrixFactorizations

@testset "RQ" begin
    @testset "LAPACK $elty" for elty in (Float32,Float64,ComplexF32,ComplexF64)
        @testset "Compare with LAPACK (square $elty)" begin
            n = 10
            A = randn(n,n)
            R, Q = @inferred rq(A)
            Ṟ, Q̄ = Our._rqfactUnblocked!(copy(A))
            fla,τla = LinearAlgebra.LAPACK.gerqf!(copy(A))
            @test Q.factors ≈ fla
            @test Q.τ ≈ τla
            @test Q.factors ≈ Q̄.factors
            @test Q.τ ≈ Q̄.τ
            @test R ≈ Ṟ
            @test Q ≈ Q̄
        end
        @testset "Compare with LAPACK (rectangular $elty)" begin
            n = 10
            A = randn(n,n+2)
            R, Q = rq(A)
            Ṟ, Q̄ = Our._rqfactUnblocked!(copy(A))
            fla,τla = LinearAlgebra.LAPACK.gerqf!(copy(A))
            @test Q.factors ≈ fla
            @test Q.τ ≈ τla
            @test Q.factors ≈ Q̄.factors
            @test Q.τ ≈ Q̄.τ
            @test R ≈ Ṟ
            @test Q ≈ Q̄

            A = randn(n+2,n)
            R, Q = rq(A)
            Ṟ, Q̄ = Our._rqfactUnblocked!(copy(A))
            fla,τla = LinearAlgebra.LAPACK.gerqf!(copy(A))
            @test Q.factors ≈ fla
            @test Q.τ ≈ τla
            @test Q.factors ≈ Q̄.factors
            @test Q.τ ≈ Q̄.τ
            @test R ≈ Ṟ
            @test Q ≈ Q̄
        end
    end
    @testset for eltya in (Float32, Float64, ComplexF32, ComplexF64, BigFloat, Complex{BigFloat}, Int)
    # @testset for eltya in (Float32,ComplexF32)
        raw_a = eltya == Int ? rand(1:7, n, n) : convert(Matrix{eltya}, eltya <: Complex ? complex.(areal, aimg) : areal)
        raw_a2 = eltya == Int ? rand(1:7, n, n) : convert(Matrix{eltya}, eltya <: Complex ? complex.(a2real, a2img) : a2real)
        asym = raw_a' + raw_a                  # symmetric indefinite
        apd  = raw_a' * raw_a                 # symmetric positive-definite
        ε = εa = eps(abs(float(one(eltya))))

        @testset for eltyb in (Float32, Float64, ComplexF32, ComplexF64, Int)
        # @info "stubbed eltyb loop"
        # @testset for eltyb in (eltya, )
            raw_b = eltyb == Int ? rand(1:5, n, 2) : convert(Matrix{eltyb}, eltyb <: Complex ? complex.(breal, bimg) : breal)
            εb = eps(abs(float(one(eltyb))))
            ε = max(εa, εb)
            tab = promote_type(eltya, eltyb)

            @testset "RQ decomposition of a Number" begin
                α = rand(eltyb)
                aα = fill(α, 1, 1)
                @test rq(α).R * rq(α).Q ≈ rq(aα).R * rq(aα).Q
                @test abs(rq(α).Q[1,1]) ≈ one(eltyb)
            end

            for (a, b) in ((raw_a, raw_b),
                (view(raw_a, 1:n-1, 1:n-1), view(raw_b, 1:n-1, 1)))
                a_1 = size(a, 1)
                @testset "RQ decomposition" begin
                    rqa   = @inferred rq(a)
                    @inferred rq(a)
                    q, r  = rqa.Q, rqa.R
                    @test_throws ErrorException rqa.Z
                    @test q'*q ≈ Matrix(I, a_1, a_1)
                    @test q*q' ≈ Matrix(I, a_1, a_1)
                    @test q'*Matrix(1.0I, a_1, a_1)' ≈ q'
                    @test q'q ≈ Matrix(I, a_1, a_1)
                    @test Matrix(1.0I, a_1, a_1)'q' ≈ q'
                    @test r*q ≈ a
                    @test a*(rqa\b) ≈ b atol=3000ε
                    @test Array(rqa) ≈ a
                    sq = size(q.factors, 2)
                    @test *(Matrix{eltyb}(I, sq, sq), adjoint(q)) * q ≈ Matrix(I, sq, sq) atol=5000ε
                    if eltya != Int
                        @test Matrix{eltyb}(I, a_1, a_1)*q ≈ convert(AbstractMatrix{tab}, q)
                        ac = copy(a)
                        # would need rectangular ldiv! method
                        @test_throws DimensionMismatch rq!(a[:, 1:5])\b == rq!(view(ac, :, 1:5))\b
                    end
                    rqstring = sprint((t, s) -> show(t, "text/plain", s), rqa)
                    rstring  = sprint((t, s) -> show(t, "text/plain", s), r)
                    qstring  = sprint((t, s) -> show(t, "text/plain", s), q)
                    @test rqstring == "$(summary(rqa))\nR factor:\n$rstring\nQ factor:\n$qstring"
                end
            end
            if eltya != Int
                @testset "Matmul with RQ factorizations" begin
                    a = raw_a

                    rqa = rq(a[:,1:n1])
                    q, r = rqa.Q, rqa.R
                    @test rmul!(copy(q'), q) ≈ Matrix(I, n1, n1)
                    @test_throws DimensionMismatch rmul!(Matrix{eltya}(I, n1+1, n1+1),q)
                    @test rmul!(copy(q), adjoint(q)) ≈ Matrix(I, n1, n1)
                    @test_throws DimensionMismatch rmul!(Matrix{eltya}(I, n1+1, n1+1),adjoint(q))
                    @test_throws ErrorException size(q,-1)
                    @test_throws DimensionMismatch q * Matrix{Int8}(I, n+4, n+4)
                end
            end
        end
        @testset "Wide RQ" begin
            m = n-2
            A = raw_a[1:m,1:n]
            R,Q = rq(A)
            @test Q'*Q ≈ Matrix(I, n, n)
            @test istriu(R)
            @test hcat(zeros(m,n-m),R)*Q ≈ A
            # test the perverse padded product
            @test R*Q ≈ A
            Qm = Matrix(Q)
            @test Qm' * Qm ≈ Matrix(I, n, n)
            @test hcat(zeros(m,n-m),R)*Qm ≈ A
        end
        @testset "Tall RQ" begin
            p = n-2
            A = raw_a[1:n,1:p]
            R,Q = rq(A)
            @test Q'*Q ≈ Matrix(I, p, p)
            @test istriu(R,p-n)
            @test R*Q ≈ A
            Qm = Matrix(Q)
            @test Qm' * Qm ≈ Matrix(I, p, p)
            @test R*Qm ≈ A
        end
    end

    @testset "transpose errors" begin
        @test_throws MethodError transpose(rq(randn(3,3)))
        @test_throws MethodError adjoint(rq(randn(3,3)))
        @test_throws MethodError transpose(rq(big.(randn(3,3))))
        @test_throws MethodError adjoint(rq(big.(randn(3,3))))
    end

    @testset "Issue 7304" begin
        A = [-√.5 -√.5; -√.5 √.5]
        Q = rq(A).Q
        @test norm(A+Q) < eps()
    end

    @testset "rq on AbstractVector" begin
        vl = [3.0, 4.0]
        for Tl in (Float32, Float64)
            for T in (Tl, Complex{Tl})
                v = convert(Vector{T}, vl)
                nv, nm = rq(v)
                @test nv*nm ≈ v
            end
        end
    end

    @testset "Issue 24589. Promotion of rational matrices" begin
        A = rand(1//1:5//5, 4,3)
        @test first(rq(A)) == first(rq(float(A)))
    end

    # omit "Issue Test Factorization fallbacks for rectangular problems"

    @testset "lmul!/rmul! $elty" for elty in (:real, :cplx)
        s = elty == :real ? 0.0 : 0.25im
        A = randn(100,100) .+ s
        R,Q = rq(A)
        x = randn(100) .+ s
        b = randn(100,2) .+ s
        @test lmul!(Q, copy(x)) ≈ Matrix(Q)*x
        @test lmul!(Q, copy(b)) ≈ Matrix(Q)*b
        @test lmul!(Q', copy(x)) ≈ Matrix(Q)'*x
        @test lmul!(Q', copy(b)) ≈ Matrix(Q)'*b
        c = randn(2,100) .+ s
        @test rmul!(copy(c), Q) ≈ c*Matrix(Q)
        @test rmul!(copy(c), Q') ≈ c*Matrix(Q')

        A = randn(103,100) .+ s
        R,Q = rq(A)
        x = randn(100) .+ s
        b = randn(100,2) .+ s
        @test lmul!(Q, copy(x)) ≈ Matrix(Q)*x
        @test lmul!(Q, copy(b)) ≈ Matrix(Q)*b
        @test lmul!(Q', copy(x)) ≈ Matrix(Q)'*x
        @test lmul!(Q', copy(b)) ≈ Matrix(Q)'*b
        c = randn(2,100) .+ s
        @test rmul!(copy(c), Q) ≈ c*Matrix(Q)
        @test rmul!(copy(c), Q') ≈ c*Matrix(Q')
    end
end
