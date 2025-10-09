using MatrixFactorizations, ArrayLayouts, Test

@testset "QL" begin
    @testset "Compare with QR" begin
        n = 10
        A = randn(n,n)
        Q̃, R = @inferred qr(A[end:-1:1,end:-1:1])
        Q, L = @inferred ql(A)
        Q̄, L̄ = MatrixFactorizations.generic_qlfactUnblocked!(copy(A))
        @test Q.factors ≈ Q̄.factors
        @test Q.τ ≈ Q̄.τ
        @test R[end:-1:1,end:-1:1] ≈ L ≈ L̄
        @test Q̃[end:-1:1,end:-1:1] ≈ Matrix(Q) ≈ Matrix(Q̄)
        @test copy(Q) ≡ Q

        A = randn(n,n+2)
        Q̃, R = qr(A[end:-1:1,end:-1:1])
        Q, L = ql(A)
        Q̄, L̄ = MatrixFactorizations.generic_qlfactUnblocked!(copy(A))
        @test Q.factors ≈ Q̄.factors
        @test Q.τ ≈ Q̄.τ
        @test R[end:-1:1,end:-1:1] ≈ L ≈ L̄
        @test Q̃[end:-1:1,end:-1:1] ≈ Matrix(Q) ≈ Matrix(Q̄)

        A = randn(n+2,n)
        Q̃, R = qr(A[end:-1:1,end:-1:1])
        Q, L = ql(A)
        Q̄, L̄ = MatrixFactorizations.generic_qlfactUnblocked!(copy(A))
        @test Q.factors ≈ Q̄.factors
        @test Q.τ ≈ Q̄.τ
        @test R[end:-1:1,end:-1:1] ≈ L ≈ L̄
        @test Q̃[end:-1:1,end:-1:1] ≈ squareQ(Q) ≈ squareQ(Q̄)
    end

    @testset for eltya in (Float32, Float64, ComplexF32, ComplexF64, BigFloat, Int)
        raw_a = eltya == Int ? rand(1:7, n, n) : convert(Matrix{eltya}, eltya <: Complex ? complex.(areal, aimg) : areal)
        raw_a2 = eltya == Int ? rand(1:7, n, n) : convert(Matrix{eltya}, eltya <: Complex ? complex.(a2real, a2img) : a2real)
        asym = raw_a' + raw_a                  # symmetric indefinite
        apd  = raw_a' * raw_a                 # symmetric positive-definite
        ε = εa = eps(abs(float(one(eltya))))

        @testset for eltyb in (Float32, Float64, ComplexF32, ComplexF64, Int)
            raw_b = eltyb == Int ? rand(1:5, n, 2) : convert(Matrix{eltyb}, eltyb <: Complex ? complex.(breal, bimg) : breal)
            εb = eps(abs(float(one(eltyb))))
            ε = max(εa, εb)
            tab = promote_type(eltya, eltyb)

            @testset "QL decomposition of a Number" begin
                α = rand(eltyb)
                aα = fill(α, 1, 1)
                @test ql(α).Q * ql(α).L ≈ ql(aα).Q * ql(aα).L
                @test abs(ql(α).Q[1,1]) ≈ one(eltyb)
            end

            for (a, b) in ((raw_a, raw_b),
                (view(raw_a, 1:n-1, 1:n-1), view(raw_b, 1:n-1, 1)))
                a_1 = size(a, 1)
                @testset "QL decomposition (without pivoting)" begin
                    qla   = @inferred ql(a)
                    @inferred ql(a)
                    q, l  = qla.Q, qla.L
                    @test_throws FieldError qla.Z
                    @test q'*squareQ(q) ≈ Matrix(I, a_1, a_1)
                    @test q*squareQ(q)' ≈ Matrix(I, a_1, a_1)
                    @test q'*Matrix(1.0I, a_1, a_1)' ≈ squareQ(q)'
                    @test squareQ(q)'q ≈ Matrix(I, a_1, a_1)
                    @test Matrix(1.0I, a_1, a_1)'q' ≈ squareQ(q)'
                    @test q*l ≈ a
                    @test a*(qla\b) ≈ b atol=5000ε
                    @test Array(qla) ≈ a
                    sq = size(q.factors, 2)
                    @test *(Matrix{eltyb}(I, sq, sq), adjoint(q)) * squareQ(q) ≈ Matrix(I, sq, sq) atol=5000ε
                    if eltya != Int
                        @test Matrix{eltyb}(I, a_1, a_1)*q ≈ squareQ(convert(AbstractQtype{tab}, q))
                        ac = copy(a)
                        @test ql!(a[:, 1:5])\b == ql!(view(ac, :, 1:5))\b
                    end
                    qlstring = sprint((t, s) -> show(t, "text/plain", s), qla)
                    rstring  = sprint((t, s) -> show(t, "text/plain", s), l)
                    qstring  = sprint((t, s) -> show(t, "text/plain", s), q)
                    @test qlstring == "$(summary(qla))\nQ factor:\n$qstring\nL factor:\n$rstring"
                end
                @testset "Thin QL decomposition (without pivoting)" begin
                    qla   = @inferred ql(a[:, 1:n1], Val(false))
                    @inferred ql(a[:, 1:n1], Val(false))
                    q,l   = qla.Q, qla.L
                    @test_throws FieldError qla.Z
                    @test q'*squareQ(q) ≈ Matrix(I, a_1, a_1)
                    @test q'*rectangularQ(q) ≈ Matrix(I, a_1, n1)
                    @test q*l ≈ a[:, 1:n1]
                    @test q*b[1:n1] ≈ rectangularQ(q)*b[1:n1] atol=100ε
                    @test q*b ≈ squareQ(q)*b atol=100ε
                    @test_throws DimensionMismatch q*b[1:n1 + 1]
                    @test_throws DimensionMismatch b[1:n1 + 1]*q'
                    sq = size(q.factors, 1)
                    @test *(LowerTriangular(Matrix{eltyb}(I, sq, sq)), adjoint(q))*squareQ(q) ≈ Matrix(I, a_1, a_1) atol=5000ε
                    if eltya != Int
                        @test Matrix{eltyb}(I, a_1, a_1)*q ≈ squareQ(convert(LinearAlgebra.AbstractQ{tab}, q))
                    end
                end
            end
            if eltya != Int
                @testset "Matmul with QL factorizations" begin
                    a = raw_a

                    qla = ql(a[:,1:n1], Val(false))
                    q, l = qla.Q, qla.L
                    @test rmul!(copy(squareQ(q)'), q) ≈ Matrix(I, n, n)
                    @test_throws DimensionMismatch rmul!(Matrix{eltya}(I, n+1, n+1),q)
                    @test rmul!(squareQ(q), adjoint(q)) ≈ Matrix(I, n, n)
                    @test_throws DimensionMismatch rmul!(Matrix{eltya}(I, n+1, n+1),adjoint(q))
                    @test_throws ErrorException size(q,-1)
                    @test_throws DimensionMismatch q * Matrix{Int8}(I, n+4, n+4)
                end
            end
        end
    end


    @testset "Issue 7304" begin
        A = [-√.5 -√.5; -√.5 √.5]
        Q = rectangularQ(ql(A).Q)
        @test norm(A+Q) < eps()
    end

    @testset "ql on AbstractVector" begin
        vl = [3.0, 4.0]
        for Tl in (Float32, Float64)
            for T in (Tl, Complex{Tl})
                v = convert(Vector{T}, vl)
                nv, nm = ql(v)
                @test nv*nm ≈ v
            end
        end
    end

    @testset "Issue 24589. Promotion of rational matrices" begin
        A = rand(1//1:5//5, 4,3)
        @test Matrix(first(ql(A))) == Matrix(first(ql(float(A))))
    end

    @testset "Issue Test Factorization fallbacks for rectangular problems" begin
        A  = randn(3,2)
        Ac = copy(A')
        b  = randn(3)
        b0 = copy(b)
        c  = randn(2)
        @test_broken A \b ≈ ldiv!(c, ql(A ), b)
    end

    @testset "Wide QL" begin
        A = randn(3,5)
        Q,L = ql(A)
        @test Q*L ≈ A
    end

    @testset "lmul!/rmul!" begin
        A = randn(100,100)
        Q,L = ql(A)
        x = randn(100)
        b = randn(100,2)
        @test lmul!(Q, copy(x)) ≈ Matrix(Q)*x
        @test lmul!(Q, copy(b)) ≈ Matrix(Q)*b
        @test lmul!(Q', copy(x)) ≈ Matrix(Q)'*x
        @test lmul!(Q', copy(b)) ≈ Matrix(Q)'*b
        c = randn(2,100)
        @test rmul!(copy(c), Q) ≈ c*Matrix(Q)
        @test rmul!(copy(c), Q') ≈ c*Matrix(Q')

        A = randn(100,103)
        Q,L = ql(A)
        x = randn(100)
        b = randn(100,2)
        @test lmul!(Q, copy(x)) ≈ Matrix(Q)*x
        @test lmul!(Q, copy(b)) ≈ Matrix(Q)*b
        @test lmul!(Q', copy(x)) ≈ Matrix(Q)'*x
        @test lmul!(Q', copy(b)) ≈ Matrix(Q)'*b
        c = randn(2,100)
        @test rmul!(copy(c), Q) ≈ c*Matrix(Q)
        @test rmul!(copy(c), Q') ≈ c*Matrix(Q')
    end

    @testset "Q * Tri" begin
        A = randn(5,5)
        Q,L = ql(A)
        @test Q * LowerTriangular(L) ≈ A
        @test LowerTriangular(L) * Q ≈ L * Q
        @test Q' * LowerTriangular(L) ≈ Q' * L
        @test LowerTriangular(L)' * Q' ≈ A'
    end

    @testset "row/col slices via copymutable" begin
        n = 5
        A = randn(n,n)
        Q, L = ql(A)
        @test Q[2,3:4] == [Q[2,k] for k=3:4]
        @test Q[3:4,2] == [Q[k,2] for k=3:4]
    end

    @testset "Q * Q" begin
        Q = ql(randn(5,5)).Q
        @test Q*Q ≈ Matrix(Q)^2
    end

    @testset "row/colsupport" begin
        n = 10
        B = Bidiagonal(randn(n), randn(n-1), :U)
        Q = ql(B).Q
        Q = MatrixFactorizations.QLPackedQ(Tridiagonal(Q.factors), Q.τ)
        @test rowsupport(Q, 4) ≡ colsupport(Q', 4) ≡ Base.OneTo(5)
        @test colsupport(Q, 4) ≡ rowsupport(Q', 4) ≡ 3:10
    end

    @testset "AbstractMatrix conversion" begin
        Q = ql(randn(5,5)).Q
        @test AbstractMatrix{Float64}(Q) isa Matrix{Float64}
    end
end