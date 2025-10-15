using LinearAlgebra, MatrixFactorizations, Random, Test

@testset "A = LUL⁻¹" begin
    @testset "A::Matrix{Int}" begin
        V = [2 3 16; 1 -1 5; 0 1 1//1]
        λ = [17//1; -2; 3]
        A = det(V)*V*Diagonal(λ)/V
        @test A == Matrix{Int}(A)
        L, U = lulinv(A, λ)
        @test A ≈ L*U/L
        @test A*L ≈ L*U
    end
    @testset "A::Matrix{Rational{Int}}" begin
        Random.seed!(0)
        V = rand(-3:3//1, 5, 5)
        λ = [-2; -1; 0; 1; 2//1]
        A = V*Diagonal(λ)/V
        L, U = lulinv(A, λ)
        @test A ≈ L*U/L
        @test A*L ≈ L*U
    end
    @testset "Full Jordan blocks" begin
        Random.seed!(0)
        V = rand(-3:3//1, 5, 5)
        λ = [-2; -2; 1; 1; 1//1]
        J = diagm(0=> λ)
        J[1, 2] = 1
        J[3, 4] = 1
        J[4, 5] = 1
        A = V*J/V
        L, U = lulinv(A, λ)
        @test A ≈ L*U/L
        @test A*L ≈ L*U
    end
    @testset "Alg ≠ geo" begin
        Random.seed!(0)
        V = rand(-3:3//1, 5, 5)
        λ = [-2; -2; 1; 1; 1//1]
        J = diagm(0 => λ)
        J[1, 2] = 1
        J[3, 4] = 1
        A = V*J/V
        L, U = lulinv(A, λ)
        @test A ≈ L*U/L
        @test A*L ≈ L*U
        λ = [1; -2; -2; 1; 1//1]
        L, U = lulinv(A, λ)
        @test A ≈ L*U/L
        @test A*L ≈ L*U
        λ = [1; -2; 1; -2; 1//1]
        L, U = lulinv(A, λ)
        @test A ≈ L*U/L
        @test A*L ≈ L*U
    end
    @testset "Index scan is correct" begin
        Random.seed!(0)
        V = rand(-3:3//1, 8, 8)
        J1 = diagm(0 => 1//1*ones(Int, 4), 1 => rand(0:1, 3))
        J2 = diagm(0 => -1//1*ones(Int, 4), 1 => rand(0:1, 3))
        J = [J1 zeros(Rational{Int}, size(J1)); zeros(Rational{Int}, size(J2)) J2]
        λ = diag(J)
        A = V*J/V
        L, U = lulinv(A, λ)
        @test A ≈ L*U/L
        @test A*L ≈ L*U
    end
    @testset "A::Matrix{Float64}" begin
        A = [4.0 3; 6 3]
        L, U = lulinv(A)
        @test A ≈ L*U/L
        @test A*L ≈ L*U
    end
    @testset "size, det, logdet, logabsdet" begin
        A = [4.0 3; 6 3]
        F = lulinv(A)
        @test size(A) == size(F)
        @test size(F, 1) == size(F, 2)
        @test F\A ≈ I
        @test A/F ≈ I
        @test det(A) ≈ det(F)
        @test_throws DomainError logdet(F)
        lad, sd = logabsdet(A)
        lad1, sd1 = logabsdet(F)
        @test lad ≈ lad1
        @test sd ≈ sd1
    end
    @testset "REPL printing" begin
            bf = IOBuffer()
            show(bf, "text/plain", lulinv([1 0; 0 1]))
            seekstart(bf)
            @test String(take!(bf)) ==
"""
LULinv{Float64, Matrix{Float64}}
L factor:
2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
U factor:
2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0"""
    end
    @testset "propertynames" begin
        names = sort!(collect(string.(Base.propertynames(lulinv([2 1; 1 2])))))
        @test names == ["L", "U"]
        allnames = sort!(collect(string.(Base.propertynames(lulinv([2 1; 1 2]), true))))
        @test allnames == ["L", "U", "factors"]
    end
    @testset "A::Matrix{Int64}, λ::Vector{Rational{Int64}}" begin
        A = [-150 334 778; -89 195 464; 5 -10 -27]
        F = lulinv(A, [17, -2, 3//1])
        @test A * F.L == F.L * F.U
        G = LULinv{Float64}(F)
        @test A * G.L ≈ G.L * G.U
    end
end
