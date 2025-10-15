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
        F = lulinv(A)
        @test F\A ≈ I
        @test A/F ≈ I
        @test det(A) ≈ det(F)
        @test_throws DomainError logdet(F)
        lad, sd = logabsdet(A)
        lad1, sd1 = logabsdet(F)
        @test lad ≈ lad1
        @test sd ≈ sd1
    end
end
