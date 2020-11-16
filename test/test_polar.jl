@testset "Polar Decomposition" begin

Random.seed!(1103)

# For each variant, decomposing A to U, H, verify:
# U is unitary
# H is hermitian and positive semi-definite
# backward error is small

for T in [Float64, Complex{Float64}]
    @testset "Newton $T" begin
        for n in [1,3,10]
            A = rand(T,n,n)
            U,H = polar(A, alg = :newton)
            @test U'*U ≈ Matrix(I,n,n) atol=1e-7
            @test ishermitian(H)
            for i in eigvals(H)
                @test i >= 0.
            end
            @test A ≈ U*H

            # also check the properties here (once should suffice)
            r = polar(A, alg = :newton)
            @test U == r.U
            @test H == r.H

            r = polar(A, alg = :newton, verbose = true)
            @test U == r.U
            @test H == r.H

            m = n + 2
            A = rand(T,m,n)
            U,H = polar(A, alg = :newton)
            @test U'*U ≈ Matrix(I,n,n) atol=1e-7
            @test ishermitian(H)
            for i in eigvals(H)
                @test i >= 0.
            end
            @test A ≈ U*H
        end
        A = rand(T,5,7)
        @test_throws ArgumentError U,H = polar(A, alg = :newton)
    end
end

for T in [Float64, Complex{Float64}]
    @testset "Halley $T" begin
        for n in [1,3,10]
            A = rand(T, n, n)
            r = polar(A, alg = :halley)
            U = r.U
            H = r.H
            @test U'*U ≈ Matrix(I,n,n) atol=1e-7
            @test ishermitian(H)
            for i in eigvals(H)
                @test i >= 0.
            end
            @test A ≈ U*H
        end
    end

    @testset "QDWH $T" begin
      for m in [1,3,10]
        B = rand(T, m, m)
        U2,H2 = polar(B, alg = :qdwh)
        @test U2'*U2 ≈ Matrix(I,m,m) atol=1e-7
        @test ishermitian(H2)
        for i in eigvals(H2)
            @test i >= 0.
        end
        @test B ≈ U2*H2
      end
    end
end

for T in [Float64, Complex{Float64}]
    @testset "SVD $T" begin
        for n in [1,3,10]
            m = n + rand(0:5)
            A = rand(T, m, n)
            U,H = polar(A, alg = :svd)
            @test U'*U ≈ Matrix(I,n,n) atol=1e-7
            @test ishermitian(H)
            for i in eigvals(H)
                @test i >= 0.
            end
            @test A ≈ U*H
        end
    end
end

for T in [Float64, Complex{Float64}]
    @testset "Newton-Schulz $T" begin
        for n in [1,3,10]
            m = n
            A = rand(T, m, n)
            nA = opnorm(A)
            A = (1.5 / nA) * A
            U,H = polar(A, alg = :schulz)
            @test U'*U ≈ Matrix(I,n,n) atol=1e-7
            @test ishermitian(H)
            for i in eigvals(H)
                @test i >= 0.
            end
            @test A ≈ U*H
        end
    end
end

for T in [Float64, Complex{Float64}]
    @testset "Hybrid $T" begin
        for n in [1,3,10]
            A = rand(T,n,n)
            U,H = polar(A, alg =:hybrid)
            @test U'*U ≈ Matrix(I,n,n) atol=1e-7
            @test ishermitian(H)
            for i in eigvals(H)
                @test i >= 0.
            end
            @test A ≈ U*H atol=1e-7
        end
    end
end

@testset "Float32" begin
    for n in [1,3,10]
        A = Array{Float32}(undef, n, n)
        copyto!(A, rand(n,n))

        r1 = polar(A, alg =:newton)
        r2 = polar(A, alg =:qdwh)
        r3 = polar(A, alg =:halley)
        r4 = polar(A, alg =:svd)
        r5 = polar(A, alg =:hybrid)

        @test r1.U ≈ r2.U
        @test r2.U ≈ r3.U
        @test r3.U ≈ r4.U
        @test r4.U ≈ r5.U
    end
end


end
