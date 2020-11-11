@testset "Polar Decomposition" begin

Random.seed!(1103)

for T in [Float64, Complex{Float64}]
    @testset "Newton $T" begin
        for n in [1,3,10]

            A = rand(T,n,n)

            r = polar(A, alg = :newton);

            # Test unitary matrix U

            U = r.U
            H = r.H

            @test U'*U ≈ Matrix(I,n,n) atol=1e-7

            # Test Hermitian positive semifefinite matrix H

            @test ishermitian(H)

            for i in eigvals(H)
                @test i >= 0.
            end

            @test A ≈ U*H

        end
    end
end

for T in [Float64, Complex{Float64}]
    @testset "Halley $T" begin
        for n in [1,3,10]

            A = rand(T, n, n)

            r = polar(A, alg = :halley);


            # Test unitary matrix U

            U = r.U
            H = r.H

            @test U'*U ≈ Matrix(I,n,n) atol=1e-7

            # Test Hermitian positive semifefinite matrix H

            @test ishermitian(H)

            for i in eigvals(H)
                @test i >= 0.
            end

            @test A ≈ U*H

        end
    end

    ##########################################################
    @testset "QDWH $T" begin
      for m in [1,3,10]

        B = rand(T, m, m)

        r2 = polar(B, alg = :qdwh);

        # Test unitary matrix U

        U2 = r2.U
        H2 = r2.H

        @test U2'*U2 ≈ Matrix(I,m,m) atol=1e-7

        # Test Hermitian positive semifefinite matrix H

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

            r = polar(A, alg = :svd);


            # Test unitary matrix U

            U = r.U
            H = r.H

            @test U'*U ≈ Matrix(I,n,n) atol=1e-7

            # Test Hermitian positive semifefinite matrix H

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

            r = polar(A, alg =:hybrid);


            # Test unitary matrix U

            U = r.U
            H = r.H

            @test U'*U ≈ Matrix(I,n,n) atol=1e-7

            # Test Hermitian positive semifefinite matrix H

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

        r1 = polar(A, alg =:newton);

        r2 = polar(A, alg =:qdwh);

        r3 = polar(A, alg =:halley);

        r4 = polar(A, alg =:svd);

        r5 = polar(A, alg =:hybrid);

        @test r1.U ≈ r2.U
        @test r2.U ≈ r3.U
        @test r3.U ≈ r4.U
        @test r4.U ≈ r5.U

    end
end


end
