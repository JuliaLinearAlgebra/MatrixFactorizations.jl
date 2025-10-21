using LinearAlgebra, MatrixFactorizations, Random, Test

@testset "Jordan Canonical Form" begin
    Random.seed!(0)
    V = rand(-3:3//1, 5, 5)
    λ = [1; 1; 1; -2; -2//1]
    J = diagm(0 => λ)
    J[1, 2] = 1
    #J[3, 4] = 1
    J[4, 5] = 1
    A = V*J/V
    F = lulinv(A, λ)
    L, U, p = F.L, F.U, F.p
    @test A[p, p] ≈ L*U/L
    E, R = MatrixFactorizations.triangular_to_psychologically_block_diagonal(UpperTriangular(U))
    @test U ≈ E*R/E
    P, B = MatrixFactorizations.psychologically_block_diagonal_to_block_diagonal(R)
    @test R ≈ P*B*P'
    EP, B = MatrixFactorizations.triangular_to_block_diagonal(UpperTriangular(U))
    @test U ≈ EP*B/EP
    PLEP, B = MatrixFactorizations.block_diagonalize(A, λ)
    @test A ≈ PLEP*B/PLEP

    @testset "Block sizes" begin
        import MatrixFactorizations: determine_block_sizes
        m = determine_block_sizes([1;;])
        @test m == [1]
        m = determine_block_sizes([1 2; 3 4])
        @test m == [1, 1]
        m = determine_block_sizes([1 2; 3 1])
        @test m == [2]
        m = determine_block_sizes([1 2 3;0 1 4; 0 0 1])
        @test m == [3]
        m = determine_block_sizes([1 2 3;0 2 4; 0 0 2])
        @test m == [1, 2]
        m = determine_block_sizes([2 2 3;0 2 4; 0 0 1])
        @test m == [2, 1]
        m = determine_block_sizes([1 2 3;0 2 4; 0 0 3])
        @test m == [1, 1, 1]
        m = determine_block_sizes([1 2 3 4;0 2 5 6; 0 0 3 7; 0 0 0 4])
        @test m == [1, 1, 1, 1]
        m = determine_block_sizes([1 2 3 4;0 2 5 6; 0 0 3 7; 0 0 0 3])
        @test m == [1, 1, 2]
        m = determine_block_sizes([1 2 3 4;0 2 5 6; 0 0 2 7; 0 0 0 3])
        @test m == [1, 2, 1]
        m = determine_block_sizes([1 2 3 4;0 3 5 6; 0 0 3 7; 0 0 0 3])
        @test m == [1, 3]
        m = determine_block_sizes([3 2 3 4;0 3 5 6; 0 0 3 7; 0 0 0 3])
        @test m == [4]
        m = determine_block_sizes([1 2 3 4;0 1 5 6; 0 0 3 7; 0 0 0 4])
        @test m == [2, 1, 1]
        m = determine_block_sizes([1 2 3 4;0 1 5 6; 0 0 3 7; 0 0 0 3])
        @test m == [2, 2]
        m = determine_block_sizes([1 2 3 4;0 1 5 6; 0 0 1 7; 0 0 0 3])
        @test m == [3, 1]
        m = determine_block_sizes(UpperTriangular([1 2 3 4;0 1 5 6; 0 0 1 7; 0 0 0 3]))
        @test m == [3, 1]
        Random.seed!(0)
        m = determine_block_sizes(rand(0:1, 10, 10))
        @test m == [1, 2, 2, 2, 1, 1, 1]
        m = determine_block_sizes(rand(0:1, 20, 20))
        @test m == [1, 3, 4, 6, 2, 2, 1, 1]
    end

    @testset "Upper triangular block to Jordan blocks" begin
        import MatrixFactorizations: upper_triangular_block_to_jordan_blocks
        Random.seed!(0)
        B = diagm(0 => 3//1*ones(Int, 12), 1 => [1,1,0,1,1,0,1,0,1,1,0]) + rand(-3:3, 12)*[zeros(Int,11); 1]'; B[end] = 3; B
        F, J = upper_triangular_block_to_jordan_blocks(B)
        @test B ≈ F*J/F

        B = diagm(0 => 4//1*ones(Int, 12)) + triu!(rand(-4:4, 12, 12), 1)
        F, J = upper_triangular_block_to_jordan_blocks(B)
        @test B ≈ F*J/F

        B = diagm(0 => 4//1*ones(Int, 12)) + triu!(rand(-4:4, 12, 12), 1)
        B = Rational{BigInt}.(B)
        F, J = upper_triangular_block_to_jordan_blocks(B)
        @test B ≈ F*J/F

        B = diagm(0 => 4//1*ones(Int, 12)) + triu!(rand(-4:4, 12, 12), 1)
        F, J = upper_triangular_block_to_jordan_blocks(B)
        @test B ≈ F*J/F

        B = [2 1 2; 0 2 1; 0 0 2//1]
        F, J = upper_triangular_block_to_jordan_blocks(B)
        @test B ≈ F*J/F

        B = [2 1 4 0; 0 2 1 0; 0 0 2 0; 0 0 0 2//1]
        F, J = upper_triangular_block_to_jordan_blocks(B)
        @test B ≈ F*J/F

        B = [2 1 4 0; 0 2 1 -1; 0 0 2 1; 0 0 0 2//1]
        F, J = upper_triangular_block_to_jordan_blocks(B)
        @test B ≈ F*J/F

        F = Matrix{Rational{Int}}(triu!(rand(1:4, 12, 12)))
        J = diagm(0 => 4//1*ones(Int, 12), 1 => rand(0:1, 11))
        B = Matrix{Rational{BigInt}}(F*J/F)
        F, J = upper_triangular_block_to_jordan_blocks(B)
        @test B ≈ F*J/F
    end

    @testset "JCF" begin
        Random.seed!(0)
        for _ in 1:10
            V = rand(-3:3//1, 8, 8)
            J1 = diagm(0 => 1//1*ones(Int, 4), 1 => rand(0:1, 3))
            J2 = diagm(0 => -1//1*ones(Int, 4), 1 => rand(0:1, 3))
            J = [J1 zeros(Rational{Int}, size(J1)); zeros(Rational{Int}, size(J2)) J2]
            λ = diag(J)
            A = V*J/V
            V, J = jordan(A, λ)
            @test A*V ≈ V*J
        end
    end

    @testset "Jordan canonical forms in the wild" begin
        @testset "https://en.wikipedia.org/wiki/Jordan_normal_form" begin
            A = [5 4 2 1; 0 1 -1 -1; -1 -1 3 0; 1 1 -1 2//1]
            λ = [1, 2, 4, 4//1]
            @test λ ≈ eigvals(A)
            V, J = jordan(A, λ)
            @test A*V == V*J
            # Next step in this code would be to completely elucidate the block structure in J:
            #JCF = BlockArrays.BlockDiagonal([JordanBlock(1//1, 1), JordanBlock(2//1, 1), JordanBlock(4//1, 2)])
            #@test J == JCF
        end
        @testset "https://math.stackexchange.com/questions/1221465/jordan-canonical-form-deployment" begin
            A = [-3 1 0 1 1; -3 1 0 1 1; -4 1 0 2 1; -3 1 0 1 1; -4 1 0 1 2//1]
            λ = [0, 0, 0, 0, 1//1]
            @test λ ≈ eigvals(A)
            V, J = jordan(A, λ)
            @test A*V == V*J
        end
        @testset "https://math.stackexchange.com/questions/3627150/find-the-jordan-form" begin
            A = [-83 -15 -68 -2 -100 216; -3 1 -2 0 -4 9; 42 7 35 1 50 -108; 43 8 35 3 50 -108; 22 4 18 1 27 -54; -10 -2 -8 0 -12 28//1]
            λ = [1, 2, 2, 2, 2, 2//1]
            @test all(det(A-λ[j]*I) == 0 for j in 1:length(λ))
            V, J = jordan(A, λ)
            @test A*V == V*J
        end
        @testset "https://www.maplesoft.com/applications/preview.aspx?id=33195" begin
            A = [-10 -13 15 -16 6 9 -3; 4 5 -6 7 -2 -4 1; -50 -60 72 -77 29 43 -13; -82 -98 118 -127 47 71 -21; 2 1 -2 2 -1 -1 0; -56 -66 80 -86 32 48 -14; 48 59 -70 75 -28 -42 13//1]
            λ = zeros(Rational{Int}, 7)
            V, J = jordan(A, λ)
            @test A*V == V*J
        end
        @testset "https://askfilo.com/user-question-answers-smart-solutions/find-the-jordan-canonical-form-j-for-the-matrix-also-find-a-3337353836383137" begin
            A = [-1 0 -2 -4; 2 1 2 4; -4 2 -1 -4; 2 -1 1 3//1]
            λ = [-1, 1, 1, 1//1]
            V, J = jordan(A, λ)
            @test A*V == V*J
        end
        @testset "https://www.bartleby.com/questions-and-answers/1.-find-the-jordan-canonical-form-for-the-following-matrices.-0-0-0-0-1-2-0-4-a.-v-2.-1.-1-2-0.-0-0-/2193cc71-c370-45ea-bfed-d5cfe17c8946" begin
            B = [0 1 0 0; -3 4 0 0; 2 -1 2 0; -1 1 1 2//1]
            λ = [1, 2, 2, 3//1]
            V, J = jordan(B, λ)
            @test B*V == V*J
            A = [1 0 0 0 0; 1 -1 0 0 -1; 1 -1 0 0 -1; 0 0 0 0 -1; -1 1 0 0 1//1]
            λ = [0, 0, 0, 0, 1//1]
            V, J = jordan(A, λ)
            @test A*V == V*J
        end
        @testset "http://buzzard.ups.edu/courses/2014spring/420demos/JCF-demo.pdf" begin
            A = Rational{BigInt}[50 -10 17 -21 7 6 1 -37 -8 -10 10 -14 27 39 8;
                 186 -207 -36 -184 -74 -71 -54 -22 -1 -93 38 -11 56 187 14;
                 3 16 9 -10 -11 8 -4 -24 4 -6 2 -14 13 6 -2;
                 -132 202 62 162 83 77 58 -23 -14 86 -28 -4 -22 -149 -3;
                 213 -237 -41 -218 -94 -82 -65 -30 2 -111 45 -16 69 216 13;
                 -608 549 27 552 195 173 139 178 34 266 -124 72 -239 -574 -57;
                 253 -299 -57 -274 -125 -108 -83 -26 3 -137 52 -15 79 258 15;
                 -196 175 0 170 50 56 38 58 21 76 -38 18 -76 -176 -23;
                 704 -651 -40 -649 -233 -206 -167 -197 -35 -313 143 -80 272 668 64;
                 322 -238 25 -251 -53 -62 -51 -126 -39 -112 62 -43 139 283 42;
                 -536 411 -28 449 129 115 98 211 52 207 -105 79 -236 -482 -61;
                 170 -190 -25 -165 -58 -67 -45 -20 -11 -77 32 -4 51 163 17;
                 1 14 9 10 9 7 6 -7 -4 5 0 -1 7 -4 1;
                 223 -195 -1 -201 -64 -61 -49 -71 -20 -92 43 -26 91 206 25;
                 -58 4 -27 29 -7 -10 1 53 12 12 -11 20 -39 -45 -10]
            λ = Rational{BigInt}[-ones(5); 2*ones(10)]
            V, J = jordan(A, λ)
            @test A*V == V*J
        end
        @testset "http://matrix.skku.ac.kr/2014-Album/JCF/2GEV_JCF_u_GEV.htm" begin
            A = [0 0 1 7 -1; -5 -6 -6 -35 5; 1 1 -7 7 -1; 0 0 0 -9 0; 2 1 -5 -42 -3//1]
            λ = [-9, -7, -7, -1, -1//1]
            V, J = jordan(A, λ)
            @test A*V == V*J
        end
        @testset "https://www.numerade.com/ask/question/1-find-a-jordan-canonical-form-of-the-following-matrix-4-1-0-0-a-h4ih-1-28093/" begin
            A = [4 1 0 0 0; -1 3 1 0 0; 1 0 2 0 0; -2 -1 -1 2 1; 1 0 0 0 2//1]
            λ = [2, 2, 3, 3, 3//1]
            @test all(det(A-λ[j]*I) == 0 for j in 1:length(λ))
            V, J = jordan(A, λ)
            @test A*V == V*J
        end
        @testset "https://www.chegg.com/homework-help/questions-and-answers/3-find-jordan-canonical-form-j-following-matrices-determine-matrix-x-xax-j-11-0-1-1-2-0-0--q25805340" begin
            A = [1 0 1; 1 0 2; 1 -1 2]
            λ = [1, 1, 1]
            @test all(det(A-λ[j]*I) == 0 for j in 1:length(λ))
            V, J = jordan(A, λ)
            @test A*V == V*J
            A = [0 0 0 1; 0 0 0 1; 1 2 0 0; 0 0 0 -1.0]
            V, J = jordan(A, eigvals(A))
            @test A*V ≈ V*J
            A = [1 2 0 0; 0 1 2 0; 0 0 1 2; 0 0 0 1//1]
            λ = diag(A) # because A is upper triangular
            V, J = jordan(A, λ)
            @test A*V == V*J
            A = [1 1 1 1 1; 0 1 1 1 1; 0 0 1 1 1; 0 0 0 0 1; 0 0 0 0 0//1]
            λ = diag(A) # because A is upper triangular
            V, J = jordan(A, λ)
            @test A*V == V*J
            A = [2 1 1 1 1 1; 0 2 1 1 1 1; 0 0 0 1 1 1; 0 0 0 0 1 1; 0 0 0 0 1 1; 0 0 0 0 1 1//1]
            λ = Rational{Int}.(eigvals(A))
            @test λ ≈ eigvals(A)
            V, J = jordan(A, λ)
            @test A*V == V*J
        end
        @testset "https://www.chegg.com/homework-help/questions-and-answers/solve-using-matlab-find-jordan-canonical-form-b-determine-matrix-diagonalizable-find-simil-q46775702" begin
            A = [0 1 -1 2 1 -1; -4 3 -1 3 1 0; 0 1 1 3 -1 -2; 0 1 -1 4 0 -2; 0 1 -1 2 2 -2; -1 1 -1 2 1 0//1]
            λ = [1, 1, 2, 2, 2, 2//1]
            V, J = jordan(A, λ)
            @test A*V == V*J
        end
        @testset "https://www.chegg.com/homework-help/questions-and-answers/question-2-calculate-jordan-canonical-form-matrix-0-0-0-2-0-0-0-2-0-0-2-1-0-0-1-0-1-1-1-1--q58058571" begin
            A = [2 0 0 0 0 0; 0 2 0 0 0 0; 1 0 2 0 0 0; -1 0 0 -3 0 0; -1 0 1 1 -3 0; -1 1 -1 1 -1 -3//1]
            λ = diag(A) # because A is lower triangular
            V, J = jordan(A, λ)
            @test A*V == V*J
        end
    end

    @testset "REPL printing" begin
            bf = IOBuffer()
            show(bf, "text/plain", jordan([1 0; 0 1]))
            seekstart(bf)
            @test String(take!(bf)) ==
"""
Jordan{Float64, Matrix{Float64}, Matrix{Float64}}
Generalized eigenvectors:
2×2 Matrix{Float64}:
 0.0  1.0
 1.0  0.0
Jordan normal form:
2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0"""
    end

    @testset "propertynames" begin
        names = sort!(collect(string.(Base.propertynames(jordan([2 1; 1 2])))))
        @test names == ["J", "V"]
        allnames = sort!(collect(string.(Base.propertynames(jordan([2 1; 1 2]), true))))
        @test allnames == ["J", "V"]
    end

    @testset "Mixed input types, conversion" begin
        A = [0 0 1 7 -1; -5 -6 -6 -35 5; 1 1 -7 7 -1; 0 0 0 -9 0; 2 1 -5 -42 -3//1]
        λ = [-9, -7, -7, -1, -1]
        V, J = jordan(A, λ)
        @test A*V == V*J
        A = [0 0 1 7 -1; -5 -6 -6 -35 5; 1 1 -7 7 -1; 0 0 0 -9 0; 2 1 -5 -42 -3]
        λ = [-9, -7, -7, -1, -1//1]
        V, J = jordan(A, λ)
        @test A*V == V*J
        F = jordan(A, λ)
        G = Jordan{Float64}(F)
        @test A*G.V ≈ G.V*G.J
    end
end
