# This file is based on LinearAlgebra/test/lu.jl, a part of Julia. License is MIT: https://julialang.org/license

using Test, LinearAlgebra, Random, MatrixFactorizations
using LinearAlgebra: ldiv!, BlasReal, BlasInt, BlasFloat, rdiv!

@testset "UL" begin
    n = 10

    # Split n into 2 parts for tests needing two matrices
    n1 = div(n, 2)
    n2 = 2*n1

    Random.seed!(1234321)

    areal = randn(n,n)/2
    aimg  = randn(n,n)/2
    breal = randn(n,2)/2
    bimg  = randn(n,2)/2
    creal = randn(n)/2
    cimg  = randn(n)/2
    dureal = randn(n-1)/2
    duimg  = randn(n-1)/2
    dlreal = randn(n-1)/2
    dlimg  = randn(n-1)/2
    dreal = randn(n)/2
    dimg  = randn(n)/2

    @testset "UL simple test" begin
        A = randn(n,n)
        U,L = ul(A,Val(false))
        @test U*L ≈ A
        U,L,p = ul(A)
        @test (U*L) ≈ A[p,:]
        P = ul(A).P
        @test P'*U*L ≈ A
        L,U,p = lu(A)
        @test (L*U) ≈ A[p,:] # for context
        P = lu(A).P
        @test P'*L*U ≈ A
    end

    @testset for eltya in (Float32, Float64, ComplexF32, ComplexF64, BigFloat, Int)
        a = eltya == Int ? rand(1:7, n, n) :
            convert(Matrix{eltya}, eltya <: Complex ? complex.(areal, aimg) : areal)
        d = if eltya == Int
            Tridiagonal(rand(1:7, n-1), rand(1:7, n), rand(1:7, n-1))
        elseif eltya <: Complex
            convert(Tridiagonal{eltya}, Tridiagonal(
                complex.(dlreal, dlimg), complex.(dreal, dimg), complex.(dureal, duimg)))
        else
            convert(Tridiagonal{eltya}, Tridiagonal(dlreal, dreal, dureal))
        end
        ε = εa = eps(abs(float(one(eltya))))

        if eltya <: BlasFloat
            @testset "UL factorization for Number" begin
                num = rand(eltya)
                @test (ul(num)...,) == (hcat(one(eltya)), hcat(num), [1])
                @test convert(Array, ul(num)) ≈ eltya[num]
            end
        end
        κ  = cond(a,1)
        κd    = cond(Array(d),1)
        # @testset "Tridiagonal UL" begin
        #     lud   = ul(d)
        #     @test LinearAlgebra.issuccess(lud)
        #     @test ul(lud) == lud
        #     @test_throws ErrorException lud.Z
        #     @test lud.L*lud.U ≈ lud.P*Array(d)
        #     @test lud.L*lud.U ≈ Array(d)[lud.p,:]
        #     @test AbstractArray(lud) ≈ d
        #     @test Array(lud) ≈ d
        #     if eltya != Int
        #         dlu = convert.(eltya, [1, 1])
        #         dia = convert.(eltya, [-2, -2, -2])
        #         tri = Tridiagonal(dlu, dia, dlu)
        #         @test_throws ArgumentError ul!(tri)
        #     end
        # end
        @testset for eltyb in (Float32, Float64, ComplexF32, ComplexF64, Int)
            b  = eltyb == Int ? rand(1:5, n, 2) :
                convert(Matrix{eltyb}, eltyb <: Complex ? complex.(breal, bimg) : breal)
            c  = eltyb == Int ? rand(1:5, n) :
                convert(Vector{eltyb}, eltyb <: Complex ? complex.(creal, cimg) : creal)
            εb = eps(abs(float(one(eltyb))))
            ε  = max(εa,εb)
            @testset "(Automatic) Square UL decomposition" begin
                lua   = factorize(a)
                let Bs = copy(b), Cs = copy(c)
                    for (bb, cc) in ((Bs, Cs), (view(Bs, 1:n, 1), view(Cs, 1:n)))
                        @test norm(a*(lua\bb) - bb, 1) < ε*κ*n*2 # Two because the right hand side has two columns
                        @test norm(a'*(lua'\bb) - bb, 1) < ε*κ*n*2 # Two because the right hand side has two columns
                        @test norm(a'*(lua'\a') - a', 1) < ε*κ*n^2
                        @test norm(a*(lua\cc) - cc, 1) < ε*κ*n # cc is a vector
                        @test norm(a'*(lua'\cc) - cc, 1) < ε*κ*n # cc is a vector
                        @test AbstractArray(lua) ≈ a
                        @test norm(transpose(a)*(transpose(lua)\bb) - bb,1) < ε*κ*n*2 # Two because the right hand side has two columns
                        @test norm(transpose(a)*(transpose(lua)\cc) - cc,1) < ε*κ*n
                    end

                    # Test whether Ax_ldiv_B!(y, UL, x) indeed overwrites y
                    resultT = typeof(oneunit(eltyb) / oneunit(eltya))

                    b_dest = similar(b, resultT)
                    c_dest = similar(c, resultT)

                    ldiv!(b_dest, lua, b)
                    ldiv!(c_dest, lua, c)
                    @test norm(b_dest - lua \ b, 1) < ε*κ*2n
                    @test norm(c_dest - lua \ c, 1) < ε*κ*n

                    ldiv!(b_dest, transpose(lua), b)
                    ldiv!(c_dest, transpose(lua), c)
                    @test norm(b_dest - transpose(lua) \ b, 1) < ε*κ*2n
                    @test norm(c_dest - transpose(lua) \ c, 1) < ε*κ*n

                    ldiv!(b_dest, adjoint(lua), b)
                    ldiv!(c_dest, adjoint(lua), c)
                    @test norm(b_dest - lua' \ b, 1) < ε*κ*2n
                    @test norm(c_dest - lua' \ c, 1) < ε*κ*n

                    if eltyb != Int && !(eltya <: Complex) || eltya <: Complex && eltyb <: Complex
                        p = Matrix(b')
                        q = Matrix(c')
                        p_dest = copy(p)
                        q_dest = copy(q)
                        rdiv!(p_dest, lua)
                        rdiv!(q_dest, lua)
                        @test norm(p_dest - p / lua, 1) < ε*κ*2n
                        @test norm(q_dest - q / lua, 1) < ε*κ*n
                    end
                end
                if eltya <: BlasFloat && eltyb <: BlasFloat
                    e = rand(eltyb,n,n)
                    @test norm(e/lua - e/a,1) < ε*κ*n^2
                end
            end
            # @testset "Tridiagonal UL" begin
            #     lud   = factorize(d)
            #     f = zeros(eltyb, n+1)
            #     @test_throws DimensionMismatch lud\f
            #     @test_throws DimensionMismatch transpose(lud)\f
            #     @test_throws DimensionMismatch lud'\f
            #     @test_throws DimensionMismatch LinearAlgebra.ldiv!(transpose(lud), f)
            #     let Bs = copy(b)
            #         for bb in (Bs, view(Bs, 1:n, 1))
            #             @test norm(d*(lud\bb) - bb, 1) < ε*κd*n*2 # Two because the right hand side has two columns
            #             if eltya <: Real
            #                 @test norm((transpose(lud)\bb) - Array(transpose(d))\bb, 1) < ε*κd*n*2 # Two because the right hand side has two columns
            #                 if eltya != Int && eltyb != Int
            #                     @test norm(LinearAlgebra.ldiv!(transpose(lud), copy(bb)) - Array(transpose(d))\bb, 1) < ε*κd*n*2
            #                 end
            #             end
            #             if eltya <: Complex
            #                 @test norm((lud'\bb) - Array(d')\bb, 1) < ε*κd*n*2 # Two because the right hand side has two columns
            #             end
            #         end
            #     end
            #     if eltya <: BlasFloat && eltyb <: BlasFloat
            #         e = rand(eltyb,n,n)
            #         @test norm(e/lud - e/d,1) < ε*κ*n^2
            #         @test norm((transpose(lud)\e') - Array(transpose(d))\e',1) < ε*κd*n^2
            #         #test singular
            #         du = rand(eltya,n-1)
            #         dl = rand(eltya,n-1)
            #         dd = rand(eltya,n)
            #         dd[1] = zero(eltya)
            #         du[1] = zero(eltya)
            #         dl[1] = zero(eltya)
            #         zT = Tridiagonal(dl,dd,du)
            #         @test !LinearAlgebra.issuccess(ul(zT; check = false))
            #     end
            # end
            # @testset "Thin UL" begin
            #     lua   = @inferred ul(a[:,1:n1])
            #     @test lua.U*lua.L ≈ lua.P'*a[:,1:n1]
            # end
            # @testset "Fat UL" begin
            #     lua   = ul(a[1:n1,:])
            #     @test lua.L*lua.U ≈ lua.P*a[1:n1,:]
            # end
        end

        # @testset "UL of Symmetric/Hermitian" begin
        #     for HS in (Hermitian(a'a), Symmetric(a'a))
        #         luhs = ul(HS)
        #         @test luhs.L*luhs.U ≈ luhs.P*Matrix(HS)
        #     end
        # end
    end

    @testset "Singular matrices" for T in (Float64, ComplexF64)
        A = T[1 2; 0 0]
        @test_throws SingularException ul(A)
        @test_throws SingularException ul!(copy(A))
        @test_throws SingularException ul(A; check = true)
        @test_throws SingularException ul!(copy(A); check = true)
        @test !issuccess(ul(A; check = false))
        @test !issuccess(ul!(copy(A); check = false))
        @test_throws ZeroPivotException ul(A, Val(false))
        @test_throws ZeroPivotException ul!(copy(A), Val(false))
        @test_throws ZeroPivotException ul(A, Val(false); check = true)
        @test_throws ZeroPivotException ul!(copy(A), Val(false); check = true)
        @test !issuccess(ul(A, Val(false); check = false))
        @test !issuccess(ul!(copy(A), Val(false); check = false))
        F = ul(A; check = false)
        @test sprint((io, x) -> show(io, "text/plain", x), F) ==
            "Failed factorization of type $(typeof(F))"
    end

    # @testset "conversion" begin
    #     Random.seed!(3)
    #     a = Tridiagonal(rand(9),rand(10),rand(9))
    #     fa = Array(a)
    #     falu = ul(fa)
    #     alu = ul(a)
    #     falu = convert(typeof(falu),alu)
    #     @test Array(alu) == fa
    #     @test AbstractArray(alu) == fa
    # end

    @testset "Rational Matrices" begin
        ## Integrate in general tests when more linear algebra is implemented in julia
        a = convert(Matrix{Rational{BigInt}}, rand(1:10//1,n,n))/n
        b = rand(1:10,n,2)
        @inferred ul(a)
        lua   = factorize(a)
        l,u,p = lua.L, lua.U, lua.p
        @test l*u ≈ a[p,:]
        @test l[invperm(p),:]*u ≈ a
        @test a*inv(lua) ≈ Matrix(I, n, n)
        let Bs = b
            for b in (Bs, view(Bs, 1:n, 1))
                @test a*(lua\b) ≈ b
            end
        end
        @test @inferred(det(a)) ≈ det(Array{Float64}(a))
    end

    @testset "Rational{BigInt} and BigFloat Hilbert Matrix" begin
        ## Hilbert Matrix (very ill conditioned)
        ## Testing Rational{BigInt} and BigFloat version
        nHilbert = 50
        H = Rational{BigInt}[1//(i+j-1) for i = 1:nHilbert,j = 1:nHilbert]
        Hinv = Rational{BigInt}[(-1)^(i+j)*(i+j-1)*binomial(nHilbert+i-1,nHilbert-j)*
            binomial(nHilbert+j-1,nHilbert-i)*binomial(i+j-2,i-1)^2
            for i = big(1):nHilbert,j=big(1):nHilbert]
        @test inv(H) == Hinv
        setprecision(2^10) do
            @test norm(Array{Float64}(inv(float(H)) - float(Hinv))) < 1e-100
        end
    end

    @testset "logdet" begin
        @test @inferred(logdet(ComplexF32[1.0f0 0.5f0; 0.5f0 -1.0f0])) === 0.22314355f0 + 3.1415927f0im
        @test_throws DomainError logdet([1 1; 1 -1])
    end

    @testset "REPL printing" begin
            bf = IOBuffer()
            show(bf, "text/plain", ul(Matrix(I, 4, 4)))
            seekstart(bf)
            @test String(take!(bf)) == """
UL{Float64,Array{Float64,2},Array{Int64,1}}
U factor:
4×4 Array{Float64,2}:
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0
 0.0  0.0  0.0  1.0
L factor:
4×4 Array{Float64,2}:
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0
 0.0  0.0  0.0  1.0"""
    end

    @testset "propertynames" begin
        names = sort!(collect(string.(Base.propertynames(ul(rand(3,3))))))
        @test names == ["L", "P", "U", "p"]
        allnames = sort!(collect(string.(Base.propertynames(ul(rand(3,3)), true))))
        @test allnames == ["L", "P", "U", "factors", "info", "ipiv", "p"]
    end

    @testset "Issue #30917. Determinant of integer matrix" begin
        @test det([1 1 0 0 1 0 0 0
                1 0 1 0 0 1 0 0
                1 0 0 1 0 0 1 0
                0 1 1 1 0 0 0 0
                0 1 0 0 0 0 1 1
                0 0 1 0 1 0 0 1
                0 0 0 1 1 1 0 0
                0 0 0 0 1 1 0 1]) ≈ 6
    end

    @testset "Issue #33177. No ldiv!(UL, Adjoint)" begin
        A = [1 0; 1 1]
        B = [1 2; 2 8]
        F = ul(B)
        @test (A  / F') * B == A
        @test (A' / F') * B == A'

        a = complex.(randn(2), randn(2))
        @test (a' / F') * B ≈ a'
        @test (transpose(a) / F') * B ≈ transpose(a)

        A = complex.(randn(2, 2), randn(2, 2))
        @test (A' / F') * B ≈ A'
        @test (transpose(A) / F') * B ≈ transpose(A)
    end

    @testset "0x0 matrix" begin
        A = ones(0, 0)
        F = ul(A)
        @test F.U == ones(0, 0)
        @test F.L == ones(0, 0)
        @test F.P == ones(0, 0)
        @test F.p == []
    end

    @testset "more ldiv! methods" begin
        for elty in (Float16, Float64, ComplexF64), transform in (transpose, adjoint)
            A = randn(elty, 5, 5)
            B = randn(elty, 5, 5)
            @test ldiv!(transform(ul(A)), copy(B)) ≈ transform(A) \ B
            @test ldiv!(transform(ul(A)), transform(copy(B))) ≈ transform(A) \ transform(B)
        end
    end

    @testset "more rdiv! methods" begin
        A = randn(5,5)
        B = randn(5,5)
        @test rdiv!(copy(A), ul(B)) ≈ A / B

        for elty in (Float16, Float64, ComplexF64), transform in (transpose, adjoint)
            A = randn(elty, 5, 5)
            C = copy(A)
            B = randn(elty, 5, 5)
            @test rdiv!(transform(A), transform(ul(B))) ≈ transform(C) / transform(B)
        end
    end

    @testset "conversions" begin
        A = randn(5,5)
        F = ul(A)
        @test ul(F) ≡ F ≡ Factorization{Float64}(F)
        @test UL{Float32}(F).factors ≈ UL{Float32,Matrix{Float32}}(F).factors ≈
                Factorization{Float32}(F).factors ≈ ul(Float32.(A)).factors
        @test copy(F).factors == F.factors
    end

    @testset "det" begin
        A = randn(5,5)
        @test det(A) ≈ det(ul(A))
        @test all(logabsdet(A) .≈ logabsdet(ul(A)))
    end

    @testset "inv" begin
        A = randn(5,5)
        @test inv(ul(A)) ≈ inv(A)
    end
end