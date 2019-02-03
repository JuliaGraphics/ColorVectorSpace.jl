module ColorVectorSpaceTests

using LinearAlgebra, Statistics
using ColorVectorSpace, Colors, FixedPointNumbers, StatsBase

using Test

const var = Statistics.var

macro test_colortype_approx_eq(a, b)
    :(test_colortype_approx_eq($(esc(a)), $(esc(b)), $(string(a)), $(string(b))))
end

n8sum(x,y) = Float64(N0f8(x)) + Float64(N0f8(y))

function test_colortype_approx_eq(a::Colorant, b::Colorant, astr, bstr)
    @test typeof(a) == typeof(b)
    n = length(fieldnames(typeof(a)))
    for i = 1:n
        @test getfield(a, i) ≈ getfield(b,i)
    end
end

@testset "Colortypes" begin

    @testset "nan" begin
        function make_checked_nan(::Type{T}) where T
            x = nan(T)
            isa(x, T) && isnan(x)
        end
        for S in (Float32, Float64)
            @test make_checked_nan(S)
            @test make_checked_nan(Gray{S})
            @test make_checked_nan(AGray{S})
            @test make_checked_nan(GrayA{S})
            @test make_checked_nan(RGB{S})
            @test make_checked_nan(ARGB{S})
            @test make_checked_nan(ARGB{S})
        end
    end

    @testset "Arithmetic with Gray" begin
        cf = Gray{Float32}(0.1)
        @test +cf == cf
        @test -cf == Gray(-0.1f0)
        ccmp = Gray{Float32}(0.2)
        @test 2*cf == ccmp
        @test cf*2 == ccmp
        @test ccmp/2 == cf
        @test 2.0f0*cf == ccmp
        @test_colortype_approx_eq cf*cf Gray{Float32}(0.01)
        @test_colortype_approx_eq cf^2 Gray{Float32}(0.01)
        @test_colortype_approx_eq cf^3.0f0 Gray{Float32}(0.001)
        @test eltype(2.0*cf) == Float64
        @test abs2(ccmp) == 0.2f0^2
        @test norm(cf) == 0.1f0
        @test sum(abs2, ccmp) == 0.2f0^2
        cu = Gray{N0f8}(0.1)
        @test 2*cu == Gray(2*cu.val)
        @test 2.0f0*cu == Gray(2.0f0*cu.val)
        f = N0f8(0.5)
        @test (f*cu).val ≈ f*cu.val
        @test cf/2.0f0 == Gray{Float32}(0.05)
        @test cu/2 == Gray(cu.val/2)
        @test cu/0.5f0 == Gray(cu.val/0.5f0)
        @test cf+cf == ccmp
        @test isfinite(cf)
        @test isfinite(Gray(true))
        @test !isinf(cf)
        @test !isnan(cf)
        @test !isfinite(Gray(NaN))
        @test !isinf(Gray(NaN))
        @test isnan(Gray(NaN))
        @test !isfinite(Gray(Inf))
        @test Gray(Inf) |> isinf
        @test !isnan(Gray(Inf))
        @test abs(Gray(0.1)) ≈ 0.1
        @test eps(Gray{N0f8}) == Gray(eps(N0f8))  # #282
        @test atan(Gray(0.1), Gray(0.2)) == atan(0.1, 0.2)

        acu = Gray{N0f8}[cu]
        acf = Gray{Float32}[cf]
        @test +acu === acu
        @test @inferred(acu./trues(1)) == acu
        @test typeof(acu./trues(1)) == Vector{typeof(cu/true)}
        @test @inferred(ones(Int, 1)./acu) == [1/cu]
        @test typeof(ones(Int, 1)./acu) == Vector{typeof(1/cu)}
        @test @inferred(acu./acu) == [1]
        @test typeof(acu./acu) == Vector{typeof(cu/cu)}
        @test typeof(acu+acf) == Vector{Gray{Float32}}
        @test typeof(acu-acf) == Vector{Gray{Float32}}
        @test typeof(acu.+acf) == Vector{Gray{Float32}}
        @test typeof(acu.-acf) == Vector{Gray{Float32}}
        @test typeof(acu.+cf) == Vector{Gray{Float32}}
        @test typeof(acu.-cf) == Vector{Gray{Float32}}
        @test typeof(2*acf) == Vector{Gray{Float32}}
        @test typeof(2 .* acf) == Vector{Gray{Float32}}
        @test typeof(0x02*acu) == Vector{Gray{Float32}}
        @test typeof(acu/2) == Vector{Gray{typeof(N0f8(0.5)/2)}}
        @test typeof(acf.^2) == Vector{Gray{Float32}}
        @test (acu./Gray{N0f8}(0.5))[1] == gray(acu[1])/N0f8(0.5)
        @test (acf./Gray{Float32}(2))[1] ≈ 0.05f0
        @test (acu/2)[1] == Gray(gray(acu[1])/2)
        @test (acf/2)[1] ≈ Gray{Float32}(0.05f0)
        @test sum(abs2, [cf, ccmp]) ≈ 0.05f0

        @test gray(0.8) == 0.8

        a = Gray{N0f8}[0.8,0.7]
        @test sum(a) == Gray(n8sum(0.8,0.7))
        @test abs( var(a) - (a[1]-a[2])^2 / 2 ) <= 0.001
        @test isapprox(a, a)
        @test real(Gray{Float32}) <: Real
        @test zero(ColorTypes.Gray) == 0
        @test oneunit(ColorTypes.Gray) == 1
        a = Gray{N0f8}[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        @test StatsBase.histrange(a,10) == 0.1f0:0.1f0:1f0

        @test typeof(float(Gray{N0f16}(0.5))) <: AbstractFloat
        @test quantile( Gray{N0f16}[0.0,0.5,1.0], 0.1) ≈ 0.10000152590218968
        @test middle(Gray(0.2)) === Gray(0.2)
        @test middle(Gray(0.2), Gray(0.4)) === Gray((0.2+0.4)/2)

        # issue #56
        @test Gray24(0.8)*N0f8(0.5) === Gray{N0f8}(0.4)
        @test Gray24(0.8)*0.5 === Gray(0.4)
        @test Gray24(0.8)/2   === Gray(0.5f0*N0f8(0.8))
        @test Gray24(0.8)/2.0 === Gray(0.4)
    end

    @testset "Comparisons with Gray" begin
        g1 = Gray{N0f8}(0.2)
        g2 = Gray{N0f8}(0.3)
        @test isless(g1, g2)
        @test !(isless(g2, g1))
        @test g1 < g2
        @test !(g2 < g1)
        @test isless(g1, 0.5)
        @test !(isless(0.5, g1))
        @test g1 < 0.5
        @test !(0.5 < g1)
        @test (@inferred(max(g1, g2)) ) == g2
        @test max(g1, 0.1) == 0.2
        @test (@inferred(min(g1, g2)) ) == g1
        @test min(g1, 0.1) == 0.1
        a = Gray{Float64}(0.9999999999999999)
        b = Gray{Float64}(1.0)

        @test isapprox(a, b)
        a = Gray{Float64}(0.99)
        @test !(isapprox(a, b, rtol = 0.01))
        @test isapprox(a, b, rtol = 0.1)
    end

    @testset "Unary operations with Gray" begin
        for g in (Gray(0.4), Gray{N0f8}(0.4))
            @test @inferred(zero(g)) === typeof(g)(0)
            @test @inferred(oneunit(g)) === typeof(g)(1)
            for op in ColorVectorSpace.unaryOps
                try
                    v = @eval $op(gray(g))  # if this fails, don't bother
                    @show op
                    @test op(g) == v
                catch
                end
            end
        end
        u = N0f8(0.4)
        @test ~Gray(u) == Gray(~u)
        @test -Gray(u) == Gray(-u)
    end

    @testset "Arithmetic with GrayA" begin
        p1 = GrayA{Float32}(Gray(0.8), 0.2)
        @test @inferred(zero(p1)) === GrayA{Float32}(0,0)
        @test @inferred(oneunit(p1)) === GrayA{Float32}(1,1)
        @test +p1 == p1
        @test -p1 == GrayA(-0.8f0, -0.2f0)
        p2 = GrayA{Float32}(Gray(0.6), 0.3)
        @test_colortype_approx_eq p1+p2 GrayA{Float32}(Gray(1.4),0.5)
        @test_colortype_approx_eq (p1+p2)/2 GrayA{Float32}(Gray(0.7),0.25)
        @test_colortype_approx_eq 0.4f0*p1+0.6f0*p2 GrayA{Float32}(Gray(0.68),0.26)
        @test_colortype_approx_eq ([p1]+[p2])[1] GrayA{Float32}(Gray(1.4),0.5)
        @test_colortype_approx_eq ([p1].+[p2])[1] GrayA{Float32}(Gray(1.4),0.5)
        @test_colortype_approx_eq ([p1].+p2)[1] GrayA{Float32}(Gray(1.4),0.5)
        @test_colortype_approx_eq ([p1]-[p2])[1] GrayA{Float32}(Gray(0.2),-0.1)
        @test_colortype_approx_eq ([p1].-[p2])[1] GrayA{Float32}(Gray(0.2),-0.1)
        @test_colortype_approx_eq ([p1].-p2)[1] GrayA{Float32}(Gray(0.2),-0.1)
        @test_colortype_approx_eq ([p1]/2)[1] GrayA{Float32}(Gray(0.4),0.1)
        @test_colortype_approx_eq (0.4f0*[p1]+0.6f0*[p2])[1] GrayA{Float32}(Gray(0.68),0.26)

        a = GrayA{N0f8}[GrayA(0.8,0.7), GrayA(0.5,0.2)]
        @test sum(a) == GrayA(n8sum(0.8,0.5), n8sum(0.7,0.2))
        @test isapprox(a, a)
        a = AGray{Float64}(1.0, 0.9999999999999999)
        b = AGray{Float64}(1.0, 1.0)

        @test a ≈ b
        a = AGray{Float64}(1.0, 0.99)
        @test !isapprox(a, b, rtol = 0.01)
        @test isapprox(a, b, rtol = 0.1)

        # issue #56
        @test AGray32(0.8,0.2)*N0f8(0.5) === AGray{N0f8}(0.4,0.1)
        @test AGray32(0.8,0.2)*0.5 === AGray(0.4,0.1)
        @test AGray32(0.8,0.2)/2   === AGray(0.5f0*N0f8(0.8),0.5f0*N0f8(0.2))
        @test AGray32(0.8,0.2)/2.0 === AGray(0.4,0.1)
    end

    @testset "Arithemtic with RGB" begin
        cf = RGB{Float32}(0.1,0.2,0.3)
        @test @inferred(zero(cf)) === RGB{Float32}(0,0,0)
        @test @inferred(oneunit(cf)) === RGB{Float32}(1,1,1)
        @test +cf == cf
        @test -cf == RGB(-0.1f0, -0.2f0, -0.3f0)
        ccmp = RGB{Float32}(0.2,0.4,0.6)
        @test 2*cf == ccmp
        @test cf*2 == ccmp
        @test ccmp/2 == cf
        @test 2.0f0*cf == ccmp
        @test eltype(2.0*cf) == Float64
        cu = RGB{N0f8}(0.1,0.2,0.3)
        @test_colortype_approx_eq 2*cu RGB(2*cu.r, 2*cu.g, 2*cu.b)
        @test_colortype_approx_eq 2.0f0*cu RGB(2.0f0*cu.r, 2.0f0*cu.g, 2.0f0*cu.b)
        f = N0f8(0.5)
        @test (f*cu).r ≈ f*cu.r
        @test cf/2.0f0 == RGB{Float32}(0.05,0.1,0.15)
        @test cu/2 ≈ RGB(cu.r/2,cu.g/2,cu.b/2)
        @test cu/0.5f0 == RGB(cu.r/0.5f0, cu.g/0.5f0, cu.b/0.5f0)
        @test cf+cf == ccmp
        @test cu * 1//2 == mapc(x->Float64(Rational(x)/2), cu)
        @test_colortype_approx_eq (cf.*[0.8f0])[1] RGB{Float32}(0.8*0.1,0.8*0.2,0.8*0.3)
        @test_colortype_approx_eq ([0.8f0].*cf)[1] RGB{Float32}(0.8*0.1,0.8*0.2,0.8*0.3)
        @test isfinite(cf)
        @test !isinf(cf)
        @test !isnan(cf)
        @test !isfinite(RGB(NaN, 1, 0.5))
        @test !isinf(RGB(NaN, 1, 0.5))
        @test isnan(RGB(NaN, 1, 0.5))
        @test !isfinite(RGB(1, Inf, 0.5))
        @test isinf(RGB(1, Inf, 0.5))
        @test !isnan(RGB(1, Inf, 0.5))
        @test abs(RGB(0.1,0.2,0.3)) ≈ 0.6
        @test sum(abs2, RGB(0.1,0.2,0.3)) ≈ 0.14
        @test norm(RGB(0.1,0.2,0.3)) ≈ sqrt(0.14)

        acu = RGB{N0f8}[cu]
        acf = RGB{Float32}[cf]
        @test typeof(acu+acf) == Vector{RGB{Float32}}
        @test typeof(acu-acf) == Vector{RGB{Float32}}
        @test typeof(acu.+acf) == Vector{RGB{Float32}}
        @test typeof(acu.-acf) == Vector{RGB{Float32}}
        @test typeof(acu.+cf) == Vector{RGB{Float32}}
        @test typeof(acu.-cf) == Vector{RGB{Float32}}
        @test typeof(2*acf) == Vector{RGB{Float32}}
        @test typeof(convert(UInt8, 2)*acu) == Vector{RGB{Float32}}
        @test typeof(acu/2) == Vector{RGB{typeof(N0f8(0.5)/2)}}
        rcu = rand(RGB{N0f8}, 3, 5)
        @test @inferred(rcu./trues(3, 5)) == rcu
        @test typeof(rcu./trues(3, 5)) == Matrix{typeof(cu/true)}

        a = RGB{N0f8}[RGB(1,0,0), RGB(1,0.8,0)]
        @test sum(a) == RGB(2.0,0.8,0)
        @test isapprox(a, a)
        a = RGB{Float64}(1.0, 1.0, 0.9999999999999999)
        b = RGB{Float64}(1.0, 1.0, 1.0)

        @test isapprox(a, b)
        a = RGB{Float64}(1.0, 1.0, 0.99)
        @test !(isapprox(a, b, rtol = 0.01))
        @test isapprox(a, b, rtol = 0.1)
        # issue #56
        @test RGB24(1,0,0)*N0f8(0.5) === RGB{N0f8}(0.5,0,0)
        @test RGB24(1,0,0)*0.5 === RGB(0.5,0,0)
        @test RGB24(1,0,0)/2   === RGB(0.5f0,0,0)
        @test RGB24(1,0,0)/2.0 === RGB(0.5,0,0)
    end

    @testset "Arithemtic with RGBA" begin
        cf = RGBA{Float32}(0.1,0.2,0.3,0.4)
        @test @inferred(zero(cf)) === RGBA{Float32}(0,0,0,0)
        @test @inferred(oneunit(cf)) === RGBA{Float32}(1,1,1,1)
        @test +cf == cf
        @test -cf == RGBA(-0.1f0, -0.2f0, -0.3f0, -0.4f0)
        ccmp = RGBA{Float32}(0.2,0.4,0.6,0.8)
        @test 2*cf == ccmp
        @test cf*2 == ccmp
        @test ccmp/2 == cf
        @test 2.0f0*cf == ccmp
        @test eltype(2.0*cf) == Float64
        cu = RGBA{N0f8}(0.1,0.2,0.3,0.4)
        @test_colortype_approx_eq 2*cu RGBA(2*cu.r, 2*cu.g, 2*cu.b, 2*cu.alpha)
        @test_colortype_approx_eq 2.0f0*cu RGBA(2.0f0*cu.r, 2.0f0*cu.g, 2.0f0*cu.b, 2.0f0*cu.alpha)
        f = N0f8(0.5)
        @test (f*cu).r ≈ f*cu.r
        @test cf/2.0f0 == RGBA{Float32}(0.05,0.1,0.15,0.2)
        @test cu/2 == RGBA(cu.r/2,cu.g/2,cu.b/2,cu.alpha/2)
        @test cu/0.5f0 == RGBA(cu.r/0.5f0, cu.g/0.5f0, cu.b/0.5f0, cu.alpha/0.5f0)
        @test cf+cf == ccmp
        @test_colortype_approx_eq (cf.*[0.8f0])[1] RGBA{Float32}(0.8*0.1,0.8*0.2,0.8*0.3,0.8*0.4)
        @test_colortype_approx_eq ([0.8f0].*cf)[1] RGBA{Float32}(0.8*0.1,0.8*0.2,0.8*0.3,0.8*0.4)
        @test isfinite(cf)
        @test !isinf(cf)
        @test !isnan(cf)
        @test isnan(RGBA(NaN, 1, 0.5, 0.8))
        @test !isinf(RGBA(NaN, 1, 0.5))
        @test isnan(RGBA(NaN, 1, 0.5))
        @test !isfinite(RGBA(1, Inf, 0.5))
        @test RGBA(1, Inf, 0.5) |> isinf
        @test !isnan(RGBA(1, Inf, 0.5))
        @test !isfinite(RGBA(0.2, 1, 0.5, NaN))
        @test !isinf(RGBA(0.2, 1, 0.5, NaN))
        @test isnan(RGBA(0.2, 1, 0.5, NaN))
        @test !isfinite(RGBA(0.2, 1, 0.5, Inf))
        @test RGBA(0.2, 1, 0.5, Inf) |> isinf
        @test !isnan(RGBA(0.2, 1, 0.5, Inf))
        @test abs(RGBA(0.1,0.2,0.3,0.2)) ≈ 0.8

        acu = RGBA{N0f8}[cu]
        acf = RGBA{Float32}[cf]
        @test typeof(acu+acf) == Vector{RGBA{Float32}}
        @test typeof(acu-acf) == Vector{RGBA{Float32}}
        @test typeof(acu.+acf) == Vector{RGBA{Float32}}
        @test typeof(acu.-acf) == Vector{RGBA{Float32}}
        @test typeof(acu.+cf) == Vector{RGBA{Float32}}
        @test typeof(acu.-cf) == Vector{RGBA{Float32}}
        @test typeof(2*acf) == Vector{RGBA{Float32}}
        @test typeof(convert(UInt8, 2)*acu) == Vector{RGBA{Float32}}
        @test typeof(acu/2) == Vector{RGBA{typeof(N0f8(0.5)/2)}}

        a = RGBA{N0f8}[RGBA(1,0,0,0.8), RGBA(0.7,0.8,0,0.9)]
        @test sum(a) == RGBA(n8sum(1,0.7),0.8,0,n8sum(0.8,0.9))
        @test isapprox(a, a)
        a = ARGB{Float64}(1.0, 1.0, 1.0, 0.9999999999999999)
        b = ARGB{Float64}(1.0, 1.0, 1.0, 1.0)

        @test isapprox(a, b)
        a = ARGB{Float64}(1.0, 1.0, 1.0, 0.99)
        @test !(isapprox(a, b, rtol = 0.01))
        @test isapprox(a, b, rtol = 0.1)
        # issue #56
        @test ARGB32(1,0,0,0.8)*N0f8(0.5) === ARGB{N0f8}(0.5,0,0,0.4)
        @test ARGB32(1,0,0,0.8)*0.5 === ARGB(0.5,0,0,0.4)
        @test ARGB32(1,0,0,0.8)/2   === ARGB(0.5f0,0,0,0.5f0*N0f8(0.8))
        @test ARGB32(1,0,0,0.8)/2.0 === ARGB(0.5,0,0,0.4)
    end

    @testset "Mixed-type arithmetic" begin
        @test RGB(1,0,0) + Gray(0.2f0) == RGB{Float32}(1.2,0.2,0.2)
        @test RGB(1,0,0) - Gray(0.2f0) == RGB{Float32}(0.8,-0.2,-0.2)
    end

    @testset "dotc" begin
        @test dotc(0.2, 0.2) == 0.2^2
        @test dotc(0.2, 0.3f0) == 0.2*0.3f0
        @test dotc(N0f8(0.2), N0f8(0.3)) == Float32(N0f8(0.2))*Float32(N0f8(0.3))
        @test dotc(Gray{N0f8}(0.2), Gray24(0.3)) == Float32(N0f8(0.2))*Float32(N0f8(0.3))
        xc, yc = RGB(0.2,0.2,0.2), RGB{N0f8}(0.3,0.3,0.3)
        @test isapprox(dotc(xc, yc) , dotc(convert(Gray, xc), convert(Gray, yc)), atol=1e-6)
        @test dotc(RGB(1,0,0), RGB(0,1,1)) == 0
    end

    @testset "typemin/max" begin
        for T in (Normed{UInt8,8}, Normed{UInt8,6}, Normed{UInt16,16}, Normed{UInt16,14}, Float32, Float64)
            @test typemin(Gray{T}) === Gray{T}(typemin(T))
            @test typemax(Gray{T}) === Gray{T}(typemax(T))
            @test typemin(Gray{T}(0.5)) === Gray{T}(typemin(T))
            @test typemax(Gray{T}(0.5)) === Gray{T}(typemax(T))
            A = maximum(Gray{T}.([1 0 0; 0 1 0]); dims=1)  # see PR#44 discussion
            @test isa(A, Matrix{Gray{T}})
            @test size(A) == (1,3)
        end
    end
end

end
