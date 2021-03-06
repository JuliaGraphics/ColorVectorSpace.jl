using LinearAlgebra, Statistics, SpecialFunctions
using ColorVectorSpace, Colors, FixedPointNumbers

using Test

n8sum(x,y) = Float64(N0f8(x)) + Float64(N0f8(y))

macro test_colortype_approx_eq(a, b)
    :(test_colortype_approx_eq($(esc(a)), $(esc(b)), $(string(a)), $(string(b))))
end

function test_colortype_approx_eq(a::Colorant, b::Colorant, astr, bstr)
    @test typeof(a) == typeof(b)
    n = length(fieldnames(typeof(a)))
    for i = 1:n
        @test getfield(a, i) ≈ getfield(b,i)
    end
end

struct RatRGB <: AbstractRGB{Rational{Int}}
    r::Rational{Int}
    g::Rational{Int}
    b::Rational{Int}
end
ColorTypes.red(c::RatRGB)   = c.r
ColorTypes.green(c::RatRGB) = c.g
ColorTypes.blue(c::RatRGB)  = c.b

@testset "Colortypes" begin

    @testset "convert" begin
        for x in (0.5, 0.5f0, NaN, NaN32, N0f8(0.5))
            @test @inferred(convert(Gray{typeof(x)}, x))  === @inferred(convert(Gray, x))  === Gray(x)
            @test @inferred(convert(RGB{typeof(x)}, x))   === @inferred(convert(RGB, x))   === RGB(x, x, x)
            # These should be fixed by a future release of ColorTypes
            @test_broken @inferred(convert(AGray{typeof(x)}, x)) === @inferred(convert(AGray, x)) === AGray(x, 1)
            @test_broken @inferred(convert(ARGB{typeof(x)}, x))  === @inferred(convert(ARGB, x))  === ARGB(x, x, x, 1)
            @test_broken @inferred(convert(GrayA{typeof(x)}, x)) === @inferred(convert(GrayA, x)) === GrayA(x, 1)
            @test_broken @inferred(convert(RGBA{typeof(x)}, x))  === @inferred(convert(RGBA, x))  === RGBA(x, x, x, 1)
        end
    end

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

    @testset "traits" begin
        @test floattype(Gray{N0f8}) === Gray{float(N0f8)}
    end

    @testset "Arithmetic with Gray" begin
        cf = Gray{Float32}(0.1)
        @test @inferred(+cf) === cf
        @test @inferred(-cf) === Gray(-0.1f0)
        @test @inferred(one(cf)*cf) === cf
        @test oneunit(cf) === Gray(1.0f0)
        ccmp = Gray{Float32}(0.2)
        @test @inferred(2*cf) === cf*2 === 2.0f0*cf === cf*2.0f0 === ccmp
        @test @inferred(ccmp/2) === cf
        @test @inferred(cf*cf) === Gray{Float32}(0.1f0*0.1f0)
        @test @inferred(Gray{N0f32}(0.5)*Gray(0.5f0)) === Gray(Float64(N0f32(0.5)) * 0.5)
        @test @inferred(cf^2 ) === Gray{Float32}(0.1f0*0.1f0)
        @test @inferred(cf^3.0f0) === Gray{Float32}(0.1f0^3.0f0)
        @test @inferred(2.0*cf) === cf*2.0 === Gray(2.0*0.1f0)
        cf64 = Gray(0.2)
        @test cf / cf64 === Gray(0.1f0/0.2)
        @test_throws MethodError cf ÷ cf
        @test cf + 0.1     === 0.1 + cf        === Gray(Float64(0.1f0) + 0.1)
        @test cf64 - 0.1f0 === -(0.1f0 - cf64) === Gray( 0.2 - Float64(0.1f0))
        @test_throws MethodError abs2(ccmp)
        @test norm(cf) == norm(cf, 2) == norm(gray(cf))
        @test norm(cf, 1)   == norm(gray(cf), 1)
        @test norm(cf, Inf) == norm(gray(cf), Inf)
        @test @inferred(abs(cf)) === Gray(0.1f0)
        cu = Gray{N0f8}(0.1)
        @test @inferred(2*cu) === cu*2 === Gray(2*gray(cu))
        @test @inferred(2.0f0*cu) === cu*2.0f0 === Gray(2.0f0*gray(cu))
        f = N0f8(0.5)
        @test @inferred(gray(f*cu)) === gray(cu*f) ===f*gray(cu)
        @test @inferred(cf/2.0f0) === Gray{Float32}(0.05)
        @test @inferred(cu/2) === Gray(cu.val/2)
        @test @inferred(cu/0.5f0) === Gray(cu.val/0.5f0)
        @test @inferred(cf+cf) === ccmp
        @test isfinite(cf)
        @test isfinite(Gray(true))
        @test !isinf(cf)
        @test !isinf(Gray(f))
        @test !isnan(cf)
        @test !isfinite(Gray(NaN))
        @test !isinf(Gray(NaN))
        @test isnan(Gray(NaN))
        @test !isfinite(Gray(Inf))
        @test Gray(Inf) |> isinf
        @test !isnan(Gray(Inf))
        @test abs(Gray(0.1)) === Gray(0.1)
        @test eps(Gray{N0f8}) === Gray(eps(N0f8))  # #282
        @test atan(Gray(0.1), Gray(0.2)) == atan(0.1, 0.2)
        @test hypot(Gray(0.2), Gray(0.3)) === hypot(0.2, 0.3)
        # Multiplication
        @test cf ⋅ cf   === gray(cf)^2
        @test cf ⋅ cf64 === gray(cf) * gray(cf64)
        @test cf ⊙ cf   === Gray(gray(cf)^2)
        @test cf ⊙ cf64 === Gray(gray(cf) * gray(cf64))
        @test cf ⊗ cf   === Gray(gray(cf)^2)
        @test cf ⊗ cf64 === Gray(gray(cf) * gray(cf64))

        acu = Gray{N0f8}[cu]
        acf = Gray{Float32}[cf]
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

        @test gray(0.8) === 0.8

        a = Gray{N0f8}[0.8,0.7]
        @test a == a
        @test a === a
        @test isapprox(a, a)
        @test sum(a) == Gray(n8sum(0.8,0.7))
        @test sum(a[1:1]) == a[1]
        @test abs( varmult(*, a) - (a[1]-a[2])^2 / 2 ) <= 0.001

        @test real(Gray{Float32}) <: Real
        @test zero(ColorTypes.Gray) == 0
        @test oneunit(ColorTypes.Gray) == 1

        @test typeof(float(Gray{N0f16}(0.5))) <: AbstractFloat
        @test quantile( Gray{N0f16}[0.0,0.5,1.0], 0.1) ≈ 0.1 atol=eps(N0f16)
        @test middle(Gray(0.2)) === Gray(0.2)
        @test middle(Gray(0.2), Gray(0.4)) === Gray((0.2+0.4)/2)

        # issue #56
        @test Gray24(0.8)*N0f8(0.5) === Gray{N0f8}(0.4)
        @test Gray24(0.8)*0.5 === Gray(0.4)
        @test Gray24(0.8)/2   === Gray(0.5f0*N0f8(0.8))
        @test Gray24(0.8)/2.0 === Gray(0.4)

        # issue #133
        @test Gray24(0.2) + Gray24(0.4) === Gray24(0.6)
        @test Gray24(1)   - Gray24(0.2) === Gray24(0.8)
        @test Gray24(1)   * Gray24(0.2) === Gray24(0.2)
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
        @test @inferred(max(g1, g2)) === g2
        @test @inferred(max(g1, Gray(0.3))) === Gray(0.3)
        @test max(g1, 0.1) === max(0.1, g1) === Float64(gray(g1))
        @test (@inferred(min(g1, g2)) ) == g1
        @test min(g1, 0.1) === min(0.1, g1) === 0.1
        a = Gray{Float64}(0.9999999999999999)
        b = Gray{Float64}(1.0)

        @test (Gray(0.3) < Gray(NaN)) == (0.3 < NaN)
        @test (Gray(NaN) < Gray(0.3)) == (NaN < 0.3)
        @test isless(Gray(0.3), Gray(NaN)) == isless(0.3, NaN)
        @test isless(Gray(NaN), Gray(0.3)) == isless(NaN, 0.3)
        @test isless(Gray(0.3), NaN) == isless(0.3, NaN)
        @test isless(Gray(NaN), 0.3) == isless(NaN, 0.3)
        @test isless(0.3, Gray(NaN)) == isless(0.3, NaN)
        @test isless(NaN, Gray(0.3)) == isless(NaN, 0.3)

        @test isapprox(a, b)
        a = Gray{Float64}(0.99)
        @test !(isapprox(a, b, rtol = 0.01))
        @test isapprox(a, b, rtol = 0.1)
    end

    @testset "Unary operations with Gray" begin
        ntested = 0
        for g in (Gray(0.4), Gray{N0f8}(0.4))
            @test @inferred(zero(g)) === typeof(g)(0)
            @test @inferred(oneunit(g)) === typeof(g)(1)
            for opgroup in (ColorVectorSpace.unaryOps, (:trunc, :floor, :round, :ceil, :eps, :bswap))
                for op in opgroup
                    op ∈ (:frexp, :exponent, :modf, :lfact) && continue
                    op === :~ && eltype(g) === Float64 && continue
                    op === :significand && eltype(g) === N0f8 && continue
                    try
                        v = @eval $op(gray($g))  # if this fails, don't bother with the next test
                        @test @eval($op($g)) === Gray(v)
                        ntested += 1
                    catch ex
                        @test ex isa Union{DomainError,ArgumentError}
                    end
                end
            end
        end
        @test ntested > 130
        @test logabsgamma(Gray(0.2)) == (Gray(logabsgamma(0.2)[1]), 1)
        for g in (Gray{N0f8}(0.4), Gray{N0f8}(0.6))
            for op in (:trunc, :floor, :round, :ceil)
                v = @eval $op(Bool, gray($g))
                @test @eval($op(Bool, $g)) === Gray(v)
            end
        end
        for (g1, g2) in ((Gray(0.4), Gray(0.3)), (Gray(N0f8(0.4)), Gray(N0f8(0.3))))
            for op in (:mod, :rem, :mod1)
                v = @eval $op(gray($g1), gray($g2))
                @test @eval($op($g1, $g2)) === Gray(v)
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

        # issue #133
        @test AGray32(1, 0.4) - AGray32(0.2, 0.2) === AGray32(0.8, 0.2)
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
        @test cu/0.5f0 ≈ RGB(cu.r/0.5f0, cu.g/0.5f0, cu.b/0.5f0)
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
        @test abs(RGB(0.1,0.2,0.3)) == RGB(0.1,0.2,0.3)
        @test_throws MethodError abs2(RGB(0.1,0.2,0.3))
        @test_throws MethodError sum(abs2, RGB(0.1,0.2,0.3))
        @test norm(RGB(0.1,0.2,0.3)) ≈ sqrt(0.14)/sqrt(3)

        @test_throws MethodError RGBX(0, 0, 1) + XRGB(1, 0, 0)

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
        @test sum(typeof(a)()) == RGB(0.0,0.0,0)
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
        # issue #133
        @test RGB24(1, 0, 0) + RGB24(0, 0, 1) === RGB24(1, 0, 1)

        # Multiplication
        @test_throws MethodError cf*cf
        cf64 = mapc(Float64, cf)
        @test cf ⋅ cf   === (red(cf)^2 + green(cf)^2 + blue(cf)^2)/3
        @test cf ⋅ cf64 === (red(cf)*red(cf64) + green(cf)*green(cf64) + blue(cf)*blue(cf64))/3
        @test cf ⊙ cf   === RGB(red(cf)^2, green(cf)^2, blue(cf)^2)
        @test cf ⊙ cf64 === RGB(red(cf)*red(cf64), green(cf)*green(cf64), blue(cf)*blue(cf64))
        c2 = rand(RGB{Float64})
        rr = cf ⊗ c2
        @test Matrix(rr) == [red(cf)*red(c2)   red(cf)*green(c2)   red(cf)*blue(c2);
                             green(cf)*red(c2) green(cf)*green(c2) green(cf)*blue(c2);
                             blue(cf)*red(c2)  blue(cf)*green(c2)  blue(cf)*blue(c2)]
        @test +rr === rr
        @test -rr === RGBRGB(-rr.rr, -rr.gr, -rr.br, -rr.rg, -rr.gg, -rr.bg, -rr.rb, -rr.gb, -rr.bb)
        @test rr + rr == 2*rr == rr*2
        @test rr - rr == zero(rr)
        io = IOBuffer()
        print(io, N0f8)
        Tstr = String(take!(io))
        cfn = RGB{N0f8}(0.1, 0.2, 0.3)
        show(io, cfn ⊗ cfn)
        spstr = Base.VERSION >= v"1.5" ? "" : " "
        @test String(take!(io)) == "RGBRGB{$Tstr}(\n 0.012N0f8  0.02N0f8   0.031N0f8\n 0.02N0f8   0.039N0f8  0.059N0f8\n 0.031N0f8  0.059N0f8  0.09N0f8$spstr)"
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
        @test abs(RGBA(0.1,0.2,0.3,0.2)) === RGBA(0.1,0.2,0.3,0.2)

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
        # issue #133
        @test ARGB32(1, 0, 0, 0.2) + ARGB32(0, 0, 1, 0.2) === ARGB32(1, 0, 1, 0.4)
    end

    @testset "Mixed-type arithmetic" begin
        @test AGray32(0.2, 0.4) + Gray24(0.2) === AGray32(0.4, 0.4N0f8+1N0f8)
        @test RGB(1, 0, 0)      + Gray(0.2f0) === RGB{Float32}(1.2, 0.2, 0.2)
        @test RGB(1, 0, 0)      - Gray(0.2f0) === RGB{Float32}(0.8, -0.2, -0.2)
        @test RGB24(1, 0, 0)    + Gray(0.2f0) === RGB{Float32}(1.2, 0.2, 0.2)
        @test RGB24(1, 0, 0)    - Gray(0.2f0) === RGB{Float32}(0.8, -0.2, -0.2)
        @test RGB(1.0f0, 0, 0)  + Gray24(0.2) === RGB{Float32}(1.2, 0.2, 0.2)
        @test RGB(1.0f0, 0, 0)  - Gray24(0.2) === RGB{Float32}(0.8, -0.2, -0.2)
        @test RGB24(1, 0, 0)    + Gray24(0.2) === RGB24(1N0f8+0.2N0f8, 0.2, 0.2)
        @test RGB24(0.4, 0, 0.2)   + AGray32(0.4, 1)   === ARGB32(0.8, 0.4, 0.6, 1N0f8+1N0f8)
        @test RGB24(0.4, 0.6, 0.5) - AGray32(0.4, 0.2) === ARGB32(0, 0.2, 0.1, 0.8)
        @test ARGB32(0.4, 0, 0.2, 0.5) + Gray24(0.4)   === ARGB32(0.8, 0.4, 0.6, 0.5N0f8+1N0f8)
        @test ARGB32(0.4, 0, 0.2, 0.5) + AGray32(0.4, 0.2) === ARGB32(0.8, 0.4, 0.6, 0.5N0f8+0.2N0f8)

        g, rgb = Gray(0.2), RGB(0.1, 0.2, 0.3)
        @test g ⋅ rgb == rgb ⋅ g ≈ 0.2*(0.1 + 0.2 + 0.3)/3
        @test g ⊙ rgb == rgb ⊙ g ≈ RGB(0.2*0.1, 0.2^2, 0.2*0.3)
        @test g ⊗ rgb == RGB(g) ⊗ rgb
        @test rgb ⊗ g == rgb ⊗ RGB(g)
    end

    @testset "Custom RGB arithmetic" begin
        cf = RatRGB(1//10, 2//10, 3//10)
        @test cf ⋅ cf   === (Float64(red(cf))^2 + Float64(green(cf))^2 + Float64(blue(cf))^2)/3
    end

    @testset "Complement" begin
        @test complement(Gray(0.2)) === Gray(0.8)
        @test complement(AGray(0.2f0, 0.7f0)) === AGray(0.8f0, 0.7f0)
        @test complement(GrayA{N0f8}(0.2, 0.7)) === GrayA{N0f8}(0.8, 0.7)
        @test_broken complement(Gray24(0.2)) === Gray24(0.8)
        @test_broken complement(AGray32(0.2, 0.7)) === AGray32(0.8, 0.7)

        @test complement(RGB(0, 0.3, 1)) === RGB(1, 0.7, 0)
        @test complement(ARGB(0, 0.3f0, 1, 0.7f0)) === ARGB(1, 0.7f0, 0, 0.7f0)
        @test complement(RGBA{N0f8}(0, 0.6, 1, 0.7)) === RGBA{N0f8}(1, 0.4, 0.0, 0.7)
        @test complement(RGB24(0, 0.6, 1)) === RGB24(1, 0.4, 0.0)
        @test complement(ARGB32(0, 0.6, 1, 0.7)) === ARGB32(1, 0.4, 0.0, 0.7)
    end

    @testset "dotc" begin
        @test dotc(0.2, 0.2) == 0.2^2
        @test dotc(Int8(3), Int16(6)) === 18
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

    @testset "Colors issue #326" begin
        A = rand(RGB{N0f8}, 2, 2)
        @test @inferred(mean(A)) == mean(map(c->mapc(FixedPointNumbers.Treduce, c), A))
    end

    @testset "Equivalence" begin
        x = 0.4
        g = Gray(x)
        c = RGB(g)
        for p in (0, 1, 2, Inf)
            @test norm(x, p) == norm(g, p) ≈ norm(c, p)
        end
        @test dot(x, x) == dot(g, g) ≈ dot(c, c)
        @test_throws MethodError mapreduce(x->x^2, +, c)   # this risks breaking equivalence & noniterability
    end

    @testset "varmult" begin
        cs = [RGB(0.2, 0.3, 0.4), RGB(0.5, 0.3, 0.2)]
        @test varmult(⋅, cs) ≈ 2*(0.15^2 + 0.1^2)/3    # the /3 is for the 3 color channels, i.e., equivalence
        @test varmult(⋅, cs; corrected=false) ≈ (0.15^2 + 0.1^2)/3
        @test varmult(⋅, cs; mean=RGB(0, 0, 0)) ≈ (0.2^2+0.3^2+0.4^2 + 0.5^2+0.3^2+0.2^2)/3
        @test varmult(⊙, cs) ≈ 2*RGB(0.15^2, 0, 0.1^2)
        @test Matrix(varmult(⊗, cs)) ≈ 2*[0.15^2 0 -0.1*0.15; 0 0 0; -0.1*0.15 0 0.1^2]

        cs = [RGB(0.1, 0.2,  0.3)  RGB(0.3, 0.5, 0.3);
              RGB(0.2, 0.21, 0.33) RGB(0.4, 0.51, 0.33);
              RGB(0.3, 0.22, 0.36) RGB(0.5, 0.52, 0.36)]
        v1 = RGB(0.1^2, 0.15^2, 0)
        @test varmult(⊙, cs, dims=2) ≈ 2*[v1, v1, v1]
        v2 = RGB(0.1^2, 0.01^2, 0.03^2)
        @test varmult(⊙, cs, dims=1) ≈ [v2 v2]
    end

    @testset "copy" begin
        g = Gray{N0f8}(0.2)
        @test copy(g) === g
        c = RGB(0.1, 0.2, 0.3)
        @test copy(c) === c
    end

end
