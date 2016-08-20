module ColorVectorSpaceTests

using FactCheck, Base.Test, ColorVectorSpace, ColorTypes, FixedPointNumbers, Compat

macro test_colortype_approx_eq(a, b)
    :(test_colortype_approx_eq($(esc(a)), $(esc(b)), $(string(a)), $(string(b))))
end

u8sum(x,y) = Float64(UFixed8(x)) + Float64(UFixed8(y))

FactCheck.roughly(x::Gray) = (y::Gray) -> isapprox(y, x)
FactCheck.roughly(x::RGB) = (y::RGB) -> isapprox(y, x)

facts("Colortypes") do
    function test_colortype_approx_eq(a::Colorant, b::Colorant, astr, bstr)
        @fact typeof(a) --> typeof(b)
        n = length(fieldnames(typeof(a)))
        for i = 1:n
            @fact getfield(a, i) --> roughly(getfield(b,i))
        end
    end

    context("nan") do
        function make_checked_nan{T}(::Type{T})
            x = nan(T)
            isa(x, T) && isnan(x)
        end
        for S in (Float32, Float64)
            @fact make_checked_nan(S) --> true
            @fact make_checked_nan(Gray{S}) --> true
            @fact make_checked_nan(AGray{S}) --> true
            @fact make_checked_nan(GrayA{S}) --> true
            @fact make_checked_nan(RGB{S}) --> true
            @fact make_checked_nan(ARGB{S}) --> true
            @fact make_checked_nan(ARGB{S}) --> true
        end
    end

    context("Arithmetic with Gray") do
        cf = Gray{Float32}(0.1)
        ccmp = Gray{Float32}(0.2)
        @fact 2*cf --> ccmp
        @fact cf*2 --> ccmp
        @fact ccmp/2 --> cf
        @fact 2.0f0*cf --> ccmp
        @test_colortype_approx_eq cf*cf Gray{Float32}(0.01)
        @test_colortype_approx_eq cf^2 Gray{Float32}(0.01)
        @test_colortype_approx_eq cf^3.0f0 Gray{Float32}(0.001)
        @fact eltype(2.0*cf) --> Float64
        @fact abs2(ccmp) --> 0.2f0^2
        @fact norm(cf) --> 0.1f0
        @fact sumabs2(ccmp) --> 0.2f0^2
        cu = Gray{U8}(0.1)
        @fact 2*cu --> Gray(2*cu.val)
        @fact 2.0f0*cu --> Gray(2.0f0*cu.val)
        f = U8(0.5)
        @fact (f*cu).val --> roughly(f*cu.val)
        @fact 2.*cf --> ccmp
        @fact cf.*2 --> ccmp
        @fact cf/2.0f0 --> Gray{Float32}(0.05)
        @fact cu/2 --> Gray(cu.val/2)
        @fact cu/0.5f0 --> Gray(cu.val/0.5f0)
        @fact cf+cf --> ccmp
        @fact cf --> isfinite
        @fact cf --> not(isinf)
        @fact cf --> not(isnan)
        @fact Gray(NaN) --> not(isfinite)
        @fact Gray(NaN) --> not(isinf)
        @fact Gray(NaN) --> isnan
        @fact Gray(Inf) --> not(isfinite)
        @fact Gray(Inf) --> isinf
        @fact Gray(Inf) --> not(isnan)
        @fact abs(Gray(0.1)) --> roughly(0.1)
        @fact eps(Gray{U8}) --> Gray(eps(U8))  # #282

        acu = Gray{U8}[cu]
        acf = Gray{Float32}[cf]
        @fact typeof(acu+acf) --> Vector{Gray{Float32}}
        @fact typeof(acu-acf) --> Vector{Gray{Float32}}
        @fact typeof(acu.+acf) --> Vector{Gray{Float32}}
        @fact typeof(acu.-acf) --> Vector{Gray{Float32}}
        @fact typeof(acu+cf) --> Vector{Gray{U8}}
        @fact typeof(acu-cf) --> Vector{Gray{U8}}
        @fact typeof(acu.+cf) --> Vector{Gray{U8}}
        @fact typeof(acu.-cf) --> Vector{Gray{U8}}
        @fact typeof(2*acf) --> Vector{Gray{Float32}}
        @fact typeof(2.*acf) --> Vector{Gray{Float32}}
        @fact typeof(0x02*acu) --> Vector{Gray{Float32}}
        @fact typeof(acu/2) --> Vector{Gray{typeof(U8(0.5)/2)}}
        @fact typeof(acf.^2) --> Vector{Gray{Float32}}
        @fact (acu/Gray{U8}(0.5))[1] --> gray(acu[1])/U8(0.5)
        @fact (acf/Gray{Float32}(2))[1] --> roughly(0.05f0)
        @fact (acu/2)[1] --> Gray(gray(acu[1])/2)
        @fact (acf/2)[1] --> roughly(Gray{Float32}(0.05f0))
        @fact sumabs2([cf,ccmp]) --> roughly(0.05f0)

        @fact gray(0.8) --> 0.8

        a = Gray{U8}[0.8,0.7]
        @fact sum(a) --> Gray(u8sum(0.8,0.7))
        @fact abs( var(a) - (a[1]-a[2])^2 / 2 ) --> less_than(0.001)
        @fact isapprox(a, a) --> true
        @fact real(Gray{Float32}) <: Real --> true
        @fact zero(ColorTypes.Gray)-->0
        @fact one(ColorTypes.Gray)-->1
        a = Gray{U8}[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        @fact histrange(a,10)-->0.1:0.1:1

    end

    context("Comparisons with Gray") do
        g1 = Gray{U8}(0.2)
        g2 = Gray{U8}(0.3)
        @fact isless(g1, g2) --> true
        @fact isless(g2, g1) --> false
        @fact g1 < g2 --> true
        @fact g2 < g1 --> false
        @fact isless(g1, 0.5) --> true
        @fact isless(0.5, g1) --> false
        @fact g1 < 0.5 --> true
        @fact 0.5 < g1 --> false
        @fact @inferred(max(g1, g2)) --> g2
        @fact @inferred(max(g1, 0.1)) --> 0.2
        @fact @inferred(min(g1, g2)) --> g1
        @fact @inferred(min(g1, 0.1)) --> 0.1
        a = Gray{Float64}(0.9999999999999999)
        b = Gray{Float64}(1.0)

        @fact isapprox(a, b) --> true
        a = Gray{Float64}(0.99)
        @fact isapprox(a, b, rtol = 0.01) --> false
        @fact isapprox(a, b, rtol = 0.1) --> true
    end

    context("Unary operations with Gray") do
        for g in (Gray(0.4), Gray{U8}(0.4))
            for op in ColorVectorSpace.unaryOps
                try
                    v = @eval $op(gray(g))  # if this fails, don't bother
                    @fact $op(g) --> v
                end
            end
        end
        u = U8(0.4)
        @fact ~Gray(u) --> Gray(~u)
        @fact -Gray(u) --> Gray(-u)
    end

    context("Arithmetic with GrayA") do
        p1 = GrayA{Float32}(Gray(0.8), 0.2)
        p2 = GrayA{Float32}(Gray(0.6), 0.3)
        @test_colortype_approx_eq p1+p2 GrayA{Float32}(Gray(1.4),0.5)
        @test_colortype_approx_eq (p1+p2)/2 GrayA{Float32}(Gray(0.7),0.25)
        @test_colortype_approx_eq 0.4f0*p1+0.6f0*p2 GrayA{Float32}(Gray(0.68),0.26)
        @test_colortype_approx_eq ([p1]+[p2])[1] GrayA{Float32}(Gray(1.4),0.5)
        @test_colortype_approx_eq ([p1].+[p2])[1] GrayA{Float32}(Gray(1.4),0.5)
        @test_colortype_approx_eq ([p1]+p2)[1] GrayA{Float32}(Gray(1.4),0.5)
        @test_colortype_approx_eq ([p1].+p2)[1] GrayA{Float32}(Gray(1.4),0.5)
        @test_colortype_approx_eq ([p1]-[p2])[1] GrayA{Float32}(Gray(0.2),-0.1)
        @test_colortype_approx_eq ([p1].-[p2])[1] GrayA{Float32}(Gray(0.2),-0.1)
        @test_colortype_approx_eq ([p1]-p2)[1] GrayA{Float32}(Gray(0.2),-0.1)
        @test_colortype_approx_eq ([p1].-p2)[1] GrayA{Float32}(Gray(0.2),-0.1)
        @test_colortype_approx_eq ([p1]/2)[1] GrayA{Float32}(Gray(0.4),0.1)
        @test_colortype_approx_eq (0.4f0*[p1]+0.6f0*[p2])[1] GrayA{Float32}(Gray(0.68),0.26)

        a = GrayA{U8}[GrayA(0.8,0.7), GrayA(0.5,0.2)]
        @fact sum(a) --> GrayA(u8sum(0.8,0.5), u8sum(0.7,0.2))
        @fact isapprox(a, a) --> true
        a = AGray{Float64}(1.0, 0.9999999999999999)
        b = AGray{Float64}(1.0, 1.0)

        @fact isapprox(a, b) --> true
        a = AGray{Float64}(1.0, 0.99)
        @fact isapprox(a, b, rtol = 0.01) --> false
        @fact isapprox(a, b, rtol = 0.1) --> true

    end

    context("Arithemtic with RGB") do
        cf = RGB{Float32}(0.1,0.2,0.3)
        ccmp = RGB{Float32}(0.2,0.4,0.6)
        @fact 2*cf --> ccmp
        @fact cf*2 --> ccmp
        @fact ccmp/2 --> cf
        @fact 2.0f0*cf --> ccmp
        @fact eltype(2.0*cf) --> Float64
        cu = RGB{U8}(0.1,0.2,0.3)
        @test_colortype_approx_eq 2*cu RGB(2*cu.r, 2*cu.g, 2*cu.b)
        @test_colortype_approx_eq 2.0f0*cu RGB(2.0f0*cu.r, 2.0f0*cu.g, 2.0f0*cu.b)
        f = U8(0.5)
        @fact (f*cu).r --> roughly(f*cu.r)
        @fact 2.*cf --> ccmp
        @fact cf.*2 --> ccmp
        @fact cf/2.0f0 --> RGB{Float32}(0.05,0.1,0.15)
        @fact cu/2 --> RGB(cu.r/2,cu.g/2,cu.b/2)
        @fact cu/0.5f0 --> RGB(cu.r/0.5f0, cu.g/0.5f0, cu.b/0.5f0)
        @fact cf+cf --> ccmp
        @fact cu * 1//2 --> roughly(RGB{Float64}(U8(0.1)/2, U8(0.2)/2, U8(0.3)/2))
        @test_colortype_approx_eq (cf*[0.8f0])[1] RGB{Float32}(0.8*0.1,0.8*0.2,0.8*0.3)
        @test_colortype_approx_eq ([0.8f0]*cf)[1] RGB{Float32}(0.8*0.1,0.8*0.2,0.8*0.3)
        @test_colortype_approx_eq (cf.*[0.8f0])[1] RGB{Float32}(0.8*0.1,0.8*0.2,0.8*0.3)
        @test_colortype_approx_eq ([0.8f0].*cf)[1] RGB{Float32}(0.8*0.1,0.8*0.2,0.8*0.3)
        @fact cf --> isfinite
        @fact cf --> not(isinf)
        @fact cf --> not(isnan)
        @fact RGB(NaN, 1, 0.5) --> not(isfinite)
        @fact RGB(NaN, 1, 0.5) --> not(isinf)
        @fact RGB(NaN, 1, 0.5) --> isnan
        @fact RGB(1, Inf, 0.5) --> not(isfinite)
        @fact RGB(1, Inf, 0.5) --> isinf
        @fact RGB(1, Inf, 0.5) --> not(isnan)
        @fact abs(RGB(0.1,0.2,0.3)) --> roughly(0.6)
        @fact sumabs2(RGB(0.1,0.2,0.3)) --> roughly(0.14)
        @fact norm(RGB(0.1,0.2,0.3)) --> roughly(sqrt(0.14))

        acu = RGB{U8}[cu]
        acf = RGB{Float32}[cf]
        @fact typeof(acu+acf) --> Vector{RGB{Float32}}
        @fact typeof(acu-acf) --> Vector{RGB{Float32}}
        @fact typeof(acu.+acf) --> Vector{RGB{Float32}}
        @fact typeof(acu.-acf) --> Vector{RGB{Float32}}
        @fact typeof(acu+cf) --> Vector{RGB{U8}}
        @fact typeof(acu-cf) --> Vector{RGB{U8}}
        @fact typeof(acu.+cf) --> Vector{RGB{U8}}
        @fact typeof(acu.-cf) --> Vector{RGB{U8}}
        @fact typeof(2*acf) --> Vector{RGB{Float32}}
        @fact typeof(convert(UInt8, 2)*acu) --> Vector{RGB{Float32}}
        @fact typeof(acu/2) --> Vector{RGB{typeof(U8(0.5)/2)}}

        a = RGB{U8}[RGB(1,0,0), RGB(1,0.8,0)]
        @fact sum(a) --> RGB(2.0,0.8,0)
        @fact isapprox(a, a) --> true
        a = RGB{Float64}(1.0, 1.0, 0.9999999999999999)
        b = RGB{Float64}(1.0, 1.0, 1.0)

        @fact isapprox(a, b) --> true
        a = RGB{Float64}(1.0, 1.0, 0.99)
        @fact isapprox(a, b, rtol = 0.01) --> false
        @fact isapprox(a, b, rtol = 0.1) --> true
    end

    context("Arithemtic with RGBA") do
        cf = RGBA{Float32}(0.1,0.2,0.3,0.4)
        ccmp = RGBA{Float32}(0.2,0.4,0.6,0.8)
        @fact 2*cf --> ccmp
        @fact cf*2 --> ccmp
        @fact ccmp/2 --> cf
        @fact 2.0f0*cf --> ccmp
        @fact eltype(2.0*cf) --> Float64
        cu = RGBA{U8}(0.1,0.2,0.3,0.4)
        @test_colortype_approx_eq 2*cu RGBA(2*cu.r, 2*cu.g, 2*cu.b, 2*cu.alpha)
        @test_colortype_approx_eq 2.0f0*cu RGBA(2.0f0*cu.r, 2.0f0*cu.g, 2.0f0*cu.b, 2.0f0*cu.alpha)
        f = U8(0.5)
        @fact (f*cu).r --> roughly(f*cu.r)
        @fact 2.*cf --> ccmp
        @fact cf.*2 --> ccmp
        @fact cf/2.0f0 --> RGBA{Float32}(0.05,0.1,0.15,0.2)
        @fact cu/2 --> RGBA(cu.r/2,cu.g/2,cu.b/2,cu.alpha/2)
        @fact cu/0.5f0 --> RGBA(cu.r/0.5f0, cu.g/0.5f0, cu.b/0.5f0, cu.alpha/0.5f0)
        @fact cf+cf --> ccmp
        @test_colortype_approx_eq (cf*[0.8f0])[1] RGBA{Float32}(0.8*0.1,0.8*0.2,0.8*0.3,0.8*0.4)
        @test_colortype_approx_eq ([0.8f0]*cf)[1] RGBA{Float32}(0.8*0.1,0.8*0.2,0.8*0.3,0.8*0.4)
        @test_colortype_approx_eq (cf.*[0.8f0])[1] RGBA{Float32}(0.8*0.1,0.8*0.2,0.8*0.3,0.8*0.4)
        @test_colortype_approx_eq ([0.8f0].*cf)[1] RGBA{Float32}(0.8*0.1,0.8*0.2,0.8*0.3,0.8*0.4)
        @fact cf --> isfinite
        @fact cf --> not(isinf)
        @fact cf --> not(isnan)
        @fact RGBA(NaN, 1, 0.5, 0.8) --> not(isfinite)
        @fact RGBA(NaN, 1, 0.5) --> not(isinf)
        @fact RGBA(NaN, 1, 0.5) --> isnan
        @fact RGBA(1, Inf, 0.5) --> not(isfinite)
        @fact RGBA(1, Inf, 0.5) --> isinf
        @fact RGBA(1, Inf, 0.5) --> not(isnan)
        @fact RGBA(0.2, 1, 0.5, NaN) --> not(isfinite)
        @fact RGBA(0.2, 1, 0.5, NaN) --> not(isinf)
        @fact RGBA(0.2, 1, 0.5, NaN) --> isnan
        @fact RGBA(0.2, 1, 0.5, Inf) --> not(isfinite)
        @fact RGBA(0.2, 1, 0.5, Inf) --> isinf
        @fact RGBA(0.2, 1, 0.5, Inf) --> not(isnan)
        @fact abs(RGBA(0.1,0.2,0.3,0.2)) --> roughly(0.8)

        acu = RGBA{U8}[cu]
        acf = RGBA{Float32}[cf]
        @fact typeof(acu+acf) --> Vector{RGBA{Float32}}
        @fact typeof(acu-acf) --> Vector{RGBA{Float32}}
        @fact typeof(acu.+acf) --> Vector{RGBA{Float32}}
        @fact typeof(acu.-acf) --> Vector{RGBA{Float32}}
        @fact typeof(acu+cf) --> Vector{RGBA{U8}}
        @fact typeof(acu-cf) --> Vector{RGBA{U8}}
        @fact typeof(acu.+cf) --> Vector{RGBA{U8}}
        @fact typeof(acu.-cf) --> Vector{RGBA{U8}}
        @fact typeof(2*acf) --> Vector{RGBA{Float32}}
        @fact typeof(convert(UInt8, 2)*acu) --> Vector{RGBA{Float32}}
        @fact typeof(acu/2) --> Vector{RGBA{typeof(U8(0.5)/2)}}

        a = RGBA{U8}[RGBA(1,0,0,0.8), RGBA(0.7,0.8,0,0.9)]
        @fact sum(a) --> RGBA(u8sum(1,0.7),0.8,0,u8sum(0.8,0.9))
        @fact isapprox(a, a) --> true
        a = ARGB{Float64}(1.0, 1.0, 1.0, 0.9999999999999999)
        b = ARGB{Float64}(1.0, 1.0, 1.0, 1.0)

        @fact isapprox(a, b) --> true
        a = ARGB{Float64}(1.0, 1.0, 1.0, 0.99)
        @fact isapprox(a, b, rtol = 0.01) --> false
        @fact isapprox(a, b, rtol = 0.1) --> true
    end
end

isinteractive() || FactCheck.exitstatus()

end
