module ColorVectorSpace

using ColorTypes, FixedPointNumbers, SpecialFunctions
using TensorCore
import TensorCore: ⊙, ⊗

using FixedPointNumbers: ShorterThanInt

import Base: ==, +, -, *, /, ^, <, ~
import Base: abs, clamp, convert, copy, div, eps, float,
    isfinite, isinf, isnan, isless, length, mapreduce, oneunit,
    promote_op, promote_rule, zero, trunc, floor, round, ceil, bswap,
    mod, mod1, rem, atan, hypot, max, min, real, typemin, typemax
# More unaryOps (mostly math functions)
import Base:      conj, sin, cos, tan, sinh, cosh, tanh,
                  asin, acos, atan, asinh, acosh, atanh,
                  sec, csc, cot, asec, acsc, acot,
                  sech, csch, coth, asech, acsch, acoth,
                  sinc, cosc, cosd, cotd, cscd, secd,
                  sind, tand, acosd, acotd, acscd, asecd,
                  asind, atand, rad2deg, deg2rad,
                  log, log2, log10, log1p, exponent, exp,
                  exp2, exp10, expm1, cbrt, sqrt,
                  significand, frexp, modf
import LinearAlgebra: norm, ⋅, dot, promote_leaf_eltypes  # norm1, norm2, normInf
import SpecialFunctions: gamma, logabsgamma, lfact
using Statistics
import Statistics: middle # and `_mean_promote`

export RGBRGB, complement, nan, dotc, dot, ⋅, hadamard, ⊙, tensor, ⊗, norm, varmult

MathTypes{T,C} = Union{AbstractRGB{T},TransparentRGB{C,T},AbstractGray{T},TransparentGray{C,T}}

## Version compatibility with ColorTypes

if !hasmethod(zero, (Type{TransparentGray},))
    zero(::Type{C}) where {C<:TransparentGray} = C(0,0)
    zero(::Type{C}) where {C<:AbstractRGB}     = C(0,0,0)
    zero(::Type{C}) where {C<:TransparentRGB}  = C(0,0,0,0)
    zero(p::Colorant) = zero(typeof(p))
end

if !hasmethod(one, (Type{TransparentGray},))
    Base.one(::Type{C}) where {C<:TransparentGray} = C(1,1)
    Base.one(::Type{C}) where {C<:AbstractRGB}     = C(1,1,1)
    Base.one(::Type{C}) where {C<:TransparentRGB}  = C(1,1,1,1)
    Base.one(p::Colorant) = one(typeof(p))
end

if !hasmethod(isfinite, (Colorant,))
    isfinite(c::Colorant) = mapreducec(isfinite, &, true, c)
    isinf(c::Colorant) = mapreducec(isinf, |, false, c)
    isnan(c::Colorant) = mapreducec(isnan, |, false, c)
end

if isdefined(ColorTypes, :nan)
    using ColorTypes: nan
else
    nan(::Type{T}) where {T<:AbstractFloat} = convert(T, NaN)
    nan(::Type{C}) where {T<:AbstractFloat, C<:MathTypes{T}} = mapc(_ -> nan(T), zero(C))
end

if which(real, (Type{<:AbstractGray},)).module === Base
    real(::Type{C}) where {C<:AbstractGray} = real(eltype(C))
end

# To help type inference
promote_rule(::Type{T}, ::Type{C}) where {T<:Real,C<:AbstractGray} = promote_type(T, eltype(C))

promote_leaf_eltypes(x::Union{AbstractArray{T},Tuple{T,Vararg{T}}}) where {T<:MathTypes} = eltype(T)

if isdefined(Statistics, :_mean_promote)
    Statistics._mean_promote(x::MathTypes, y::MathTypes) = mapc(FixedPointNumbers.Treduce, y)
end

## Traits and key utilities

# Return eltypes for arithmetic operations
multype(::Type{A}, ::Type{B}) where {A,B} = coltype(typeof(zero(A)*zero(B)))
sumtype(::Type{A}, ::Type{B}) where {A,B} = coltype(typeof(zero(A)+zero(B)))
divtype(::Type{A}, ::Type{B}) where {A,B} = coltype(typeof(zero(A)/oneunit(B)))
powtype(::Type{A}, ::Type{B}) where {A,B} = coltype(typeof(zero(A)^zero(B)))
sumtype(a::Colorant, b::Colorant) = coltype(sumtype(eltype(a),eltype(b)))

coltype(::Type{T}) where {T<:Fractional} = T
coltype(::Type{T}) where {T<:Number}     = floattype(T)

acctype(::Type{T}) where {T<:FixedPoint}        = floattype(T)
acctype(::Type{T}) where {T<:ShorterThanInt}    = Int
acctype(::Type{Rational{T}}) where {T<:Integer} = typeof(zero(T)/oneunit(T))
acctype(::Type{T}) where {T<:Real}              = T

acctype(::Type{T1}, ::Type{T2}) where {T1,T2} = acctype(promote_type(T1, T2))

acc(x::Number) = convert(acctype(typeof(x)), x)

# Scalar binary RGB operations require the same RGB type for each element,
# otherwise we don't know which to return
color_rettype(::Type{A}, ::Type{B}) where {A<:AbstractRGB,B<:AbstractRGB} = _color_rettype(base_colorant_type(A), base_colorant_type(B))
color_rettype(::Type{A}, ::Type{B}) where {A<:AbstractGray,B<:AbstractGray} = _color_rettype(base_colorant_type(A), base_colorant_type(B))
color_rettype(::Type{A}, ::Type{B}) where {A<:TransparentRGB,B<:TransparentRGB} = _color_rettype(base_colorant_type(A), base_colorant_type(B))
color_rettype(::Type{A}, ::Type{B}) where {A<:TransparentGray,B<:TransparentGray} = _color_rettype(base_colorant_type(A), base_colorant_type(B))
_color_rettype(::Type{C}, ::Type{C}) where {C<:Colorant} = C

color_rettype(c1::Colorant, c2::Colorant) = color_rettype(typeof(c1), typeof(c2))

arith_colorant_type(::C) where {C<:Colorant} = arith_colorant_type(C)
function arith_colorant_type(::Type{C}) where {C<:Colorant}
    Cb = base_colorant_type(C)
    isconcretetype(C) && C === Cb && return _arith_colorant_type(C) # non-parametric
    return Cb
end
_arith_colorant_type(::Type{<:AbstractGray})    = Gray
_arith_colorant_type(::Type{<:TransparentGray}) = AGray
_arith_colorant_type(::Type{<:AbstractGrayA})   = GrayA
_arith_colorant_type(::Type{<:AbstractRGB})     = RGB
_arith_colorant_type(::Type{<:TransparentRGB})  = ARGB
_arith_colorant_type(::Type{<:AbstractRGBA})    = RGBA

parametric(::Type{C}, ::Type{T}) where {C,T} = C{T}
parametric(::Type{C}, ::Type{T}) where {T, C<:Colorant{T}} = C # e.g. parametric(RGB24, N0f8) == RGB24

rettype(::typeof(+), a::C, b::C) where {C <: Colorant} = C
rettype(::typeof(-), a::C, b::C) where {C <: Colorant} = C
rettype(::typeof(+), a, b) = parametric(color_rettype(a, b), sumtype(a, b))
rettype(::typeof(-), a, b) = parametric(color_rettype(a, b), sumtype(a, b))
rettype(::typeof(*), a, b) = parametric(color_rettype(a, b), multype(eltype(a), eltype(b))) # gray * gray
rettype(::typeof(*), a::Real, b) = arith_colorant_type(b){multype(typeof(a), eltype(b))}
rettype(::typeof(/), a, b::Real) = arith_colorant_type(a){divtype(eltype(a), typeof(b))}
rettype(::typeof(^), a, b)          = arith_colorant_type(a){powtype(eltype(a), typeof(b))}
rettype(::typeof(^), a, b::Integer) = arith_colorant_type(a){powtype(eltype(a), Int)}

# Useful for leveraging iterator algorithms. Don't use this externally, as the implementation may change.
channels(c::AbstractGray)    = (gray(c),)
channels(c::TransparentGray) = (gray(c), alpha(c))
channels(c::AbstractRGB)     = (red(c), green(c), blue(c))
channels(c::TransparentRGB)  = (red(c), green(c), blue(c), alpha(c))

## Generic algorithms
Base.add_sum(c1::MathTypes,c2::MathTypes) = mapc(Base.add_sum, c1, c2)
Base.reduce_first(::typeof(Base.add_sum), c::MathTypes) = mapc(x->Base.reduce_first(Base.add_sum, x), c)
function Base.reduce_empty(::typeof(Base.add_sum), ::Type{T}) where {T<:MathTypes}
    z = Base.reduce_empty(Base.add_sum, eltype(T))
    return zero(base_colorant_type(T){typeof(z)})
end


## Rounding & mod
for f in (:trunc, :floor, :round, :ceil, :eps, :bswap)
    @eval $f(g::Gray{T}) where {T} = Gray{T}($f(gray(g)))
end
eps(::Type{Gray{T}}) where {T} = Gray(eps(T))

for f in (:trunc, :floor, :round, :ceil)
    @eval $f(::Type{T}, g::Gray) where {T<:Integer} = Gray{T}($f(T, gray(g)))
end

for f in (:mod, :rem, :mod1)
    @eval $f(x::Gray, m::Gray) = Gray($f(gray(x), gray(m)))
end

dotc(x::T, y::T) where {T<:Real} = acc(x)*acc(y)
dotc(x::Real, y::Real) = dotc(promote(x, y)...)

"""
    y = complement(x)

Take the complement `1-x` of `x`.  If `x` is a color with an alpha channel,
the alpha channel is left untouched. Don't forget to add a dot when `x` is
an array: `complement.(x)`
"""
complement(x::Union{Number,Colorant}) = oneunit(x)-x
complement(x::TransparentColor) = typeof(x)(complement(color(x)), alpha(x))


## Math on Colors. These implementations encourage inlining and,
## for the case of Normed types, nearly halve the number of multiplications (for RGB)

# Common
copy(c::MathTypes) = c
(*)(c::MathTypes, f::Real) = (*)(f, c)
(+)(c::MathTypes) = mapc(+, c)
(+)(c::MathTypes{Bool}) = c
(-)(c::MathTypes) = mapc(-, c)
(-)(c::MathTypes{Bool}) = c
(/)(c::MathTypes, f::Real) = (one(f)/f)*c
(/)(c::MathTypes, f::Integer) = (one(eltype(c))/f)*c
abs(c::MathTypes) = mapc(abs, c)
norm(c::MathTypes, p::Real=2) = (cc = channels(c); norm(cc, p)/(p == 0 ? length(cc) : length(cc)^(1/p)))

## Mixed types
(+)(a::MathTypes, b::MathTypes) = (+)(promote(a, b)...)
(-)(a::MathTypes, b::MathTypes) = (-)(promote(a, b)...)


# Scalar RGB
(*)(f::Real, c::AbstractRGB)    = rettype(*, f, c)(f*red(c), f*green(c), f*blue(c))
(*)(f::Real, c::TransparentRGB) = rettype(*, f, c)(f*red(c), f*green(c), f*blue(c), f*alpha(c))
function (*)(f::Real, c::AbstractRGB{T}) where T<:Normed
    fs = f*(1/reinterpret(oneunit(T)))
    rettype(*, f, c)(fs*reinterpret(red(c)), fs*reinterpret(green(c)), fs*reinterpret(blue(c)))
end
function (*)(f::Normed, c::AbstractRGB{T}) where T<:Normed
    fs = reinterpret(f)*(1/widen(reinterpret(oneunit(T)))^2)
    rettype(*, f, c)(fs*reinterpret(red(c)), fs*reinterpret(green(c)), fs*reinterpret(blue(c)))
end
function (/)(c::AbstractRGB{T}, f::Real) where T<:Normed
    fs = (one(f)/reinterpret(oneunit(T)))/f
    rettype(/, c, f)(fs*reinterpret(red(c)), fs*reinterpret(green(c)), fs*reinterpret(blue(c)))
end
function (/)(c::AbstractRGB{T}, f::Integer) where T<:Normed
    fs = (1/reinterpret(oneunit(T)))/f
    rettype(/, c, f)(fs*reinterpret(red(c)), fs*reinterpret(green(c)), fs*reinterpret(blue(c)))
end
(+)(a::AbstractRGB, b::AbstractRGB) = rettype(+, a, b)(red(a)+red(b), green(a)+green(b), blue(a)+blue(b))
(-)(a::AbstractRGB, b::AbstractRGB) = rettype(-, a, b)(red(a)-red(b), green(a)-green(b), blue(a)-blue(b))
(+)(a::TransparentRGB, b::TransparentRGB) = rettype(+, a, b)(red(a)+red(b), green(a)+green(b), blue(a)+blue(b), alpha(a)+alpha(b))
(-)(a::TransparentRGB, b::TransparentRGB) = rettype(-, a, b)(red(a)-red(b), green(a)-green(b), blue(a)-blue(b), alpha(a)-alpha(b))

# New multiplication operators
(⋅)(x::AbstractRGB, y::AbstractRGB)  = (T = acctype(eltype(x), eltype(y)); T(red(x))*T(red(y)) + T(green(x))*T(green(y)) + T(blue(x))*T(blue(y)))/3
(⋅)(x::Union{AbstractRGB,AbstractGray}, y::Union{AbstractRGB,AbstractGray})  = ⋅(promote(x, y)...)
(⊙)(x::C, y::C) where C<:AbstractRGB = base_color_type(C)(red(x)*red(y), green(x)*green(y), blue(x)*blue(y))
(⊙)(x::Union{AbstractRGB,AbstractGray}, y::Union{AbstractRGB,AbstractGray})  = ⊙(promote(x, y)...)
# ⊗ defined below


# These constants come from squaring the conversion to grayscale
# (rec601 luma), and normalizing
dotc(x::T, y::T) where {T<:AbstractRGB} = 0.200f0 * acc(red(x))*acc(red(y)) + 0.771f0 * acc(green(x))*acc(green(y)) + 0.029f0 * acc(blue(x))*acc(blue(y))
dotc(x::AbstractRGB, y::AbstractRGB) = dotc(promote(x, y)...)

# Scalar Gray
const unaryOps = (:~, :conj, :abs,
                  :sin, :cos, :tan, :sinh, :cosh, :tanh,
                  :asin, :acos, :atan, :asinh, :acosh, :atanh,
                  :sec, :csc, :cot, :asec, :acsc, :acot,
                  :sech, :csch, :coth, :asech, :acsch, :acoth,
                  :sinc, :cosc, :cosd, :cotd, :cscd, :secd,
                  :sind, :tand, :acosd, :acotd, :acscd, :asecd,
                  :asind, :atand, :rad2deg, :deg2rad,
                  :log, :log2, :log10, :log1p, :exponent, :exp,
                  :exp2, :exp10, :expm1, :cbrt, :sqrt,
                  :significand,
                  :gamma, :lfact, :frexp, :modf,
                  :(SpecialFunctions.erf), :(SpecialFunctions.erfc),
                  :(SpecialFunctions.erfcx), :(SpecialFunctions.erfi), :(SpecialFunctions.dawson),
                  :(SpecialFunctions.airyai),
                  :(SpecialFunctions.airyaiprime), :(SpecialFunctions.airybi), :(SpecialFunctions.airybiprime),
                  :(SpecialFunctions.besselj0), :(SpecialFunctions.besselj1), :(SpecialFunctions.bessely0), :(SpecialFunctions.bessely1),
                  :(SpecialFunctions.eta), :(SpecialFunctions.zeta), :(SpecialFunctions.digamma))
for op in unaryOps
    @eval ($op)(c::AbstractGray) = Gray($op(gray(c)))
end
function logabsgamma(c::AbstractGray)
    lagc, s = logabsgamma(gray(c))
    return Gray(lagc), s
end

middle(c::AbstractGray) = arith_colorant_type(c)(middle(gray(c)))
middle(x::C, y::C) where {C<:AbstractGray} = arith_colorant_type(C)(middle(gray(x), gray(y)))

(*)(f::Real, c::AbstractGray)    = rettype(*, f, c)(f*gray(c))
(*)(f::Real, c::TransparentGray) = rettype(*, f, c)(f*gray(c), f*alpha(c))
(/)(n::Number, c::AbstractGray) = base_color_type(c)(n/gray(c))
(+)(a::AbstractGray,    b::AbstractGray)    = rettype(+, a, b)(gray(a)+gray(b))
(+)(a::TransparentGray, b::TransparentGray) = rettype(+, a, b)(gray(a)+gray(b), alpha(a)+alpha(b))
(-)(a::AbstractGray,    b::AbstractGray)    = rettype(-, a, b)(gray(a)-gray(b))
(-)(a::TransparentGray, b::TransparentGray) = rettype(-, a, b)(gray(a)-gray(b), alpha(a)-alpha(b))
(*)(a::AbstractGray, b::AbstractGray) = rettype(*, a, b)(gray(a)*gray(b))
(^)(a::AbstractGray, b::Integer) = rettype(^, a, b)(gray(a)^convert(Int,b))
(^)(a::AbstractGray, b::Real)    = rettype(^, a, b)(gray(a)^b)
(/)(a::C, b::C) where C<:AbstractGray = base_color_type(C)(gray(a)/gray(b))
(/)(a::AbstractGray, b::AbstractGray) = /(promote(a, b)...)
(+)(a::AbstractGray, b::Number) = base_color_type(a)(gray(a)+b)
(+)(a::Number, b::AbstractGray) = b+a
(-)(a::AbstractGray, b::Number) = base_color_type(a)(gray(a)-b)
(-)(a::Number, b::AbstractGray) = base_color_type(b)(a-gray(b))

(⋅)(x::AbstractGray, y::AbstractGray) = gray(x)*gray(y)
(⊙)(x::C, y::C) where C<:AbstractGray = base_color_type(C)(gray(x)*gray(y))
(⊙)(x::AbstractGray, y::AbstractGray) = ⊙(promote(x, y)...)
(⊗)(x::AbstractGray, y::AbstractGray) = ⊙(x, y)

max(a::T, b::T) where {T<:AbstractGray} = T(max(gray(a),gray(b)))
max(a::AbstractGray, b::AbstractGray) = max(promote(a,b)...)
max(a::Number, b::AbstractGray) = max(promote(a,b)...)
max(a::AbstractGray, b::Number) = max(promote(a,b)...)
min(a::T, b::T) where {T<:AbstractGray} = T(min(gray(a),gray(b)))
min(a::AbstractGray, b::AbstractGray) = min(promote(a,b)...)
min(a::Number, b::AbstractGray) = min(promote(a,b)...)
min(a::AbstractGray, b::Number) = min(promote(a,b)...)

atan(x::AbstractGray, y::AbstractGray)  = atan(gray(x), gray(y))
hypot(x::AbstractGray, y::AbstractGray) = hypot(gray(x), gray(y))

if which(<, Tuple{AbstractGray,AbstractGray}).module === Base  # planned for ColorTypes 0.11
    (<)(g1::AbstractGray, g2::AbstractGray) = gray(g1) < gray(g2)
    (<)(c::AbstractGray, r::Real) = gray(c) < r
    (<)(r::Real, c::AbstractGray) = r < gray(c)
end
if !hasmethod(isless, Tuple{AbstractGray,Real})  # planned for ColorTypes 0.11
    isless(c::AbstractGray, r::Real) = isless(gray(c), r)
    isless(r::Real, c::AbstractGray) = isless(r, gray(c))
end

dotc(x::T, y::T) where {T<:AbstractGray} = acc(gray(x))*acc(gray(y))
dotc(x::AbstractGray, y::AbstractGray) = dotc(promote(x, y)...)

typemin(::Type{C}) where {C<:AbstractGray} = C(typemin(eltype(C)))
typemax(::Type{C}) where {C<:AbstractGray} = C(typemax(eltype(C)))
typemin(c::AbstractGray) = typemin(typeof(c))
typemax(c::AbstractGray) = typemax(typeof(c))

## RGB tensor products

"""
    RGBRGB(rr, gr, br, rg, gg, bg, rb, gb, bb)

Represent the [tensor product](https://en.wikipedia.org/wiki/Tensor_product) of two RGB values.

# Example

```jldoctest
julia> a, b = RGB(0.2f0, 0.3f0, 0.5f0), RGB(0.77f0, 0.11f0, 0.22f0)
(RGB{Float32}(0.2f0,0.3f0,0.5f0), RGB{Float32}(0.77f0,0.11f0,0.22f0))

julia> a ⊗ b
RGBRGB{Float32}:
 0.154  0.022  0.044
 0.231  0.033  0.066
 0.385  0.055  0.11
"""
struct RGBRGB{T}
    rr::T
    gr::T
    br::T
    rg::T
    gg::T
    bg::T
    rb::T
    gb::T
    bb::T
end
function RGBRGB{T}(mat::AbstractMatrix) where T
    size(mat) == (3, 3) || throw(DimensionMismatch("`mat` must be a 3×3 matrix."))
    @inbounds RGBRGB{T}(NTuple{9, T}(T(mat[I]) for I in CartesianIndices(mat))...)
end
RGBRGB(mat::AbstractMatrix{T}) where T = RGBRGB{T}(mat)

Base.eltype(::Type{RGBRGB{T}}) where T = T
Base.Matrix{T}(p::RGBRGB) where T = T[p.rr p.rg p.rb;
                                      p.gr p.gg p.gb;
                                      p.br p.bg p.bb]
Base.Matrix(p::RGBRGB{T}) where T = Matrix{T}(p)

function Base.show(io::IO, @nospecialize(p::RGBRGB))
    print(io, "RGBRGB{", eltype(p), "}([")
    ioc = IOContext(io, :typeinfo => eltype(p))
    print(ioc, p.rr, " ", p.rg, " ", p.rb, "; ")
    print(ioc, p.gr, " ", p.gg, " ", p.gb, "; ")
    print(ioc, p.br, " ", p.bg, " ", p.bb, "])")
end
function Base.show(io::IO, ::MIME"text/plain", @nospecialize(p::RGBRGB))
    println(io, "RGBRGB{", eltype(p), "}:")
    ioc = IOContext(io, :typeinfo => eltype(p), :compact => get(io, :compact, true))
    Base.print_matrix(ioc, Matrix(p))
end

Base.zero(::Type{RGBRGB{T}}) where T = (z = zero(T); RGBRGB(z, z, z, z, z, z, z, z, z))
Base.zero(a::RGBRGB) = zero(typeof(a))

+(a::RGBRGB) = a
-(a::RGBRGB) = RGBRGB(-a.rr, -a.gr, -a.br, -a.rg, -a.gg, -a.bg, -a.rb, -a.gb, -a.bb)
+(a::RGBRGB, b::RGBRGB) = RGBRGB(a.rr + b.rr, a.gr + b.gr, a.br + b.br,
                                 a.rg + b.rg, a.gg + b.gg, a.bg + b.bg,
                                 a.rb + b.rb, a.gb + b.gb, a.bb + b.bb)
-(a::RGBRGB, b::RGBRGB) = +(a, -b)
*(α::Real, a::RGBRGB) = RGBRGB(α*a.rr, α*a.gr, α*a.br, α*a.rg, α*a.gg, α*a.bg, α*a.rb, α*a.gb, α*a.bb)
*(a::RGBRGB, α::Real) = α*a
/(a::RGBRGB, α::Real) = (1/α)*a

function ⊗(a::AbstractRGB, b::AbstractRGB)
    ar, ag, ab = red(a), green(a), blue(a)
    br, bg, bb = red(b), green(b), blue(b)
    agbr, abbg, arbb, abbr, arbg, agbb = ag*br, ab*bg, ar*bb, ab*br, ar*bg, ag*bb
    return RGBRGB(ar*br, agbr, abbr, arbg, ag*bg, abbg, arbb, agbb, ab*bb)
end
⊗(a::Union{AbstractRGB,AbstractGray}, b::Union{AbstractRGB,AbstractGray}) = ⊗(promote(a, b)...)

"""
    varmult(op, itr; corrected::Bool=true, mean=Statistics.mean(itr), dims=:)

Compute the variance of elements of `itr`, using `op` as the multiplication operator.
The keyword arguments behave identically to those of `Statistics.var`.

# Example

```julia
julia> cs = [RGB(0.2, 0.3, 0.4), RGB(0.5, 0.3, 0.2)]
2-element Array{RGB{Float64},1} with eltype RGB{Float64}:
 RGB{Float64}(0.2,0.3,0.4)
 RGB{Float64}(0.5,0.3,0.2)

julia> varmult(⋅, cs)
0.021666666666666667

julia> varmult(⊙, cs)
RGB{Float64}(0.045,0.0,0.020000000000000004)

julia> varmult(⊗, cs)
RGBRGB{Float64}:
  0.045  0.0  -0.03
  0.0    0.0   0.0
 -0.03   0.0   0.02
```
"""
function varmult(op, itr; corrected::Bool=true, dims=:, mean=Statistics.mean(itr; dims=dims))
    if dims === (:)
        v = mapreduce(c->(Δc = c-mean; op(Δc, Δc)), +, itr; dims=dims)
        n = length(itr)
    else
        # TODO: avoid temporary creation
        v = mapreduce(Δc->op(Δc, Δc), +, itr .- mean; dims=dims)
        n = length(itr) // length(v)
    end
    return v / (corrected ? max(1, n-1) : max(1, n))
end

function __init__()
    if isdefined(Base, :Experimental) && isdefined(Base.Experimental, :register_error_hint)
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
            if exc.f === _color_rettype && length(argtypes) >= 2
                # Color is not necessary, this is just to show it's possible.
                A, B = argtypes
                A !== B && print(io, "\nIn binary operation with $A and $B, the return type is ambiguous")
            end
        end
    end
end

## Precompilation

if Base.VERSION >= v"1.4.2"
    include("precompile.jl")
    _precompile_()
end

end
