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
import SpecialFunctions: gamma, lgamma, lfact
using Statistics
import Statistics: middle, _mean_promote

export RGBRGB, nan, dotc, dot, ⋅, hadamard, ⊙, tensor, ⊗, norm, varmult

MathTypes{T,C} = Union{AbstractRGB{T},TransparentRGB{C,T},AbstractGray{T},TransparentGray{C,T}}

## Version compatibility with ColorTypes

if !hasmethod(zero, (Type{AGray{N0f8}},))
    zero(::Type{C}) where {C<:TransparentGray} = C(0,0)
    zero(::Type{C}) where {C<:AbstractRGB}     = C(0,0,0)
    zero(::Type{C}) where {C<:TransparentRGB}  = C(0,0,0,0)
    zero(p::Colorant) = zero(typeof(p))
end

if !hasmethod(one, (Gray{N0f8},))
    Base.one(::Type{C}) where {C<:AbstractGray}    = C(1)
    Base.one(::Type{C}) where {C<:TransparentGray} = C(1,1)
    Base.one(::Type{C}) where {C<:AbstractRGB}     = C(1,1,1)
    Base.one(::Type{C}) where {C<:TransparentRGB}  = C(1,1,1,1)
    Base.one(p::Colorant) = one(typeof(p))
end

if !hasmethod(oneunit, (Type{AGray{N0f8}},))
    oneunit(::Type{C}) where {C<:TransparentGray} = C(1,1)
    oneunit(::Type{C}) where {C<:AbstractRGB}     = C(1,1,1)
    oneunit(::Type{C}) where {C<:TransparentRGB}  = C(1,1,1,1)
    oneunit(p::Colorant) = oneunit(typeof(p))
end

# Real values are treated like grays
if !hasmethod(gray, (Number,))
    ColorTypes.gray(x::Real) = x
end

## Traits and key utilities

# Return types for arithmetic operations
multype(::Type{A}, ::Type{B}) where {A,B} = coltype(typeof(zero(A)*zero(B)))
sumtype(::Type{A}, ::Type{B}) where {A,B} = coltype(typeof(zero(A)+zero(B)))
divtype(::Type{A}, ::Type{B}) where {A,B} = coltype(typeof(zero(A)/oneunit(B)))
powtype(::Type{A}, ::Type{B}) where {A,B} = coltype(typeof(zero(A)^zero(B)))
multype(a::Colorant, b::Colorant) = coltype(multype(eltype(a),eltype(b)))
sumtype(a::Colorant, b::Colorant) = coltype(sumtype(eltype(a),eltype(b)))
divtype(a::Colorant, b::Colorant) = coltype(divtype(eltype(a),eltype(b)))

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
_color_rettype(::Type{A}, ::Type{B}) where {A<:Colorant,B<:Colorant} = error("binary operation with $A and $B, return type is ambiguous")
_color_rettype(::Type{C}, ::Type{C}) where {C<:Colorant} = C

color_rettype(c1::Colorant, c2::Colorant) = color_rettype(typeof(c1), typeof(c2))

arith_colorant_type(::C) where {C<:Colorant} = arith_colorant_type(C)
arith_colorant_type(::Type{C}) where {C<:Colorant} = base_colorant_type(C)
arith_colorant_type(::Type{Gray24}) = Gray
arith_colorant_type(::Type{AGray32}) = AGray
arith_colorant_type(::Type{RGB24}) = RGB
arith_colorant_type(::Type{ARGB32}) = ARGB

parametric(::Type{C}, ::Type{T}) where {C,T} = C{T}
parametric(::Type{Gray24}, ::Type{N0f8}) = Gray24
parametric(::Type{RGB24}, ::Type{N0f8}) = RGB24
parametric(::Type{AGray32}, ::Type{N0f8}) = AGray32
parametric(::Type{ARGB32}, ::Type{N0f8}) = ARGB32

# Useful for leveraging iterator algorithms. Don't use this externally, as the implementation may change.
channels(c::AbstractGray)    = (gray(c),)
channels(c::TransparentGray) = (gray(c), alpha(c))
channels(c::AbstractRGB)     = (red(c), green(c), blue(c))
channels(c::TransparentRGB)  = (red(c), green(c), blue(c), alpha(c))

## Math on colors and color types

nan(::Type{T}) where {T<:AbstractFloat} = convert(T, NaN)
nan(::Type{C}) where {C<:MathTypes} = _nan(eltype(C), C)
_nan(::Type{T}, ::Type{C}) where {T<:AbstractFloat,C<:AbstractGray} = (x = convert(T, NaN); C(x))
_nan(::Type{T}, ::Type{C}) where {T<:AbstractFloat,C<:TransparentGray} = (x = convert(T, NaN); C(x,x))
_nan(::Type{T}, ::Type{C}) where {T<:AbstractFloat,C<:AbstractRGB} = (x = convert(T, NaN); C(x,x,x))
_nan(::Type{T}, ::Type{C}) where {T<:AbstractFloat,C<:TransparentRGB} = (x = convert(T, NaN); C(x,x,x,x))



## Generic algorithms
Base.add_sum(c1::MathTypes,c2::MathTypes) = mapc(Base.add_sum, c1, c2)
Base.reduce_first(::typeof(Base.add_sum), c::MathTypes) = mapc(x->Base.reduce_first(Base.add_sum, x), c)
function Base.reduce_empty(::typeof(Base.add_sum), ::Type{T}) where {T<:MathTypes}
    z = Base.reduce_empty(Base.add_sum, eltype(T))
    return zero(base_colorant_type(T){typeof(z)})
end

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

## Math on Colors. These implementations encourage inlining and,
## for the case of Normed types, nearly halve the number of multiplications (for RGB)

# Scalar RGB
copy(c::AbstractRGB) = c
(+)(c::AbstractRGB) = mapc(+, c)
(+)(c::TransparentRGB) = mapc(+, c)
(-)(c::AbstractRGB) = mapc(-, c)
(-)(c::TransparentRGB) = mapc(-, c)
(*)(f::Real, c::AbstractRGB) = arith_colorant_type(c){multype(typeof(f),eltype(c))}(f*red(c), f*green(c), f*blue(c))
(*)(f::Real, c::TransparentRGB) = arith_colorant_type(c){multype(typeof(f),eltype(c))}(f*red(c), f*green(c), f*blue(c), f*alpha(c))
function (*)(f::Real, c::AbstractRGB{T}) where T<:Normed
    fs = f*(1/reinterpret(oneunit(T)))
    arith_colorant_type(c){multype(typeof(f),T)}(fs*reinterpret(red(c)), fs*reinterpret(green(c)), fs*reinterpret(blue(c)))
end
function (*)(f::Normed, c::AbstractRGB{T}) where T<:Normed
    fs = reinterpret(f)*(1/widen(reinterpret(oneunit(T)))^2)
    arith_colorant_type(c){multype(typeof(f),T)}(fs*reinterpret(red(c)), fs*reinterpret(green(c)), fs*reinterpret(blue(c)))
end
function (/)(c::AbstractRGB{T}, f::Real) where T<:Normed
    fs = (one(f)/reinterpret(oneunit(T)))/f
    arith_colorant_type(c){divtype(typeof(f),T)}(fs*reinterpret(red(c)), fs*reinterpret(green(c)), fs*reinterpret(blue(c)))
end
function (/)(c::AbstractRGB{T}, f::Integer) where T<:Normed
    fs = (1/reinterpret(oneunit(T)))/f
    arith_colorant_type(c){divtype(typeof(f),T)}(fs*reinterpret(red(c)), fs*reinterpret(green(c)), fs*reinterpret(blue(c)))
end
(+)(a::AbstractRGB{S}, b::AbstractRGB{T}) where {S,T} = parametric(color_rettype(a, b), sumtype(S,T))(red(a)+red(b), green(a)+green(b), blue(a)+blue(b))
(-)(a::AbstractRGB{S}, b::AbstractRGB{T}) where {S,T} = parametric(color_rettype(a, b), sumtype(S,T))(red(a)-red(b), green(a)-green(b), blue(a)-blue(b))
(+)(a::TransparentRGB, b::TransparentRGB) =
    parametric(color_rettype(a, b), sumtype(a,b))(red(a)+red(b), green(a)+green(b), blue(a)+blue(b), alpha(a)+alpha(b))
(-)(a::TransparentRGB, b::TransparentRGB) =
    parametric(color_rettype(a, b), sumtype(a,b))(red(a)-red(b), green(a)-green(b), blue(a)-blue(b), alpha(a)-alpha(b))
(*)(c::AbstractRGB, f::Real) = (*)(f, c)
(*)(c::TransparentRGB, f::Real) = (*)(f, c)
(/)(c::AbstractRGB, f::Real) = (one(f)/f)*c
(/)(c::TransparentRGB, f::Real) = (one(f)/f)*c
(/)(c::AbstractRGB, f::Integer) = (one(eltype(c))/f)*c
(/)(c::TransparentRGB, f::Integer) = (one(eltype(c))/f)*c


# New multiplication operators
(⋅)(x::AbstractRGB, y::AbstractRGB)  = (T = acctype(eltype(x), eltype(y)); T(red(x))*T(red(y)) + T(green(x))*T(green(y)) + T(blue(x))*T(blue(y)))/3
(⊙)(x::C, y::C) where C<:AbstractRGB = base_color_type(C)(red(x)*red(y), green(x)*green(y), blue(x)*blue(y))
(⊙)(x::AbstractRGB, y::AbstractRGB)  = ⊙(promote(x, y)...)
# ⊗ defined below

isfinite(c::Colorant{T}) where {T<:Normed} = true
isfinite(c::Colorant) = mapreducec(isfinite, &, true, c)
isnan(c::Colorant{T}) where {T<:Normed} = false
isnan(c::Colorant) = mapreducec(isnan, |, false, c)
isinf(c::Colorant{T}) where {T<:Normed} = false
isinf(c::Colorant) = mapreducec(isinf, |, false, c)
abs(c::MathTypes) = mapc(abs, c)
norm(c::MathTypes, p::Real=2) = (cc = channels(c); norm(cc, p)/(p == 0 ? length(cc) : length(cc)^(1/p)))

# function Base.rtoldefault(::Union{C1,Type{C1}}, ::Union{C2,Type{C2}}, atol::Real) where {C1<:MathTypes,C2<:MathTypes}
#     T1, T2 = eltype(C1), eltype(C2)
#     @show T1, T2
#     return Base.rtoldefault(eltype(C1), eltype(C2), atol)
# end

promote_leaf_eltypes(x::Union{AbstractArray{T},Tuple{T,Vararg{T}}}) where {T<:MathTypes} = eltype(T)

# These constants come from squaring the conversion to grayscale
# (rec601 luma), and normalizing
dotc(x::T, y::T) where {T<:AbstractRGB} = 0.200f0 * acc(red(x))*acc(red(y)) + 0.771f0 * acc(green(x))*acc(green(y)) + 0.029f0 * acc(blue(x))*acc(blue(y))
dotc(x::AbstractRGB, y::AbstractRGB) = dotc(promote(x, y)...)

# Scalar Gray
copy(c::AbstractGray) = c
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
                  :significand, :lgamma,
                  :gamma, :lfact, :frexp, :modf,
                  :(SpecialFunctions.erf), :(SpecialFunctions.erfc),
                  :(SpecialFunctions.erfcx), :(SpecialFunctions.erfi), :(SpecialFunctions.dawson),
                  :(SpecialFunctions.airy), :(SpecialFunctions.airyai),
                  :(SpecialFunctions.airyprime), :(SpecialFunctions.airyaiprime), :(SpecialFunctions.airybi), :(SpecialFunctions.airybiprime),
                  :(SpecialFunctions.besselj0), :(SpecialFunctions.besselj1), :(SpecialFunctions.bessely0), :(SpecialFunctions.bessely1),
                  :(SpecialFunctions.eta), :(SpecialFunctions.zeta), :(SpecialFunctions.digamma))
for op in unaryOps
    @eval ($op)(c::AbstractGray) = Gray($op(gray(c)))
end

middle(c::AbstractGray) = arith_colorant_type(c)(middle(gray(c)))
middle(x::C, y::C) where {C<:AbstractGray} = arith_colorant_type(C)(middle(gray(x), gray(y)))

_mean_promote(x::MathTypes, y::MathTypes) = mapc(FixedPointNumbers.Treduce, y)

(*)(f::Real, c::AbstractGray) = arith_colorant_type(c){multype(typeof(f),eltype(c))}(f*gray(c))
(*)(f::Real, c::TransparentGray) = arith_colorant_type(c){multype(typeof(f),eltype(c))}(f*gray(c), f*alpha(c))
(*)(c::AbstractGray, f::Real) = (*)(f, c)
(*)(c::TransparentGray, f::Real) = (*)(f, c)
(/)(c::AbstractGray, f::Real) = (one(f)/f)*c
(/)(n::Number, c::AbstractGray) = base_color_type(c)(n/gray(c))
(/)(c::TransparentGray, f::Real) = (one(f)/f)*c
(/)(c::AbstractGray, f::Integer) = (one(eltype(c))/f)*c
(/)(c::TransparentGray, f::Integer) = (one(eltype(c))/f)*c
(+)(a::AbstractGray{S}, b::AbstractGray{T}) where {S,T} = parametric(color_rettype(a,b), sumtype(S,T))(gray(a)+gray(b))
(+)(a::TransparentGray, b::TransparentGray) = parametric(color_rettype(a,b), sumtype(eltype(a),eltype(b)))(gray(a)+gray(b),alpha(a)+alpha(b))
(-)(a::AbstractGray{S}, b::AbstractGray{T}) where {S,T} = parametric(color_rettype(a,b), sumtype(S,T))(gray(a)-gray(b))
(-)(a::TransparentGray, b::TransparentGray) = parametric(color_rettype(a,b), sumtype(eltype(a),eltype(b)))(gray(a)-gray(b),alpha(a)-alpha(b))
(*)(a::AbstractGray{S}, b::AbstractGray{T}) where {S,T} = parametric(color_rettype(a,b), multype(S,T))(gray(a)*gray(b))
(^)(a::AbstractGray{S}, b::Integer) where {S} = arith_colorant_type(a){powtype(S,Int)}(gray(a)^convert(Int,b))
(^)(a::AbstractGray{S}, b::Real) where {S} = arith_colorant_type(a){powtype(S,typeof(b))}(gray(a)^b)
(+)(c::AbstractGray) = c
(+)(c::TransparentGray) = c
(-)(c::AbstractGray) = typeof(c)(-gray(c))
(-)(c::TransparentGray) = typeof(c)(-gray(c),-alpha(c))
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

# function Base.isapprox(x::AbstractArray{Cx},
#                        y::AbstractArray{Cy};
#                        rtol::Real=Base.rtoldefault(eltype(Cx),eltype(Cy),0),
#                        atol::Real=0,
#                        norm::Function=norm) where {Cx<:MathTypes,Cy<:MathTypes}
#     d = norm(x - y)
#     if isfinite(d)
#         return d <= atol + rtol*max(norm(x), norm(y))
#     else
#         # Fall back to a component-wise approximate comparison
#         return all(ab -> isapprox(ab[1], ab[2]; rtol=rtol, atol=atol), zip(x, y))
#     end
# end

dotc(x::T, y::T) where {T<:AbstractGray} = acc(gray(x))*acc(gray(y))
dotc(x::AbstractGray, y::AbstractGray) = dotc(promote(x, y)...)

float(::Type{T}) where {T<:Gray} = typeof(float(zero(T)))

# Mixed types
(+)(a::MathTypes, b::MathTypes) = (+)(promote(a, b)...)
(-)(a::MathTypes, b::MathTypes) = (-)(promote(a, b)...)

real(::Type{C}) where {C<:AbstractGray} = real(eltype(C))

# To help type inference
promote_rule(::Type{T}, ::Type{C}) where {T<:Real,C<:AbstractGray} = promote_type(T, eltype(C))

typemin(::Type{T}) where {T<:ColorTypes.AbstractGray} = T(typemin(eltype(T)))
typemax(::Type{T}) where {T<:ColorTypes.AbstractGray} = T(typemax(eltype(T)))

typemin(::T) where {T<:ColorTypes.AbstractGray} = T(typemin(eltype(T)))
typemax(::T) where {T<:ColorTypes.AbstractGray} = T(typemax(eltype(T)))

## RGB tensor products

"""
    RGBRGB(rr, gr, br, rg, gg, bg, rb, gb, bb)

Represent the [tensor product](https://en.wikipedia.org/wiki/Tensor_product) of two RGB values.

# Example

```jldoctest
julia> a, b = RGB(0.2f0, 0.3f0, 0.5f0), RGB(0.77f0, 0.11f0, 0.22f0)
(RGB{Float32}(0.2f0,0.3f0,0.5f0), RGB{Float32}(0.77f0,0.11f0,0.22f0))

julia> a ⊗ b
RGBRGB{Float32}(
 0.154f0  0.022f0  0.044f0
 0.231f0  0.033f0  0.066f0
 0.385f0  0.055f0  0.11f0 )
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
Base.eltype(::Type{RGBRGB{T}}) where T = T
Base.Matrix{T}(p::RGBRGB) where T = T[p.rr p.rg p.rb;
                                      p.gr p.gg p.gb;
                                      p.br p.bg p.bb]
Base.Matrix(p::RGBRGB{T}) where T = Matrix{T}(p)

function Base.show(io::IO, p::RGBRGB)
    print(io, "RGBRGB{", eltype(p), "}(\n")
    Base.print_matrix(io, Matrix(p))
    print(io, ')')
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
RGBRGB{Float64}(
  0.045  0.0  -0.03
  0.0    0.0   0.0
 -0.03   0.0   0.020000000000000004)
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

## Precompilation

if Base.VERSION >= v"1.4.2"
    include("precompile.jl")
    _precompile_()
end

end
