__precompile__(true)

module ColorVectorSpace

using ColorTypes, FixedPointNumbers

import Base: ==, +, -, *, /, .+, .-, .*, ./, ^, .^, <, ~
import Base: abs, abs2, clamp, convert, copy, div, eps, isfinite, isinf,
    isnan, isless, length, mapreduce, norm, one, promote_array_type,
    promote_op, promote_rule, zero, trunc, floor, round, ceil, bswap,
    mod, rem, atan2, hypot, max, min, varm, real, histrange, rand

# The unaryOps
import Base:      conj, sin, cos, tan, sinh, cosh, tanh,
                  asin, acos, atan, asinh, acosh, atanh,
                  sec, csc, cot, asec, acsc, acot,
                  sech, csch, coth, asech, acsch, acoth,
                  sinc, cosc, cosd, cotd, cscd, secd,
                  sind, tand, acosd, acotd, acscd, asecd,
                  asind, atand, rad2deg, deg2rad,
                  log, log2, log10, log1p, exponent, exp,
                  exp2, expm1, cbrt, sqrt, erf,
                  erfc, erfcx, erfi, dawson,
                  significand, lgamma,
                  gamma, lfact, frexp, modf, airy, airyai,
                  airyprime, airyaiprime, airybi, airybiprime,
                  besselj0, besselj1, bessely0, bessely1,
                  eta, zeta, digamma

typealias AbstractGray{T} Color{T,1}
typealias TransparentRGB{C<:AbstractRGB,T}   TransparentColor{C,T,4}
typealias TransparentGray{C<:AbstractGray,T} TransparentColor{C,T,2}
typealias TransparentRGBFloat{C<:AbstractRGB,T<:AbstractFloat} TransparentColor{C,T,4}
typealias TransparentGrayFloat{C<:AbstractGray,T<:AbstractFloat} TransparentColor{C,T,2}
typealias TransparentRGBUFixed{C<:AbstractRGB,T<:UFixed} TransparentColor{C,T,4}
typealias TransparentGrayUFixed{C<:AbstractGray,T<:UFixed} TransparentColor{C,T,2}

typealias MathTypes{T,C} Union{AbstractRGB{T},TransparentRGB{C,T},AbstractGray{T},TransparentGray{C,T}}
typealias MathTypesFloat{C,T<:Union{Float16,Float32,Float64}} MathTypes{T,C}

## Generic algorithms
mapreduce(f, op::Base.ShortCircuiting, a::MathTypes) = f(a)  # ambiguity
mapreduce(f, op, a::MathTypes) = f(a)

for f in (:trunc, :floor, :round, :ceil, :eps, :bswap)
    @eval $f{T}(g::Gray{T}) = Gray{T}($f(gray(g)))
    @eval @vectorize_1arg Gray $f
end
eps{T}(::Type{Gray{T}}) = Gray(eps(T))
@vectorize_1arg AbstractGray isfinite
@vectorize_1arg AbstractGray isinf
@vectorize_1arg AbstractGray isnan
@vectorize_1arg AbstractGray abs
@vectorize_1arg AbstractGray abs2
for f in (:trunc, :floor, :round, :ceil)
    @eval $f{T<:Integer}(::Type{T}, g::Gray) = Gray{T}($f(T, gray(g)))
    @eval $f{T<:Integer,G<:Gray,Ti}(::Type{T}, A::SparseMatrixCSC{G,Ti}) = error("not defined") # fix ambiguity warning
end

for f in (:mod, :rem, :mod1)
    @eval $f(x::Gray, m::Gray) = Gray($f(gray(x), gray(m)))
end

# Real values are treated like grays
ColorTypes.gray(x::Real) = x

# Return types for arithmetic operations
multype(a::Type,b::Type) = typeof(one(a)*one(b))
sumtype(a::Type,b::Type) = typeof(one(a)+one(b))
divtype(a::Type,b::Type) = typeof(one(a)/one(b))
powtype(a::Type,b::Type) = typeof(one(a)^one(b))
multype(a::Colorant, b::Colorant) = multype(eltype(a),eltype(b))
sumtype(a::Colorant, b::Colorant) = sumtype(eltype(a),eltype(b))
divtype(a::Colorant, b::Colorant) = divtype(eltype(a),eltype(b))
powtype(a::Colorant, b::Colorant) = powtype(eltype(a),eltype(b))

# Scalar binary RGB operations require the same RGB type for each element,
# otherwise we don't know which to return
color_rettype{A<:AbstractRGB,B<:AbstractRGB}(::Type{A}, ::Type{B}) = _color_rettype(base_colorant_type(A), base_colorant_type(B))
color_rettype{A<:AbstractGray,B<:AbstractGray}(::Type{A}, ::Type{B}) = _color_rettype(base_colorant_type(A), base_colorant_type(B))
color_rettype{A<:TransparentRGB,B<:TransparentRGB}(::Type{A}, ::Type{B}) = _color_rettype(base_colorant_type(A), base_colorant_type(B))
color_rettype{A<:TransparentGray,B<:TransparentGray}(::Type{A}, ::Type{B}) = _color_rettype(base_colorant_type(A), base_colorant_type(B))
_color_rettype{A<:Colorant,B<:Colorant}(::Type{A}, ::Type{B}) = error("binary operation with $A and $B, return type is ambiguous")
_color_rettype{C<:Colorant}(::Type{C}, ::Type{C}) = C

color_rettype(c1::Colorant, c2::Colorant) = color_rettype(typeof(c1), typeof(c2))

## Math on Colors. These implementations encourage inlining and,
## for the case of UFixed types, nearly halve the number of multiplications (for RGB)

# Scalar RGB
copy(c::AbstractRGB) = c
(*)(f::Real, c::AbstractRGB) = base_colorant_type(c){multype(typeof(f),eltype(c))}(f*red(c), f*green(c), f*blue(c))
(*)(f::Real, c::TransparentRGB) = base_colorant_type(c){multype(typeof(f),eltype(c))}(f*red(c), f*green(c), f*blue(c), f*alpha(c))
function (*){T<:UFixed}(f::Real, c::AbstractRGB{T})
    fs = f*(1/reinterpret(one(T)))
    base_colorant_type(c){multype(typeof(f),T)}(fs*reinterpret(red(c)), fs*reinterpret(green(c)), fs*reinterpret(blue(c)))
end
function (*){T<:UFixed}(f::UFixed, c::AbstractRGB{T})
    fs = reinterpret(f)*(1/widen(reinterpret(one(T)))^2)
    base_colorant_type(c){multype(typeof(f),T)}(fs*reinterpret(red(c)), fs*reinterpret(green(c)), fs*reinterpret(blue(c)))
end
function (/){T<:UFixed}(c::AbstractRGB{T}, f::Real)
    fs = (one(f)/reinterpret(one(T)))/f
    base_colorant_type(c){divtype(typeof(f),T)}(fs*reinterpret(red(c)), fs*reinterpret(green(c)), fs*reinterpret(blue(c)))
end
function (/){T<:UFixed}(c::AbstractRGB{T}, f::Integer)
    fs = (1/reinterpret(one(T)))/f
    base_colorant_type(c){divtype(typeof(f),T)}(fs*reinterpret(red(c)), fs*reinterpret(green(c)), fs*reinterpret(blue(c)))
end
(+){S,T}(a::AbstractRGB{S}, b::AbstractRGB{T}) = color_rettype(a, b){sumtype(S,T)}(red(a)+red(b), green(a)+green(b), blue(a)+blue(b))
(-){S,T}(a::AbstractRGB{S}, b::AbstractRGB{T}) = color_rettype(a, b){sumtype(S,T)}(red(a)-red(b), green(a)-green(b), blue(a)-blue(b))
(+)(a::TransparentRGB, b::TransparentRGB) =
    color_rettype(a, b){sumtype(a,b)}(red(a)+red(b), green(a)+green(b), blue(a)+blue(b), alpha(a)+alpha(b))
(-)(a::TransparentRGB, b::TransparentRGB) =
    color_rettype(a, b){sumtype(a,b)}(red(a)-red(b), green(a)-green(b), blue(a)-blue(b), alpha(a)-alpha(b))
(*)(c::AbstractRGB, f::Real) = (*)(f, c)
(*)(c::TransparentRGB, f::Real) = (*)(f, c)
(.*)(f::Real, c::AbstractRGB) = (*)(f, c)
(.*)(f::Real, c::TransparentRGB) = (*)(f, c)
(.*)(c::AbstractRGB, f::Real) = (*)(f, c)
(.*)(c::TransparentRGB, f::Real) = (*)(f, c)
(/)(c::AbstractRGB, f::Real) = (one(f)/f)*c
(/)(c::TransparentRGB, f::Real) = (one(f)/f)*c
(/)(c::AbstractRGB, f::Integer) = (one(eltype(c))/f)*c
(/)(c::TransparentRGB, f::Integer) = (one(eltype(c))/f)*c
(./)(c::AbstractRGB, f::Real) = (/)(c, f)
(./)(c::TransparentRGB, f::Real) = (/)(c, f)

isfinite{T<:UFixed}(c::Colorant{T}) = true
isfinite{T<:AbstractFloat}(c::AbstractRGB{T}) = isfinite(red(c)) && isfinite(green(c)) && isfinite(blue(c))
isfinite(c::TransparentRGBFloat) = isfinite(red(c)) && isfinite(green(c)) && isfinite(blue(c)) && isfinite(alpha(c))
isnan{T<:UFixed}(c::Colorant{T}) = false
isnan{T<:AbstractFloat}(c::AbstractRGB{T}) = isnan(red(c)) || isnan(green(c)) || isnan(blue(c))
isnan(c::TransparentRGBFloat) = isnan(red(c)) || isnan(green(c)) || isnan(blue(c)) || isnan(alpha(c))
isinf{T<:UFixed}(c::Colorant{T}) = false
isinf{T<:AbstractFloat}(c::AbstractRGB{T}) = isinf(red(c)) || isinf(green(c)) || isinf(blue(c))
isinf(c::TransparentRGBFloat) = isinf(red(c)) || isinf(green(c)) || isinf(blue(c)) || isinf(alpha(c))
abs(c::AbstractRGB) = abs(red(c))+abs(green(c))+abs(blue(c)) # should this have a different name?
abs{T<:UFixed}(c::AbstractRGB{T}) = Float32(red(c))+Float32(green(c))+Float32(blue(c)) # should this have a different name?
abs(c::TransparentRGB) = abs(red(c))+abs(green(c))+abs(blue(c))+abs(alpha(c)) # should this have a different name?
abs{T<:UFixed}(c::TransparentRGB{T}) = Float32(red(c))+Float32(green(c))+Float32(blue(c))+Float32(alpha(c)) # should this have a different name?
abs2(c::AbstractRGB) = red(c)^2+green(c)^2+blue(c)^2
abs2{T<:UFixed}(c::AbstractRGB{T}) = Float32(red(c))^2+Float32(green(c))^2+Float32(blue(c))^2

one{C<:AbstractRGB}(::Type{C})     = C(1,1,1)
one{C<:TransparentRGB}(::Type{C})  = C(1,1,1,1)
zero{C<:AbstractRGB}(::Type{C})    = C(0,0,0)
zero{C<:TransparentRGB}(::Type{C}) = C(0,0,0,0)
zero{C<:YCbCr}(::Type{C}) = C(0,0,0)
zero{C<:HSV}(::Type{C}) = C(0,0,0)
one(p::Colorant) = one(typeof(p))
zero(p::Colorant) = zero(typeof(p))
typemin{C<:AbstractRGB}(::Type{C}) = zero(C)
typemax{C<:AbstractRGB}(::Type{C}) = one(C)
typemin{C<:AbstractGray}(::Type{C}) = zero(C)
typemax{C<:AbstractGray}(::Type{C}) = one(C)
typemin{C<:TransparentGray}(::Type{C}) = zero(C)
typemax{C<:TransparentGray}(::Type{C}) = one(C)
typemin{C<:TransparentRGB}(::Type{C}) = zero(C)
typemax{C<:TransparentRGB}(::Type{C}) = one(C)

# Arrays
(+){CV<:AbstractRGB}(A::AbstractArray{CV}, b::AbstractRGB) = (.+)(A, b)
(+){CV<:AbstractRGB}(b::AbstractRGB, A::AbstractArray{CV}) = (.+)(b, A)
(-){CV<:AbstractRGB}(A::AbstractArray{CV}, b::AbstractRGB) = (.-)(A, b)
(-){CV<:AbstractRGB}(b::AbstractRGB, A::AbstractArray{CV}) = (.-)(b, A)
(*){T<:Number}(A::AbstractArray{T}, b::AbstractRGB) = A.*b
(*){T<:Number}(b::AbstractRGB, A::AbstractArray{T}) = A.*b
(.+){C<:AbstractRGB}(A::AbstractArray{C}, b::AbstractRGB) = plus(A, b)
(.+){C<:AbstractRGB}(b::AbstractRGB, A::AbstractArray{C}) = plus(b, A)
(.-){C<:AbstractRGB}(A::AbstractArray{C}, b::AbstractRGB) = minus(A, b)
(.-){C<:AbstractRGB}(b::AbstractRGB, A::AbstractArray{C}) = minus(b, A)
(.*){T<:Number}(A::AbstractArray{T}, b::AbstractRGB) = mul(A, b)
(.*){T<:Number}(b::AbstractRGB, A::AbstractArray{T}) = mul(b, A)

(+){CV<:TransparentRGB}(A::AbstractArray{CV}, b::TransparentRGB) = (.+)(A, b)
(+){CV<:TransparentRGB}(b::TransparentRGB, A::AbstractArray{CV}) = (.+)(b, A)
(-){CV<:TransparentRGB}(A::AbstractArray{CV}, b::TransparentRGB) = (.-)(A, b)
(-){CV<:TransparentRGB}(b::TransparentRGB, A::AbstractArray{CV}) = (.-)(b, A)
(*){T<:Number}(A::AbstractArray{T}, b::TransparentRGB) = A.*b
(*){T<:Number}(b::TransparentRGB, A::AbstractArray{T}) = A.*b
(.+){C<:TransparentRGB}(A::AbstractArray{C}, b::TransparentRGB) = plus(A, b)
(.+){C<:TransparentRGB}(b::TransparentRGB, A::AbstractArray{C}) = plus(b, A)
(.-){C<:TransparentRGB}(A::AbstractArray{C}, b::TransparentRGB) = minus(A, b)
(.-){C<:TransparentRGB}(b::TransparentRGB, A::AbstractArray{C}) = minus(b, A)
(.*){T<:Number}(A::AbstractArray{T}, b::TransparentRGB) = mul(A, b)
(.*){T<:Number}(b::TransparentRGB, A::AbstractArray{T}) = mul(b, A)

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
                  :exp2, :expm1, :cbrt, :sqrt, :erf,
                  :erfc, :erfcx, :erfi, :dawson,
                  :significand, :lgamma,
                  :gamma, :lfact, :frexp, :modf, :airy, :airyai,
                  :airyprime, :airyaiprime, :airybi, :airybiprime,
                  :besselj0, :besselj1, :bessely0, :bessely1,
                  :eta, :zeta, :digamma)
for op in unaryOps
    @eval ($op)(c::AbstractGray) = $op(gray(c))
end

(*)(f::Real, c::AbstractGray) = base_colorant_type(c){multype(typeof(f),eltype(c))}(f*gray(c))
(*)(f::Real, c::TransparentGray) = base_colorant_type(c){multype(typeof(f),eltype(c))}(f*gray(c), f*alpha(c))
(*)(c::AbstractGray, f::Real) = (*)(f, c)
(.*)(f::Real, c::AbstractGray) = (*)(f, c)
(.*)(c::AbstractGray, f::Real) = (*)(f, c)
(*)(c::TransparentGray, f::Real) = (*)(f, c)
(.*)(f::Real, c::TransparentGray) = (*)(f, c)
(.*)(c::TransparentGray, f::Real) = (*)(f, c)
(/)(c::AbstractGray, f::Real) = (one(f)/f)*c
(/)(n::Number, c::AbstractGray) = n/gray(c)
(/)(c::TransparentGray, f::Real) = (one(f)/f)*c
(/)(c::AbstractGray, f::Integer) = (one(eltype(c))/f)*c
(/)(c::TransparentGray, f::Integer) = (one(eltype(c))/f)*c
(./)(c::AbstractGray, f::Real) = c/f
(./)(n::Number, c::AbstractGray) = n/gray(c)
(./)(c::TransparentGray, f::Real) = c/f
(+){S,T}(a::AbstractGray{S}, b::AbstractGray{T}) = color_rettype(a,b){sumtype(S,T)}(gray(a)+gray(b))
(+)(a::TransparentGray, b::TransparentGray) = color_rettype(a,b){sumtype(eltype(a),eltype(b))}(gray(a)+gray(b),alpha(a)+alpha(b))
(-){S,T}(a::AbstractGray{S}, b::AbstractGray{T}) = color_rettype(a,b){sumtype(S,T)}(gray(a)-gray(b))
(-)(a::TransparentGray, b::TransparentGray) = color_rettype(a,b){sumtype(eltype(a),eltype(b))}(gray(a)-gray(b),alpha(a)-alpha(b))
(*){S,T}(a::AbstractGray{S}, b::AbstractGray{T}) = color_rettype(a,b){multype(S,T)}(gray(a)*gray(b))
(^){S}(a::AbstractGray{S}, b::Integer) = base_colorant_type(a){powtype(S,Int)}(gray(a)^convert(Int,b))
(^){S}(a::AbstractGray{S}, b::Real) = base_colorant_type(a){powtype(S,typeof(b))}(gray(a)^b)
(.^){S}(a::AbstractGray{S}, b) = a^b
(+)(c::AbstractGray) = c
(+)(c::TransparentGray) = c
(-)(c::AbstractGray) = typeof(c)(-gray(c))
(-)(c::TransparentGray) = typeof(c)(-gray(c),-alpha(c))
(/)(a::AbstractGray, b::AbstractGray) = gray(a)/gray(b)
div(a::AbstractGray, b::AbstractGray) = div(gray(a), gray(b))
(+)(a::AbstractGray, b::Number) = gray(a)+b
(-)(a::AbstractGray, b::Number) = gray(a)-b
(+)(a::Number, b::AbstractGray) = a+gray(b)
(-)(a::Number, b::AbstractGray) = a-gray(b)
(.+)(a::AbstractGray, b::Number) = gray(a)+b
(.-)(a::AbstractGray, b::Number) = gray(a)-b
(.+)(a::Number, b::AbstractGray) = a+gray(b)
(.-)(a::Number, b::AbstractGray) = a-gray(b)
max{T<:AbstractGray}(a::T, b::T) = T(max(gray(a),gray(b)))
max(a::AbstractGray, b::AbstractGray) = max(promote(a,b)...)
max(a::Number, b::AbstractGray) = max(promote(a,b)...)
max(a::AbstractGray, b::Number) = max(promote(a,b)...)
min{T<:AbstractGray}(a::T, b::T) = T(min(gray(a),gray(b)))
min(a::AbstractGray, b::AbstractGray) = min(promote(a,b)...)
min(a::Number, b::AbstractGray) = min(promote(a,b)...)
min(a::AbstractGray, b::Number) = min(promote(a,b)...)

isfinite{T<:AbstractFloat}(c::AbstractGray{T}) = isfinite(gray(c))
isfinite(c::TransparentGrayFloat) = isfinite(gray(c)) && isfinite(alpha(c))
isnan{T<:AbstractFloat}(c::AbstractGray{T}) = isnan(gray(c))
isnan(c::TransparentGrayFloat) = isnan(gray(c)) && isnan(alpha(c))
isinf{T<:AbstractFloat}(c::AbstractGray{T}) = isinf(gray(c))
isinf(c::TransparentGrayFloat) = isinf(gray(c)) && isnan(alpha(c))
norm(c::AbstractGray) = abs(gray(c))
abs(c::TransparentGray) = abs(gray(c))+abs(alpha(c)) # should this have a different name?
abs(c::TransparentGrayUFixed) = Float32(gray(c)) + Float32(alpha(c)) # should this have a different name?
abs2(c::AbstractGray) = gray(c)^2
abs2{T<:UFixed}(c::AbstractGray{T}) = Float32(gray(c))^2
abs2(c::TransparentGray) = gray(c)^2+alpha(c)^2
abs2(c::TransparentGrayUFixed) = Float32(gray(c))^2 + Float32(alpha(c))^2
atan2(x::Gray, y::Gray) = atan2(convert(Real, x), convert(Real, y))
hypot(x::Gray, y::Gray) = hypot(convert(Real, x), convert(Real, y))

(<)(g1::AbstractGray, g2::AbstractGray) = gray(g1) < gray(g2)
(<)(c::AbstractGray, r::Real) = gray(c) < r
(<)(r::Real, c::AbstractGray) = r < gray(c)
isless(g1::AbstractGray, g2::AbstractGray) = isless(gray(g1), gray(g2))
isless(c::AbstractGray, r::Real) = isless(gray(c), r)
isless(r::Real, c::AbstractGray) = isless(r, gray(c))
Base.isapprox(x::AbstractGray, y::AbstractGray; kwargs...) = isapprox(gray(x), gray(y); kwargs...)

zero{C<:TransparentGray}(::Type{C}) = C(0,0)
zero{C<:Gray}(::Type{C}) = C(0)
one{C<:TransparentGray}(::Type{C}) = C(1,1)
one{C<:Gray}(::Type{C}) = C(1)

# rand
rand{G<:AbstractGray}(::Type{G}) = G(rand())
rand{G<:TransparentGray}(::Type{G}) = G(rand(), rand())
rand{C<:AbstractRGB}(::Type{C}) = C(rand(), rand(), rand())
rand{C<:TransparentRGB}(::Type{C}) = C(rand(), rand(), rand(), rand())
function rand{C<:MathTypesFloat}(::Type{C}, sz::Dims)
    reinterpret(C, rand(eltype(C), (sizeof(C)÷sizeof(eltype(C)), sz...)), sz)
end
function rand{C<:MathTypes{U8}}(::Type{C}, sz::Dims)
    reinterpret(C, rand(UInt8, (sizeof(C), sz...)), sz)
end
function rand{C<:MathTypes{U16}}(::Type{C}, sz::Dims)
    reinterpret(C, rand(UInt16, (sizeof(C)>>1, sz...)), sz)
end

 # Arrays
(+){CV<:AbstractGray}(A::AbstractArray{CV}, b::AbstractGray) = (.+)(A, b)
(+){CV<:AbstractGray}(b::AbstractGray, A::AbstractArray{CV}) = (.+)(b, A)
(-){CV<:AbstractGray}(A::AbstractArray{CV}, b::AbstractGray) = (.-)(A, b)
(-){CV<:AbstractGray}(b::AbstractGray, A::AbstractArray{CV}) = (.-)(b, A)
(*){T<:Number}(A::AbstractArray{T}, b::AbstractGray) = A.*b
(*){T<:Number}(b::AbstractGray, A::AbstractArray{T}) = A.*b
(/){C<:AbstractGray}(A::AbstractArray{C}, b::AbstractGray) = A./b
(.+){C<:AbstractGray}(A::AbstractArray{C}, b::AbstractGray) = plus(A, b)
(.+){C<:AbstractGray}(b::AbstractGray, A::AbstractArray{C}) = plus(b, A)
(.-){C<:AbstractGray}(A::AbstractArray{C}, b::AbstractGray) = minus(A, b)
(.-){C<:AbstractGray}(b::AbstractGray, A::AbstractArray{C}) = minus(b, A)
(.*){T<:Number}(A::AbstractArray{T}, b::AbstractGray) = mul(A, b)
(.*){T<:Number}(b::AbstractGray, A::AbstractArray{T}) = mul(b, A)
(./){C<:AbstractGray}(A::AbstractArray{C}, b::AbstractGray) = divd(A, b)

Base.@vectorize_2arg Gray max
Base.@vectorize_2arg Gray min
for f in (:min, :max)
    @eval begin
        ($f){T<:Gray}(x::Number, y::AbstractArray{T}) =
            reshape([ ($f)(x, y[i]) for i in eachindex(y) ], size(y))
        ($f){T<:Gray}(x::AbstractArray{T}, y::Number) =
            reshape([ ($f)(x[i], y) for i in eachindex(x) ], size(x))
    end
end

(+){CV<:TransparentGray}(A::AbstractArray{CV}, b::TransparentGray) = (.+)(A, b)
(+){CV<:TransparentGray}(b::TransparentGray, A::AbstractArray{CV}) = (.+)(b, A)
(-){CV<:TransparentGray}(A::AbstractArray{CV}, b::TransparentGray) = (.-)(A, b)
(-){CV<:TransparentGray}(b::TransparentGray, A::AbstractArray{CV}) = (.-)(b, A)
(*){T<:Number}(A::AbstractArray{T}, b::TransparentGray) = A.*b
(*){T<:Number}(b::TransparentGray, A::AbstractArray{T}) = A.*b
(.+){C<:TransparentGray}(A::AbstractArray{C}, b::TransparentGray) = plus(A, b)
(.+){C<:TransparentGray}(b::TransparentGray, A::AbstractArray{C}) = plus(b, A)
(.-){C<:TransparentGray}(A::AbstractArray{C}, b::TransparentGray) = minus(A, b)
(.-){C<:TransparentGray}(b::TransparentGray, A::AbstractArray{C}) = minus(b, A)
(.*){T<:Number}(A::AbstractArray{T}, b::TransparentGray) = mul(A, b)
(.*){T<:Number}(b::TransparentGray, A::AbstractArray{T}) = mul(b, A)

varm{C<:AbstractGray}(v::AbstractArray{C}, s::AbstractGray; corrected::Bool=true) =
        varm(map(gray,v),gray(s); corrected=corrected)
real{C<:AbstractGray}(::Type{C}) = real(eltype(C))

# Called plus/minus instead of plus/sub because `sub` already has a meaning!
function plus(A::AbstractArray, b::Colorant)
    bT = convert(eltype(A), b)
    out = similar(A)
    plus!(out, A, bT)
end
plus(b::Colorant, A::AbstractArray) = plus(A, b)
function minus(A::AbstractArray, b::Colorant)
    bT = convert(eltype(A), b)
    out = similar(A)
    minus!(out, A, bT)
end
function minus(b::Colorant, A::AbstractArray)
    bT = convert(eltype(A), b)
    out = similar(A)
    minus!(out, bT, A)
end
function mul{T<:Number}(A::AbstractArray{T}, b::Colorant)
    bT = typeof(b*one(T))
    out = similar(A, bT)
    mul!(out, A, b)
end
mul{T<:Number}(b::Colorant, A::AbstractArray{T}) = mul(A, b)
function divd{C<:AbstractGray}(A::AbstractArray{C}, b::AbstractGray)
    bT = typeof(zero(C)/b)
    out = similar(A, bT)
    div!(out, A, b)
end

for (func, op) in ((:plus!, :+),
                   (:minus!, :-),
                   (:mul!, :*),
                   (:div!, :/))
    @eval begin
        function $func{T,N}(out, A::AbstractArray{T,N}, b)
            Rout, RA = eachindex(out), eachindex(A)
            if Rout == RA
                for I in RA
                    @inbounds out[I] = $op(A[I], b)
                end
            else
                for (Iout, IA) in zip(Rout, RA)
                    @inbounds out[Iout] = $op(A[IA], b)
                end
            end
            out
        end
    end
end

# This needs separate implementation because we can take -b of unsigned types
function minus!{T,N}(out, b::Colorant, A::AbstractArray{T,N})
    Rout, RA = eachindex(out), eachindex(A)
    if Rout == RA
        for I in RA
            @inbounds out[I] = b - A[I]
        end
    else
        for (Iout, IA) in zip(Rout, RA)
            @inbounds out[Iout] = b - A[IA]
        end
    end
    out
end

#histrange for Gray type
Base.histrange{T}(v::AbstractArray{Gray{T}}, n::Integer) = histrange(convert(Array{Float32}, map(gray, v)), n)

# Promotions for reductions
if VERSION < v"0.5.0-dev+3701"
    Base.r_promote{T<:FixedPoint}(::Base.AddFun, c::MathTypes{T}) = convert(base_colorant_type(typeof(c)){Float64}, c)
    Base.r_promote{T<:FixedPoint}(::Base.MulFun, c::MathTypes{T}) = convert(base_colorant_type(typeof(c)){Float64}, c)
else
    Base.r_promote{T<:FixedPoint}(::typeof(+), c::MathTypes{T}) = convert(base_colorant_type(typeof(c)){Float64}, c)
    Base.r_promote{T<:FixedPoint}(::typeof(*), c::MathTypes{T}) = convert(base_colorant_type(typeof(c)){Float64}, c)
end

# To help type inference
promote_array_type{T<:Real,C<:MathTypes}(F, ::Type{T}, ::Type{C}) = base_colorant_type(C){Base.promote_array_type(F, T, eltype(C))}
promote_op{C<:MathTypes,T<:Real}(F, ::Type{C}, ::Type{T}) = typeof(F(zero(C), zero(T)))
promote_op{C<:MathTypes,T<:Real}(F, ::Type{T}, ::Type{C}) = typeof(F(zero(T), zero(C)))
promote_rule{C1<:Colorant,C2<:Colorant}(::Type{C1}, ::Type{C2}) = color_rettype(C1,C2){promote_type(eltype(C1), eltype(C2))}
promote_rule{T<:Real,C<:AbstractGray}(::Type{T}, ::Type{C}) = promote_type(T, eltype(C))

@deprecate sumsq abs2

end
