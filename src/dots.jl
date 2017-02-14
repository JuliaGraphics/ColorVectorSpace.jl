# dot operations
import Base: .+, .-, .*, ./, .^
    (.*)(f::Real, c::AbstractRGB) = (*)(f, c)
(.*)(f::Real, c::TransparentRGB) = (*)(f, c)
(.*)(c::AbstractRGB, f::Real) = (*)(f, c)
(.*)(c::TransparentRGB, f::Real) = (*)(f, c)
(./)(c::AbstractRGB, f::Real) = (/)(c, f)
(./)(c::TransparentRGB, f::Real) = (/)(c, f)
(.+){C<:AbstractRGB}(A::AbstractArray{C}, b::AbstractRGB) = plus(A, b)
(.+){C<:AbstractRGB}(b::AbstractRGB, A::AbstractArray{C}) = plus(b, A)
(.-){C<:AbstractRGB}(A::AbstractArray{C}, b::AbstractRGB) = minus(A, b)
(.-){C<:AbstractRGB}(b::AbstractRGB, A::AbstractArray{C}) = minus(b, A)
(.*){T<:Number}(A::AbstractArray{T}, b::AbstractRGB) = mul(A, b)
(.*){T<:Number}(b::AbstractRGB, A::AbstractArray{T}) = mul(b, A)
(.+){C<:TransparentRGB}(A::AbstractArray{C}, b::TransparentRGB) = plus(A, b)
(.+){C<:TransparentRGB}(b::TransparentRGB, A::AbstractArray{C}) = plus(b, A)
(.-){C<:TransparentRGB}(A::AbstractArray{C}, b::TransparentRGB) = minus(A, b)
(.-){C<:TransparentRGB}(b::TransparentRGB, A::AbstractArray{C}) = minus(b, A)
(.*){T<:Number}(A::AbstractArray{T}, b::TransparentRGB) = mul(A, b)
(.*){T<:Number}(b::TransparentRGB, A::AbstractArray{T}) = mul(b, A)
(.*)(f::Real, c::AbstractGray) = (*)(f, c)
(.*)(c::AbstractGray, f::Real) = (*)(f, c)
(.*)(f::Real, c::TransparentGray) = (*)(f, c)
(.*)(c::TransparentGray, f::Real) = (*)(f, c)
(./)(c::AbstractGray, f::Real) = c/f
(./)(n::Number, c::AbstractGray) = n/gray(c)
(./)(c::TransparentGray, f::Real) = c/f
(.^){S}(a::AbstractGray{S}, b) = a^b
(.+)(a::AbstractGray, b::Number) = gray(a)+b
(.-)(a::AbstractGray, b::Number) = gray(a)-b
(.+)(a::Number, b::AbstractGray) = a+gray(b)
(.-)(a::Number, b::AbstractGray) = a-gray(b)

(.+){C<:AbstractGray}(A::AbstractArray{C}, b::AbstractGray) = plus(A, b)
(.+){C<:AbstractGray}(b::AbstractGray, A::AbstractArray{C}) = plus(b, A)
(.-){C<:AbstractGray}(A::AbstractArray{C}, b::AbstractGray) = minus(A, b)
(.-){C<:AbstractGray}(b::AbstractGray, A::AbstractArray{C}) = minus(b, A)
(.*){T<:Number}(A::AbstractArray{T}, b::AbstractGray) = mul(A, b)
(.*){T<:Number}(b::AbstractGray, A::AbstractArray{T}) = mul(b, A)
(./){C<:AbstractGray}(A::AbstractArray{C}, b::AbstractGray) = divd(A, b)

(.+){C<:TransparentGray}(A::AbstractArray{C}, b::TransparentGray) = plus(A, b)
(.+){C<:TransparentGray}(b::TransparentGray, A::AbstractArray{C}) = plus(b, A)
(.-){C<:TransparentGray}(A::AbstractArray{C}, b::TransparentGray) = minus(A, b)
(.-){C<:TransparentGray}(b::TransparentGray, A::AbstractArray{C}) = minus(b, A)
(.*){T<:Number}(A::AbstractArray{T}, b::TransparentGray) = mul(A, b)
(.*){T<:Number}(b::TransparentGray, A::AbstractArray{T}) = mul(b, A)

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
