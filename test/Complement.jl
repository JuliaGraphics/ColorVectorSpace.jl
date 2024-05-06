using ColorVectorSpace
using Colors
using Test

@testset "Complement - RGBA" begin
    for _color in (RGBA(0.2, 0.5, 0.9, 0.4), RGBA(0.1, 0.2, 0.3, 0.4), RGBA(1.0, 0.9, 0.8, 0.7))
        comp = Complement(_color)
        comp2 = complement(_color)
        @test comp isa Complement{RGBA{Float64}, Float64, 4}
        @test color_type(comp) == Complement{RGB{Float64}, Float64, 4}
        @test base_color_type(comp) == Complement{RGB, <:Any, 4}
        @test base_colorant_type(comp) == Complement{RGBA, <:Any, 4}
        @test red(comp) ≈ red(comp2)
        @test green(comp) ≈ green(comp2)
        @test blue(comp) ≈ blue(comp2)
        @test alpha(comp) == alpha(comp2)
        @test oneunit(comp) == Complement(zero(_color))
        @test zero(comp) == Complement(one(_color))
        @test isnan(nan(typeof(comp)))
        @test nan(typeof(comp)) isa typeof(comp)
        @test complement(comp) == _color
        @test convert(typeof(_color), comp) == complement(_color)
        @test convert(Complement, comp) == comp
        @test convert(Complement, _color) == Complement(complement(_color))
        @test convert(Complement{RGBA{Float64}}, _color) == Complement(complement(_color))
        @test convert(Complement{RGBA{Float64}, Float64}, _color) == Complement(complement(_color))
        @test convert(Complement{RGBA{Float64}, Float64, 4}, _color) == Complement(complement(_color))
        if VERSION ≥ v"1.10"
            @test reinterpret(typeof(comp), _color) == comp
            @test reinterpret(Complement, _color) == comp
        end
        @test all(reinterpret(typeof(comp), [_color]) .≈ [comp])
    end
end

@testset "Complement - RGB" begin
    for _color in (RGB(0.2, 0.5, 0.9), RGB(0.1, 0.2, 0.3), RGB(1.0, 0.9, 0.8))
        comp = Complement(_color)
        comp2 = complement(_color)
        @test comp isa Complement{RGB{Float64}, Float64, 3}
        @test color_type(comp) == Complement{RGB{Float64}, Float64, 3}
        @test base_color_type(comp) == Complement{RGB, <:Any, 3}
        @test base_colorant_type(comp) == Complement{RGB, <:Any, 3}
        @test red(comp) ≈ red(comp2)
        @test green(comp) ≈ green(comp2)
        @test blue(comp) ≈ blue(comp2)
        @test alpha(comp) == alpha(comp2)
        @test oneunit(comp) == Complement(zero(_color))
        @test zero(comp) == Complement(one(_color))
        @test isnan(nan(typeof(comp)))
        @test nan(typeof(comp)) isa typeof(comp)
        @test complement(comp) == _color
        @test convert(typeof(_color), comp) == complement(_color)
        @test convert(Complement, comp) == comp
        @test convert(Complement, _color) == Complement(complement(_color))
        @test convert(Complement{RGB{Float64}}, _color) == Complement(complement(_color))
        @test convert(Complement{RGB{Float64}, Float64}, _color) == Complement(complement(_color))
        @test convert(Complement{RGB{Float64}, Float64, 3}, _color) == Complement(complement(_color))
        @test reinterpret(typeof(comp), _color) == comp
        @test reinterpret(Complement, _color) == comp
        @test all(reinterpret(typeof(comp), [_color]) .≈ [comp])
    end
end

@testset "Complement - Gray" begin
    for _color in (Gray(0.2), Gray(0.5), Gray(0.9))
        comp = Complement(_color)
        comp2 = complement(_color)
        @test comp isa Complement{Gray{Float64}, Float64, 1}
        @test color_type(comp) == Complement{Gray{Float64}, Float64, 1}
        @test base_color_type(comp) == Complement{Gray, <:Any, 1}
        @test base_colorant_type(comp) == Complement{Gray, <:Any, 1}
        @test gray(comp) ≈ gray(comp2)
        @test alpha(comp) == alpha(comp2)
        @test oneunit(comp) == Complement(zero(_color))
        @test zero(comp) == Complement(one(_color))
        @test isnan(nan(typeof(comp)))
        @test nan(typeof(comp)) isa typeof(comp)
        @test complement(comp) == _color
        @test convert(typeof(_color), comp) == complement(_color)
        @test convert(Complement, comp) == comp
        @test convert(Complement, _color) == Complement(complement(_color))
        @test convert(Complement{Gray{Float64}}, _color) == Complement(complement(_color))
        @test convert(Complement{Gray{Float64}, Float64}, _color) == Complement(complement(_color))
        @test convert(Complement{Gray{Float64}, Float64, 1}, _color) == Complement(complement(_color))

        if VERSION ≥ v"1.10"
            @test reinterpret(typeof(comp), _color) == comp
            @test reinterpret(Complement, _color) == comp
        end
        @test all(reinterpret(typeof(comp), [_color]) .≈ [comp])
    end
end

@testset "ComplementArray" begin
    arr = Complement.(Gray.(0.1:0.1:1.0))
    comp_arr = ComplementArray(arr)
    @test parent(comp_arr) == arr
    @test size(comp_arr) == size(arr)
    @test gray.(arr) == gray.(comp_arr)

    comp_arr[1] = Gray(0.15)
    @test gray.(arr) == gray.(comp_arr)
    @test IndexStyle(comp_arr) == IndexStyle(arr)

    arr = Complement.(RGB.(0.1:0.1:1.0, 0.01:0.1:1.0, 0.08:0.1:1.0))
    comp_arr = ComplementArray(arr)
    @test parent(comp_arr) == arr
    @test size(comp_arr) == size(arr)
    @test red.(arr) == red.(comp_arr)
    @test green.(arr) == green.(comp_arr)
    @test blue.(arr) == blue.(comp_arr)

    comp_arr[1] = RGB(0.73)
    @test red.(arr) == red.(comp_arr)
    @test IndexStyle(comp_arr) == IndexStyle(arr)
end
