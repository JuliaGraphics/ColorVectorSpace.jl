# ColorVectorSpace

[![Build Status](https://travis-ci.org/JuliaGraphics/ColorVectorSpace.jl.svg?branch=master)](https://travis-ci.org/JuliaGraphics/ColorVectorSpace.jl)

This package is an add-on to [ColorTypes](), and provides fast
mathematical operations for objects with types such as `RGB` and
`Gray`.

## Introduction

Colorspaces such as RGB, unlike XYZ, are technically non-linear; the
"colorimetrically correct" approach when averaging two RGBs is to
first convert each to XYZ, average them, and then convert back to RGB.

However, particularly in image processing it is common to ignore this
concern, and for the sake of performance treat an RGB as if it were a
3-vector.  This package provides such operations.

## Usage

```jl
using ColorTypes, ColorVectorSpace
```

That's it. Just by loading `ColorVectorSpace`, most basic mathematical
operations will "just work" on `AbstractRGB`, `AbstractGray`,
`TransparentRGB`, and `TransparentGray` objects.

If you discover missing operations, please open an issue, or better
yet submit a pull request.
