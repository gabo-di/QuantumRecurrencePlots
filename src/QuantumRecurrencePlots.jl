module QuantumRecurrencePlots

# ---- imports ----
using Revise
using LinearAlgebra
using UnPack
using FFTW
using Gridap
using FastGaussQuadrature


# ---- includes ----
includet("utils.jl")
includet("PolynomialUtils.jl")
includet("SchrodingerMatrix.jl")

# ---- exports ----
export make_ε_x,
       make_ε_t,
       Laguerre,
       Legendre,
       Hermite,
       Chebyshev


end
