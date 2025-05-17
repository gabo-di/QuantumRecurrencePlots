module QuantumRecurrencePlots

# ---- imports ----
using Revise
using LinearAlgebra, SparseArrays
using UnPack
using FFTW
using Gmsh
using Gridap, GridapGmsh
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
       Hermite_hat,
       Chebyshev,
       MSV_matrix_1D


end
