module QuantumRecurrencePlots

# ---- imports ----
using DrWatson
using Revise
using LinearAlgebra, SparseArrays
using UnPack
using FFTW
using Gmsh
using Gridap, GridapGmsh
using FastGaussQuadrature
using SpecialFunctions
using Roots


# ---- includes ----
include("utils.jl")
include("PolynomialUtils.jl")
include("SchrodingerMatrix.jl")

# ---- exports ----
export make_ε_x,
       make_ε_t,
       Laguerre,
       Legendre,
       Hermite,
       Hermite_hat,
       Chebyshev,
       MSP_matrix_1D,
       MSP_matrix_2D,
       elliptic_billiard,
       robnik_billiard,
       make_gmsh_billiard,
       initialize_gridap

end
