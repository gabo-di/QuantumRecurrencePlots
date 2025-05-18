########################################################################
# Quantum billiard in an arbitrary smooth outer boundary
# -------------------------------------------------------
# Requirements:
#   ] add Gmsh Gridap LinearAlgebra SparseArrays
# Make sure the gmsh binary is in your PATH (Gmsh.jl will download it
# automatically on first use if not found).
########################################################################

using Revise
using DrWatson
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using UnPack
using Gmsh
using Gridap, GridapGmsh
using LinearAlgebra, SparseArrays
using Arpack, JacobiDavidson, KrylovKit

# includet(srcdir("QuantumRecurrencePlots.jl"))

# parmeters for gmsh
p_gmsh =  ( nθ = 400, # sampling resolution of boundary curve should be high enough for smooth CAD curve
            l_min = 0.01, # min characteristic length
            l_max = 0.02); # max characteristic length

# parameters for billiard boundary
# p_bill    = (R = 1.0, ε = 0.0);   # Robnik with ε=0.2  (ε=0 → circle)
p_bill    = (a = 1.0, b = sqrt(0.51));   # Elliptic (a=1, b=1 → circle)
p_bill    = (a = 1.0, b = 1.0);   # Elliptic (a=1, b=1 → circle)

# parameters for gridap
p_gridap = (order = 3, # cubic Lagrange elements
            bc_type = :Neumann  # boundary type :Dirichlet  or :Neumann
            );

# gather parameters
p = (p_gmsh = p_gmsh,
     p_bill = p_bill,
     p_gridap = p_gridap);

# create the billiard mesh with gmsh
rho = elliptic_billiard;
name = "elliptic_billiard";
gmsh_file = make_gmsh_billiard(rho, name, p);

# read file and create FE space on gridap
model, V = initialize_gridap(gmsh_file, p);

# get the mass and stiffnes matrix
M, S, _ = MSV_matrix_2D(V, model, p);

# eigenvalue
# using JacobiDavidson
pschur, resudials = jdqz(S, M, pairs=5, verbosity=1, target=Near(0.0 + 0.0*im), max_iter=1000, tolerance=1e-8); # smallest
pschur, resudials = jdqz(S, M, pairs=5, verbosity=1, max_iter=1000, tolerance=1e-8); # largest

# using Arpack
λ, ψ_coeffs = eigs(S, M; nev=5, which=:LR,check=1,maxiter=1000,tol=1e-5, sigma=1.0e-6); # smallest
λ, ψ_coeffs = eigs(S, M; nev=5, which=:LR,check=1,maxiter=1000,tol=1e-5); # biggest

# using KrylovKit
a = geneigsolve((S,M), 5, :SR; verbosity=3, tol=1e-8, maxiter=300) # smallest
a = geneigsolve((S,M), 5; verbosity=3, tol=1e-8) # largest

# ---------------------------------------------------------------------
# 5.  Crank–Nicolson time‑stepper (unitary in M‑norm)
# ---------------------------------------------------------------------
Δt  = 1e-3
A = M + 0.5im*Δt*S
B = M - 0.5im*Δt*S
solver = lu(A)                      # factor once, reuse every step

function step!(ψ)
    ψ .= solver \ (B*ψ)
    return ψ
end

# ---------------------------------------------------------------------
# 6.  Initial wave‑packet (normalise in M‑norm)
# ---------------------------------------------------------------------
# Example: Gaussian centered at (xc,yc) with momentum (kx,ky)
xc, yc = 0.0, -0.2
σ      = 0.05
kx, ky = 0.0, 20.0

ϕ(x, y) = exp(-( (x-xc)^2 + (y-yc)^2 ) / (2σ^2)) * exp(1im*(kx*x + ky*y))
ψ0_vec  = nodal_projection(ϕ, V)             # Gridap utility
normM   = real(ψ0_vec' * (M * ψ0_vec))
ψ       = ψ0_vec / sqrt(normM)               # M‑normalised coefficients

# ---------------------------------------------------------------------
# 7.  Time loop (save snapshots every Nplot steps)
# ---------------------------------------------------------------------
Nsteps, Nplot = 10000, 200
for n in 1:Nsteps
    step!(ψ)
    if n % Nplot == 0
        println("t = $(n*Δt)  |ψ|ₘ² = ", real(ψ'*(M*ψ)))
        # --- optional: write to VTK for Paraview ---------------------
        # writevtk(trian, "out/ψ_$n"; cellfields=["Re"=>real(ψ), "Im"=>imag(ψ)])
    end
end

