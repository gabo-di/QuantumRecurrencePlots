########################################################################
# Quantum billiard in an arbitrary smooth outer boundary
########################################################################

using DrWatson
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using Arpack
using CairoMakie, GridapMakie
using SpecialFunctions

function main()
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
                bc_type = :Dirichlet  # boundary type :Dirichlet  or :Neumann
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
    @unpack M, S = MSP_matrix_2D(V, model, p);

    # eigenvalue
    # using JacobiDavidson
    # pschur, resudials = jdqz(S, M, pairs=5, verbosity=1, target=Near(0.0 + 0.0*im), max_iter=1000, tolerance=1e-8); # smallest
    # pschur, resudials = jdqz(S, M, pairs=5, verbosity=1, max_iter=1000, tolerance=1e-8); # largest

    # using Arpack
    λ, ψ_coeffs = eigs(S, M; nev=5, which=:LR,check=1,maxiter=1000,tol=1e-5, sigma=100); # smallest
    # λ, ψ_coeffs = eigs(S, M; nev=5, which=:LR,check=1,maxiter=1000,tol=1e-5); # biggest

    # using KrylovKit
    # a = geneigsolve((S,M), 5, :SR; verbosity=3, tol=1e-8, maxiter=300) # smallest
    # a = geneigsolve((S,M), 5; verbosity=3, tol=1e-8) # largest

    # plot an eigenvector 
    i = 2;
    fig = Figure(size = (600,600));
    ax = Axis(fig[1,1]; title = "Eigen‑energy = $(λ[i])")
    a = plot!(ax, FEFunction(V, real(ψ_coeffs[:,i])))
    Colorbar(fig[1,2], a)
    display(fig)

    # plot another function
    l1 = QuantumRecurrencePlots.approx_besselroots(1,1)[1]
    ψ₀(p) = besselj1(sqrt(p[1]^2 + p[2]^2)*l1)*p[1]/sqrt(p[1]^2 + p[2]^2)
    ψ1_fe = interpolate(p->real(ψ₀(p)), V)
    fig, _, a = plot(abs(ψ1_fe))
    Colorbar(fig[1,2], a)
    display(fig)
    ψ1 = get_free_dof_values(ψ1_fe)
    ψ1 = ψ1 / norm(ψ1)
    ψ1_ = real(ψ_coeffs[:,2])
    ψ1_ = ψ1_ / norm(ψ1_)
    # @test isapprox(abs(dot(ψ1_, ψ1)), 1; atol=1e-8)
end
