using Revise
using DrWatson
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using FFTW
using CairoMakie
using LinearAlgebra

"""
    solve_harmonic_oscillator(x, k, dt, tmax, ω, ψ_initial)
    
Solves the time-dependent Schrödinger equation for a harmonic oscillator
using the split-step Fourier method.

Parameters:
- x: Spatial grid
- k: Momentum grid
- dt: Time step
- tmax: Maximum simulation time
- ω: Frequency of the harmonic oscillator
- ψ_initial: Initial wavefunction
"""
function solve_harmonic_oscillator(x, t, p, ψ_initial)
    @unpack π_k, ε_x, ε_t = p
    @unpack k_fft, P_fft, P_ifft = p

    dt = t[2] - t[1]
    dx = x[2] - x[1]
    nt = length(t)
    
    # Potential energy (harmonic oscillator)
    V = 1/2 * π_k * ε_t / ε_x ^ 2 .* x.^2
    
    # Kinetic energy in Fourier space
    T = ε_x ^ 2 /(2* ε_t) .* k_fft.^2
    
    # Define evolution operators
    exp_V = exp.(-im * V * dt/2)
    exp_T = exp.(-im * T * dt)
    
    # Initialize wavefunction
    ψ = copy(ψ_initial)
    
    # Normalize
    ψ = ψ ./ sqrt(sum(abs2.(ψ)) * dx)

    # Time evolution using split-step Fourier method
    for i in 1:(nt-1)
        # Apply half-step of potential
        ψ = ψ .* exp_V
        
        # Apply full step of kinetic energy in Fourier space
        ψ_k = P_fft * ψ
        ψ_k = ψ_k .* exp_T
        ψ = P_ifft *ψ_k
        
        # Apply second half-step of potential
        ψ = ψ .* exp_V
    end
    
    return ψ
end

# Plot the results using Makie
function plot_comparison(x, tmax, ψ_numerical, ψ_analytical)
    fig = Figure(resolution=(1200, 800))
    
    # Real part comparison
    ax1 = Axis(fig[1, 1], 
        title="Real Part Comparison", 
        xlabel="Position", 
        ylabel="Real(ψ)")
    
    lines!(ax1, x, real.(ψ_numerical), color=:blue, linewidth=2, label="Numerical")
    lines!(ax1, x, real.(ψ_analytical), color=:red, linestyle=:dash, linewidth=2, label="Analytical")
    axislegend(ax1, position=:rt)
    
    # Imaginary part comparison
    ax2 = Axis(fig[1, 2], 
        title="Imaginary Part Comparison", 
        xlabel="Position", 
        ylabel="Imag(ψ)")
    
    lines!(ax2, x, imag.(ψ_numerical), color=:blue, linewidth=2, label="Numerical")
    lines!(ax2, x, imag.(ψ_analytical), color=:red, linestyle=:dash, linewidth=2, label="Analytical")
    axislegend(ax2, position=:rt)
    
    # Probability density comparison
    ax3 = Axis(fig[2, 1:2], 
        title="Probability Density Comparison", 
        xlabel="Position", 
        ylabel="|ψ|²")
    
    lines!(ax3, x, abs2.(ψ_numerical), color=:blue, linewidth=2, label="Numerical")
    lines!(ax3, x, abs2.(ψ_analytical), color=:red, linestyle=:dash, linewidth=2, label="Analytical")
    axislegend(ax3, position=:rt)
    
    # Error analysis
    norm_error = norm(ψ_numerical - ψ_analytical) / norm(ψ_analytical)
    
    Label(fig[0, 1:2], "Harmonic Oscillator Coherent State: After t = $(round(tmax, digits=3)) (Error: $(round(norm_error, digits=6)))",
          fontsize=20)
    
    return fig
end


using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using LinearAlgebra
using SparseArrays
using Plots

"""
    solve_harmonic_oscillator_fem(L, n_cells, degree, ω)
    
Solves the quantum harmonic oscillator using finite element method in Gridap.

Parameters:
- L: Half-width of the domain [-L,L]
- n_cells: Number of cells in the mesh
- degree: Polynomial degree of basis functions
- ω: Frequency of the harmonic oscillator
"""
function solve_harmonic_oscillator_fem(L=10.0, n_cells=200, degree=3, ω=1.0)
    # 1. Create the domain and mesh
    domain = (-L, L)
    partition = (n_cells,)
    model = CartesianDiscreteModel(domain, partition)
    
    # 2. Define the FE spaces
    reffe = ReferenceFE(lagrangian, Float64, degree)
    V = FESpace(
        model,
        reffe;
        conformity=:H1,
        dirichlet_tags="boundary"  # Set Dirichlet BC at the boundaries
    )
    
    # Get test and trial spaces
    U = TrialFESpace(V, 0.0)  # u = 0 at boundaries
    
    # 3. Define quadrature rule
    Ω = Triangulation(model)
    degree_quad = 2*degree  # Higher for accuracy
    dΩ = Measure(Ω, degree_quad)
    
    # 4. Define bilinear forms
    # Mass matrix: ∫ u⋅v dΩ
    a_M(u, v) = ∫(u*v)*dΩ
    
    # Stiffness matrix (kinetic energy): (1/2)∫ ∇u⋅∇v dΩ
    a_S(u, v) = ∫(0.5*∇(u)⋅∇(v))*dΩ
    
    # Potential energy matrix: (1/2)ω²∫ x²⋅u⋅v dΩ
    function a_V(u, v)
        potential(x) = 0.5*ω^2*x[1]^2  # Harmonic potential V(x) = (1/2)ω²x²
        return ∫(potential∘get_physical_coordinate(Ω)*u*v)*dΩ
    end
    
    # 5. Assemble matrices
    M = assemble_matrix(a_M, U, V)
    S = assemble_matrix(a_S, U, V)
    V_mat = assemble_matrix(a_V, U, V)
    
    # 6. Construct the Hamiltonian matrix: H = S + V
    H = S + V_mat
    
    # Convert to sparse matrices for eigenvalue solvers
    M_sparse = sparse(M)
    H_sparse = sparse(H)
    
    # Return domain information, spaces, and matrices
    return model, V, U, M_sparse, S_sparse, V_mat, H_sparse
end

# Solve the system
L = 10.0  # Domain half-width
n_cells = 300  # Number of cells
degree = 4  # Polynomial degree (higher = more accurate)
ω = 1.0  # Harmonic oscillator frequency

model, V, U, M, S, V_mat, H = solve_harmonic_oscillator_fem(L, n_cells, degree, ω)

# Solve the eigenvalue problem using Arpack
using Arpack

println("Solving eigenvalue problem...")
λ, ψ_coeffs = eigs(H, M; nev=10, which=:SM, sigma=0.0, maxiter=1000)
println("Eigenvalues: ", real.(λ))

# Expected eigenvalues for harmonic oscillator: Eₙ = ω(n + 1/2), n = 0,1,2,...
expected = ω * (0:9 .+ 0.5)
println("Expected eigenvalues: ", expected)
println("Relative errors: ", abs.(λ .- expected) ./ expected)

# Plot eigenfunctions
function plot_eigenfunctions(model, V, ψ_coeffs, λ, num_to_plot=4)
    # Create FE function for visualization
    uh_fem = FEFunction(V, ψ_coeffs)
    
    # Extract x-coordinates and function values
    coords = get_node_coordinates(get_fe_dof_basis(V))
    x_coords = [p[1] for p in coords]
    
    # Sort coordinates and get sorting indices
    sorted_indices = sortperm(x_coords)
    x_sorted = x_coords[sorted_indices]
    
    # Plot the eigenfunctions
    p = plot(layout=(num_to_plot, 1), size=(800, 200*num_to_plot))
    
    for i in 1:num_to_plot
        # Normalize eigenfunction
        ψ_i = ψ_coeffs[:, i]
        ψ_i = ψ_i / sqrt(ψ_i' * M * ψ_i)
        
        # Create FE function
        ψ_fem = FEFunction(V, ψ_i)
        
        # Evaluate at the sorted coordinates
        ψ_values = [ψ_fem(Point(x)) for x in x_sorted]
        
        # Plot with appropriate title
        plot!(p[i], x_sorted, ψ_values, 
              title="n=$(i-1), E=$(round(real(λ[i]), digits=3))", 
              label="", legend=false, 
              xlabel=i==num_to_plot ? "Position x" : "", 
              ylabel="ψ$(i-1)(x)")
    end
    
    return p
end

# Plot the first few eigenfunctions
p = plot_eigenfunctions(model, V, ψ_coeffs, λ, 6)
display(p)

function main()
    # Set up simulation parameters
    p = (
        N = 2048,      # Number of grid points (higher for better accuracy)
        nt = 1000,     # number of time steps 
        τ = 0.22121,    # τ times period of oscillation is the final time
        # the following parameters have units
        L_0 = 10.0,    # Length scale
        E_0 = 1.0,     # Energy scale
        T_0 = 1.0,     # Time scale
        ħ = 1.0,       # hbar
        m = 1.0,       # mass of particle
        k = 1.0        # harmonic potential mω^2
    )

    # prepare the adimensional parameters, not all scales are independent
    p = make_ε_x(p)
    p = make_ε_t(p)
    p = QuantumRecurrencePlots.make_harmonicPotential_π_k(p)

    # now consider all adimensional variables
     
    # Coherent state parameter
    α = 2.0 + 1.0*im # small so it is in energy range
    p = merge(p, Dict(:α => α))
    x_max = sqrt(2*p.ε_x^2 / p.ε_t) * abs(p.α)

    # prepare grid 
    L = 2*4*x_max  # x_max gives the size scale, L must be greater than this  
    x = LinRange(-L/2, L/2 - L/p.N, p.N)

    # prepare parameters for FFT
    p = QuantumRecurrencePlots.makeParsFFT_1D(x, p)


    # Time parameters
    tmax = p.τ*QuantumRecurrencePlots.get_periodHarmonicPotential(p) # Evolve for one full period
    t0 = 0.0 # initial time
    t = LinRange(t0, tmax, p.nt)

    # Generate initial coherent state
    ψ_initial = QuantumRecurrencePlots.coherent_state_1D(x, t0, p)

    # Solve using split-step Fourier method
    ψ_numerical = solve_harmonic_oscillator(x, t, p, ψ_initial)

    # Compute analytical solution at final time
    ψ_analytical = QuantumRecurrencePlots.coherent_state_1D(x, tmax, p)


    # Generate and save the plot
    fig = plot_comparison(x, tmax, ψ_numerical, ψ_analytical)

    # Display the figure if running interactively
    display(fig)
end

main()
