# ---------------------------------------------------------------------
# 5.  Crank–Nicolson time‑stepper (unitary in M‑norm)
# ---------------------------------------------------------------------
Δt  = 1e-3
A = M + 0.5im*Δt*S
B = M - 0.5im*Δt*S
solver = factorize(A)                      # factor once, reuse every step

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



using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using LinearAlgebra
using SparseArrays
using CairoMakie 

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
    
    # Convert to sparse matrices for eigenvalue solvers
    
    # Return domain information, spaces, and matrices
    return model, V, M, S, V_mat
end

# Solve the system
L = 10.0  # Domain half-width
n_cells = 300  # Number of cells
degree = 4  # Polynomial degree (higher = more accurate)
ω = 1.0  # Harmonic oscillator frequency

model, V, M, S, V_mat = solve_harmonic_oscillator_fem(L, n_cells, degree, ω)

# Solve the eigenvalue problem using Arpack
using Arpack

println("Solving eigenvalue problem...")
λ, ψ_coeffs = eigs(S+V_mat, M; nev=10, which=:LR, check=1, sigma=1e-6, maxiter=1000, tol=1e-5)
println("Eigenvalues: ", real.(λ))

# Expected eigenvalues for harmonic oscillator: Eₙ = ω(n + 1/2), n = 0,1,2,...
expected = ω * (collect(0:size(λ,1)-1) .+ 0.5)
println("Expected eigenvalues: ", expected)
println("Relative errors: ", abs.(λ .- expected) ./ expected)

# Plot eigenfunctions
function plot_eigenfunctions(L, V, ψ_coeffs, λ, num_to_plot=4)
    # Create a fine grid for plotting
    n_plot_points = 1000
    x_plot = range(-L, L, length=n_plot_points)

    # Plot the eigenfunctions
    fig = Figure(size=(800,200*num_to_plot))
    
    for i in 1:num_to_plot
        ax = Axis(fig[i,1])
         
        # Normalize eigenfunction
        ψ_i = ψ_coeffs[:, i]
        ψ_i = ψ_i / sqrt(abs(ψ_i' * M * ψ_i))  # Normalize with respect to mass matrix
        
        # Create FE function
        ψ_fem = FEFunction(V, real(ψ_i))
        
        # Evaluate at the plotting points
        ψ_values = zeros(n_plot_points)
        for (j, x) in enumerate(x_plot)
            try
                ψ_values[j] = ψ_fem(Gridap.Point(x))
            catch
                ψ_values[j] = 0.0  # Handle points outside domain
            end
        end
        
        # Plot with appropriate title
        plot!(ax, x_plot, ψ_values)
    end
    
    return fig
end

# Plot the first few eigenfunctions
fig = plot_eigenfunctions(L, V,  ψ_coeffs, λ, 4)
display(fig)
