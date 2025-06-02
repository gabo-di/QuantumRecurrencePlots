using DrWatson
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using FFTW
using CairoMakie
using LinearAlgebra
using Arpack
using Gridap
using Infiltrator

"""
    solve_harmonic_oscillator_fft(x, k, dt, tmax, ω, ψ_initial)
    
Solves the time-dependent Schrödinger equation for a harmonic oscillator
using the split-step Fourier method.

Parameters:
- x: Spatial grid
- t: Time grid
- p: parameters for simulation
- ψ_initial: Initial wavefunction
"""
function solve_harmonic_oscillator_fft(x, t, p, ψ_initial)
    @unpack π_k, ε_x, ε_t = p
    @unpack k_fft, P_fft, P_ifft = p

    dt = t[2] - t[1]
    dx = x[2] - x[1]
    nt = length(t)
    
    # Potential energy (harmonic oscillator) 
    f(x) = QuantumRecurrencePlots.harmonicPotential(x, p)
    V = f.(x) 
    # V = 1/2 * π_k * ε_t / ε_x ^ 2 .* x.^2
    
    # Kinetic energy in Fourier space
    kin(x) = QuantumRecurrencePlots.kineticEnergy(x, p)
    T = kin.(k_fft)
    # T = ε_x ^ 2 /(2* ε_t) .* k_fft.^2
    
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

function to_plot_femfunctions(L, V, ψ_coeffs, M)
    # Create a fine grid for plotting
    n_plot_points = 1000
    x_plot = range(-L, L, length=n_plot_points)

    
    # Normalize eigenfunction
    ψ_i = ψ_coeffs[:]
    ψ_i = ψ_i / sqrt(abs(ψ_i' * M * ψ_i))  # Normalize with respect to mass matrix
   
    # Create FE function
    ψ_fem = FEFunction(V, ψ_i)
   
    # Evaluate at the plotting points
    T = ComplexF64
    ψ_values = zeros(T, n_plot_points)
    for (j, x) in enumerate(x_plot)
        ψ_values[j] = ψ_fem(Gridap.Point(x))
    end
   
   
    return x_plot, ψ_values
end

# Plot the results using Makie
function plot_comparison(x, tmax, ψ_numerical, ψ_analytical)
    fig = Figure(size=(1200, 800))
    
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

function main()
    # Solve the system
    p = (
        L_0 = 1.0,    # Length scale
        E_0 = 10.0,     # Energy scale
        T_0 = 1.0,     # Time scale
        ħ = 1.0,       # hbar
        m = 2.0,       # mass of particle
        k = 1.0,        # harmonic potential mω^2
        nt = 10000,     # number of time steps 
        τ = 4.22121,    # τ times period of oscillation is the final time
    );
    #
    # prepare the adimensional parameters, not all scales are independent
    p = make_ε_x(p);
    p = make_ε_t(p);
    p = QuantumRecurrencePlots.make_harmonicPotential_π_k(p);


    # now consider all adimensional variables
     
    # Coherent state parameter
    α = 1.0 + 0.0*im; # small so it is in energy range
    p = merge(p, Dict(:α => α));
    x_max = sqrt(2*p.ε_x^2 / p.ε_t) * abs(p.α);

    # prepare grid 
    L = 2*4*x_max;  # x_max gives the size scale, L must be greater than this  

    p_gridap = (
        L = L,
        n_cells = 300,  # Number of cells
        order = 4,  # Polynomial degree (higher = more accurate)
        bc_type = :Dirichlet
    );

    
    p = merge(p, (p_gridap = p_gridap,));

    model, V = initialize_gridap_1D(p, ComplexF64);

    # preapare the matrices
    f(x) = QuantumRecurrencePlots.harmonicPotential(x, p);
    msp = MSP_matrix_1D(V, model, p, f);

    tmax = p.τ*QuantumRecurrencePlots.get_periodHarmonicPotential(p); # Evolve for one full period
    t0 = 0.0 # initial time
    t = LinRange(t0, tmax, p.nt);
    ψ_initial(x) = QuantumRecurrencePlots.coherent_state_1D(x[1], t0, p);
    ψ_0 = get_free_dof_values(interpolate(ψ_initial, V))

    x, ψ_numerical = to_plot_femfunctions(L, V, solve_harmonic_oscillator_MSP(msp, V, t, p, ψ_0), msp.M)

    # Compute analytical solution at final time
    ψ_analytical(x) = QuantumRecurrencePlots.coherent_state_1D(x[1], tmax, p)
    x, ψ_f = to_plot_femfunctions(L, V,  get_free_dof_values(interpolate(ψ_analytical, V)), msp.M)


    # Generate and save the plot
    fig = plot_comparison(x, tmax, ψ_numerical, ψ_f)

    # Display the figure if running interactively
    display(fig)

    # Eigen system
    # begin
    #     @unpack M, S, P = msp 
    #     H = P + S * 1/2 * p.ε_x ^ 2 /p.ε_t # we reescale the stiffnes matrix with the adimensional parameters 
    #
    #
    #     # Solve the eigenvalue problem using Arpack
    #
    #     println("Solving eigenvalue problem...")
    #     λ, ψ_coeffs = eigs(H, M; nev=10, which=:LR, sigma=1e-6, tol=1e-5, maxiter=1000)
    #     # λ, ψ_coeffs = eigs(H, M; nev=10, which=:SM, tol=1e-5, maxiter=1000)
    #     println("Eigenvalues: ", real.(λ))
    #
    #     # Expected eigenvalues for harmonic oscillator: Eₙ = ω(n + 1/2), n = 0,1,2,...
    #     expected = p.π_k * (collect(0:9) .+ 0.5)
    #     println("Expected eigenvalues: ", expected)
    #     println("Relative errors: ", abs.(λ .- expected) ./ expected)
    #     
    #     # Plot the first few eigenfunctions
    #     p = plot_femfunctions(L, V, ψ_coeffs[:,2], msp.M)
    #     display(p)
    # end
end


function solve_harmonic_oscillator_MSP(msp::MSP, V, t, p, ψ_initial)
    @unpack M, S, P = msp
    @unpack π_k, ε_x, ε_t = p

    dt = t[2] - t[1]
    nt = length(t)
    H = P + S * 1/2 * p.ε_x ^ 2 /p.ε_t # we reescale the stiffnes matrix with the adimensional parameters 

    # precompute matrices for Crank-Nicolson
    A = factorize(M + im*dt/2*H)
    B = (M - im*dt/2*H)

    ψ_fem = copy(ψ_initial)

    for i in 1:(nt-1)
        ψ_fem = A \ (B * ψ_fem)
    end
    return ψ_fem
end


function main_()
    # Set up simulation parameters
    p = (
        N = 2048*2,      # Number of grid points (higher for better accuracy)
        nt = 10000,     # number of time steps 
        τ = 4.22121,    # τ times period of oscillation is the final time
        # the following parameters have units
        L_0 = 1.0,    # Length scale
        E_0 = 10.0,     # Energy scale
        T_0 = 1.0,     # Time scale
        ħ = 1.0,       # hbar
        m = 2.0,       # mass of particle
        k = 1.0        # harmonic potential mω^2
    )

    # prepare the adimensional parameters, not all scales are independent
    p = make_ε_x(p)
    p = make_ε_t(p)
    p = QuantumRecurrencePlots.make_harmonicPotential_π_k(p)

    # now consider all adimensional variables
     
    # Coherent state parameter
    α = 1.0 + 0.0*im # small so it is in energy range
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
    ψ_numerical = solve_harmonic_oscillator_fft(x, t, p, ψ_initial)

    # Compute analytical solution at final time
    ψ_analytical = QuantumRecurrencePlots.coherent_state_1D(x, tmax, p)


    # Generate and save the plot
    fig = plot_comparison(x, tmax, ψ_numerical, ψ_analytical)

    # Display the figure if running interactively
    display(fig)
end
