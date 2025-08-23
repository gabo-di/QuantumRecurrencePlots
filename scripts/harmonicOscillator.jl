using DrWatson
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using CairoMakie
using LinearAlgebra
using Arpack
using Gridap
using Roots


#########
# utils #
#########

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

######################################
# coherent state harmonic oscillator #
######################################
# fem
function main()
    # Solve the system
    p = (
        L_0 = 1.0,    # Length scale
        E_0 = 10.0,     # Energy scale
        T_0 = 1.0,     # Time scale
        ħ = 1.0,       # hbar
        m = 2.0,       # mass of particle
        k_2 = 1.0,        # harmonic potential mω^2
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
    L = 2*8*x_max;  # x_max gives the size scale, L must be greater than this  

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
    ψ_initial(x) = QuantumRecurrencePlots.harmonic_coherent_state_1D(x[1], t0, p);
    ψ_0 = get_free_dof_values(interpolate(ψ_initial, V))

    # x, ψ_numerical = to_plot_femfunctions(L, V, solve_harmonic_oscillator_MSP(msp, V, t, p, ψ_0), msp.M)
    x, ψ_numerical = to_plot_femfunctions(L, V, QuantumRecurrencePlots.solve_schr_CrNi(msp, t, p, ψ_0), msp.M)

    # Compute analytical solution at final time
    ψ_analytical(x) = QuantumRecurrencePlots.harmonic_coherent_state_1D(x[1], tmax, p)
    x, ψ_f = to_plot_femfunctions(L, V,  get_free_dof_values(interpolate(ψ_analytical, V)), msp.M)


    # Generate and save the plot
    fig = Figure(size=(1200, 800))
    QuantumRecurrencePlots.plot_comparison_1D!(fig, x, tmax, ψ_numerical, ψ_f)

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

# ssfm
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
        k_2 = 1.0        # harmonic potential mω^2
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
    L = 2*8*x_max  # x_max gives the size scale, L must be greater than this  
    x = LinRange(-L/2, L/2 - L/p.N, p.N)

    # prepare parameters for FFT
    p = QuantumRecurrencePlots.makeParsFFT_1D(x, p)


    # Time parameters
    tmax = p.τ*QuantumRecurrencePlots.get_periodHarmonicPotential(p) # Evolve for one full period
    t0 = 0.0 # initial time
    t = LinRange(t0, tmax, p.nt)

    # Generate initial coherent state
    ψ_initial = QuantumRecurrencePlots.harmonic_coherent_state_1D(x, t0, p)

    # Solve using split-step Fourier method
    f(x) = QuantumRecurrencePlots.harmonicPotential(x, p)
    ψ_numerical = QuantumRecurrencePlots.solve_schr_SSFM_Yoshida(x, t, p, ψ_initial, f)
    

    # Compute analytical solution at final time
    ψ_analytical = QuantumRecurrencePlots.harmonic_coherent_state_1D(x, tmax, p)


    # Generate and save the plot
    fig = Figure(size=(1200, 800))
    QuantumRecurrencePlots.plot_comparison_1D!(fig, x, tmax, ψ_numerical, ψ_analytical)

    # Display the figure if running interactively
    display(fig)
end

####################################
# eigen states harmonic oscillator #
####################################
function main_1()
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
        k_2 = 1.0        # harmonic potential mω^2
    )

    # prepare the adimensional parameters, not all scales are independent
    p = make_ε_x(p)
    p = make_ε_t(p)
    p = QuantumRecurrencePlots.make_harmonicPotential_π_k(p)

end


# ssfm
function main__()
    p = (
        N = 512,      # Number of grid points (higher for better accuracy)
        L_0 = 100.0,    # Length scale
        E_0 = 10.0,     # Energy scale
        T_0 = 14.0,     # Time scale
        ħ = 12.0,       # hbar
        m = 2.0,       # mass of particle
        k_2 = 13.0,        # harmonic potential mω^2
        nt = 10000,     # number of time steps 
        τ = 4.22121,    # τ times period of oscillation is the final time
    );
    #
    # prepare the adimensional parameters, not all scales are independent
    p = make_ε_x(p);
    p = make_ε_t(p);
    p = QuantumRecurrencePlots.make_harmonicPotential_π_k(p);

    # now consider all adimensional variables

    n = 10; # eigen state
    x_max = fzero(x->exp(-x^2/2)*x^n - 0.001, sqrt(2*n-1));
    L = 2*x_max/sqrt(sqrt(p.π_k)*p.ε_t/p.ε_x^2);  # x_max gives the size scale, L must be greater than this  
    x = LinRange(-L/2, L/2 - L/p.N, p.N);


    # prepare parameters for FFT
    p = QuantumRecurrencePlots.makeParsFFT_1D(x, p)
     
    # Time parameters
    tmax = p.τ*QuantumRecurrencePlots.get_periodHarmonicPotential(p) # Evolve for one full period
    t0 = 0.0 # initial time
    t = LinRange(t0, tmax, p.nt)

    psi_0 = QuantumRecurrencePlots.harmonic_eigen_state_1D(x, t0, n, p);

    # Solve using split-step Fourier method
    f(x) = QuantumRecurrencePlots.harmonicPotential(x, p)
    psi_n = QuantumRecurrencePlots.solve_schr_SSFM_Yoshida(x, t, p, psi_0, f)

    psi_t = QuantumRecurrencePlots.harmonic_eigen_state_1D(x, tmax, n, p)
    
    
    # Generate and save the plot
    fig = Figure(size=(1200, 800))
    QuantumRecurrencePlots.plot_comparison_1D!(fig, x, tmax, psi_n, psi_t)

    # Display the figure if running interactively
    display(fig)
end
