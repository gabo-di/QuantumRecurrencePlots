using DrWatson
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using FFTW
using CairoMakie
using LinearAlgebra
using Roots
using Infiltrator


function main()
    p = (
        N = 1024,      # Number of grid points (higher for better accuracy)
        L_0 = 1.0,    # Length scale
        E_0 = 1.0,     # Energy scale
        T_0 = 1.0,     # Time scale
        ħ = 1.0,       # hbar
        m = 1.0,       # mass of particle
        k = 1.0,        # harmonic potential mω^2
        nt = 100,     # number of time steps 
        τ = 0.3,    # τ times period of oscillation is the final time
    );
    #
    # prepare the adimensional parameters, not all scales are independent
    p = make_ε_x(p);
    p = make_ε_t(p);
    p = QuantumRecurrencePlots.make_harmonicPotential_π_k(p);

    # now consider all adimensional variables

    n = 6; # eigen state for size of grid
    x_max = fzero(x->exp(-x^2/2)*x^(n) - 0.001, sqrt(2*n-1));
    L = 2*x_max/sqrt(sqrt(p.π_k)*p.ε_t/p.ε_x^2);  # x_max gives the size scale, L must be greater than this  
    x = LinRange(-L/2, L/2 - L/p.N, p.N);


    # prepare parameters for FFT
    p = QuantumRecurrencePlots.makeParsFFT_1D(x, p)
     
    # Time parameters
    tmax = p.τ*QuantumRecurrencePlots.get_periodHarmonicPotential(p) # Evolve for one full period
    t0 = 0.0 # initial time
    t = LinRange(t0, tmax, p.nt)

    # c = [1,1,sqrt(2), sqrt(6)] # eigenstates hermite_hat are He_n / sqrt(n!)
    c = [1,1]  
    psi_0 = QuantumRecurrencePlots.eigen_state_sum_1D(x, t0, c, p);
    psi_0 .= psi_0 / sqrt(sum(abs2.(psi_0)))

    # Solve using split-step Fourier method
    f(x) = QuantumRecurrencePlots.harmonicPotential(x, p)
    psi_n = QuantumRecurrencePlots.solve_schr_SSFM_Yoshida(x, t, p, psi_0, f)

    psi_t = QuantumRecurrencePlots.eigen_state_sum_1D(x, tmax, c, p)
    psi_t .= psi_t / sqrt(sum(abs2.(psi_t)))
    
    
    # Generate and save the plot
    fig = Figure(size=(1200, 800))
    QuantumRecurrencePlots.plot_comparison_1D!(fig, x, tmax, psi_n, psi_t)

    # Display the figure if running interactively
    display(fig)
end
 
