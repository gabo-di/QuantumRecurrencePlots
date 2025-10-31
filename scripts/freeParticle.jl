using DrWatson
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using CairoMakie
using LinearAlgebra
using Infiltrator
using OrdinaryDiffEq
using FastGaussQuadrature

"""
free particle using fourier space evolution
"""
function main()
    p = (
        L_0 = 1.0,    # Length scale
        E_0 = 10.0,     # Energy scale
        T_0 = 1.0,     # Time scale
        ħ = 1.0,       # hbar
        m = 2.0,       # mass of particle
        v_0 = 1.0,      # velocity of particle (does not have units)
        x_0 = 0.0,      # particles initial position (does not have units)
        σ²_x = 1.0,     # width of initial distribution (does not have units)
        N = 2048 * 2,      # Number of grid points (higher for better accuracy)
        τ = 2.22121    # τ is the final time
    )

    # prepare the adimensional parameters, not all scales are independent
    p = make_ε_x(p)
    p = make_ε_t(p)
    p = QuantumRecurrencePlots.make_freeParticle_pars(p)

    # now consider all adimensional parameters

    # prepare grid
    xmax = p.x_0 + p.v_0 * p.τ  # maximum classical displacement
    σ² = p.ε_x^2 / p.ε_t
    σ_0 = sqrt(σ² / p.π_σ²)
    σ_max = sqrt(σ² / p.π_σ² * (1 + p.τ^2 * p.π_σ² / 2))
    mini = min(-6 * σ_0, xmax - 6 * σ_max)
    maxi = max(6 * σ_0, xmax + 6 * σ_max)
    x = LinRange(mini, maxi, p.N)

    # prepare parameters for FFT
    p = QuantumRecurrencePlots.makeParsFFT_1D(x, p)

    # Generate initial gaussian state
    ψ_initial = QuantumRecurrencePlots.free_gaussian_state_1D(x, 0.0, p)

    # solve in Fourier space
    ψ = copy(ψ_initial)

    # Kinetic energy in Fourier space
    kin(x) = QuantumRecurrencePlots.kineticEnergy(x, p)
    T = kin.(p.k_fft)
    exp_T = exp.(-im * T * p.τ)

    # time evolution
    ψ_k = p.P_fft * ψ
    ψ_k = ψ_k .* exp_T
    ψ = p.P_ifft * ψ_k

    # Compute analytical solution at final time
    ψ_analytical = QuantumRecurrencePlots.free_gaussian_state_1D(x, p.τ, p)

    # Generate and save the plot
    fig = Figure(size = (1200, 800))
    QuantumRecurrencePlots.plot_comparison_1D!(fig, x, p.τ, ψ, ψ_analytical)

    # Display the figure if running interactively
    display(fig)
end
