using DrWatson
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using FFTW
using CairoMakie
using LinearAlgebra
using Infiltrator
using Gridap
using FastGaussQuadrature
using NonlinearSolve
using Arpack
using NonlinearSolveHomotopyContinuation
import HomotopyContinuation as HC

# harmonic oscillator
function main()
    p = (
        N = 1024,      # Number of grid points (higher for better accuracy)
        nt = 10000,     # number of time steps
        L_0 = 100.0,    # Length scale
        E_0 = 10.0,     # Energy scale
        T_0 = 14.0,     # Time scale
        ħ = 12.0,       # hbar
        m = 2.0,       # mass of particle
        k_2 = 13.0,        # harmonic potential mω^2
        τ = 4.22121    # τ times period of oscillation is the final time
    )

    # prepare the adimensional parameters, not all scales are independent
    p = make_ε_x(p)
    p = make_ε_t(p)
    p = QuantumRecurrencePlots.make_harmonicPotential_π_k(p)

    # now consider all adimensional variables

    # Make grid finding optimal size knowing "last eigenstate"
    pp = (n = 10, tol = 1e-4) # eigen state for size of grid
    u0 = sqrt(pp.n * 2 - 1) # initial guess
    f_to_opt(x, p) = exp(-x^2 / 2) * x^p.n - p.tol
    prob = NonlinearProblem(f_to_opt, u0, pp)
    x_max = NonlinearSolve.solve(prob, SimpleNewtonRaphson())[1]
    L = 4 * x_max / sqrt(sqrt(p.π_k_2) * p.ε_t / p.ε_x^2)  # x_max gives the size scale, L must be greater than this
    x = LinRange(-L / 2, L / 2 - L / p.N, p.N)

    # prepare parameters for FFT
    p = QuantumRecurrencePlots.makeParsFFT_1D(x, p)

    # Time parameters
    tmax = p.τ * QuantumRecurrencePlots.get_periodHarmonicPotential(p) # Evolve for one full period
    t0 = 0.0 # initial time
    t = LinRange(t0, tmax, p.nt)

    # psi initial
    c = [1, 1] # gives pole of order 1 initially in x = -1 / sqrt(2)
    c = [1, 1, sqrt(2), sqrt(6)] # gives 3 pole of order 1 initially in x = [-2, 0, 1] ./ sqrt(2)
    psi_0 = QuantumRecurrencePlots.harmonic_eigen_state_sum_1D(x, t0, c, p)
    # psi_0 .= psi_0 / sqrt(sum(abs2.(psi_0)));

    # Solve using split-step Fourier method
    f(x) = QuantumRecurrencePlots.harmonicPotential(x, p)
    kin(x) = QuantumRecurrencePlots.kineticEnergy(x, p)

    psi_n = QuantumRecurrencePlots.solve_schr_SSFM_Yoshida(x, t, p, psi_0, f, kin)

    # Analytical solution
    psi_t = QuantumRecurrencePlots.harmonic_eigen_state_sum_1D(x, tmax, c, p)
    # psi_t .= psi_t / sqrt(sum(abs2.(psi_t)));

    # Generate and save the plot
    fig = Figure(size = (1200, 800))
    QuantumRecurrencePlots.plot_comparison_1D!(fig, x, tmax, psi_n, psi_t)

    # Display the figure if running interactively
    display(fig)

    n_h = 100
    m_h = 100
    x_h, w_h = gausshermite(n_h, normalize = true)
    # a = QuantumRecurrencePlots.Hermite_hat(m_h);
    a = QuantumRecurrencePlots.Hermite(m_h)

    α = QuantumRecurrencePlots._harmonicPotential_xscale(p)
    f_(x) = QuantumRecurrencePlots.harmonicPotential(x, p) / α^2
    msp = MSP_matrix_1D(x_h, w_h, a, f_)
    H = msp.P + msp.S * 1 / 2 * p.ε_x^2 / p.ε_t * α^2 # we reescale the stiffnes matrix with the adimensional parameters
    # λ, ψ_coeffs = eigs(H, msp.M; nev=round(Int, m_h/3), which=:LR,check=1,maxiter=1000,tol=1e-5, sigma=1e-3); # smallest
    λ, ψ_coeffs = eigen(H, msp.M) # note that arpack gives awfull results when M is not the identity

    fig = Figure(size = (800, 800))
    ax = Axis(fig[1, 1])
    n_eigs = 20#round(Int, m_h/3)
    for i in 1:n_eigs #axes(ψ_coeffs, 2)
        nn = maximum(sqrt.(abs2.(ψ_coeffs[:, i])))
        lines!(ax, (range(0, n_eigs - 1)), log10.(abs2.(ψ_coeffs[1:n_eigs, i] ./ nn)),
            label = "$(i-1) eigenstate")
    end
    axislegend(ax, position = :rt)
    ylims!(ax, -10, 1)

    display(fig)
    return λ, ψ_coeffs, msp, p
end

# quartic potential
function main_()
    p = (
        N = 1024,      # Number of grid points (higher for better accuracy)
        nt = 10000,     # number of time steps
        L_0 = 100.0,    # Length scale
        E_0 = 10.0,     # Energy scale
        T_0 = 14.0,     # Time scale
        ħ = 12.0,       # hbar
        m = 2.0,       # mass of particle
        k_4 = 13.0,        # harmonic potential mω^2
        τ = 4.22121    # τ times period of oscillation is the final time
    )

    # prepare the adimensional parameters, not all scales are independent
    p = make_ε_x(p)
    p = make_ε_t(p)
    p = QuantumRecurrencePlots.make_quarticPotential_π_k(p)

    # now consider all adimensional variables

    # Make grid finding optimal size knowing "last eigenstate"
    pp = (n = 10, tol = 1e-4) # eigen state for size of grid
    u0 = sqrt(pp.n * 2 - 1) # initial guess
    f_to_opt(x, p) = exp(-x^2 / 2) * x^p.n - p.tol
    prob = NonlinearProblem(f_to_opt, u0, pp)
    x_max = NonlinearSolve.solve(prob, SimpleNewtonRaphson())[1]
    L = 4 * x_max / sqrt(p.ε_t / p.ε_x^2)  # x_max gives the size scale, L must be greater than this
    x = LinRange(-L / 2, L / 2 - L / p.N, p.N)

    n_h = 100
    m_h = 100
    x_h, w_h = gausshermite(n_h, normalize = true)
    a = QuantumRecurrencePlots.Hermite(m_h)

    α = QuantumRecurrencePlots._quarticPotential_xscale(p)
    f_(x) = QuantumRecurrencePlots.quarticPotential(x, p) / α^4
    msp = MSP_matrix_1D(x_h, w_h, a, f_)
    H = msp.P + msp.S * 1 / 2 * p.ε_x^2 / p.ε_t * α^2 # we reescale the stiffnes matrix with the adimensional parameters
    # λ, ψ_coeffs = eigs(H, msp.M; nev=round(Int, m_h/3), which=:LR,check=1,maxiter=10000,tol=1e-8, sigma=1e-3); # smallest
    λ, ψ_coeffs = eigen(H, msp.M) # note that arpack gives awfull results when M is not the identity

    fig = Figure(size = (800, 800))
    ax = Axis(fig[1, 1])
    n_eigs = 20#round(Int, m_h/3)
    for i in 1:n_eigs #axes(ψ_coeffs, 2)
        nn = maximum(sqrt.(abs2.(ψ_coeffs[:, i])))
        lines!(ax, (range(0, n_eigs - 1)), log10.(abs2.(ψ_coeffs[1:n_eigs, i] ./ nn)),
            label = "$(i-1) eigenstate")
    end
    axislegend(ax, position = :rt)
    ylims!(ax, -10, 1)

    display(fig)
    return λ, ψ_coeffs, msp, p
end

function main_1()
    alg = HomotopyContinuationJL{true}(; threading = false, autodiff = false)
    rhs = function (u, p)
        return u * u - p[1] * u + p[2]
    end
    jac = function (u, p)
        return 2u - p[1]
    end
    fn = NonlinearFunction(rhs; jac)
    prob = NonlinearProblem(fn, 1.0 + im, [5.0, 6.0])
    f, hcsys = NonlinearSolveHomotopyContinuation.homotopy_continuation_preprocessing(
        prob, alg)
    orig_sol = HC.solve(hcsys; alg.kwargs...)
    allsols = HC.solutions(orig_sol)
end
