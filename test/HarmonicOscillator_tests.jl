using DrWatson
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using LinearAlgebra
using Arpack
using NonlinearSolve
using FastGaussQuadrature

@testset "Time Integrators" begin
    @testset "SSFM coherent state" begin
        p = (
            N = 1024 * 2,      # Number of grid points (higher for better accuracy)
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
        α = 1.0 + 0.0 * im # small so it is in energy range
        p = merge(p, Dict(:α => α))
        x_max = sqrt(2 * p.ε_x^2 / p.ε_t) * abs(p.α)

        # prepare grid
        L = 2 * 8 * x_max  # x_max gives the size scale, L must be greater than this
        x = LinRange(-L / 2, L / 2 - L / p.N, p.N)

        # Solve using split-step Fourier method
        f(x) = QuantumRecurrencePlots.harmonicPotential(x, p)
        kin(x) = QuantumRecurrencePlots.kineticEnergy(x, p)

        # prepare parameters for FFT
        p = QuantumRecurrencePlots.makeParsFFT_1D(x, p)

        # Time parameters
        tmax = p.τ * QuantumRecurrencePlots.get_periodHarmonicPotential(p) # Evolve for one full period
        t0 = 0.0 # initial time
        t = LinRange(t0, tmax, p.nt)

        # Generate initial coherent state
        ψ_initial = QuantumRecurrencePlots.harmonic_coherent_state_1D(x, t0, p)

        # Compute analytical solution at final time
        ψ_analytical = QuantumRecurrencePlots.harmonic_coherent_state_1D(x, tmax, p)
        @testset "Second Order Method" begin
            ψ_numerical = QuantumRecurrencePlots.solve_schr_SSFM(x, t, p, ψ_initial, f, kin)
            @test isapprox(
                norm(ψ_numerical - ψ_analytical) / norm(ψ_analytical), 0; atol = 2e-5)
        end

        @testset "Yoshida Method" begin
            ψ_numerical = QuantumRecurrencePlots.solve_schr_SSFM_Yoshida(
                x, t, p, ψ_initial, f, kin)
            @test isapprox(
                norm(ψ_numerical - ψ_analytical) / norm(ψ_analytical), 0; atol = 1e-8)
        end
    end

    if false  #skip because takes time
        @testset "Cranck Nicolson coherent state" begin
            p = (
                N = 1024 * 2,      # Number of grid points (higher for better accuracy)
                nt = 100000,     # number of time steps
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
            α = 1.0 + 0.0 * im # small so it is in energy range
            p = merge(p, Dict(:α => α))
            x_max = sqrt(2 * p.ε_x^2 / p.ε_t) * abs(p.α)

            # prepare grid
            L = 2 * 8 * x_max  # x_max gives the size scale, L must be greater than this
            x = LinRange(-L / 2, L / 2 - L / p.N, p.N)

            p_gridap = (
                L = L,
                n_cells = 300,  # Number of cells
                order = 4,  # Polynomial degree (higher = more accurate)
                bc_type = :Dirichlet
            )

            p = merge(p, (p_gridap = p_gridap,))

            model, V = initialize_gridap_1D(p, ComplexF64)

            # preapare the matrices
            f(x) = QuantumRecurrencePlots.harmonicPotential(x, p)
            msp = MSP_matrix_1D(V, model, p, f)

            tmax = p.τ * QuantumRecurrencePlots.get_periodHarmonicPotential(p) # Evolve for one full period
            t0 = 0.0 # initial time
            t = LinRange(t0, tmax, p.nt)
            ψ_initial(x) = QuantumRecurrencePlots.harmonic_coherent_state_1D(x[1], t0, p)
            ψ_0 = get_free_dof_values(interpolate(ψ_initial, V))

            ψ_numerical = QuantumRecurrencePlots.to_plot_femfunctions_1D(
                x, V, QuantumRecurrencePlots.solve_schr_CrNi(msp, t, p, ψ_0), msp.M)

            # Compute analytical solution at final time
            ψ_analytical(x) = QuantumRecurrencePlots.harmonic_coherent_state_1D(
                x[1], tmax, p)
            ψ_f = QuantumRecurrencePlots.to_plot_femfunctions_1D(
                x, V, get_free_dof_values(interpolate(ψ_analytical, V)), msp.M)

            @test isapprox(norm(ψ_numerical - ψ_f) / norm(ψ_f), 0; atol = 1e-5)
        end
    end

    @testset "SSFM single eigen state" begin
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
        #
        # prepare the adimensional parameters, not all scales are independent
        p = make_ε_x(p)
        p = make_ε_t(p)
        p = QuantumRecurrencePlots.make_harmonicPotential_π_k(p)

        # now consider all adimensional variables

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

        psi_0 = QuantumRecurrencePlots.harmonic_eigen_state_1D(x, t0, pp.n, p)

        # Solve using split-step Fourier method
        f(x) = QuantumRecurrencePlots.harmonicPotential(x, p)
        kin(x) = QuantumRecurrencePlots.kineticEnergy(x, p)

        psi_t = QuantumRecurrencePlots.harmonic_eigen_state_1D(x, tmax, pp.n, p)
        @testset "Second Order Method" begin
            psi_n = QuantumRecurrencePlots.solve_schr_SSFM(x, t, p, psi_0, f, kin)
            @test isapprox(norm(psi_n - psi_t) / norm(psi_t), 0; atol = 1e-4)
        end

        @testset "Yoshida Method" begin
            psi_n = QuantumRecurrencePlots.solve_schr_SSFM_Yoshida(x, t, p, psi_0, f, kin)
            @test isapprox(norm(psi_n - psi_t) / norm(psi_t), 0; atol = 1e-8)
        end
    end

    @testset "SSFM superposition of eigen states" begin
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
        #
        # prepare the adimensional parameters, not all scales are independent
        p = make_ε_x(p)
        p = make_ε_t(p)
        p = QuantumRecurrencePlots.make_harmonicPotential_π_k(p)

        # now consider all adimensional variables

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

        c = [1, 1, 1, 1]
        psi_0 = QuantumRecurrencePlots.harmonic_eigen_state_sum_1D(x, t0, c, p)

        # Solve using split-step Fourier method
        f(x) = QuantumRecurrencePlots.harmonicPotential(x, p)
        kin(x) = QuantumRecurrencePlots.kineticEnergy(x, p)

        psi_t = QuantumRecurrencePlots.harmonic_eigen_state_sum_1D(x, tmax, c, p)

        @testset "Second Order Method" begin
            psi_n = QuantumRecurrencePlots.solve_schr_SSFM(x, t, p, psi_0, f, kin)
            @test isapprox(norm(psi_n - psi_t) / norm(psi_t), 0; atol = 1e-4)
        end

        @testset "Yoshida Method" begin
            psi_n = QuantumRecurrencePlots.solve_schr_SSFM_Yoshida(x, t, p, psi_0, f, kin)
            @test isapprox(norm(psi_n - psi_t) / norm(psi_t), 0; atol = 1e-8)
        end
    end
end

@testset "Eigen states" begin
    p_1 = (
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

    p_1 = make_ε_x(p_1)
    p_1 = make_ε_t(p_1)
    p_1 = QuantumRecurrencePlots.make_harmonicPotential_π_k(p_1)

    # different parameter set
    p_2 = (
        N = 1024,      # Number of grid points (higher for better accuracy)
        nt = 10000,     # number of time steps
        L_0 = 1.0,    # Length scale
        E_0 = 1.0,     # Energy scale
        T_0 = 1.0,     # Time scale
        ħ = 1.0,       # hbar
        m = 1.0,       # mass of particle
        k_2 = 1.0,        # harmonic potential mω^2
        τ = 4.22121    # τ times period of oscillation is the final time
    )

    p_2 = make_ε_x(p_2)
    p_2 = make_ε_t(p_2)
    p_2 = QuantumRecurrencePlots.make_harmonicPotential_π_k(p_2)
    # now consider all adimensional variables
    n_h = 100
    m_h = 100
    x_h, w_h = gausshermite(n_h, normalize = true)
    a = QuantumRecurrencePlots.Hermite(m_h)

    α_1 = QuantumRecurrencePlots._harmonicPotential_xscale(p_1)
    f_1(x) = QuantumRecurrencePlots.harmonicPotential(x, p_1) / α_1^2
    msp_1 = MSP_matrix_1D(x_h, w_h, a, f_1)
    H_1 = msp_1.P + msp_1.S * 1 / 2 * p_1.ε_x^2 / p_1.ε_t * α_1^2 # we reescale the stiffnes matrix with the adimensional parameters
    λ_1, ψ_coeffs_1 = eigen(H_1, msp_1.M) # note that arpack gives awfull results when M is not the identity

    α_2 = QuantumRecurrencePlots._harmonicPotential_xscale(p_2)
    f_2(x) = QuantumRecurrencePlots.harmonicPotential(x, p_2) / α_2^2
    msp_2 = MSP_matrix_1D(x_h, w_h, a, f_2)
    H_2 = msp_2.P + msp_2.S * 1 / 2 * p_2.ε_x^2 / p_2.ε_t * α_2^2 # we reescale the stiffnes matrix with the adimensional parameters
    λ_2, ψ_coeffs_2 = eigen(H_2, msp_2.M) # note that arpack gives awfull results when M is not the identity

    @testset "Eigen energy scale" begin
        @test isapprox(λ_1 ./ QuantumRecurrencePlots._harmonicPotential_Escale(p_1),
            [0.5 + i for i in 0:(length(λ_1) - 1)]; atol = 1e-8)
        @test isapprox(λ_1 ./ QuantumRecurrencePlots._harmonicPotential_Escale(p_1),
            λ_2 ./ QuantumRecurrencePlots._harmonicPotential_Escale(p_2); atol = 1e-8)
    end

    @testset "Eigen vectors parameter invariance" begin
        @test isapprox(ψ_coeffs_2[1, 1:20] ./ ψ_coeffs_2[1, 1],
            ψ_coeffs_1[1, 1:20] ./ ψ_coeffs_1[1, 1]; atol = 1e-6)
        @test isapprox(ψ_coeffs_2[2, 1:20] ./ ψ_coeffs_2[2, 2],
            ψ_coeffs_1[2, 1:20] ./ ψ_coeffs_1[2, 2]; atol = 1e-6)
        @test isapprox(ψ_coeffs_2[3, 1:20] ./ ψ_coeffs_2[3, 3],
            ψ_coeffs_1[3, 1:20] ./ ψ_coeffs_1[3, 3]; atol = 1e-6)
        @test isapprox(ψ_coeffs_2[4, 1:20] ./ ψ_coeffs_2[4, 4],
            ψ_coeffs_1[4, 1:20] ./ ψ_coeffs_1[4, 4]; atol = 1e-6)
        @test isapprox(ψ_coeffs_2[5, 1:20] ./ ψ_coeffs_2[5, 5],
            ψ_coeffs_1[5, 1:20] ./ ψ_coeffs_1[5, 5]; atol = 1e-6)
    end

    @testset "Evolution in eigensystem" begin
        pp = (n = 10, tol = 1e-4) # eigen state for size of grid
        u0 = sqrt(pp.n * 2 - 1) # initial guess
        f_to_opt(x, p) = exp(-x^2 / 2) * x^p.n - p.tol
        prob = NonlinearProblem(f_to_opt, u0, pp)
        x_max = NonlinearSolve.solve(prob, SimpleNewtonRaphson())[1]
        L = 4 * x_max / sqrt(sqrt(p_1.π_k_2) * p_1.ε_t / p_1.ε_x^2)  # x_max gives the size scale, L must be greater than this
        x = LinRange(-L / 2, L / 2 - L / p_1.N, p_1.N)
        tmax = p_1.τ * QuantumRecurrencePlots.get_periodHarmonicPotential(p_1) # Evolve for one full period
        t0 = 0.0 # initial time
        t = LinRange(t0, tmax, p_1.nt)

        in = 5
        c = ones(in)
        e = λ_1[1:in]
        s = ψ_coeffs_1[1:(in * 2), 1:in]
        psi_0 = QuantumRecurrencePlots.hermite_expansion_state_sum_1D(
            x .* α_1, t0, c, e, s) .*
                sqrt(α_1)
        psi_t = QuantumRecurrencePlots.hermite_expansion_state_sum_1D(
            x .* α_1, tmax, c, e, s) .*
                sqrt(α_1)
        p_1 = QuantumRecurrencePlots.makeParsFFT_1D(x .* α_1, p_1)
        kin_(x) = QuantumRecurrencePlots.kineticEnergy(x, p_1) * α_1^2
        f_(x) = QuantumRecurrencePlots.harmonicPotential(x, p_1) / α_1^2
        psi_n = QuantumRecurrencePlots.solve_schr_SSFM_Yoshida(
            x .* α_1, t, p_1, psi_0, f_, kin_)
        psi_a = QuantumRecurrencePlots.harmonic_eigen_state_sum_1D(x, tmax, s * c, p_1) # note that in this way they have same value
        @test isapprox(psi_t, psi_a)
        @test isapprox(psi_t, psi_n)
    end
end
