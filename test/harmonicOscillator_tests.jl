using DrWatson
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using LinearAlgebra
using Arpack
using Gridap
using Roots
using FastGaussQuadrature

@testset "Time Integrators" begin 
    @testset "SSFM coherent state" begin
        p = (
            N = 1024*2,      # Number of grid points (higher for better accuracy)
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

        
        # Solve using split-step Fourier method
        f(x) = QuantumRecurrencePlots.harmonicPotential(x, p)

        # prepare parameters for FFT
        p = QuantumRecurrencePlots.makeParsFFT_1D(x, p)


        # Time parameters
        tmax = p.τ*QuantumRecurrencePlots.get_periodHarmonicPotential(p) # Evolve for one full period
        t0 = 0.0 # initial time
        t = LinRange(t0, tmax, p.nt)

        # Generate initial coherent state
        ψ_initial = QuantumRecurrencePlots.harmonic_coherent_state_1D(x, t0, p)
        
        # Compute analytical solution at final time
        ψ_analytical = QuantumRecurrencePlots.harmonic_coherent_state_1D(x, tmax, p)
        @testset "Second Order Method" begin
            ψ_numerical = QuantumRecurrencePlots.solve_schr_SSFM(x, t, p, ψ_initial, f)
            @test isapprox(norm(ψ_numerical - ψ_analytical)/norm(ψ_analytical), 0; atol=2e-5)
        end

        @testset "Yoshida Method" begin
            ψ_numerical = QuantumRecurrencePlots.solve_schr_SSFM_Yoshida(x, t, p, ψ_initial, f)
            @test isapprox(norm(ψ_numerical - ψ_analytical)/norm(ψ_analytical), 0; atol=1e-8)
        end
    end

    if false  #skip because takes time
        @testset "Cranck Nicolson coherent state" begin
            p = (
                N = 1024*2,      # Number of grid points (higher for better accuracy)
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
            x = LinRange(-L/2, L/2 - L/p.N, p.N)

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

            function to_plot_femfunctions(x, V, ψ_coeffs, M)
                # Create a fine grid for plotting
                n_plot_points = length(x)

                
                # Normalize eigenfunction
                ψ_i = ψ_coeffs[:]
                ψ_i = ψ_i / sqrt(abs(ψ_i' * M * ψ_i))  # Normalize with respect to mass matrix
               
                # Create FE function
                ψ_fem = FEFunction(V, ψ_i)
               
                # Evaluate at the plotting points
                T = ComplexF64
                ψ_values = zeros(T, n_plot_points)
                for (j, x) in enumerate(x)
                    ψ_values[j] = ψ_fem(Gridap.Point(x))
                end
               
                return ψ_values
            end

            ψ_numerical = to_plot_femfunctions(x, V, QuantumRecurrencePlots.solve_schr_CrNi(msp, t, p, ψ_0), msp.M)

            # Compute analytical solution at final time
            ψ_analytical(x) = QuantumRecurrencePlots.harmonic_coherent_state_1D(x[1], tmax, p)
            ψ_f = to_plot_femfunctions(x, V,  get_free_dof_values(interpolate(ψ_analytical, V)), msp.M)


            @test isapprox(norm(ψ_numerical - ψ_f)/norm(ψ_f), 0; atol=1e-5)
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
        L = 4*x_max/sqrt(sqrt(p.π_k_2)*p.ε_t/p.ε_x^2);  # x_max gives the size scale, L must be greater than this  
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

        psi_t = QuantumRecurrencePlots.harmonic_eigen_state_1D(x, tmax, n, p)
        @testset "Second Order Method" begin
            psi_n = QuantumRecurrencePlots.solve_schr_SSFM(x, t, p, psi_0, f)
            @test isapprox(norm(psi_n - psi_t)/norm(psi_t), 0; atol=1e-4)
        end

        @testset "Yoshida Method" begin
            psi_n = QuantumRecurrencePlots.solve_schr_SSFM_Yoshida(x, t, p, psi_0, f)
            @test isapprox(norm(psi_n - psi_t)/norm(psi_t), 0; atol=1e-8)
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
            τ = 4.22121,    # τ times period of oscillation is the final time
        );
        #
        # prepare the adimensional parameters, not all scales are independent
        p = make_ε_x(p);
        p = make_ε_t(p);
        p = QuantumRecurrencePlots.make_harmonicPotential_π_k(p);

        # now consider all adimensional variables

        n = 6; # eigen state
        x_max = fzero(x->exp(-x^2/2)*x^n - 0.001, sqrt(2*n-1));
        L = 4*x_max/sqrt(sqrt(p.π_k_2)*p.ε_t/p.ε_x^2);  # x_max gives the size scale, L must be greater than this  
        x = LinRange(-L/2, L/2 - L/p.N, p.N);


        # prepare parameters for FFT
        p = QuantumRecurrencePlots.makeParsFFT_1D(x, p)
         
        # Time parameters
        tmax = p.τ*QuantumRecurrencePlots.get_periodHarmonicPotential(p) # Evolve for one full period
        t0 = 0.0 # initial time
        t = LinRange(t0, tmax, p.nt)

        c = [1,1,sqrt(2), sqrt(6)]; # gives 3 pole of order 1 initially in x = [-2, 0, 1] ./ sqrt(2)
        psi_0 = QuantumRecurrencePlots.harmonic_eigen_state_sum_1D(x, t0, c, p);

        # Solve using split-step Fourier method
        f(x) = QuantumRecurrencePlots.harmonicPotential(x, p)

        psi_t = QuantumRecurrencePlots.harmonic_eigen_state_sum_1D(x, tmax, c, p);

        @testset "Second Order Method" begin
            psi_n = QuantumRecurrencePlots.solve_schr_SSFM(x, t, p, psi_0, f)
            @test isapprox(norm(psi_n - psi_t)/norm(psi_t), 0; atol=1e-4)
        end

        @testset "Yoshida Method" begin
            psi_n = QuantumRecurrencePlots.solve_schr_SSFM_Yoshida(x, t, p, psi_0, f)
            @test isapprox(norm(psi_n - psi_t)/norm(psi_t), 0; atol=1e-8)
        end
    end
end

@testset "Eigen states" begin
    p = (
        N = 1024,      # Number of grid points (higher for better accuracy)
        nt = 10000,     # number of time steps 
        L_0 = 100.0,    # Length scale
        E_0 = 10.0,     # Energy scale
        T_0 = 14.0,     # Time scale
        ħ = 12.0,       # hbar
        m = 2.0,       # mass of particle
        k_2 = 13.0,        # harmonic potential mω^2
        τ = 4.22121,    # τ times period of oscillation is the final time
    );

    p = (
        N = 1024,      # Number of grid points (higher for better accuracy)
        nt = 10000,     # number of time steps 
        L_0 = 100.0,    # Length scale
        E_0 = 10.0,     # Energy scale
        T_0 = 1.0,     # Time scale
        ħ = 1.0,       # hbar
        m = 1.0,       # mass of particle
        k_2 = 1.0,        # harmonic potential mω^2
        τ = 4.22121,    # τ times period of oscillation is the final time
    );
    p = make_ε_x(p);
    p = make_ε_t(p);
    p = QuantumRecurrencePlots.make_harmonicPotential_π_k(p);

    # now consider all adimensional variables

    n = 6; # eigen state for size of grid
    x_max = fzero(x->exp(-x^2/2)*x^(n) - 0.001, sqrt(2*n-1));
    L = 4*x_max/sqrt(p.ε_t/p.ε_x^2);  # x_max gives the size scale, L must be greater than this  
    x = LinRange(-L/2, L/2 - L/p.N, p.N);

    n_h = 100;
    m_h = 50;
    x_h, w_h = gausshermite(n_h, normalize=true);
    a = QuantumRecurrencePlots.Hermite_hat(m_h);

    f(x) = QuantumRecurrencePlots.harmonicPotential(x, p);
    msp = MSP_matrix_1D(x_h, w_h, a, f);
    H = msp.P + msp.S * 1/2 * p.ε_x ^ 2 /p.ε_t; # we reescale the stiffnes matrix with the adimensional parameters 
    λ, ψ_coeffs = eigs(H, msp.M; nev=n-2, which=:LR,check=1,maxiter=1000,tol=1e-5, sigma=1e-3); # smallest

end
