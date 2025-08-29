using DrWatson
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using LinearAlgebra
using Arpack
using NonlinearSolve
using FastGaussQuadrature

@testset "Eigen states" begin
    p_1 = (
        N = 1024,      # Number of grid points (higher for better accuracy)
        nt = 10000,     # number of time steps 
        L_0 = 100.0,    # Length scale
        E_0 = 10.0,     # Energy scale
        T_0 = 14.0,     # Time scale
        ħ = 12.0,       # hbar
        m = 2.0,       # mass of particle
        k_4 = 13.0,        # harmonic potential mω^2
        τ = 4.22121,    # τ times period of oscillation is the final time
    );

    p_1 = make_ε_x(p_1);
    p_1 = make_ε_t(p_1);
    p_1 = QuantumRecurrencePlots.make_quarticPotential_π_k(p_1);

    # different parameter set
    p_2 = (
        N = 1024,      # Number of grid points (higher for better accuracy)
        nt = 10000,     # number of time steps 
        L_0 = 1.0,    # Length scale
        E_0 = 1.0,     # Energy scale
        T_0 = 1.0,     # Time scale
        ħ = 1.0,       # hbar
        m = 1.0,       # mass of particle
        k_4 = 1.0,        # harmonic potential mω^2
        τ = 4.22121,    # τ times period of oscillation is the final time
    );

    p_2 = make_ε_x(p_2);
    p_2 = make_ε_t(p_2);
    p_2 = QuantumRecurrencePlots.make_quarticPotential_π_k(p_2);
    # now consider all adimensional variables
    
    @testset "Eigen energy scale" begin
        n_h = 100;
        m_h = 100;
        x_h, w_h = gausshermite(n_h, normalize=true);
        a = QuantumRecurrencePlots.Hermite_hat(m_h);

        α_1 = QuantumRecurrencePlots._quarticPotential_xscale(p_1);
        f_1(x) = QuantumRecurrencePlots.quarticPotential(x, p_1)/α_1^4;
        msp_1 = MSP_matrix_1D(x_h, w_h, a, f_1);
        H_1 = msp_1.P + msp_1.S * 1/2 * p_1.ε_x ^ 2 /p_1.ε_t * α_1^2; # we reescale the stiffnes matrix with the adimensional parameters 
        λ_1, ψ_coeffs_1 = eigs(H_1, msp_1.M; nev=round(Int,m_h/3), which=:LR,check=1,maxiter=1000,tol=1e-5, sigma=1e-3); # smallest

        α_2 = QuantumRecurrencePlots._quarticPotential_xscale(p_2);
        f_2(x) = QuantumRecurrencePlots.quarticPotential(x, p_2)/α_2^4;
        msp_2 = MSP_matrix_1D(x_h, w_h, a, f_2);
        H_2 = msp_2.P + msp_2.S * 1/2 * p_2.ε_x ^ 2 /p_2.ε_t * α_2^2; # we reescale the stiffnes matrix with the adimensional parameters 
        λ_2, ψ_coeffs_2 = eigs(H_2, msp_2.M; nev=round(Int,m_h/3), which=:LR,check=1,maxiter=1000,tol=1e-5, sigma=1e-3); # smallest

        @test isapprox(λ_1 ./ QuantumRecurrencePlots._quarticPotential_Escale(p_1),
                        λ_2 ./ QuantumRecurrencePlots._quarticPotential_Escale(p_2); atol=1e-6) # some times atol=1e-8 does not work is strange since we do not use random numbers
    end
end
