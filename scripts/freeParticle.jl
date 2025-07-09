using DrWatson
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using CairoMakie
using LinearAlgebra
using Infiltrator


#########
# utils #
#########

function plot_mean_observables(ψ_analytical)
    return nothing
end


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
    
    Label(fig[0, 1:2], "Free Particle Gaussian State: After t = $(round(tmax, digits=3)) (Error: $(round(norm_error, digits=6)))",
          fontsize=20)
    
    return fig
end

#############################################
# many particles in free particle evolution #
#############################################

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
        σ²_x = 1.0,     # width of initial distribution (does not have units)
        nt = 10,     # number of time steps 
        N = 2048*2,      # Number of grid points (higher for better accuracy)
        τ = 2.22121,    # τ is the final time
    );

    # prepare the adimensional parameters, not all scales are independent
    p = make_ε_x(p);
    p = make_ε_t(p);
    p = QuantumRecurrencePlots.make_freeParticle_pars(p);

    # now consider all adimensional parameters
    
    # prepare grid
    xmax = p.x_0 + p.v_0 * p.τ  # maximum classical displacement
    σ² = p.ε_x^2/p.ε_t
    σ_0 = sqrt( σ²/p.π_σ²)
    σ_max = sqrt(σ²/p.π_σ²* ( 1 + p.τ^2*p.π_σ² /2 ))
    mini = min(-6*σ_0, xmax - 6 * σ_max)
    maxi = max(6*σ_0, xmax + 6 * σ_max )
    x = LinRange(mini, maxi, p.N)
    
    # prepare parameters for FFT
    p = QuantumRecurrencePlots.makeParsFFT_1D(x, p)

    # Generate initial gaussian state
    ψ_initial = QuantumRecurrencePlots.gaussian_state_1D(x, 0.0, p)

    # solve in Fourier space
    ψ = copy(ψ_initial)

    # Kinetic energy in Fourier space
    kin(x) = QuantumRecurrencePlots.kineticEnergy(x, p)
    T = kin.(p.k_fft)
    exp_T = exp.(-im * T * p.τ)

    # time evolution
    ψ_k = p.P_fft * ψ
    ψ_k = ψ_k .* exp_T
    ψ = p.P_ifft *ψ_k

    # Compute analytical solution at final time
    ψ_analytical = QuantumRecurrencePlots.gaussian_state_1D(x, p.τ, p)

    # Generate and save the plot
    fig = plot_comparison(x, p.τ, ψ, ψ_analytical)

    # Display the figure if running interactively
    display(fig)
end


