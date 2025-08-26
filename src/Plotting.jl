using Makie

function plot_comparison_1D!(fig, x, tmax, ψ_numerical, ψ_analytical)
    # fig = Figure(size=(1200, 800))
    
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
    
    Label(fig[0, 1:2], "Solution after t = $(round(tmax, digits=3)) (Error: $(round(norm_error, digits=6)))",
          fontsize=20)
    
    return nothing
    # return fig
end


function to_plot_femfunctions_1D(x, V, ψ_coeffs, M)
    # Create a fine grid for plotting
    n_plot_points = length(x)

    
    # Normalize eigenfunction
    ψ_i = ψ_coeffs[:]
    ψ_i = ψ_i / sqrt(abs(ψ_i' * M * ψ_i))  # Normalize with respect to mass matrix
   
    # Create FE function
    ψ_fem = FEFunction(V, ψ_i)
   
    # Evaluate at the plotting points
    T = eltype(ψ_coeffs) 
    ψ_values = zeros(T, n_plot_points)
    for (j, x) in enumerate(x)
        ψ_values[j] = ψ_fem(Gridap.Point(x))
    end
   
    return ψ_values
end
