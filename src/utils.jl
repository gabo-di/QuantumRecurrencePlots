#################
# General Stuff #
#################

"""
    δ = kronecker(n::Int, m::Int)

Delta kronecker
"""
function kronecker(n::Int, m::Int)
    if n==m
        return 1
    else
        return 0
    end
end

function makeParsFFT_1D(x, p)
    N = length(x)
    dx = x[2] - x[1]

    k_fft = 2pi/dx * fftfreq(N)
    P_fft = plan_fft(x)
    P_ifft = plan_ifft(x)

    p_ = Dict{Symbol,Any}()
    @pack! p_ = k_fft, P_fft, P_ifft
    return merge(p,p_)
end

#####################################
# Adimensional Schrodinger Equation # 
#####################################

"""
    p = make_ε_x(p)

calculates and saves the parameter ε_x in p
"""
function make_ε_x(p)
    @unpack m, E_0, L_0, ħ = p   
    ε_x = sqrt(ħ^2 / (m*L_0^2*E_0))

    p_ = Dict{Symbol,Any}()
    @pack! p_ = ε_x 
    return merge(p,p_)
end

"""
    p = make_ε_t(p)

calculates and saves the parameter ε_t in p
"""
function make_ε_t(p)
    @unpack T_0, E_0, ħ = p
    ε_t = ħ/(T_0*E_0)

    p_ = Dict{Symbol,Any}()
    @pack! p_ = ε_t 
    return merge(p,p_)
end

#########################
# For Circular Billiard #
#########################

jprime(ν,x) = 0.5*(besselj(ν - 1, x) - besselj(ν + 1, x))

function besseljprime_zeros(ν::Real, N::Int; step = 0.5, atol = 1e-12)
    ν ≥ 0 || error("Order ν must be ≥ 0")
    roots = Float64[]
    xL    = 1e-6              # skip trivial x=0 only for ν=0
    fL    = jprime(ν, xL)

    while length(roots) < N
        # march to the right until the sign changes
        xR = xL + step
        fR = jprime(ν, xR)

        # enlarge step adaptively if fL and fR have same sign too often
        tries = 0
        while sign(fL) == sign(fR)
            xL, xR = xR, xR + step
            fL, fR = fR, jprime(ν, xR)
            tries += 1
            tries < 50 || error("Unable to bracket next root — increase `step`")
        end

        # bracket found -> Brent
        root = find_zero((x)-> jprime(ν,x), (xL,xR), Order1(), atol=atol)
        push!(roots, root)


        # prepare next iteration
        xL, fL = root, jprime(ν, root + eps())   # move off the root slightly
    end
    return roots
end


###################################
# For Quantum Harmonic Oscillator #
###################################

"""
    ψ = coherent_state_1D(x, t, p)
    
Analytical solution for a coherent state of a harmonic oscillator on 1D.

Parameters:
- x: Spatial grid points
- t: Scalar Time
- p: parameter container with entries (adimensional units):
    - α: Complex coherent state parameter (α = |α|exp(iφ))
    - π_k: Squared Frequency of the harmonic oscillator
"""
function coherent_state_1D(x, t, p)
    @unpack α, π_k, ε_x, ε_t = p
    ω = sqrt(π_k)

    # Width parameter
    σ² = ε_x^2/(ε_t*ω)
    
    # Time-dependent parameters
    x_t = abs(α) * sqrt(2*ε_x^2 /(ε_t * ω)) * cos(angle(α) - ω*t)
    p_t = abs(α) * sqrt(2*ε_t*ω /ε_x^2) * sin(angle(α) - ω*t)
    
    # Phase factor
    ϕ_t = -ω*t/2
    
    # Construct the wavefunction
    prefactor = (1/(π*σ²))^(1/4)
    gaussian = exp.(-(x .- x_t).^2 ./ (2σ²))
    phase = exp.(im .* (p_t .* (x .- x_t/2) .+ ϕ_t))
    
    return prefactor .* gaussian .* phase
end

function make_harmonicPotential_π_k(p)
    @unpack m, T_0, L_0, k = p   
    π_k = k/m * T_0^2

    p_ = Dict{Symbol,Any}()
    @pack! p_ = π_k 
    return merge(p,p_)
end

function get_periodHarmonicPotential(p)
    @unpack π_k = p
    w_ = sqrt(π_k)
    return 2pi/w_
end

function harmonicPotential(x, p)
    @unpack π_k, ε_x, ε_t = p
    return 1/2 * π_k * ε_t / ε_x ^ 2 * x^2
end

function kineticEnergy(k_fft, p)
    @unpack ε_x, ε_t = p
    return 1/2 * ε_x ^ 2 /ε_t * k_fft^2
end

#############################
# For Quantum Free Particle #
#############################

function gaussian_state_1D(x, t, p)
    @unpack x_0, v_0, π_σ², ε_x, ε_t = p

    # Width parameter
    σ² = ε_x^2/(ε_t)
    
    # Time-dependent parameters
    x_t = x_0 + v_0*t
    
    # Phase factor
    ϕ_t = v_0^2*t/(2*σ²)
    
    # Construct the wavefunction
    prefactor = (1/(2π*σ²/π_σ²*(1 + im*t*π_σ²/2 )^2))^(1/4)
    gaussian = exp.(-(x .- x_t).^2 ./ (4σ²/π_σ² * (1 + im*t*π_σ²/2 ) ))
    phase = exp.(im .* (v_0 .* (x .- x_t)/σ² .+ ϕ_t))
    
    return prefactor .* gaussian .* phase
end

function make_freeParticle_pars(p)
    @unpack σ²_x, ε_x, ε_t = p
    x_0 = 0

    # Width parameter
    σ² = ε_x^2/(ε_t)

    π_σ² = σ² / σ²_x

    p_ = Dict{Symbol,Any}()
    @pack! p_ = π_σ², x_0 
    return merge(p,p_)
end
