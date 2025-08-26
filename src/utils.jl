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

function kineticEnergy(k_fft, p)
    @unpack ε_x, ε_t = p
    return 1/2 * ε_x ^ 2 /ε_t * k_fft^2
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


####################
# Time integration #
####################
function solve_schr_SSFM(x, t, p, ψ_initial, f)
    @unpack ε_x, ε_t = p
    @unpack k_fft, P_fft, P_ifft = p

    dt = t[2] - t[1]
    nt = length(t)
    
    # Potential energy 
    V = f.(x) 
    # V = 1/2 * π_k_2 * ε_t / ε_x ^ 2 .* x.^2 (harmonic oscillator)
    
    # Kinetic energy in Fourier space
    kin(x) = QuantumRecurrencePlots.kineticEnergy(x, p)
    T = kin.(k_fft)
    # T = ε_x ^ 2 /(2* ε_t) .* k_fft.^2
    
    # Define evolution operators
    exp_V = exp.(-im * V * dt/2)
    exp_T = exp.(-im * T * dt)
    
    # Initialize wavefunction
    ψ = copy(ψ_initial)
    
    # Normalize
    # ψ = ψ ./ sqrt(sum(abs2.(ψ))*dx)

    ψ_k = P_fft * ψ # buffer

    # Time evolution using split-step Fourier method
    for i in 1:(nt-1)
        # Apply half-step of potential
        ψ .= ψ .* exp_V
        
        # Apply full step of kinetic energy in Fourier space
        ψ_k .= P_fft * ψ
        ψ_k .= ψ_k .* exp_T
        ψ .= P_ifft *ψ_k
        
        # Apply second half-step of potential
        ψ .= ψ .* exp_V
    end
    
    return ψ
end

function solve_schr_SSFM_Yoshida(x, t, p, ψ_initial, f)
    @unpack ε_x, ε_t = p
    @unpack k_fft, P_fft, P_ifft = p

    dt = t[2] - t[1]
    dx = x[2] - x[1]
    nt = length(t)
    
    # Potential energy 
    V = f.(x) 
    # V = 1/2 * π_k_2 * ε_t / ε_x ^ 2 .* x.^2 (harmonic oscillator)
    
    # Kinetic energy in Fourier space
    kin(x) = QuantumRecurrencePlots.kineticEnergy(x, p)
    T = kin.(k_fft)
    # T = ε_x ^ 2 /(2* ε_t) .* k_fft.^2
    
    x_1 = 1 / (2 - cbrt(2))
    x_0 = -cbrt(2) / (2 - cbrt(2))


    # Define evolution operators
    exp_V_1 = exp.(-im * V * dt*x_1/2)
    exp_T_1 = exp.(-im * T * dt*x_1)
    exp_V_0 = exp.(-im * V * dt*x_0/2)
    exp_T_0 = exp.(-im * T * dt*x_0)
    
    # Initialize wavefunction
    ψ = copy(ψ_initial)
    
    # Normalize
    # ψ = ψ ./ sqrt(sum(abs2.(ψ))*dx)

    ψ_k = P_fft * ψ # buffer

    # Time evolution using split-step Fourier method
    for i in 1:(nt-1)
        # Apply half-step of potential x_1
        ψ .= ψ .* exp_V_1
        
        ψ_k .= P_fft * ψ
        ψ_k .= ψ_k .* exp_T_1
        ψ .= P_ifft *ψ_k
        
        ψ .= ψ .* exp_V_1


        # Apply half-step of potential x_0
        ψ .= ψ .* exp_V_0
        
        ψ_k .= P_fft * ψ
        ψ_k .= ψ_k .* exp_T_0
        ψ .= P_ifft *ψ_k
        
        ψ .= ψ .* exp_V_0

        # Apply half-step of potential x_1
        ψ .= ψ .* exp_V_1
        
        ψ_k .= P_fft * ψ
        ψ_k .= ψ_k .* exp_T_1
        ψ .= P_ifft *ψ_k
        
        ψ .= ψ .* exp_V_1

    end
    
    return ψ
end

function solve_schr_CrNi(msp, t, p, ψ_initial)
    @unpack M, S, P = msp
    @unpack ε_x, ε_t = p

    dt = t[2] - t[1]
    nt = length(t)
    H = P + S * 1/2 * p.ε_x ^ 2 /p.ε_t # we reescale the stiffnes matrix with the adimensional parameters 

    # precompute matrices for Crank-Nicolson
    A = factorize(M + im*dt/2*H)
    B = (M - im*dt/2*H)

    ψ_fem = copy(ψ_initial)

    for i in 1:(nt-1)
        ψ_fem .= A \ (B * ψ_fem)
    end
    return ψ_fem
end


#########################
# For Circular Billiard #
#########################

jprime(x,ν) = 0.5*(besselj(ν - 1, x) - besselj(ν + 1, x))

function besseljprime_zeros(ν::Real, N::Int; step = 0.5, atol = 1e-12)
    ν ≥ 0 || error("Order ν must be ≥ 0")
    roots = Float64[]
    xL    = 1e-6              # skip trivial x=0 only for ν=0
    fL    = jprime(xL, ν)
    prob = IntervalNonlinearProblem(jprime, (xL, xL+step), ν)

    while length(roots) < N
        # march to the right until the sign changes
        xR = xL + step
        fR = jprime(xR, ν)

        # enlarge step adaptively if fL and fR have same sign too often
        tries = 0
        while sign(fL) == sign(fR)
            xL, xR = xR, xR + step
            fL, fR = fR, jprime(xR, ν)
            tries += 1
            tries < 50 || error("Unable to bracket next root — increase `step`")
        end

        # bracket found -> Brent
        # root = find_zero((x)-> jprime(x, ν), (xL,xR), Order1(), atol=atol)
        root = NonlinearSolve.solve(remake(prob, tspan=(xL, xR)), ITP(); abstol = atol)[1]

        push!(roots, root)


        # prepare next iteration
        xL, fL = root, jprime(root + eps(), ν)   # move off the root slightly
    end
    return roots
end


###################################
# For Quantum Harmonic Oscillator #
###################################

"""
    ψ = harmonic_coherent_state_1D(x, t, p)
    
Analytical solution for a coherent state of a harmonic oscillator on 1D.

Parameters:
- x: Spatial grid points
- t: Scalar Time
- p: parameter container with entries (adimensional units):
    - α: Complex coherent state parameter (α = |α|exp(iφ))
    - π_k_2: Squared Frequency of the harmonic oscillator
"""
function harmonic_coherent_state_1D(x, t, p)
    @unpack α, π_k_2, ε_x, ε_t = p
    ω = sqrt(π_k_2)

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

function harmonic_eigen_state_sum_1D(x, t, c, p)
    @unpack π_k_2, ε_x, ε_t = p
    ω = sqrt(π_k_2)

    # number of eigenstates
    n = length(c)

    # Width parameter
    σ² = ε_x^2/(ε_t*ω)

    gaussian = exp.(-(x).^2 ./ (2σ²))

    # ...v sqrt(2) comes from He_n -> H_n
    pol = Hermite_hat(n)(sqrt(2/σ²) .* x)
    ss = zeros(ComplexF64, length(x))
    for i in eachindex(c)
        phase = exp( - im*t*(i-1/2)*ω ) # i starts in 1, but energy considers 0
        ss .+= pol[:,i] .* c[i] .* phase
    end


    prefactor = (1/(π*σ²)) ^ (1/4)

    return prefactor .* ss .* gaussian 
end

function harmonic_eigen_state_1D(x, t, n, p)
    # note eigen state is n = 0 1 2 3 so need to add 1 
    c = zeros(n+1)
    c[end] = 1
    harmonic_eigen_state_sum_1D(x, t, c, p)
end

function make_harmonicPotential_π_k(p)
    @unpack m, T_0, L_0, k_2 = p   
    π_k_2 = k_2/m * T_0^2

    p_ = Dict{Symbol,Any}()
    @pack! p_ = π_k_2 
    return merge(p,p_)
end

function get_periodHarmonicPotential(p)
    @unpack π_k_2 = p
    w_ = sqrt(π_k_2)
    return 2pi/w_
end

function harmonicPotential(x, p)
    @unpack π_k_2, ε_x, ε_t = p
    return 1/2 * π_k_2 * ε_t / ε_x ^ 2 * x^2
end


#############################
# For Quantum Free Particle #
#############################

function free_gaussian_state_1D(x, t, p)
    @unpack x_0, v_0, π_σ², ε_x, ε_t = p

    # Width parameter
    σ² = ε_x^2/(ε_t)
    
    # Time-dependent parameters
    x_t = v_0*t
    
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

    # Width parameter
    σ² = ε_x^2/(ε_t)

    π_σ² = σ² / σ²_x

    p_ = Dict{Symbol,Any}()
    @pack! p_ = π_σ² 
    return merge(p,p_)
end

#################################
# For Quantum Quartic Potential #
#################################

function quarticPotential(x, p)
    @unpack E_0, L_0, k_4, ε_t = p   
    π_k_4 = k_4 * L_0^4 / (E_0 * ε_t)  
    return 1/4 * π_k_4 * x^4
end




####################
# Linear regresion #
####################
# this comes from https://github.com/st--/LinearRegression.jl/tree/main  with the idea to optimze parts of it
# important is to note this post https://discourse.julialang.org/t/efficient-way-of-doing-linear-regression/31232/33 where they discuss the use of svd(X) \ y and consider tolerances
# see more here https://github.com/JuliaLang/LinearAlgebra.jl/pull/1387/files

function _append_bias_column(X)
    # avoids any conversions at this point
    ones_column = ones(eltype(X), size(X, 1))
    return [ones_column X]
end

function linregress(X, y, weights=nothing) 
    # Currently assumes that, if `X` is a Matrix, that `size(X) == (length(y),num_coefs)` (i.e., each row describes the features for one observation).

    size(X, 1) == size(y, 1) || throw(DimensionMismatch("size of X and y does not match in first dimension"))
    X = _append_bias_column(X)
    coeffs = if weights === nothing
        _lin_solve( X, y)
    else
        length(weights) == size(y, 1) || throw(DimensionMismatch("size of y and weights does not match in first dimension"))
        W = Diagonal(weights)
        _lin_solve( X, y, W)
    end
    return coeffs
end

function _lin_solve(X, y, W)
    # √W X \ √W y
    Wsqrt = sqrt(W)
    return _lin_solve(Wsqrt * X, Wsqrt * y)
end

function _lin_solve( X, y)
    return X \ y
end
