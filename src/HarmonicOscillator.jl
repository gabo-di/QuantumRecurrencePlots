function harmonicPotential_1D(u, p, x, t) 
    @unpack ε_t, ε_x, π_k = p
    return 1/4 * π_k * (ε_t^2/ε_x)^2 * x.^2
end

function kineticEnergyFFT_1D(u, p, x, t)
    @unpack ε_x, P_fft, P_ifft, k_fft = p
    return - ε_x^2 * P_ifft * ((-k_fft .^ 2) .* (P_fft * u))
end

function makeParsFFT_1D(x, p)
    N = length(x)
    dx = x[2] - x[1]
    # L = N * dx ~ 10 L_0

    k_fft = 2pi/dx * fftfreq(N)
    P_fft = plan_fft(x)
    P_ifft = plan_ifft(x)

    p_ = Dict{Symbol,Any}()
    @pack! p_ = k_fft, P_fft, P_ifft
    return merge(p,p_)
end

function make_harmonicPotential_π_k(p)
    @unpack m, T_0, L_0, k = p   
    π_k = k/(m * T_0^(-2))

    p_ = Dict{Symbol,Any}()
    @pack! p_ = π_k 
    return merge(p,p_)
end

