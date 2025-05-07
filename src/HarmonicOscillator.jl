using UnPack
using FFTW

function harmonicPotential_1D(u, p, x, t) 
    @unpack k = p
    return 1/2 * k * x.^2
end

function kineticEnergyFFT_1D(u, p, x, t)
    @unpack ħ, m, P_fft, P_ifft, k_fft = p
    return - ħ^2/m * P_ifft * ((-k_fft .^ 2) .* (P_fft * u))
end

function makeParsFFT_1D(x, p)
    N = length(x)
    dx = x[2] - x[1]
    L = N * dx

    k_fft = 2pi/dx * fftfreq(N)
    P_fft = plan_fft(x)
    P_ifft = plan_ifft(x)

    p_ = Dict{Symbol,Any}()
    @pack! p_ = k_fft, P_fft, P_ifft
    return merge(p,p_)
end
