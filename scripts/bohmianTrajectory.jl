# bohmian trajectory
# Incomplete !!!
θ = π
mode_0_re = FEFunction(V, real(ψ_coeffs[:,1]))
mode_0_im = FEFunction(V, imag(ψ_coeffs[:,1]))
mode_1_re = FEFunction(V, real(ψ_coeffs[:,2] * exp(-im*θ)))
mode_1_im = FEFunction(V, imag(ψ_coeffs[:,2] * exp(-im*θ)))
ϕ_re(t) = (cos(t*real(λ[1]))*mode_0_re + cos(t*real(λ[2]))*mode_1_re -
            sin(t*real(λ[1]))*mode_0_im - sin(t*real(λ[2]))*mode_1_im)
ϕ_im(t) = (cos(t*real(λ[1]))*mode_0_im + cos(t*real(λ[2]))*mode_1_im +
            sin(t*real(λ[1]))*mode_0_re + sin(t*real(λ[2]))*mode_1_re)

function v_Bohm!(du, u, p, t)
    @unpack ϕ_re, ϕ_im = p
    x = Gridap.Point(u...)
    v = (ϕ_re(t)(x)*∇(ϕ_im(t))(x) - ϕ_im(t)(x)*∇(ϕ_re(t))(x)) ./ (ϕ_re(t)(x)^2 + ϕ_im(t)(x)^2)
    for i in eachindex(du)
        du[i] = v[i] 
    end
    return nothing
end
