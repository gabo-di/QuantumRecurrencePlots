using DrWatson
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using FFTW
using GLMakie
using LinearAlgebra
using Infiltrator
using Gridap
using FastGaussQuadrature
using NonlinearSolve
using Arpack
using NonlinearSolveHomotopyContinuation
import HomotopyContinuation as HC
using OrdinaryDiffEq

# harmonic oscillator
function main()
    p = (
        N = 1024,      # Number of grid points (higher for better accuracy)
        nt = 1000,     # number of time steps
        L_0 = 100.0,    # Length scale
        E_0 = 10.0,     # Energy scale
        T_0 = 14.0,     # Time scale
        ħ = 12.0,       # hbar
        m = 2.0,       # mass of particle
        k_2 = 13.0,        # harmonic potential mω^2
        nsols = 50          # number of times to find the zeros >=2
    )

    # prepare the adimensional parameters, not all scales are independent
    p = make_ε_x(p)
    p = make_ε_t(p)
    p = QuantumRecurrencePlots.make_harmonicPotential_π_k(p)

    # now consider all adimensional variables

    n_h = 100
    m_h = 100
    x_h, w_h = gausshermite(n_h, normalize = true)
    a = QuantumRecurrencePlots.Hermite(m_h)

    α = QuantumRecurrencePlots._harmonicPotential_xscale(p)
    f_(x) = QuantumRecurrencePlots.harmonicPotential(x, p) / α^2
    msp = MSP_matrix_1D(x_h, w_h, a, f_)
    # we reescale the stiffnes matrix with the adimensional parameters
    H = msp.P + msp.S * 1 / 2 * p.ε_x^2 / p.ε_t * α^2
    λ, ψ_coeffs = eigen(H, msp.M)

    in = 4
    c = [0, 4, 0, 1] #these are the coefficientes in eigenbasis
    c = [1, 1, 1, 1] #these are the coefficientes in eigenbasis
    # c = [2, 2.01, 1, 0] #these are the coefficientes in eigenbasis
    e_ = QuantumRecurrencePlots._harmonicPotential_Escale(p)
    e = λ[1:in] ./ e_
    function prepare_s(ψ_coeffs, in)
        # truncate the eigenbasis
        s = ψ_coeffs[1:(in * 2), 1:in]
        # normalize
        for i in axes(s, 2)
            s[:, i] ./= (norm(s[:, i]) .* sign(s[argmax(abs.(s[:, i])), i]))
        end
        # make sure to delete small contributions
        for i in eachindex(s)
            if abs(s[i]) < 1e-8
                s[i] = 0
            end
        end
        return s
    end
    s = prepare_s(ψ_coeffs, in)

    t0 = 0.0
    tmax = 2pi / e[1]
    dt = (tmax - t0) / (p.nsols - 1)
    seed = UInt32(42)
    allsols = []
    t = []

    alg = HomotopyContinuationJL{true}(; threading = false, autodiff = false)
    fn = QuantumRecurrencePlots.make_nonlinearfunction_hermite_expansion_1D(c, e, s)
    prob = NonlinearProblem(fn, 0.0, [t0])
    _, hcsys = NonlinearSolveHomotopyContinuation.homotopy_continuation_preprocessing(
        prob, alg)
    orig_sol = HC.solve(hcsys; alg.kwargs..., seed = seed)
    sol_0 = sort(HC.solutions(orig_sol); by = real)
    push!(allsols, sol_0)
    push!(t, t0)

    _, hcsys_ = NonlinearSolveHomotopyContinuation.homotopy_continuation_preprocessing(
        remake(prob, p = [dt]), alg)

    H = HC.StraightLineHomotopy(hcsys, hcsys_)

    sol_1 = HC.solutions(HC.solve(H, sol_0; alg.kwargs..., seed = seed))
    push!(allsols, sol_1)
    push!(t, dt)

    # for i in 1:(p.nsols - 2)
    #     H.target.p[1] = (1 + i) * dt
    #     push!(t, (1 + i) * dt)
    #     sol = HC.solve(H, sol_0; alg.kwargs..., seed = seed)
    #     push!(allsols, sort(HC.solutions(sol); by = real))
    # end

    for i in 1:(p.nsols - 2)
        H.target.p[1] = (1 + i) * dt
        H.start.p[1] = (i) * dt
        push!(t, (1 + i) * dt)
        sol = HC.solve(H, allsols[end]; alg.kwargs..., seed = seed)
        push!(allsols, HC.solutions(sol))
    end

    function poles_mov!(du, u, p, t)
        @unpack σ² = p
        for i in eachindex(du)
            du[i] = -u[i] / 2
            for j in eachindex(du)
                if i != j
                    du[i] += 1 / (u[i] - u[j])
                end
            end
            du[i] *= -im * σ² / 2
        end
    end
    tspan = (t0, dt * (p.nsols - 1))
    u0 = [x[1] for x in sol_0]
    pp = (σ² = 2 * p.ε_x^2 / p.ε_t * α^2 / e_,)
    prob_diff = ODEProblem(poles_mov!, u0, tspan, pp)
    sol_diff = OrdinaryDiffEq.solve(prob_diff, Vern7())

    ts = range(t[1], t[end], 1000)
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, ts, [x[1] for x in real.(sol_diff.(ts))])

    scatter!(ax, t, [x[1][1] for x in real.(allsols)])

    lines!(ax, ts, [x[2] for x in real.(sol_diff.(ts))])

    scatter!(ax, t, [x[2][1] for x in real.(allsols)])

    # lines!(ax, ts, [x[3] for x in real.(sol_diff.(ts))])

    if length(allsols[end]) == 3
        scatter!(ax, t, [x[3][1] for x in real.(allsols)])
    end

    display(fig)

    return allsols, sol_diff, fn
end

# quartic potential
function main_()
    p = (
        N = 1024,      # Number of grid points (higher for better accuracy)
        nt = 10000,     # number of time steps
        L_0 = 100.0,    # Length scale
        E_0 = 10.0,     # Energy scale
        T_0 = 14.0,     # Time scale
        ħ = 12.0,       # hbar
        m = 2.0,       # mass of particle
        k_4 = 13.0,        # harmonic potential mω^2
        τ = 4.22121,    # τ times period of oscillation is the final time
        nsols = 50          # number of times to find the zeros >=2
    )

    # prepare the adimensional parameters, not all scales are independent
    p = make_ε_x(p)
    p = make_ε_t(p)
    p = QuantumRecurrencePlots.make_quarticPotential_π_k(p)

    # now consider all adimensional variables

    n_h = 100
    m_h = 100
    x_h, w_h = gausshermite(n_h, normalize = true)
    a = QuantumRecurrencePlots.Hermite(m_h)

    α = QuantumRecurrencePlots._quarticPotential_xscale(p)
    f_(x) = QuantumRecurrencePlots.quarticPotential(x, p) / α^4
    msp = MSP_matrix_1D(x_h, w_h, a, f_)
    H = msp.P + msp.S * 1 / 2 * p.ε_x^2 / p.ε_t * α^2 # we reescale the stiffnes matrix with the adimensional parameters
    λ, ψ_coeffs = eigen(H, msp.M) # note that arpack gives awfull results when M is not the identity

    in = 8
    c = zeros(in)
    # c[round(Int, in / 2 + 1):in] .= 0 #last eigenvalues can have some errors
    c[1:3] .= 1
    e_ = QuantumRecurrencePlots._quarticPotential_Escale(p)
    e = λ[1:in] ./ e_
    @show diff(e)
    function prepare_s(ψ_coeffs, in)
        # truncate the eigenbasis
        s = ψ_coeffs[1:(in * 2), 1:in]
        # normalize
        for i in axes(s, 2)
            s[:, i] ./= (norm(s[:, i]) .* sign(s[argmax(abs.(s[:, i])), i]))
        end
        # make sure to delete small contributions
        for i in eachindex(s)
            if abs(s[i]) < 1e-12
                s[i] = 0
            end
        end
        return s
    end
    s = prepare_s(ψ_coeffs, in)

    t0 = 0.0
    tmax = 2pi / e[1]
    dt = (tmax - t0) / (p.nsols - 1) * 1
    seed = UInt32(42)
    allsols = []
    t = []

    alg = HomotopyContinuationJL{true}(; threading = false, autodiff = false)
    fn = QuantumRecurrencePlots.make_nonlinearfunction_hermite_expansion_1D(c, e, s)
    prob = NonlinearProblem(fn, 0.0, [t0])
    _, hcsys = NonlinearSolveHomotopyContinuation.homotopy_continuation_preprocessing(
        prob, alg)
    orig_sol = HC.solve(hcsys; alg.kwargs..., seed = seed)
    sol_0 = sort(HC.solutions(orig_sol); by = x -> abs2.(x))
    push!(allsols, sol_0)
    push!(t, t0)

    _, hcsys_ = NonlinearSolveHomotopyContinuation.homotopy_continuation_preprocessing(
        remake(prob, p = [dt]), alg)

    H = HC.StraightLineHomotopy(hcsys, hcsys_)

    sol_1 = HC.solutions(HC.solve(H, sol_0; alg.kwargs..., seed = seed))
    push!(allsols, sol_1)
    push!(t, dt)

    # for i in 1:(p.nsols - 2)
    #     H.target.p[1] = (1 + i) * dt
    #     push!(t, (1 + i) * dt)
    #     sol = HC.solve(H, sol_0; alg.kwargs..., seed = seed)
    #     push!(allsols, sort(HC.solutions(sol); by = real))
    # end

    for i in 1:(p.nsols - 2)
        H.target.p[1] = (1 + i) * dt
        H.start.p[1] = (i) * dt
        push!(t, (1 + i) * dt)
        sol = HC.solve(H, allsols[end]; alg.kwargs..., seed = seed)
        push!(allsols, HC.solutions(sol))
    end

    function poles_mov!(du, u, p, t)
        @unpack σ², fp_u = p
        for i in eachindex(du)
            du[i] = -u[i] / 2 + sum(1 ./ (u[i] .- fp_u))
            for j in eachindex(du)
                if i != j
                    du[i] += 1 / (u[i] - u[j])
                end
            end
            du[i] *= -im * σ² / 2
        end
    end
    tspan = (t0, dt * (p.nsols - 1))
    nm_p = 2
    u0 = [x[1] for x in sol_0[1:nm_p]]
    pp = (σ² = 2 * p.ε_x^2 / p.ε_t * α^2 / e_,
        fp_u = [x[1] for x in sol_0[(nm_p + 1):end]])
    prob_diff = ODEProblem(poles_mov!, u0, tspan, pp)
    sol_diff = OrdinaryDiffEq.solve(prob_diff, Vern7(), dtmax = dt / 10)

    ts = range(t[1], t[end], 1000)
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, ts, [x[1] for x in real.(sol_diff.(ts))])

    scatter!(ax, t, [x[1][1] for x in real.(allsols)])

    lines!(ax, ts, [x[2] for x in real.(sol_diff.(ts))])

    scatter!(ax, t, [x[2][1] for x in real.(allsols)])

    # lines!(ax, ts, [x[3] for x in real.(sol_diff.(ts))])

    scatter!(ax, t, [x[3][1] for x in real.(allsols)])

    # lines!(ax, ts, [x[4] for x in real.(sol_diff.(ts))])

    scatter!(ax, t, [x[4][1] for x in real.(allsols)])

    # lines!(ax, ts, [x[5] for x in real.(sol_diff.(ts))])

    scatter!(ax, t, [x[5][1] for x in real.(allsols)])

    # lines!(ax, ts, [x[6] for x in real.(sol_diff.(ts))])

    scatter!(ax, t, [x[6][1] for x in real.(allsols)])

    # lines!(ax, ts, [x[7] for x in real.(sol_diff.(ts))])

    scatter!(ax, t, [x[7][1] for x in real.(allsols)])

    display(fig)

    # it seems that zeros in HC depends a lot on the parameter in and s<small
    # and the evolution on poles_mov! is very unstable, probably because we do not
    # know the actual poles, or the actual exp(fx), we can see from c[1]=1 that should
    # not have poles but we find them, so maybe this hermite expansion is not the best one
    # or if it is, we are evolving grongly the state, maybe use Yoshida method instead of
    # eigensystem, because this has error, we have a test to show that, the problem is to track
    # the poles, in quartic potential, but maybe we can prove the equations on harmonic potential
    # well all works fine
    # USE SOME FIXED POLES ON QUARTIC

    return allsols, sol_diff, fn
end
