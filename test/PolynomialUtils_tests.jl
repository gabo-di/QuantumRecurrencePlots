using DrWatson, Test
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using FastGaussQuadrature
using LinearAlgebra

@testset "Polynomial basis tests" begin
    n = 5

    @testset "Gauss Hermite" begin
        x, w = gausshermite(n, normalize = true)
        a = Hermite(n)
        V = a(x)
        A = V' * Diagonal(w) * V
        B = Diagonal([a.f_norm(i, i) for i in 0:(n - 1)])
        @test isapprox(A, B, atol = 1e-10)
    end

    @testset "Gauss Hermite derivative" begin
        f(x) = cos(-x^2 / 2) * exp(-x^2 / 2)
        df(x) = x * (sin(-x^2 / 2) - cos(-x^2 / 2)) * exp(-x^2 / 2)
        m = 32
        x, w = gausshermite(m, normalize = true)
        a = Hermite(m)
        V = a(x)
        A = QuantumRecurrencePlots.derivative_polybasis(a)

        # for this function does not work a.f_norm because we need m>20 so use diag(V' * Diagonal(w) * V)
        dfx1 = (V' * Diagonal(w) * df.(x)) ./ ([a.f_norm(i, i) for i in 0:(m - 1)])
        dfx2 = A * ((V' * Diagonal(w) * f.(x)) ./ ([a.f_norm(i, i) for i in 0:(m - 1)]))
        @test isapprox(dfx1, dfx2, atol = 1e-7)
    end

    @testset "Gauss Laguerre" begin
        x, w = gausslaguerre(n)
        a = Laguerre(n)
        V = a(x)
        A = V' * Diagonal(w) * V
        B = Diagonal([a.f_norm(i, i) for i in 0:(n - 1)])
        @test isapprox(A, B, atol = 1e-10)
    end

    @testset "Gauss Legendre" begin
        x, w = gausslegendre(n)
        a = Legendre(n)
        V = a(x)
        A = V' * Diagonal(w) * V
        B = Diagonal([a.f_norm(i, i) for i in 0:(n - 1)])
        @test isapprox(A, B, atol = 1e-10)
    end

    @testset "Gauss Chebyshev" begin
        x, w = gausschebyshev(n)
        a = Chebyshev(n)
        V = a(x)
        A = V' * Diagonal(w) * V
        B = Diagonal([a.f_norm(i, i) for i in 0:(n - 1)])
        @test isapprox(A, B, atol = 1e-10)
    end

    @testset "Gauss Hermite_hat" begin
        x, w = gausshermite(n, normalize = true)
        a = Hermite_hat(n)
        V = a(x)
        A = V' * Diagonal(w) * V
        B = Diagonal([a.f_norm(i, i) for i in 0:(n - 1)])
        @test isapprox(A, B, atol = 1e-10)
    end

    @testset "Gauss Hermite_hat derivative" begin
        f(x) = cos(-x^2 / 2) * exp(-x^2 / 2)
        df(x) = x * (sin(-x^2 / 2) - cos(-x^2 / 2)) * exp(-x^2 / 2)
        m = 90 # note that we need much more terms to achieve the same convergence as Hermite
        x, w = gausshermite(m, normalize = true)
        a = Hermite_hat(m)
        V = a(x)
        A = QuantumRecurrencePlots.derivative_polybasis(a)

        dfx1 = (V' * Diagonal(w) * df.(x))
        dfx2 = A * ((V' * Diagonal(w) * f.(x)))
        maximum(abs.(dfx1 - dfx2))
        @test isapprox(dfx1, dfx2, atol = 1e-8)
    end
end

@testset "Finding Quantum Harmonic Oscillator eigensystem tests" begin
    @testset "Hermite solution" begin
        n = 100
        m = 100
        x, w = gausshermite(n, normalize = true)
        a = Hermite(m)
        @unpack M, S, P = MSP_matrix_1D(x, w, a, x -> x^2 / 4)
        k = eigen(S + P, M)
        @test isapprox(k.values, [0.5 + i for i in 0:(m - 2)], atol = 1e-8)
    end

    @testset "Hermite_hat solution" begin
        n = 100
        m = 100
        x, w = gausshermite(n, normalize = true)
        a = Hermite_hat(m)
        @unpack M, S, P = MSP_matrix_1D(x, w, a, x -> x^2 / 4)
        k = eigen(S + P)
        @test isapprox(k.values, [0.5 + i for i in 0:(m - 2)], atol = 1e-8)

        # eigen values do not change but approximation for higher modes is bad
        k = eigen(S / 2 + 2 * P)
        ss = 30
        @test isapprox(k.values[1:ss], [0.5 + i for i in 0:(ss - 1)], atol = 1e-8)

        # eigenvalues do change
        k = eigen(S / 4 + P)
        ss = 30
        @test isapprox(k.values[1:ss], [0.25 + 0.5 * i for i in 0:(ss - 1)], atol = 1e-8)
    end
end
