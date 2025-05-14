using DrWatson, Test
@quickactivate "QuantumRecurrencePlots"

using QuantumRecurrencePlots
using FastGaussQuadrature

# Here you include files using `srcdir`
# include(srcdir("file.jl"))

# Run test suite
println("Starting tests")
ti = time()

@testset "QuantumRecurrencePlots tests" begin
    @test 1==1 
end

@testset "Polynomial basis tests" begin
    n = 5
    
    @testset "Gauss Hermite" begin
        x, w = gausshermite(n, normalize=true);
        a = Hermite(n);
        V = a(x);
        A = V' * Diagonal(w) * V;
        B = Diagonal([a.f_norm(i,i) for i in 0:(n-1)]);
        @test isapprox(A, B, atol=1e-10)
        
        # coef = ((V' * Diagonal(w) * f.(x)) ./ ([a.f_norm(i,i) for i in 0:(n-1)]))
        f(x) = exp(-x^2/2);
        df(x) = -x*exp(-x^2/2);
        m = 20;
        x, w = gausshermite(m, normalize=true);
        a = Hermite(m);
        V = a(x);
        A = derivative_polybasis(a);
        dfx1 = (V' * Diagonal(w) * df.(x)) ./ ([a.f_norm(i,i) for i in 0:(m-1)]);
        dfx2 = A * ((V' * Diagonal(w) * f.(x)) ./ ([a.f_norm(i,i) for i in 0:(m-1)]));
        @test isapprox(dfx1, dfx2, atol=1e-6)
    end

    @testset "Gauss Laguerre" begin
        x, w = gausslaguerre(n);
        a = Laguerre(n);
        V = a(x);
        A = V' * Diagonal(w) * V;
        B = Diagonal([a.f_norm(i,i) for i in 0:(n-1)]);
        @test isapprox(A, B, atol=1e-10)
    end

    @testset "Gauss Legendre" begin
        x, w = gausslegendre(n);
        a = Legendre(n);
        V = a(x);
        A = V' * Diagonal(w) * V;
        B = Diagonal([a.f_norm(i,i) for i in 0:(n-1)]);
        @test isapprox(A, B, atol=1e-10)
    end

    @testset "Gauss Chebyshev" begin
        x, w = gausschebyshev(n);
        a = Chebyshev(n);
        V = a(x);
        A = V' * Diagonal(w) * V;
        B = Diagonal([a.f_norm(i,i) for i in 0:(n-1)]);
        @test isapprox(A, B, atol=1e-10)
    end

    @testset "Gauss Hermite_hat" begin
        x, w = gausshermite(n, normalize=true);
        a = Hermite_hat(n);
        V = a(x);
        A = V' * Diagonal(w) * V;
        B = Diagonal([a.f_norm(i,i) for i in 0:(n-1)]);
        @test isapprox(A, B, atol=1e-10)
    end
end

@testset "Quantum Harmonic Oscillator tests" begin
    @testset "Hermite solution" begin
        n = 100;
        m = 100;
        x, w = gausshermite(n, normalize=true);
        a = Hermite(m);
        M, S, V = MSV_matrix_1D(x,w,a,x->x^2/4);
        k = eigen(truncate_matrix(S+V), truncate_matrix(M));    
        isapprox(k.values, [0.5 + i for i in 0:(m-2)], atol=1e-8)
    end

    @testset "Hermite_hat solution" begin
        n = 100;
        m = 100;
        x, w = gausshermite(n, normalize=true);
        a = Hermite_hat(m);
        M, S, V = MSV_matrix_1D(x,w,a,x->x^2/4);
        k = eigen(truncate_matrix(S+V)); 
        isapprox(k.values, [1.25 + 1.75*i for i in 0:(m-2)], atol=1e-8)
    end
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")
