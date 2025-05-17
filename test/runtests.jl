using DrWatson, Test
@quickactivate "QuantumRecurrencePlots"

using FastGaussQuadrature
using LinearAlgebra

# Here you include files using `srcdir`
includet(srcdir("QuantumRecurrencePlots.jl"))

# Run test suite
println("Starting tests")
ti = time()

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
        # f(x) = exp(-x^2/2);
        # df(x) = -x*exp(-x^2/2);
        # m = 20;
        # x, w = gausshermite(m, normalize=true);
        # a = Hermite(m);
        # V = a(x);
        # A = derivative_polybasis(a);
        # dfx1 = (V' * Diagonal(w) * df.(x)) ./ ([a.f_norm(i,i) for i in 0:(m-1)]);
        # dfx2 = A * ((V' * Diagonal(w) * f.(x)) ./ ([a.f_norm(i,i) for i in 0:(m-1)]));
        # @test isapprox(dfx1, dfx2, atol=1e-6)
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
        @test isapprox(k.values, [0.5 + i for i in 0:(m-2)], atol=1e-8)
    end

    @testset "Hermite_hat solution" begin
        n = 100;
        m = 100;
        x, w = gausshermite(n, normalize=true);
        a = Hermite_hat(m);
        M, S, V = MSV_matrix_1D(x,w,a,x->x^2/4);
        k = eigen(truncate_matrix(S+V)); 
        @test isapprox(k.values, [0.5 + i for i in 0:(m-2)], atol=1e-8)

        # eigen values do not change but approximation for higher modes is bad
        k = eigen(truncate_matrix(S/2+2*V)); 
        ss = 30;
        @test isapprox(k.values[1:ss], [0.5 + i for i in 0:(ss-1)], atol=1e-8)


        # eigenvalues do change
        k = eigen(truncate_matrix(S/4+V)); 
        ss = 30;
        @test isapprox(k.values[1:ss], [0.25 + 0.5*i for i in 0:(ss-1)], atol=1e-8)
    end
end

@testset "Circular billiard" begin
    @testset "Dirichlet boundary" begin
        # parmeters for gmsh
        p_gmsh =  ( nθ = 400, # sampling resolution of boundary curve should be high enough for smooth CAD curve
                    l_min = 0.01, # min characteristic length
                    l_max = 0.02); # max characteristic length

        # parameters for billiard boundary
        # p_bill    = (R = 1.0, ε = 0.0);   # Robnik with ε=0.2  (ε=0 → circle)
        p_bill    = (a = 1.0, b = sqrt(0.51));   # Elliptic (a=1, b=1 → circle)
        p_bill    = (a = 1.0, b = 1.0);   # Elliptic (a=1, b=1 → circle)

        # parameters for gridap
        p_gridap = (order = 3, # cubic Lagrange elements
                    bc_type = :Dirichlet  # boundary type :Dirichlet  or :Neumann
                    );

        # gather parameters
        p = (p_gmsh = p_gmsh,
             p_bill = p_bill,
             p_gridap = p_gridap);

        # create the billiard mesh with gmsh
        rho = elliptic_billiard;
        name = "elliptic_billiard";
        gmsh_file = make_gmsh_billiard(rho, name, p);

        # read file and create FE space on gridap
        model, V = initialize_gridap(gmsh_file, p);

        # get the mass and stiffnes matrix
        M, S, _ = MSV_matrix_2D(V, model, p);

        # smaller eigenvalues 
        λ, ψ_coeffs = eigs(S/2, M; nev=10, which=:LR,check=1,maxiter=1000,tol=1e-6, sigma=1.0e-6) # smalle
        
        # theoretical values
        l = vcat(approx_besselroots(0,2) .^2 ./ 2, approx_besselroots(1,2) .^ 2 ./ 2, approx_besselroots(2,2) .^ 2 ./ 2, approx
        sort!(l)

        atol = p_gmsh.l_min
        
        @test isapprox(λ[1], l[1]; atol=atol)
        @test isapprox(λ[2], l[2]; atol=atol)
        @test isapprox(λ[3], l[2]; atol=atol)
        @test isapprox(λ[4], l[3]; atol=atol)
        @test isapprox(λ[5], l[3]; atol=atol)
        @test isapprox(λ[6], l[4]; atol=atol)
        # some times does not finde the degenerate eigenvalue
        @test isapprox(λ[7], l[5]; atol=atol)
    end

    @testset "Neumann boundary" begin
        # parmeters for gmsh
        p_gmsh =  ( nθ = 400, # sampling resolution of boundary curve should be high enough for smooth CAD curve
                    l_min = 0.01, # min characteristic length
                    l_max = 0.02); # max characteristic length

        # parameters for billiard boundary
        # p_bill    = (R = 1.0, ε = 0.0);   # Robnik with ε=0.2  (ε=0 → circle)
        p_bill    = (a = 1.0, b = sqrt(0.51));   # Elliptic (a=1, b=1 → circle)
        p_bill    = (a = 1.0, b = 1.0);   # Elliptic (a=1, b=1 → circle)

        # parameters for gridap
        p_gridap = (order = 3, # cubic Lagrange elements
                    bc_type = :Neumann  # boundary type :Dirichlet  or :Neumann
                    );

        # gather parameters
        p = (p_gmsh = p_gmsh,
             p_bill = p_bill,
             p_gridap = p_gridap);

        # create the billiard mesh with gmsh
        rho = elliptic_billiard;
        name = "elliptic_billiard";
        gmsh_file = make_gmsh_billiard(rho, name, p);

        # read file and create FE space on gridap
        model, V = initialize_gridap(gmsh_file, p);

        # get the mass and stiffnes matrix
        M, S, _ = MSV_matrix_2D(V, model, p);

        # smaller eigenvalues 
        λ, ψ_coeffs = eigs(S/2, M; nev=10, which=:LR,check=1,maxiter=1000,tol=1e-6, sigma=1.0e-6) # smalle
        
        # theoretical values
        l = vcat(approx_besselroots(0,2) .^2 ./ 2, approx_besselroots(1,2) .^ 2 ./ 2, approx_besselroots(2,2) .^ 2 ./ 2, approx
        sort!(l)

        atol = p_gmsh.l_min
        
        @test isapprox(λ[1], l[1]; atol=atol)
        @test isapprox(λ[2], l[2]; atol=atol)
        @test isapprox(λ[3], l[2]; atol=atol)
        @test isapprox(λ[4], l[3]; atol=atol)
        @test isapprox(λ[5], l[3]; atol=atol)
        @test isapprox(λ[6], l[4]; atol=atol)
        # some times does not finde the degenerate eigenvalue
        @test isapprox(λ[7], l[5]; atol=atol)
    end

end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")
