using DrWatson, Test
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using LinearAlgebra
using Gridap
using Arpack
using FastGaussQuadrature, SpecialFunctions

@testset "Circular billiard" begin
    @testset "Make circular billiard" begin
        # parmeters for gmsh
        p_gmsh =  ( nθ = 400, # sampling resolution of boundary curve should be high enough for smooth CAD curve
                    l_min = 0.01, # min characteristic length
                    l_max = 0.02); # max characteristic length

        # parameters for billiard boundary
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
        name = "circular_billiard";
        gmsh_file = make_gmsh_billiard(rho, name, p);
        @test isequal(gmsh_file, datadir("billiards", "circular_billiard.msh"))
    end

    @testset "Dirichlet boundary eigensystem Float64" begin
        # parmeters for gmsh
        p_gmsh =  ( nθ = 400, # sampling resolution of boundary curve should be high enough for smooth CAD curve
                    l_min = 0.01, # min characteristic length
                    l_max = 0.02); # max characteristic length

        # parameters for billiard boundary
        p_bill    = (a = 1.0, b = 1.0);   # Elliptic (a=1, b=1 → circle)

        # parameters for gridap
        p_gridap = (order = 3, # cubic Lagrange elements
                    bc_type = :Dirichlet  # boundary type :Dirichlet  or :Neumann
                    );

        # gather parameters
        p = (p_gmsh = p_gmsh,
             p_bill = p_bill,
             p_gridap = p_gridap);

        gmsh_file = datadir("billiards", "circular_billiard.msh")

        # read file and create FE space on gridap
        model, V = initialize_gridap(gmsh_file, p, Float64);

        # get the mass and stiffnes matrix
        @unpack M, S = MSP_matrix_2D(V, model, p);

        # smaller eigenvalues 
        λ, ψ_coeffs = eigs(S/2, M; nev=10, which=:LR,check=1,maxiter=1000,tol=1e-6, sigma=1.0e-6) # smalle
        
        # theoretical values
        l = vcat([approx_besselroots(i,2) .^2 ./2 for i in 0:3]...)
        sort!(l)

        atol = p_gmsh.l_min
        
        @testset "eigenvalues" begin
            @test isapprox(λ[1], l[1]; atol=atol)
            @test isapprox(λ[2], l[2]; atol=atol)
            @test isapprox(λ[3], l[2]; atol=atol)
            @test isapprox(λ[4], l[3]; atol=atol)
            @test isapprox(λ[5], l[3]; atol=atol)
            @test isapprox(λ[6], l[4]; atol=atol)
            # some times does not finde the degenerate eigenvalue
            @test isapprox(λ[7], l[5]; atol=atol)
        end
        
        # eigenvectors
        l0 = sqrt(2*l[1])
        ψ₀(p) = besselj0(sqrt(p[1]^2 + p[2]^2)*l0)
        ψ0_fe = interpolate(p->real(ψ₀(p)), V)
        ψ0 = get_free_dof_values(ψ0_fe)
        ψ0 = ψ0 / norm(ψ0)
        ψ0_ = real(ψ_coeffs[:,1])
        ψ0_ = ψ0_ / norm(ψ0_)
        
        l1 = sqrt(2*l[2])
        ψ₁(p) = besselj1(sqrt(p[1]^2 + p[2]^2)*l1)*p[1]/sqrt(p[1]^2 + p[2]^2)
        ψ1_fe = interpolate(p->real(ψ₁(p)), V)
        ψ1 = get_free_dof_values(ψ1_fe)
        ψ1 = ψ1 / norm(ψ1)
        ψ1_ = real(ψ_coeffs[:,2])
        ψ1_ = ψ1_ / norm(ψ1_)

        @testset "eigenvectors" begin
            @test isapprox(abs(dot(ψ0_, ψ0)), 1; atol=1e-8)
            @test isapprox(abs(dot(ψ1_, ψ1)), 1; atol=1e-8)
        end
    end

    @testset "Neumann boundary eigensystem Float64" begin
        # parmeters for gmsh
        p_gmsh =  ( nθ = 400, # sampling resolution of boundary curve should be high enough for smooth CAD curve
                    l_min = 0.01, # min characteristic length
                    l_max = 0.02); # max characteristic length

        # parameters for billiard boundary
        p_bill    = (a = 1.0, b = 1.0);   # Elliptic (a=1, b=1 → circle)

        # parameters for gridap
        p_gridap = (order = 3, # cubic Lagrange elements
                    bc_type = :Neumann  # boundary type :Dirichlet  or :Neumann
                    );

        # gather parameters
        p = (p_gmsh = p_gmsh,
             p_bill = p_bill,
             p_gridap = p_gridap);

        gmsh_file = datadir("billiards", "circular_billiard.msh")

        # read file and create FE space on gridap
        model, V = initialize_gridap(gmsh_file, p, Float64);

        # get the mass and stiffnes matrix
        @unpack M, S = MSP_matrix_2D(V, model, p);

        # smaller eigenvalues 
        λ, ψ_coeffs = eigs(S/2, M; nev=10, which=:LR,check=1,maxiter=1000,tol=1e-6, sigma=1.0e-6) # smalle
        
        # theoretical values
        l = vcat([QuantumRecurrencePlots.besseljprime_zeros(i,2) .^2 ./2 for i in 0:5]...);
        sort!(l)

        atol = p_gmsh.l_min
        
        @testset "eigenvalues" begin
            @test isapprox(λ[1], l[1]; atol=atol)
            @test isapprox(λ[2], l[1]; atol=atol)
            @test isapprox(λ[3], l[2]; atol=atol)
            @test isapprox(λ[4], l[3]; atol=atol)
            @test isapprox(λ[5], l[4]; atol=atol)
            @test isapprox(λ[6], l[5]; atol=atol)
            @test isapprox(λ[7], l[6]; atol=atol)
        end
        
        # eigenvectors
        l5 = sqrt(2*l[4]);
        ψ₀(p) = besselj0(sqrt(p[1]^2 + p[2]^2)*l5);
        ψ5_fe = interpolate(p->real(ψ₀(p)), V);
        ψ5 = get_free_dof_values(ψ5_fe);
        ψ5 = ψ5 / norm(ψ5);
        ψ5_ = real(ψ_coeffs[:,5]);
        ψ5_ = ψ5_ / norm(ψ5_);

        @testset "eigenvectors" begin
            @test isapprox(abs(dot(ψ5_, ψ5)), 1; atol=1e-8)
        end
    end

    @testset "Dirichlet boundary eigensystem ComplexF64" begin
        # parmeters for gmsh
        p_gmsh =  ( nθ = 400, # sampling resolution of boundary curve should be high enough for smooth CAD curve
                    l_min = 0.01, # min characteristic length
                    l_max = 0.02); # max characteristic length

        # parameters for billiard boundary
        p_bill    = (a = 1.0, b = 1.0);   # Elliptic (a=1, b=1 → circle)

        # parameters for gridap
        p_gridap = (order = 3, # cubic Lagrange elements
                    bc_type = :Dirichlet  # boundary type :Dirichlet  or :Neumann
                    );

        # gather parameters
        p = (p_gmsh = p_gmsh,
             p_bill = p_bill,
             p_gridap = p_gridap);

        gmsh_file = datadir("billiards", "circular_billiard.msh")

        # read file and create FE space on gridap
        model, V = initialize_gridap(gmsh_file, p, ComplexF64);

        # get the mass and stiffnes matrix
        @unpack M, S = MSP_matrix_2D(V, model, p);

        # smaller eigenvalues 
        λ, ψ_coeffs = eigs(S/2, M; nev=10, which=:LR,check=1,maxiter=1000,tol=1e-6, sigma=1.0e-6) # smalle
        
        # theoretical values
        l = vcat([approx_besselroots(i,2) .^2 ./2 for i in 0:3]...)
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

    @testset "Neumann boundary eigensystem ComplexF64" begin
        # parmeters for gmsh
        p_gmsh =  ( nθ = 400, # sampling resolution of boundary curve should be high enough for smooth CAD curve
                    l_min = 0.01, # min characteristic length
                    l_max = 0.02); # max characteristic length

        # parameters for billiard boundary
        p_bill    = (a = 1.0, b = 1.0);   # Elliptic (a=1, b=1 → circle)

        # parameters for gridap
        p_gridap = (order = 3, # cubic Lagrange elements
                    bc_type = :Neumann  # boundary type :Dirichlet  or :Neumann
                    );

        # gather parameters
        p = (p_gmsh = p_gmsh,
             p_bill = p_bill,
             p_gridap = p_gridap);

        gmsh_file = datadir("billiards", "circular_billiard.msh")

        # read file and create FE space on gridap
        model, V = initialize_gridap(gmsh_file, p, ComplexF64);

        # get the mass and stiffnes matrix
        @unpack M, S = MSP_matrix_2D(V, model, p);

        # smaller eigenvalues 
        λ, ψ_coeffs = eigs(S/2, M; nev=10, which=:LR,check=1,maxiter=1000,tol=1e-6, sigma=1.0e-6) # smalle
        
        # theoretical values
        l = vcat([QuantumRecurrencePlots.besseljprime_zeros(i,2) .^2 ./2 for i in 0:5]...);
        sort!(l)

        atol = p_gmsh.l_min
        
        @test isapprox(λ[1], l[1]; atol=atol)
        @test isapprox(λ[2], l[1]; atol=atol)
        @test isapprox(λ[3], l[2]; atol=atol)
        @test isapprox(λ[4], l[3]; atol=atol)
        @test isapprox(λ[5], l[4]; atol=atol)
        @test isapprox(λ[6], l[5]; atol=atol)
        @test isapprox(λ[7], l[6]; atol=atol)
    end
end
