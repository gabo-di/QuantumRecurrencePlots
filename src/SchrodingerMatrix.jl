function MSV_matrix_1D(x, w, a::PolyBasis{:hermite}, f=nothing)
    V = a(x);
    n = size(V,2)
    M = zeros(n,n)
    S = zeros(n,n)
    P = zeros(n,n)
    for i in 0:(n-1)
        for j in 0:(n-1)
            m = dot(w, V[:,i+1] .* V[:,j+1])

            s = 1/4*kronecker(j+1,i+1)*dot(w, V[:,i+1].*V[:,i+1])*(i+1) - j/4*kronecker(j-1,i+1)*dot(w, V[:,i+1].*V[:,i+1])*(i+1) - i/4*kronecker(j+1,i-1)*dot(w, V[:,j+1].*V[:,j+1])*(j+1)
            if i>1
                s += i*j/4*kronecker(j-1,i-1)*dot(w, V[:,i-1].*V[:,i-1])*(i-1)
            elseif i==1
                s += i*j/4*kronecker(j-1,i-1) 
            end

            if !isnothing(f)
                p = dot(w, V[:,i+1] .* V[:,j+1] .* f.(x))
                P[j+1,i+1] = p
                P[i+1,j+1] = p
            end

            S[j+1, i+1] = s
            S[i+1, j+1] = s

            M[j+1,i+1] = m
            M[i+1,j+1] = m
        end
    end
    return M, S, P
end

function MSV_matrix_1D(x, w, a::PolyBasis{:hermite_hat}, f=nothing)
    # no = [1/sqrt(a.f_norm(i,i)) for i in 0:(a.n-1)];
    # V = a(x) .* no';
    V = a(x);
    n = size(V,2)
    M = zeros(n,n)
    S = zeros(n,n)
    P = zeros(n,n)
    for i in 0:(n-1)
        for j in 0:(n-1)
            m = kronecker(i,j)

            s = i/4*kronecker(j-1,i-1) - sqrt(j*(j-1))/4*kronecker(j-1,i+1) - sqrt(i*(i-1))/4*kronecker(j+1,i-1) + (i+1)/4*kronecker(j+1,i+1)

            if !isnothing(f)
                p = dot(w, V[:,i+1] .* V[:,j+1] .* f.(x))
                P[j+1,i+1] = p
                P[i+1,j+1] = p
            end

            S[j+1, i+1] = s
            S[i+1, j+1] = s

            M[j+1,i+1] = m
            M[i+1,j+1] = m
        end
    end
    return M, S, P
end

function MSV_matrix_2D(V::Gridap.FESpaces.UnconstrainedFESpace, model::Gridap.Geometry.UnstructuredDiscreteModel, p, f=nothing)
    @unpack p_gridap = p
    @unpack order, bc_type = p_gridap

    # Define the integration domain and measure
    Ω = Triangulation(model)
    degree = 2*order                      # exact quadrature for mass/stiffness
    dΩ = Measure(Ω, degree)

    # Define the bilinear forms for mass and stiffness
    a_M(u,v) = ∫( u*v )*dΩ        # Mass matrix
    a_S(u,v) = ∫( ∇(u)⋅∇(v) )*dΩ  # Stiffness matrix

    # Assemble sparse mass (M) and stiffness (S) matrices
    U = TrialFESpace(V)  # Create trial space from the test spaces
    # Assemble the matrices
    M = assemble_matrix(a_M, U, V)  # Mass matrix
    S = assemble_matrix(a_S, U, V)  # Stiffness matri

    # take care of potential energy
    if !isnothing(f)
        p_M(u,v) = ∫( u*f*v )*dΩ        # Potential Energy matrix
        P = assemble_matrix(p_M, U, V)
    else
        P = nothing
    end

    return M, S, P
end

function robnik_billiard(θ, p)
    @unpack R, ε = p
    return R * (1 + ε*cos(θ))
end

function elliptic_billiard(θ, p)
    @unpack a, b = p
    s, c = sincos(θ)
    return a*b / sqrt((b*c)^2 + (a*s)^2)
end

function make_gmsh_billiard(rho, name, p)
    @unpack p_gmsh, p_bill = p
    @unpack nθ, l_min, l_max = p_gmsh # should be high enough for smooth CAD curve

    # ---------------------------------------------------------------------
    # Build the 2‑D CAD model in Gmsh
    # ---------------------------------------------------------------------
    Gmsh.initialize()
    Gmsh.gmsh.option.setNumber("General.Terminal", 1)      # verbose
    gmsh_model = Gmsh.gmsh.model
    gmsh_geo   = gmsh_model.geo
    gmsh_model.add("quantum_billiard")

    # a)  Sample boundary points in CAD *Cartesian* coordinates
    point_tags = Int[]
    for (i, θ) in enumerate(range(0, 2π, length=nθ+1)[1:end-1])  # exclude 2π duplicate
        r = rho(θ, p_bill)
        x, y = r*cos(θ), r*sin(θ)
        push!(point_tags, gmsh_geo.addPoint(x, y, 0.0, 0.0))     # mesh size 0.0 => auto
    end
    push!(point_tags, point_tags[1])     # close the curve

    # b)  Interpolate points with a periodic spline
    spline_tag  = gmsh_geo.addSpline(point_tags)

    # c)  Create a curve loop and a plane surface
    loop_tag    = gmsh_geo.addCurveLoop([spline_tag])
    surf_tag    = gmsh_geo.addPlaneSurface([loop_tag])

    # d)  Synchronise geometric kernel and set mesh options
    gmsh_geo.synchronize()

    # e) Add a physical group for the billiard wall (the spline curve)
    wall_tag = 1  # You can use any positive integer
    gmsh_model.addPhysicalGroup(1, [spline_tag], wall_tag)  # 1 = dimension (curve)
    gmsh_model.setPhysicalName(1, wall_tag, "billiard_wall")  # Give it a name

    # f) Also add a physical group for the surface (domain)
    domain_tag = 2
    gmsh_model.addPhysicalGroup(2, [surf_tag], domain_tag)  # 2 = dimension (surface)
    gmsh_model.setPhysicalName(2, domain_tag, "billiard_domain")

    # Optional: control element size
    Gmsh.gmsh.option.setNumber("Mesh.CharacteristicLengthMin", l_min)
    Gmsh.gmsh.option.setNumber("Mesh.CharacteristicLengthMax", l_max)

    # e)  Generate triangular 2‑D mesh and export to memory
    Gmsh.gmsh.model.mesh.generate(2)
    gmsh_file = datadir("billiards",name*".msh")
    Gmsh.gmsh.write(gmsh_file)
    Gmsh.finalize()

    return gmsh_file
end

function initialize_gridap(gmsh_file, p)
    @unpack p_gridap = p
    @unpack order, bc_type = p_gridap

    model = GmshDiscreteModel(gmsh_file)
    reffe = ReferenceFE(lagrangian, Float64, order)

    if bc_type == :Dirichlet
        V = TestFESpace(
              model, reffe;
              conformity=:H1,
              dirichlet_tags="billiard_wall")      # ψ = 0 on wall
    elseif bc_type == :Neumann
        V = TestFESpace(
              model, reffe;
              conformity=:H1)             # keep all DOFs
    else
        println("Bad boundary type ",:bc_type)
        V = nothing
    end

    return model, V
end
