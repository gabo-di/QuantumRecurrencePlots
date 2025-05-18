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

"""
    a = truncate_matrix(a, i=1, j=1)

Returns the matrix dropping the last i rows and j columns.
"""
function truncate_matrix(a, i=1, j=1)
    return a[1:end-i, 1:end-j]
end

"""
    p = make_ε_x(p)

calculates and saves the parameter ε_x in p
"""
function make_ε_x(p)
    @unpack m, E_0, L_0, ħ = p   
    ε_x = ħ^2 / (2*m*L_0^2*E_0)

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

jprime(ν,x) = 0.5*(besselj(ν - 1, x) - besselj(ν + 1, x))

function besseljprime_zeros(ν::Real, N::Int; step = 0.5, atol = 1e-12)
    ν ≥ 0 || error("Order ν must be ≥ 0")
    roots = Float64[]
    xL    = 1e-6              # skip trivial x=0 only for ν=0
    fL    = jprime(ν, xL)

    while length(roots) < N
        # march to the right until the sign changes
        xR = xL + step
        fR = jprime(ν, xR)

        # enlarge step adaptively if fL and fR have same sign too often
        tries = 0
        while sign(fL) == sign(fR)
            xL, xR = xR, xR + step
            fL, fR = fR, jprime(ν, xR)
            tries += 1
            tries < 50 || error("Unable to bracket next root — increase `step`")
        end

        # bracket found -> Brent
        root = find_zero((x)-> jprime(ν,x), (xL,xR), Order1(), atol=atol)
        push!(roots, root)


        # prepare next iteration
        xL, fL = root, jprime(ν, root + eps())   # move off the root slightly
    end
    return roots
end
