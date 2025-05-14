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
