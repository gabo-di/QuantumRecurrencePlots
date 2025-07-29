using BaryRational
using Random 
using CairoMakie

function main()
    n = 10
    x = randn(n)
    y = randn(n)
    z = complex.(x, y)

    z0 = 1.0 - 1.0im
    w = sqrt(im)
    p = 2
    println(p)
    f = z -> -w*(z -z0)^p
    fz = f.(z)

    r = aaa(z, fz; verbose=true)
    gridx = LinRange(-5, 5, 100)
    gridy = LinRange(-5, 5, 100)
    err = [abs(f(complex(x, y)) - r(complex(x, y))) for x in gridx, y in gridy]
    ff = [real(r(complex(x, y))) for x in gridx, y in gridy]
    println("\n")
    println(maximum(err), "max err")
    println(minimum(err), "min err")
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y", width=600, height=300)
    heatmap!(ax, gridx, gridy, ff)
    display(fig)
end



