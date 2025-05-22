using SafeTestsets

# Run test suite
println("Starting tests")
ti = time()

@safetestset "Polynomial Utils" begin include("PolynomialUtils_tests.jl") end

@safetestset "Gridap Billiard" begin include("GridapBilliard_tests.jl") end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")
