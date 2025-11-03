using DrWatson, Test
@quickactivate "QuantumRecurrencePlots"
using QuantumRecurrencePlots
using LinearAlgebra
using NonlinearSolve
using NonlinearSolveHomotopyContinuation
import HomotopyContinuation as HC

@testset "Find zeros" begin
    alg = HomotopyContinuationJL{true}(; threading = false, autodiff = false)
    T = Float64
    e = T[1 / 2, 1, 3 / 2, 2]
    s = diagm(ones(T, 4))
    @testset "First order zero" begin
        c = T[1, 1, 1, 1]
        fn = QuantumRecurrencePlots.make_nonlinearfunction_hermite_expansion_1D(c, e, s)
        prob = NonlinearProblem(fn, 0.0, [0.0])
        _, hcsys = NonlinearSolveHomotopyContinuation.homotopy_continuation_preprocessing(
            prob, alg)
        orig_sol = HC.solve(hcsys; alg.kwargs...)
        allsols = sort(HC.solutions(orig_sol); by = real)

        knownsols = [T[-2], T[0], T[1]]
        knownsols__ = [T[-1], T[0], T[2]]

        @testset "simple solve" begin
            @test isapprox(knownsols, allsols)
        end

        @testset "Homotopy continuation" begin
            _, hcsys_ = NonlinearSolveHomotopyContinuation.homotopy_continuation_preprocessing(
                remake(prob, p = [2pi / e[1]]), alg)
            H = HC.StraightLineHomotopy(hcsys, hcsys_)
            sol_ = HC.solve(H, allsols; alg.kwargs...)
            allsols_ = sort(HC.solutions(sol_); by = real)
            @test isapprox(knownsols, allsols_)

            H.target.p[1] = 2pi / e[2]
            sol__ = HC.solve(H, allsols; alg.kwargs...)
            allsols__ = sort(HC.solutions(sol__); by = real)
            @test isapprox(knownsols__, allsols__)
        end
    end

    @testset "Second order zero" begin
        c = [2, -2, 1, 0]
        fn = QuantumRecurrencePlots.make_nonlinearfunction_hermite_expansion_1D(c, e, s)
        prob = NonlinearProblem(fn, 0.0, [0.0])
        _, hcsys = NonlinearSolveHomotopyContinuation.homotopy_continuation_preprocessing(
            prob, alg)
        # note that not always finds the multiplicity of the solution
        # that is why we set the seed, but usually orig_sol.path_results does
        # work for this problem
        orig_sol = HC.solve(hcsys; alg.kwargs..., seed = 0x8cf9809c)
        allsols = sort(
            HC.solutions(orig_sol; only_nonsingular = false, multiple_results = true);
            by = real)

        knownsols = [T[1], T[1]]
        @testset "simple solve" begin
            @test isapprox(knownsols, allsols)
        end

        @testset "Homotopy continuation" begin
            # the continuation can not start from the singular value from before
            # so start a new problem
            _, hcsys_ = NonlinearSolveHomotopyContinuation.homotopy_continuation_preprocessing(
                remake(prob, p = [0.0001]), alg)
            sol_ = HC.solve(hcsys_, allsols; alg.kwargs...)
            allsols_ = sort(HC.solutions(sol_); by = real)
            @test isapprox(knownsols, allsols_)

            # now that is not singular make the continuation
            H = HC.StraightLineHomotopy(hcsys_, hcsys)
            H.target.p[1] = 0.001
            sol__ = HC.solve(H, allsols_; alg.kwargs...)
            allsols__ = sort(HC.solutions(sol__); by = real)
            @test isapprox(knownsols__, allsols__)
        end
    end
end
