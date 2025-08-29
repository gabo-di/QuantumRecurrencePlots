"""
    PolyBasis{name, F0, F1, A0, A1, A2, FW, FN}

saves necessary data to make a polynomial of a given basis
"""
struct PolyBasis{name, F0, F1, A0, A1, A2, FW, FN}
    n::Int # how many polys
    f0::F0 # poly 0
    f1::F1 # poly 1
    # we generate the following polynomials considering a2(n,x)*P_{n+1}(x) = a1(n,x)*P_{n}(x) + a0(n,x)*P_{n-1}  
    a0::A0  
    a1::A1  
    a2::A2  
    # maybe add weight function interval and normalization??
    f_weight::FW # weight function on normalization
    f_norm::FN # normalization
    # \int_{x0}^{x1} P_{n}(x) P_{m}(x) f_weight(x) dx = f_norm(n,m)  

end


"""
    y = (basis::PolyBasis)(x::AbstractArray{T}) where {T}

applies each basis element to each element of x
y is of size (size(x), basis.n)
"""
function (basis::PolyBasis)(x::AbstractArray{T}) where {T}
    # returns array of shape (size(x)..., basis.n)
    if basis.n == 1 # Fast path
        return reshape(basis.f0.(x), size(x)..., 1)
    end

    y = zeros(T, prod(size(x)), basis.n)
    y[:,1] .= basis.f0.(x[:])
    y[:,2] .= basis.f1.(x[:])

    for i in 3:1:basis.n
        y[:,i] = (basis.a0.(i-2, x[:]) .* y[:,i-2] .+ basis.a1.(i-2, x[:]) .* y[:,i-1]) ./ basis.a2.(i-2, x[:])
    end

    # z = basis.f_norm.(x[:])
    return reshape(y, size(x)..., basis.n)
end

function (basis::PolyBasis)(x::T) where {T<:Number}
    basis(T[x])
end


function Chebyshev(n::Int, T::Type{<:AbstractFloat}=Float64) 
    # interval [-1 ,1]
    # weight = 1/sqrt(1-x^2)
    # ortogonality = π/2 kronecker(n,m)  or π if n = m = 0
    function f0(x) 
        1
    end
    function f1(x) 
        x
    end
    function a0(j,x)
        -1
    end
    function a1(j,x)
        2*x
    end
    function a2(j,x)
        1
    end
    function f_weight(x)
        1/sqrt(1-x^2)
    end
    function f_norm(n::Int,m::Int)
        if n==m==0
            return T(pi)
        else
            return T(pi/2*kronecker(n,m))
        end
    end

    PolyBasis{:chebyshev, typeof(f0), typeof(f1), typeof(a0), typeof(a1), typeof(a2), typeof(f_weight), typeof(f_norm)}(n, f0, f1, a0, a1, a2, f_weight, f_norm)
end

function Legendre(n::Int, T::Type{<:AbstractFloat}=Float64) 
    # interval [-1 ,1]
    # weight = 1
    # ortogonality = 2/(2*n+1) kronecker(n,m)
    function f0(x) 
        1
    end
    function f1(x) 
        x
    end
    function a0(j,x)
        -j
    end
    function a1(j,x)
        (2*j+1)*x
    end
    function a2(j,x)
        (j+1)
    end
    function f_weight(x)
        1
    end
    function f_norm(n::Int,m::Int)
        T(2/(2*n+1)*kronecker(n,m))
    end

    PolyBasis{:legendre, typeof(f0), typeof(f1), typeof(a0), typeof(a1), typeof(a2), typeof(f_weight), typeof(f_norm)}(n, f0, f1, a0, a1, a2, f_weight, f_norm)
end

function Laguerre(n::Int, T::Type{<:AbstractFloat}=Float64)
    # interval [0, ∞)
    # weight = exp(-x)
    # ortogonality = kronecker(n,m)
    function f0(x) 
        1
    end
    function f1(x) 
        1 - x
    end
    function a0(j,x)
        -j
    end
    function a1(j,x)
        (2*j+1-x)
    end
    function a2(j,x)
        (j+1)
    end
    function f_weight(x)
        return exp(-x)
    end
    function f_norm(n::Int, m::Int)
        T(kronecker(n,m))
    end

    PolyBasis{:laguerre, typeof(f0), typeof(f1), typeof(a0), typeof(a1), typeof(a2), typeof(f_weight), typeof(f_norm)}(n, f0, f1, a0, a1, a2, f_weight, f_norm)
end

function Hermite(n::Int; T::Type{<:AbstractFloat}=Float64) 
    # Probabilist's Hermite polynomials
    # interval (-∞, ∞)
    # weight = exp(-x^2 / 2)
    # ortogonality = n! * kronecker(n,m)
    function f0(x) 
        1
    end
    function f1(x) 
        x
    end
    function a0(j,x)
        -j
    end
    function a1(j,x)
        x 
    end
    function a2(j,x)
        1 
    end
    function f_weight(x)
        exp(-x^2/2)/sqrt(2pi)
    end
    function f_norm(n::Int, m::Int)
        if n <= 20 
            T(kronecker(n,m) * factorial(n))
        else # correction for big numbers
            T(kronecker(n,m) * gamma(n+1))
        end
    end

    PolyBasis{:hermite, typeof(f0), typeof(f1), typeof(a0), typeof(a1), typeof(a2), typeof(f_weight), typeof(f_norm)}(n, f0, f1, a0, a1, a2, f_weight, f_norm)
end

function derivative_polybasis(a::PolyBasis{:hermite})
    # Bidiagonal(A, :U)
    diagm(1 =>[i for i in 1:(a.n-1)])
end

function Hermite_hat(n::Int; T::Type{<:AbstractFloat}=Float64)
    # Normailzed Probabilist's Hermite polynomials
    # interval (-∞, ∞)
    # weight = exp(-x^2 / 2)
    # ortogonality = kronecker(n,m)
    function f0(x) 
        1
    end
    function f1(x) 
        x
    end
    function a0(j,x)
        -sqrt(j/(j+1))
    end
    function a1(j,x)
        x/sqrt(j+1) 
    end
    function a2(j,x)
        1 
    end
    function f_weight(x)
        exp(-x^2/2)/sqrt(2pi)
    end
    function f_norm(n::Int, m::Int)
        T(kronecker(n,m))
    end

    PolyBasis{:hermite_hat, typeof(f0), typeof(f1), typeof(a0), typeof(a1), typeof(a2), typeof(f_weight), typeof(f_norm)}(n, f0, f1, a0, a1, a2, f_weight, f_norm)
end

function derivative_polybasis(a::PolyBasis{:hermite_hat})
    # Bidiagonal(A, :U)
    diagm(1 =>[sqrt(i) for i in 1:(a.n-1)])
end
