function MSV_matrix_1D(x, w, a::PolyBasis{:hermite}, f=nothing)
    # no = [1/sqrt(a.f_norm(i,i)) for i in 0:(a.n-1)];
    # V = a(x) .* no';
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

            s = i/4*kronecker(j-1,i-1) - sqrt(j*(j-1))/4*kronecker(j-1,i+1) - sqrt(i*(i-1))/4*kronecker(j+1,i-1) + (i+1)*kronecker(j+1,i+1)

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

function truncate_matrix(a)
    return a[1:end-1, 1:end-1]
end

function M_matrix_1D(x, w, a::PolyBasis{:hermite}, f=nothing)
    # no = [1/sqrt(a.f_norm(i,i)) for i in 0:(a.n-1)];
    # V = a(x) .* no';
    V = a(x);
    n = size(V,2)
    M = zeros(n,n)
    for i in 0:(n-1)
        for j in 0:(n-1)
            m = dot(w, V[:,i+1] .* V[:,j+1])

            M[j+1,i+1] = m
            M[i+1,j+1] = m
        end
    end
    return M
end

function S_matrix_1D(x, w, a::PolyBasis{:hermite})
    V = a(x);
    n = size(V,2)
    M = zeros(n,n)
    for i in 0:(n-1)
        for j in i:(n-1)
            r = 1/4*kronecker(j+1,i+1)*dot(w, V[:,i+1].*V[:,i+1])*(i+1) - j/4*kronecker(j-1,i+1)*dot(w, V[:,i+1].*V[:,i+1])*(i+1) - i/4*kronecker(j+1,i-1)*dot(w, V[:,j+1].*V[:,j+1])*(j+1)
            # r = 1/4*kronecker(j+1,i+1)*factorial(i+1) - j/4*kronecker(j-1,i+1)*factorial(i+1) - i/4*kronecker(j+1,i-1)*factorial(j+1)
            if i>1
                r += i*j/4*kronecker(j-1,i-1)*dot(w, V[:,i-1].*V[:,i-1])*(i-1)
            elseif i==1
                r += i*j/4*kronecker(j-1,i-1) 
            end
            M[j+1, i+1] = r
            M[i+1, j+1] = r
        end
    end
    return M
end

function V_matrix_1D(x, w, a::PolyBasis, f)
    V = a(x);
    n = size(V,2)
    M = zeros(n,n)
    for i in 1:n
        for j in i:n
            r = dot(w, V[:,i] .* V[:,j] .* f.(x))
            M[j,i] = r
            M[i,j] = r
        end
    end
    return M
end
