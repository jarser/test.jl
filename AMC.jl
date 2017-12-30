module AMC

export vech, matE, matK, matN, matL, matD

function vech(A)
  A[tril(trues(A))]
end

function matE(n, i, j)
  I = eye(n)
  E = I[:, i] * I[:, j]'
end

function matK(n)
  K = zeros(n ^ 2, n ^ 2)
  for i in 1:n, j in 1:n
    K += kron(matE(n, i, j), matE(n, i, j)')
  end
  K
end

function matN(n)
  I = eye(n ^ 2)
  K = matK(n)
  N = 0.5 * (I + K)
end

function matL(n)
  I = eye(Int(0.5 * n * (n + 1)))
  L = zeros(Int(0.5 * n * (n + 1)), n ^ 2)
  for j in 1:n, i in j:n
    L[:, :] += I[:, Int((j - 1) * n + i - 0.5 * j * (j - 1))] * vec(matE(n, i, j))'
  end
  L
end

function matD(n)
  K = matK(n)
  N = matN(n)
  L = matL(n)
  D = 2.0 * N * L' - L' * L * K * L'
end

end
