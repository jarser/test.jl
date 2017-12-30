module VAR

using Distributions: Normal, TDist, ccdf

using ForwardDiff

using AMC: matK, matL, matD

struct VARe
  p::Int64
  K::Int64
  T::Int64
  η::Int64
  L::Array{Float64}
  Y::Array{Float64}
  Z::Array{Float64}
  U::Array{Float64}
  S::Array{Float64}
  Σml::Array{Float64}
  Σls::Array{Float64}
  A::Array{Float64}
  B::Array{Float64}
  Ja::Array{Float64}
  cA::Array{Float64}
  Jb::Array{Float64}
  cB::Array{Float64}
  ν::Array{Float64}
  μ::Array{Float64}
end

struct VARc
  B::Array{Float64}
  Bσ::Array{Float64}
  Bs::Array{Float64}
  Bp::Array{Float64}
end

struct VARs
  Λ::Vector{Complex{Float64}}
  Λ_abs::Vector{Float64}
  stablility::AbstractString
end

struct VARf
  H::Int64
  F::Array{Float64}
  method::AbstractString
end

struct VARi
  H::Int64
  Φ::Array{Float64}
  Φ_stdv::Array{Float64}
  Σ::Array{Float64}
  G::Array{Float64}
end

struct VARci
  H::Int64
  Ψ::Array{Float64}
  Ψ_stdv::Array{Float64}
  Ψ_∞::Array{Float64}
  Ψ_∞_stdv::Array{Float64}
  F::Array{Float64}
  F_∞::Array{Float64}
end

struct VARoi
  H::Int64
  Φ::Array{Float64}
  Θ::Array{Float64}
  Θ_stdv::Array{Float64}
  Σα::Array{Float64}
  Σσ::Array{Float64}
  G::Array{Float64}
  Cc::Array{Float64}
  Cb::Array{Float64}
end

struct VARcoi
  H::Int64
  Ψ::Array{Float64}
  Ξ::Array{Float64}
  Ξ_stdv::Array{Float64}
  Ξ_∞::Array{Float64}
  Ξ_∞_stdv::Array{Float64}
  F::Array{Float64}
  F_∞::Array{Float64}
  Bc::Array{Float64}
  Bc_∞::Array{Float64}
  Bb::Array{Float64}
  Bb_∞::Array{Float64}
end

function estimate(Y, p)
  K = size(Y, 1)
  T = size(Y, 2) - p
  η = T - (K * p + 1)
  L = zeros(K * p, T + p)
  for l in 1:p
    L[(l - 1) * K + 1:l * K, :] = Y * diagm(ones(T + p - l), l)
  end
  L = L[:, p + 1:end]
  Y = Y[:, p + 1:end]
  Z = [ones(T)'; L]
  B = (Y * Z') * inv(Z * Z')
  U = (Y - B * Z)
  S = U * U'
  Σml = (1 / T) * S
  Σls = (1 / η) * S
  ν = B[:, 1]
  A = B[:, 2:end]
  Ja = [eye(K) zeros(K, K * (p - 1))]
  cA = zeros(K * p, K * p)
  cA[K + 1:end, 1:K * (p - 1)] = eye(K * (p - 1))
  cA[1:K, 1:end] = B[:, 2:K * p + 1]
  Jb = [zeros(K, 1) eye(K) zeros(K, K * (p - 1))]
  cB = zeros(K * p + 1, K * p + 1)
  cB[1, :] = [1.0 zeros(1, K * p)]
  cB[2:end, 1] = [ν; zeros(K * (p - 1), 1)]
  cB[2:end, 2:end] = cA
  μ = Ja * inv(eye(K * p) - cA) * [ν; zeros(K * (p - 1))]
  VARe(p, K, T, η, L, Y, Z, U, S, Σml, Σls, A, B, Ja, cA, Jb, cB, ν, μ)
end

function coefficients(M::VARe, small = 0)
  B = M.B
  if small == 0
    Bσ = reshape(sqrt.(diag(kron(inv(M.Z * M.Z'), M.Σml))), size(M.B))
    Bs = M.B ./ Bσ
    Bp = 2.0 * ccdf.(Normal(), abs.(Bs))
    B = round.(M.B, 5)
    Bσ = round.(Bσ, 5)
    Bs = round.(Bs, 5)
    Bp = round.(Bp, 5)
  elseif small == 1
    Bσ = reshape(sqrt.(diag(kron(inv(M.Z * M.Z'), M.Σls))), size(M.B))
    Bs = M.B ./ Bσ
    Bp = 2.0 * ccdf.(TDist(M.η), abs.(Bs))
    B = round.(M.B, 5)
    Bσ = round.(Bσ, 5)
    Bs = round.(Bs, 5)
    Bp = round.(Bp, 5)
  end
  VARc(B, Bσ, Bs, Bp)
end

function stable(M::VARe)
  Λ = eigvals(M.cA)
  Λ_abs = abs.(Λ)
  if maximum(Λ_abs, 1)[1] < 1.0
    stability = "All eigenvalues lie inside the unit circle."
  else
    stability = "The system is not stationary."
  end
  VARs(Λ, Λ_abs, stability)
end

function forecast(M::VARe, H)
  F = zeros(M.K * M.p + 1, H + 1)
  Zt = [1.0; vec(flipdim(M.Y[:, end - M.p + 1:end], 2))]
  F[:, 1] = Zt
  for h in 2:H + 1
    F[:, h] = M.cB ^ (h - 1) * F[:, 1]
  end
  F = M.Jb * F
  F = F[:, 2:end]
  method = "iterative"
  VARf(H, F, method)
end

function irf(M::VARe, H, stdv = 0)
  cAn = zeros(M.K * M.p, M.K * M.p, H)
  Φ = zeros(M.K, M.K, H)
  for h in 1:H
    cAn[:, :, h] = M.cA ^ (h - 1)
    Φ[:, :, h] =  M.Ja * cAn[:, :, h] * M.Ja'
  end
  if stdv == 0
    Φ_stdv = zeros(size(Φ))
    Jz = [zeros(M.K * M.p, 1) eye(M.K * M.p)]
    ΓY0i = Jz * inv(M.Z * M.Z' / M.T) * Jz'
    Σ = kron(ΓY0i, M.Σml)
    G = zeros(M.K * M.K, (M.K * M.p) * M.K, H)
  elseif stdv == 1
    Jz = [zeros(M.K * M.p, 1) eye(M.K * M.p)]
    ΓY0i = Jz * inv(M.Z * M.Z' / M.T) * Jz'
    Σ = kron(ΓY0i, M.Σml)
    G = zeros(M.K * M.K, (M.K * M.p) * M.K, H)
    GΣG = zeros(M.K ^ 2, M.K ^ 2, H)
    for h in 1:H - 1
      for m in 1:h
        G[:, :, h + 1] += kron(M.Ja * cAn[:, :, h + 1 - m]', Φ[:, :, m])
      end
      GΣG[:, :, h + 1] = G[:, :, h + 1] * Σ * G[:, :, h + 1]'
    end
    GΣG = GΣG / M.T
    Φ_stdv = zeros(size(Φ))
    for h in 1:H
      Φ_stdv[:, :, h] = reshape(sqrt.(diag(GΣG[:, :, h])), size(M.Σml))
    end
  end
  VARi(H, Φ, Φ_stdv, Σ, G)
end

function cirf(M::VARe, R::VARi, stdv = 0)
  Ψ = cumsum(R.Φ, 3)
  Ψ_∞ = M.Ja * inv(eye(M.K * M.p) - M.cA) * M.Ja'
  if stdv == 0
    Ψ_stdv = zeros(size(Ψ))
    Ψ_∞_stvd = zeros(size(Ψ_∞))
    F = zeros(size(R.G))
    F_∞ = zeros(size(R.G))
  elseif stdv == 1
    F = cumsum(R.G, 3)
    FΣF = zeros(M.K ^ 2, M.K ^ 2, R.H)
    for h in 1:R.H - 1
      FΣF[:, :, h + 1] = F[:, :, h + 1] * R.Σ * F[:, :, h + 1]'
    end
    FΣF = FΣF / M.T
    Ψ_stdv = zeros(size(Ψ))
    for h in 1:R.H
      Ψ_stdv[:, :, h] = reshape(sqrt.(diag(FΣF[:, :, h])), size(M.Σml))
    end
    F_∞ = kron(repmat(Ψ_∞, 1, M.p), Ψ_∞)
    F_∞ΣF_∞ = (F_∞) * R.Σ * (F_∞)' / M.T
    Ψ_∞_stdv = reshape(sqrt.(diag(F_∞ΣF_∞)), size(M.Σml))
  end
  VARci(R.H, Ψ, Ψ_stdv, Ψ_∞, Ψ_∞_stdv, F, F_∞)
end

function oirf(M::VARe, H, stdv = 0)
  P = chol(M.Σls)'
  cAn = zeros(M.K * M.p, M.K * M.p, H)
  Φ = zeros(M.K, M.K, H)
  Θ = zeros(M.K, M.K, H)
  for h in 1:H
    cAn[:, :, h] = M.cA ^ (h - 1)
    Φ[:, :, h] =  M.Ja * cAn[:, :, h] * M.Ja'
    Θ[:, :, h] =  Φ[:, :, h] * P
  end
  Jz = [zeros(M.K * M.p, 1) eye(M.K * M.p)]
  ΓY0i = Jz * inv(M.Z * M.Z' / M.T) * Jz'
  Σα = kron(ΓY0i, M.Σls)
  Σσ = 2.0 * (inv(matD(M.K)' * matD(M.K)) * matD(M.K)') * kron(M.Σls, M.Σls) * (inv(matD(M.K)' * matD(M.K)) * matD(M.K)')'
  if stdv == 0
    G = zeros(M.K * M.K, (M.K * M.p) * M.K, H)
    Cc = zeros(M.K * M.K, (M.K * M.p) * M.K, H)
    Cb = zeros(M.K ^ 2, Int(0.5 * M.K * (M.K + 1)), H)
    Θ_stdv = zeros(size(Θ))
  elseif stdv == 1
    G = zeros(M.K * M.K, (M.K * M.p) * M.K, H)
    for h in 1:H - 1, m in 1:h
      G[:, :, h + 1] += kron(M.Ja * cAn[:, :, h + 1 - m]', Φ[:, :, m])
    end
    Cc = zeros(M.K * M.K, (M.K * M.p) * M.K, H)
    PI = kron(P', eye(M.K))
    for h in 1:H - 1
      Cc[:, :, h + 1] = PI * G[:, :, h + 1]
    end
    Cb = zeros(M.K ^ 2, Int(0.5 * M.K * (M.K + 1)), H)
    Ha = matL(M.K)' * inv(matL(M.K) * (eye(M.K ^ 2) + matK(M.K)) * kron(P, eye(M.K)) * matL(M.K)')
    IK = eye(M.K)
    for h in 1:H
      Cb[:, :, h] = kron(IK, Φ[:, :, h]) * Ha
    end
    CcΣCc = zeros(M.K ^ 2, M.K ^ 2, H)
    CbΣCb = zeros(M.K ^ 2, M.K ^ 2, H)
    for h in 1:H
      CcΣCc[:, :, h] = Cc[:, :, h] * Σα * Cc[:, :, h]'
      CbΣCb[:, :, h] = Cb[:, :, h] * Σσ * Cb[:, :, h]'
    end
    AΣA = (CcΣCc + CbΣCb) / M.T
    Θ_stdv = zeros(size(Θ))
    for h in 1:H
      Θ_stdv[:, :, h] = reshape(sqrt.(diag(AΣA[:, :, h])), size(M.Σml))
    end
  end
  VARoi(H, Φ, Θ, Θ_stdv, Σα, Σσ, G, Cc, Cb)
end

function coirf(M::VARe, O::VARoi, stdv = 0)
  P = chol(M.Σls)'
  Ψ = cumsum(O.Φ, 3)
  Ψ_∞ = M.Ja * inv(eye(M.K * M.p) - M.cA) * M.Ja'
  Ξ = cumsum(O.Θ, 3)
  Ξ_∞ = Ψ_∞ * P
  if stdv == 0
    Ξ_stdv = zeros(size(Ξ))
    Ξ_∞_stvd = zeros(size(Ξ_∞))
    F = zeros(size(O.G))
    F_∞ = zeros(size(O.G))
    Bc = zeros(size(O.Cc))
    Bb = zeros(size(O.Cb))
    Bc_∞ = zeros(size(O.Cc))
    Bb_∞ = zeros(size(O.Cb))
  elseif stdv == 1
    F = cumsum(O.G, 3)
    Bc = zeros(size(O.Cc))
    Bb = zeros(size(O.Cb))
    PI = kron(P', eye(M.K))
    IK = eye(M.K)
    Ha = matL(M.K)' * inv(matL(M.K) * (eye(M.K ^ 2) + matK(M.K)) * kron(P, eye(M.K)) * matL(M.K)')
    for h in 1:O.H
      Bc[:, :, h] = PI * F[:, :, h]
      Bb[:, :, h] = kron(eye(M.K), Ψ[:, :, h]) * Ha
    end
    BcΣBc = zeros(M.K ^ 2, M.K ^ 2, O.H)
    BbΣBb = zeros(M.K ^ 2, M.K ^ 2, O.H)
    for h in 1:O.H
      BcΣBc[:, :, h] = Bc[:, :, h] * O.Σα * Bc[:, :, h]'
      BbΣBb[:, :, h] = Bb[:, :, h] * O.Σσ * Bb[:, :, h]'
    end
    DΣD = (BcΣBc + BbΣBb) / M.T
    Ξ_stdv = zeros(size(Ξ))
    for h in 1:O.H
      Ξ_stdv[:, :, h] = reshape(sqrt.(diag(DΣD[:, :, h])), size(M.Σml))
    end
    F_∞ = kron(repmat(Ψ_∞, 1, M.p), Ψ_∞)
    Bc_∞ = PI * F_∞
    Bb_∞ = kron(eye(M.K), Ψ_∞) * Ha
    D_∞ΣD_∞ = ((Bc_∞) * O.Σα * (Bc_∞)' + (Bb_∞) * O.Σσ * (Bb_∞)') / M.T
    Ξ_∞_stdv = reshape(sqrt.(diag(D_∞ΣD_∞)), size(M.Σml))
  end
  VARcoi(O.H, Ψ, Ξ, Ξ_stdv, Ξ_∞, Ξ_∞_stdv, F, F_∞, Bc, Bc_∞, Bb, Bb_∞)
end

function girf(M::VARe, H, stdv = 0)
  cAn = zeros(M.K * M.p, M.K * M.p, H)
  Φ = zeros(M.K, M.K, H)
  Θ = zeros(M.K, M.K, H)
  for h in 1:H
    cAn[:, :, h] = M.cA ^ (h - 1)
    Φ[:, :, h] =  M.Ja * cAn[:, :, h] * M.Ja'
    for j in 1:M.K
      e = zeros(M.K, 1)
      e[j] = 1
      Θ[:, j, h] =  (M.Σls[j, j]) ^ (-0.5) * Φ[:, :, h] * M.Σls * e
    end
  end
  Θ
end

function fevd(M::VARe, stdv = 0)
  P = chol(M.Σls)'
  cAn = zeros(M.K * M.p, M.K * M.p, M.T)
  Φ = zeros(M.K, M.K, M.T)
  Θ = zeros(M.K, M.K, M.T)
  for h in 1:M.T
    cAn[:, :, h] = M.cA ^ (h - 1)
    Φ[:, :, h] =  M.Ja * cAn[:, :, h] * M.Ja'
    Θ[:, :, h] =  Φ[:, :, h] * P
  end
  MSE = zeros(M.K, M.K, M.T)
  for h in 1:M.T, j in 1:h
    MSE[:, :, h] += Θ[:, :, h] * Θ[:, :, h]'
  end
  MSE
end

end

## ADVERTENCIA ## CUIDADO CON EVIEWS Y EL AJUSTE DE GRADOS DE LIBERTAD DE LA COVARIANZA;
## ESTA SETTEADO PARA LEAST SQUARES --->>> CAMBIAR A COND MAX LIKELIHOOD

# function comp(h, A, M::VARe)
#   X = zeros(eltype(A), M.K * M.p, M.K * M.p)
#   X[M.K + 1:end, 1:M.K * (M.p - 1)] = eye(M.K * (M.p - 1))
#   X[1:M.K, 1:end] = A
#   vec(M.Ja * X ^ (h - 1) * M.Ja')
# end
#
# function compD(h, A, M::VARe)
#   ForwardDiff.jacobian(x -> comp(h, x, M), A)
# end

# function irfAD(M::VARe, H, stdv = 0) <<<--- AD version is still too slow
#   cAn = zeros(M.K * M.p, M.K * M.p, H)
#   Φ = zeros(M.K, M.K, H)
#   for h in 1:H
#     cAn[:, :, h] = M.cA ^ (h - 1)
#     Φ[:, :, h] =  M.Ja * cAn[:, :, h] * M.Ja'
#   end
#   if stdv == 0
#     Φ_stdv = zeros(size(Φ))
#     Jz = [zeros(M.K * M.p, 1) eye(M.K * M.p)]
#     ΓY0i = Jz * inv(M.Z * M.Z' / M.T) * Jz'
#     Σ = kron(ΓY0i, M.Σml)
#     G = zeros(M.K * M.K, (M.K * M.p) * M.K, H)
#   elseif stdv == 1
#     Jz = [zeros(M.K * M.p, 1) eye(M.K * M.p)]
#     ΓY0i = Jz * inv(M.Z * M.Z' / M.T) * Jz'
#     Σ = kron(ΓY0i, M.Σml)
#     G = zeros(M.K * M.K, (M.K * M.p) * M.K, H)
#     GΣG = zeros(M.K ^ 2, M.K ^ 2, H)
#     for h in 1:H
#       G[:, :, h] = compD(h, M.A, M)
#       GΣG[:, :, h] = G[:, :, h] * Σ * G[:, :, h]'
#     end
#     GΣG = GΣG / M.T
#     Φ_stdv = zeros(size(Φ))
#     for h in 1:H
#       Φ_stdv[:, :, h] = reshape(sqrt.(diag(GΣG[:, :, h])), size(M.Σml))
#     end
#   end
#   VARi(H, Φ, Φ_stdv, Σ, G)
# end

# function irf2(M::VARe, H, stdv = 0) <<<--- inefficient
#   Φ = zeros(M.K, M.K, H)
#   for h in 1:H
#     Φ[:, :, h] =  M.Ja * (M.cA ^ (h - 1)) * M.Ja'
#   end
#   if stdv == 0
#     Φ_stdv = zeros(size(Φ))
#   elseif stdv == 1
#     ΓY0 =  M.L * M.L' / M.T
#     GΣG = zeros(M.K ^ 2, M.K ^ 2, H)
#     for h in 1:H - 1
#       for m in 1:h
#         for n in 1:h
#           GΣG[:, :, h + 1] += kron(M.Ja * (M.cA') ^ (h - 1 - (m - 1)) * inv(ΓY0) * (M.cA) ^ (h - 1 - (n - 1)) * M.Ja', Φ[:, :, m] * M.Σml * Φ[:, :, n]')
#         end
#       end
#     end
#     GΣG = GΣG / M.T
#     Φ_stdv = zeros(size(Φ))
#     for h in 1:H
#       Φ_stdv[:, :, h] = reshape(sqrt.(diag(GΣG[:, :, h])), size(M.Σml))
#     end
#   end
#   VARi(H, Φ, Φ_stdv)
# end

# function irf3(M::VARe, H, stdv = 0) <<<--- better than irf2 but still more inefficient than irf
#   Φ = zeros(M.K, M.K, H)
#   for h in 1:H
#     Φ[:, :, h] =  M.Ja * (M.cA ^ (h - 1)) * M.Ja'
#   end
#   if stdv == 0
#     Φ_stdv = zeros(size(Φ))
#   elseif stdv == 1
#     Jz = [zeros(M.K, 1) eye(M.K * M.p)]
#     ΓY0i = Jz * inv(M.Z * M.Z' / M.T) * Jz'
#     Σ = kron(ΓY0i, M.Σml)
#     G = zeros(M.K * M.K, (M.K * M.p) * M.K, H)
#     GΣG = zeros(M.K ^ 2, M.K ^ 2, H)
#     for h in 1:H - 1
#       for m in 1:h
#         G[:, :, h + 1] += kron(M.Ja * (M.cA') ^ (h - 1 - (m - 1)), Φ[:, :, m])
#       end
#       GΣG[:, :, h + 1] = G[:, :, h + 1] * Σ * G[:, :, h + 1]'
#     end
#     GΣG = GΣG / M.T
#     Φ_stdv = zeros(size(Φ))
#     for h in 1:H
#       Φ_stdv[:, :, h] = reshape(sqrt.(diag(GΣG[:, :, h])), size(M.Σml))
#     end
#   end
#   VARi(H, Φ, Φ_stdv)
# end
