# Load Turing.
using Turing
using Stheno, AbstractGPs, KernelFunctions, Random, Plots

# Load other dependencies
using Distributions, LinearAlgebra
using VegaLite, DataFrames, StatsPlots
using DelimitedFiles

Random.seed!(1789);

oil_matrix = readdlm("Data.txt", Float64);
labels = readdlm("DataLabels.txt", Float64);
labels = mapslices(x -> findmax(x)[2], labels, dims=2);

@model function GPLVM(Y, kernel_function, ndim=4,::Type{T} = Float64) where {T}

  # Dimensionality of the problem.
  N, D = size(Y)
  # dimensions of latent space
  K = ndim
  noise = 1e-3

  # Priors
  α ~ MvLogNormal(MvNormal(K, 1.0))
  σ ~ LogNormal(0.0, 1.0)
  Z ~ filldist(Normal(0., 1.), K, N)

  kernel = kernel_function(α, σ)

  ## Standard
  #  gp = GP(kernel)
  #  prior = gp(ColVecs(Z), noise)

  ## SPARSE GP
  xu = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0] # inducing points
  gp = Stheno.wrap(GP(kernel), GPC())
  fobs = gp(ColVecs(Z), noise)
  finducing = gp(xu, noise)
  prior = SparseFiniteGP(fobs, finducing)

  Y ~ filldist(prior, D)
end


sekernel(α, σ²) = σ² * SqExponentialKernel() ∘ ARDTransform(α)

Y = oil_matrix
# note we keep the problem very small for reasons of runtime
ndim=2
n_data=80
n_features=size(oil_matrix)[2];

gplvm = GPLVM(oil_matrix[1:n_data,:], sekernel, ndim)

# takes about 4hrs
chain = sample(gplvm, NUTS(), 500)
z_mean = permutedims(reshape(mean(group(chain, :Z))[:,2], (ndim, n_data)))'
alpha_mean = mean(group(chain, :α))[:,2]

df_gplvm = DataFrame(z_mean', :auto)
rename!(df_gplvm, Symbol.( ["z"*string(i) for i in collect(1:ndim)]))
df_gplvm[!,:sample] = 1:n_data
df_gplvm[!,:labels] = labels[1:n_data]
alpha_indices = sortperm(alpha_mean, rev=true)[1:2]
df_gplvm[!,:ard1] = z_mean[alpha_indices[1], :]
df_gplvm[!,:ard2] = z_mean[alpha_indices[2], :]

#  p1 = df_gplvm|>  @vlplot(:point, x=:z1, y=:z2, color="labels:n")
#  p2 = df_gplvm |>  @vlplot(:point, x=:ard1, y=:ard2, color="labels:n")

