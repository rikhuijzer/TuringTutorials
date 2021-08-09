---
title: "Probabilistic Principal Component Analysis"
permalink: "/:collection/:name/"
---


Principal component analysis is a fundamental technique to analyse and visualise data.
You may have come across it in many forms and names.
Here, we give a probabilistic perspective on PCA with some biologically motivated examples.
For more details and a mathematical derivation, we recommend Bishop's textbook (Christopher M. Bishop, Pattern Recognition and Machine Learning, 2006).

```julia
using Turing
using Distributions, LinearAlgebra

# Packages for visualization
using VegaLite, DataFrames, StatsPlots

# Import Fisher's iris example data set
using RDatasets

# Set a seed for reproducibility.
using Random
Random.seed!(1789);
```




## A basic example

The idea of PCA is to find a latent variable $z$ that can be used to describe hidden structure in our dataset.
We use a simple Gaussian prior for
$$
P(z) = \mathcal{N}(z | 0, I)
$$
and similarly, the conditional distribution
$$
P(x | z) = \mathcal{N}(x | W z + \mu, \sigma^2 I)
$$
is modeled via a Gaussian distribution.

### A biologically motivated example

We'll generate synthetic data to explore the models.
The simulation is inspired by biological measurement of
expression of genes in cells, and so you can think of the two variables as cells and genes.
While the human genome is (mostly) identical between all the cells in your body, there exist interesting differences in gene expression in different human tissues and disease conditions.
Similarly, one way to investigate certain diseases is to look at differences in gene expression in cells from patients and healthy controls (usually from the same tissue).

Usually, we can assume that the changes in gene expression only affect a subset of all genes (and these can be linked to diseases in some way).
One of the challenges of this kind of data is to see the underlying structure, e.g. to make the connection between a certain state (healthy/disease) and gene expression.
The problem is that the space we are investigating is very large (Up to 20000 genes across 1000s of cells). So in order to find structure in this data, we usually need to project the data into a lower dimensional space.

Here, we simulate this problem. Admittedly, this is a very simplistic example with far fewer cells and genes and a very straighforward relationship. We try to capture the essence of the problem, but then, unfortunately, real life problems tend to be much more messy.

If you are not that interested in the biology, the more abstract problem formulation is to project a high-dimensional space onto a different space - in the case of PCA - a space where most of the variation is concentrated in the first few dimensions. So you will use PCA to explore underlying structure in your data that is not necessarily obvious from looking at the raw data itself.

```julia
n_cells = 60
n_genes = 9
mu_1 = 10. * ones(n_genes÷3)
mu_0 = zeros(n_genes÷3)
S = I(n_genes÷3)
mvn_0 = MvNormal(mu_0, S)
mvn_1 = MvNormal(mu_1, S)

# create a diagonal block like expression matrix, with some non-informative genes;
# not all features/genes are informative, some might just not differ very much between cells)
expression_matrix = transpose(vcat(
  hcat(rand(mvn_1, n_cells÷2),
       rand(mvn_0, n_cells÷2)),
  hcat(rand(mvn_0, n_cells÷2),
       rand(mvn_0, n_cells÷2)),
  hcat(rand(mvn_0, n_cells÷2),
       rand(mvn_1, n_cells÷2))))


df_exp = DataFrame(expression_matrix, :auto)
df_exp[!,:cell] = 1:n_cells

DataFrames.stack(df_exp, 1:n_genes) |>
    @vlplot(:rect, x="cell:o", color=:value,
      encoding={y={field="variable",
                   type="nominal",
                   sort="-x",
                   axis={title="gene"}
                  }
               })
```

![](figures/11-probabilistic-pca_2_1.png)



Here, you see the simulated data. You can see two groups of cells that differ in the expression of genes. While the difference between the two groups of cells here is fairly obvious from looking at the raw data, in practice and with large enough data sets, it is often impossible to spot the differences from the raw data alone. If you have some patience and compute resources you can increase the size of the dataset, or play around with the noise levels to make the problem increasingly harder.

### pPCA model
```julia
@model function pPCA(x, ::Type{TV} = Array{Float64}) where {TV}
  # Dimensionality of the problem.
  N, D = size(x)

  # latent variable z
  z ~ filldist(Normal(0.0, 1.0), D, N)

  # side note for the curious
  # we use the more concise filldist syntax partly for compactness, but also for compatibility with other AD
  # backends, see the [Turing Performance Tipps](https://turing.ml/dev/docs/using-turing/performancetips)
  # w = TV{2}(undef, D, D)
  # for d in 1:D
  #  w[d, :] ~ MvNormal(D, 1.)
  # end

  # weights/loadings W
  w ~ filldist(Normal(0.0, 1.0), D, D)

  # mean offset
  m ~ MvNormal(D, 1.0)
  mu = (w * z .+ m)'
  for d in 1:D
    x[:,d] ~ MvNormal(mu[:,d], 1.)
  end
end;
```




### pPCA inference

Here, we run the inference with the NUTS sampler. Feel free to try different samplers.

```julia
ppca = pPCA(expression_matrix)
chain = sample(ppca, NUTS(), 500);
```





### pPCA control

A quick sanity check. We reconstruct the input data from our parameter estimates, using the posterior mean as parameter estimates.

```julia
# Extract parameter estimates for plotting - mean of posterior
w = permutedims(reshape(mean(group(chain, :w))[:,2], (n_genes, n_genes)))
z = permutedims(reshape(mean(group(chain, :z))[:,2], (n_genes, n_cells)))'
mu = mean(group(chain, :m))[:,2]

X = w * z

df_rec = DataFrame(X', :auto)
df_rec[!,:cell] = 1:n_cells

DataFrames.stack(df_rec, 1:n_genes) |>
    @vlplot(:rect, x="cell:o", color=:value,
    encoding={y={field="variable",
                 type="nominal",
                 sort="-x",
                 axis={title="gene"}
                }
             })
```

![](figures/11-probabilistic-pca_5_1.png)


We can see the same pattern that we saw in the input data.
This is what we expect, as PCA is essentially a lossless transformation, i.e. the new space contains the same information as the input space as long as we keep all the dimensions.

And finally, we plot the data in a lower dimensional space. The interesting insight here is that we can project the information from the input space into a two-dimensional representation, without lossing the essential information about the two groups of cells in the input data.

```julia
df_pca = DataFrame(z', :auto)
rename!(df_pca, Symbol.( ["z"*string(i) for i in collect(1:n_genes)]))
df_pca[!,:cell] = 1:n_cells

DataFrames.stack(df_pca, 1:n_genes) |>
  @vlplot(:rect, "cell:o", "variable:o", color=:value)

df_pca[!,:type] = repeat([1, 2], inner = n_cells÷2)
df_pca |>  @vlplot(:point, x=:z1, y=:z2, color="type:n")
```

![](figures/11-probabilistic-pca_6_1.png)


We can see the two groups are well separated in this 2-D space. Another way to put it, 2 dimensions is enough to display the main structure of the data.


## Number of components

A direct question arises from this is: How many dimensions do we want to keep in order to represent the latent structure in the data?
This is a very central question for all latent factor models, i.e. how many dimensions are needed to represent that data in the latent space.
In the case of PCA, there exist a lot of heuristics to make that choice.
By using the pPCA model, this can be accomplished very elegantly, with a technique called *Automatic Relevance Determination*(ARD).
Essentially, we are using a specific prior over the factor loadings W that allows us to prune away dimensions in the
latent space. The prior is determined by a precision hyperparameter $\alpha$. Here, smaller values of $\alpha$ correspond to more important components.
You can find more details about this in the Bishop book mentioned in the introduction.


```julia
@model function pPCA_ARD(x, ::Type{TV} = Array{Float64}) where {TV}
  # Dimensionality of the problem.
  N, D = size(x)

  # latent variable z
  z ~ filldist(Normal(), D, N)

  # weights/loadings w with Automatic Relevance Determination part
  alpha ~ filldist(Gamma(1., 1.), D)
  w ~ filldist(MvNormal(1. ./ sqrt.(alpha)), D)

  mu = (w * z)'

  tau ~ Gamma(1.0, 1.0);
  for d in 1:D
    x[:,d] ~ MvNormal(mu[:, d], 1. / sqrt(tau))
  end
end;
```


```julia
ppca_ARD = pPCA_ARD(expression_matrix)
chain = sample(ppca_ARD, NUTS(), 500)

StatsPlots.plot(group(chain, :alpha))
```

![](figures/11-probabilistic-pca_8_1.png)



Here we look at the convergence of the chains for the $\alpha$ parameter. This parameter determines the relevance of individual components. We can see that the chains have converged and the posterior of the alpha parameters is centered around much smaller values in two instances. Below, we will use the mean of the small values to select the *relevant* dimensions - we can clearly see based on the values of $\alpha$ that there should be two dimensions in this example.

```julia
# Extract paramter estimates for plotting - mean of posterior
w = permutedims(reshape(mean(group(chain, :w))[:,2], (n_genes,n_genes)))
z = permutedims(reshape(mean(group(chain, :z))[:,2], (n_genes, n_cells)))'
α = mean(group(chain, :alpha))[:,2]
α
```

```
9-element Vector{Float64}:
 0.25576143792759143
 0.2581926916026036
 0.27122756902110895
 4.357986513425213
 3.9485910549054957
 4.3886567845446125
 0.27628194041153903
 0.2623082444567099
 0.25699286794405307
```




We can inspect alpha to see which elements are small, i.e. have a high relevance.

```julia
alpha_indices = sortperm(α)[1:2]
X = w[alpha_indices, alpha_indices] * z[alpha_indices,:]

df_rec = DataFrame(X', :auto)
df_rec[!,:cell] = 1:n_cells

#  #  DataFrames.stack(df_rec, 1:n_genes) |> @vlplot(:rect, "cell:o", "variable:o", color=:value) |> save("reconstruction.pdf")
DataFrames.stack(df_rec, 1:2) |> @vlplot(:rect, "cell:o", "variable:o", color=:value)

df_pre = DataFrame(z', :auto)
rename!(df_pre, Symbol.( ["z"*string(i) for i in collect(1:n_genes)]))
df_pre[!,:cell] = 1:n_cells

DataFrames.stack(df_pre, 1:n_genes) |> @vlplot(:rect, "cell:o", "variable:o", color=:value)

df_pre[!,:type] = repeat([1, 2], inner = n_cells÷2)
df_pre[!,:ard1] = df_pre[:, alpha_indices[1]]
df_pre[!,:ard2] = df_pre[:, alpha_indices[2]]
df_pre |>  @vlplot(:point, x=:ard1, y=:ard2, color="type:n")
```

![](figures/11-probabilistic-pca_10_1.png)



This plot is very similar to the low-dimensional plot above, but choosing the *relevant* dimensions based on the values of $\alpha$. When you are in doubt about the number of dimensions to project onto, ARD might provide an answer to that question.

## Batch effects

A second, common aspect apart from the dimensionality of the PCA space, is the issue of confounding factors or [batch effects](https://en.wikipedia.org/wiki/Batch_effect).
A batch effect occurs when non-biological factors in an experiment cause changes in the data produced by the experiment.
As an example, we will look at Fisher's famous Iris data set.

The data set consists of 50 samples each from three species of Iris (Iris setosa, Iris virginica and Iris versicolor).
Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. [RDatasets.jl](https://github.com/JuliaStats/RDatasets.jl) contains the Iris dataset.

An example for a batch effect in this case might be different scientists using a different measurement method to determine the length and width of the flowers. This can lead to a systematic bias in the measurement unrelated to the actual experimental variable - the species in this case.

```julia
# Example data set - generate synthetic gene expression data

# dataset available in RDatasets
data = dataset("datasets", "iris")
species = data[!, "Species"]

# we extract the four measured quantities
d = 4
dat = data[!, 1:d]
# and the number of measurements
n = size(dat)[1];
```




First, let's look at the original data using the pPCA model.

```julia
ppca = pPCA(dat)

# Here we use a different sampler, we don't always have to use NUTS:
# Hamiltonian Monte Carlo (HMC) sampler parameters
ϵ = 0.05
τ = 10
chain = sample(ppca, HMC(ϵ, τ), 1000)

# Extract parameter estimates for plotting - mean of posterior
w = permutedims(reshape(mean(group(chain, :w))[:,2], (d,d)))
z = permutedims(reshape(mean(group(chain, :z))[:,2], (d, n)))'
mu = mean(group(chain, :m))[:,2]

X = w * z
# X = w * z .+ mu

df_rec = DataFrame(X', :auto)
df_rec[!,:species] = species
DataFrames.stack(df_rec, 1:d) |> @vlplot(:rect, "species:o", "variable:o", color=:value)

df_iris = DataFrame(z', :auto)
rename!(df_iris, Symbol.( ["z"*string(i) for i in collect(1:d)]))
df_iris[!,:sample] = 1:n
df_iris[!,:species] = species

df_iris |>  @vlplot(:point, x=:z1, y=:z2, color="species:n")
```

![](figures/11-probabilistic-pca_12_1.png)


We can see that the setosa species is more clearly separated from the other two species, which overlap
considerably.

We now simulate a batch effect; imagine the person taking the measurement uses two different rulers and they are slightly off.
Again, in practice there are many different reasons for why batch effects occur and it is not always clear what is really at the basis of them,
nor can they always be tackled via the experimental setup. So we need methods to deal with them.

```julia
## Introduce batch effect
batch = rand(Binomial(1, 0.5), 150)
effect = rand(Normal(2.4, 0.6), 150)
batch_dat = dat .+ batch .* effect

ppca_batch = pPCA(batch_dat)
chain = sample(ppca_batch, HMC(ϵ, τ), 1000)
describe(chain)[1]

z = permutedims(reshape(mean(group(chain, :z))[:,2], (d, n)))'
df_pre = DataFrame(z', :auto)
rename!(df_pre, Symbol.( ["z"*string(i) for i in collect(1:d)]))
df_pre[!,:sample] = 1:n
df_pre[!,:species] = species
df_pre[!,:batch] = batch

df_pre |>  @vlplot(:point, x=:z1, y=:z2, color="species:n", shape=:batch)
```

![](figures/11-probabilistic-pca_13_1.png)


The batch effect makes it much harder to distinguish the species. And importantly, if we are not aware of the
batches, this might lead us to make wrong conclusions about the data.

In order to correct for the batch effect, we need to know about the assignment of measurement to batch.
In our example, this means knowing which ruler was used for which measurement, here encoded via the batch variable.

```julia
@model function pPCA_residual(x, batch, ::Type{TV} = Array{Float64}) where {TV}

  # Dimensionality of the problem.
  N, D = size(x)

  # latent variable z
  z ~ filldist(Normal(), D, N)

  # weights/loadings w
  w ~ filldist(Normal(), D, D)

  # covariate vector
  w_batch = TV{1}(undef, D)
  w_batch ~ MvNormal(D, 1.)

  # mean offset
  m = TV{1}(undef, D)
  m ~ MvNormal(D, 1.0)
  mu = m .+ w * z + w_batch .* batch'

  for d in 1:D
    x[:,d] ~ MvNormal(mu'[:,d], 1.)
  end

end;

ppca_residual = pPCA_residual(batch_dat, convert(Vector{Float64}, batch))
chain = sample(ppca_residual, HMC(ϵ, τ), 1000);
```




This model is described in considerably more detail [here](https://arxiv.org/abs/1106.4333).

```julia

z = permutedims(reshape(mean(group(chain, :z))[:,2], (d, n)))'
df_post = DataFrame(z', :auto)
rename!(df_post, Symbol.( ["z"*string(i) for i in collect(1:d)]))
df_post[!,:sample] = 1:n
df_post[!,:species] = species
df_post[!,:batch] = batch

df_post |>  @vlplot(:point, x=:z1, y=:z2, color="species:n", shape=:batch)
```

![](figures/11-probabilistic-pca_15_1.png)



We can now see, that the data are better separated in the latent space by accounting for the batch effect. It is not perfect, but definitely an improvement over the previous plot.

## Rotation Invariant Householder Parameterization for Bayesian PCA

While PCA is a very old technique, nevertheless, it is still object of current research. Here, we
demonstrate this by implementing a recent publication that removes the rotational symmetry from the posterior,
thus speeding up inference. For more details, please read the
[original publication](http://proceedings.mlr.press/v97/nirwan19a.html)

```julia
## helper functions for Householder transform
function V_low_tri_plus_diag(Q::Int, V)
  for q in 1:Q
    V[:,q] = V[:,q] ./ sqrt(sum(V[:,q].^2))
  end
  return(V)
end

function Householder(k::Int, V)
  v = V[:, k]
  sgn = sign(v[k]);

  v[k] += sgn;
  H = LinearAlgebra.I - (2.0 / dot(v,v) * (v * v'))
  H[k:end, k:end] = -1. * sgn .* H[k:end, k:end]

  return(H)
end

function H_prod_right(V)
  D, Q = size(V)

  H_prod = zeros(Real,D,D,Q+1)
  H_prod[:,:,1] = Diagonal(repeat([1.0], D))

  for q in 1:Q
    H_prod[:,:,q+1] = Householder(Q - q + 1, V) * H_prod[:,:,q]
  end

  return(H_prod)
end

function orthogonal_matrix(D::Int64, Q::Int64, V)
  V = V_low_tri_plus_diag(Q, V)
  H_prod = H_prod_right(V)

  return(H_prod[:,1:Q,Q+1])
end

@model function pPCA_householder(x, K::Int, ::Type{T} = Float64) where {T}

  # Dimensionality of the problem.
  D, N = size(x)
  @assert K <= D

  # parameters
  sigma_noise ~ LogNormal(0.0, 0.5)
  v ~ filldist(Normal(0.0, 1.0), Int(D*K - K*(K-1)/2))
  sigma ~ Bijectors.ordered(MvLogNormal(MvNormal(K, 1.0)))

  v_mat = zeros(T, D, K)
  v_mat[tril!(trues(size(v_mat)))] .= v
  U = orthogonal_matrix(D, Q, v_mat)

  W = zeros(T, D, K)
  W += U * Diagonal(sigma)

  Kmat = zeros(T, D, D)
  Kmat += W * W'
  for d in 1:D
    Kmat[d,d] = Kmat[d, d] + sigma_noise^2 + 1e-12;
  end
  L = LinearAlgebra.cholesky(Kmat).L

  for q in 1:Q
    r = sqrt.(sum(dot(v_mat[:,q], v_mat[:,q])))
    Turing.@addlogprob! (-log(r)*(D-q))
  end

  Turing.@addlogprob! -0.5*sum(sigma.^2) + (D-Q-1)*sum(log.(sigma))
  for qi in 1:Q
    for qj in (qi+1):Q
      Turing.@addlogprob! log(sigma[Q-qi+1]^2) - sigma[Q-qj+1]^2
    end
  end
  Turing.@addlogprob! sum(log.(2.0*sigma))

  L_full = zeros(T, D, D)
  L_full += L * transpose(L)
  # fix numerical instability (non-posdef matrix)
  for d in 1:D
    for k in (d+1):D
      L_full[d,k] = L_full[k,d]
    end
  end

  x ~ filldist(MvNormal(L_full), N);
end;

# Dimensionality of latent space
Random.seed!(1789);
Q=2
n_samples=700
ppca_householder = pPCA_householder(Matrix(dat)', Q)
chain = sample(ppca_householder, NUTS(), n_samples);

# Extract mean of v from chain
N,D = size(dat)
vv = mean(group(chain, :v))[:,2]
v_mat = zeros(Real, D, Q)
v_mat[tril!(trues(size(v_mat)))] .= vv
sigma = mean(group(chain, :sigma))[:,2]
U_n = orthogonal_matrix(D, Q, v_mat)
W_n = U_n * (LinearAlgebra.I(Q) .* sigma)

# Create array with projected values
z = W_n' * transpose(Matrix(dat))
df_post = DataFrame(convert(Array{Float64}, z)', :auto)
rename!(df_post, Symbol.( ["z"*string(i) for i in collect(1:Q)]))
df_post[!,:sample] = 1:n
df_post[!,:species] = species

df_post |> @vlplot(:point, x=:z1, y=:z2, color="species:n")
```

![](figures/11-probabilistic-pca_16_1.png)



We observe a speedup in inference, and similar results to the previous implementations. Finally, we will look at the uncertainity that is associated with the samples. We will do this by sampling from the posterior projections. In order to do so, we need to apply some normalizations to the projections.

```julia
## Create data projections for each step of chain
vv = collect(get(chain, [:v]).v)
v_mat = zeros(Real, D, Q)
vv_mat = zeros(Float64, n_samples, D, Q)
for i in 1:n_samples
  index = BitArray(zeros(n_samples, D, Q))
  index[i,:,:] = tril!(trues(size(v_mat)))
  tmp = zeros(size(vv)[1])
  for j in 1:(size(vv)[1])
    tmp[j] = vv[j][i]
  end
    vv_mat[index] .= tmp
end

ss = collect(get(chain, [:sigma]).sigma)
sigma = zeros(Q, n_samples)
for d in 1:Q
  sigma[d,:] = Array(ss[d])
end

samples_raw = Array{Float64}(undef, Q, N, n_samples)
for i in 1:n_samples
  U_ni = orthogonal_matrix(D, Q, vv_mat[i,:,:])
  W_ni = U_ni * (LinearAlgebra.I(Q) .* sigma[:,i])
  z_n = W_ni' * transpose(Matrix(dat))
  samples_raw[:,:,i] = z_n
end

# initialize a 3D plot with 1 empty series
plt = plot(
    [100, 200, 300],
    xlim = (-4.00, 7.00),
    ylim = (-100.00, 0.00),
    group = ["Setosa", "Versicolor", "Virginica"],
    markercolor = ["red", "blue", "black"],
    title = "Visualization",
    seriestype = :scatter,
)

anim = @animate for i=1:n_samples
  scatter!(plt, samples_raw[1, 1:50, i], samples_raw[2, 1:50, i], color="red", seriesalpha=0.1, label="")
  scatter!(plt, samples_raw[1, 51:100, i], samples_raw[2, 51:100, i], color="blue", seriesalpha=0.1, label="")
  scatter!(plt, samples_raw[1, 101:150, i], samples_raw[2, 101:150, i], color="black", seriesalpha=0.1, label="")
end
gif(anim, "anim_fps.gif", fps = 5)
```

![](figures/11-probabilistic-pca_17_1.gif)



Here we see the density of the (normalized) projections from the chain to illustrate the uncertainty in the
projections. Below, we show the same plot with un-normalized samples.

We can see quite clearly that it is possible to separate Setosa from Versicolor and
Virginica species, but that the latter two cannot be clearly separated based on the pPCA projection. This can be
shown even more clearly by using a kernel density estimate and plotting the contours of that estimate.

```julia
# kernel density estimate
using StatsPlots, KernelDensity
dens = kde((vec(samples_raw[1, :, :]),vec(samples_raw[2, :, :])))
StatsPlots.plot(dens)
```

![](figures/11-probabilistic-pca_18_1.png)


## Appendix
 This tutorial is part of the TuringTutorials repository, found at: <https://github.com/TuringLang/TuringTutorials>.

To locally run this tutorial, do the following commands:
```julia, eval = false
using TuringTutorials
TuringTutorials.weave_file("11-probabilistic-pca", "11-probabilistic-pca.jmd")
```

Computer Information:
```
Julia Version 1.6.1
Commit 6aaedecc44 (2021-04-23 05:59 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, skylake)
Environment:
  JULIA_NUM_THREADS = 8

```

Package Information:

```
      Status `~/TuringDev/TuringTutorials/tutorials/11-probabilistic-pca/Project.toml`
  [76274a88] Bijectors v0.9.7
  [a93c6f00] DataFrames v1.2.2
  [31c24e10] Distributions v0.25.11
  [ce6b1742] RDatasets v0.7.5
  [f3b207a7] StatsPlots v0.14.26
  [fce5fe82] Turing v0.17.0
  [112f6efa] VegaLite v2.6.0
  [37e2e46d] LinearAlgebra

```
