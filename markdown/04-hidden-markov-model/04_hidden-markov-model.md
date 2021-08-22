---
redirect_from: "tutorials/4-bayeshmm/"
title: "Bayesian Hidden Markov Models"
permalink: "/:collection/:name/"
---

This tutorial illustrates training Bayesian [Hidden Markov Models](https://en.wikipedia.org/wiki/Hidden_Markov_model) (HMM) using Turing. The main goals are learning the transition matrix, emission parameter, and hidden states. For a more rigorous academic overview on Hidden Markov Models, see [An introduction to Hidden Markov Models and Bayesian Networks](http://mlg.eng.cam.ac.uk/zoubin/papers/ijprai.pdf) (Ghahramani, 2001).

Let's load the libraries we'll need. We also set a random seed (for reproducibility) and the automatic differentiation backend to forward mode (more [here](http://turing.ml/docs/autodiff/) on why this is useful).


```julia
# Load libraries.
using Turing, StatsPlots, Random

# Turn off progress monitor.
Turing.setprogress!(false);

# Set a random seed and use the forward_diff AD mode.
Random.seed!(12345678);
```




## Simple State Detection

In this example, we'll use something where the states and emission parameters are straightforward.


```julia
# Define the emission parameter.
y = [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
N = length(y);  K = 3;

# Plot the data we just made.
plot(y, xlim = (0,30), ylim = (-1,5), size = (500, 250))
```

![](figures/04_hidden-markov-model_2_1.png)



We can see that we have three states, one for each height of the plot (1, 2, 3). This height is also our emission parameter, so state one produces a value of one, state two produces a value of two, and so on.

Ultimately, we would like to understand three major parameters:

1. The transition matrix. This is a matrix that assigns a probability of switching from one state to any other state, including the state that we are already in.
2. The emission matrix, which describes a typical value emitted by some state. In the plot above, the emission parameter for state one is simply one.
3. The state sequence is our understanding of what state we were actually in when we observed some data. This is very important in more sophisticated HMM models, where the emission value does not equal our state.

With this in mind, let's set up our model. We are going to use some of our knowledge as modelers to provide additional information about our system. This takes the form of the prior on our emission parameter.

$$
m_i \sim \mathrm{Normal}(i, 0.5) \quad \text{where} \quad m = \{1,2,3\}
$$

Simply put, this says that we expect state one to emit values in a Normally distributed manner, where the mean of each state's emissions is that state's value. The variance of 0.5 helps the model converge more quickly — consider the case where we have a variance of 1 or 2. In this case, the likelihood of observing a 2 when we are in state 1 is actually quite high, as it is within a standard deviation of the true emission value. Applying the prior that we are likely to be tightly centered around the mean prevents our model from being too confused about the state that is generating our observations.

The priors on our transition matrix are noninformative, using `T[i] ~ Dirichlet(ones(K)/K)`. The Dirichlet prior used in this way assumes that the state is likely to change to any other state with equal probability. As we'll see, this transition matrix prior will be overwritten as we observe data.


```julia
# Turing model definition.
@model BayesHmm(y, K) = begin
    # Get observation length.
    N = length(y)

    # State sequence.
    s = tzeros(Int, N)

    # Emission matrix.
    m = Vector(undef, K)

    # Transition matrix.
    T = Vector{Vector}(undef, K)

    # Assign distributions to each element
    # of the transition matrix and the
    # emission matrix.
    for i = 1:K
        T[i] ~ Dirichlet(ones(K)/K)
        m[i] ~ Normal(i, 0.5)
    end

    # Observe each point of the input.
    s[1] ~ Categorical(K)
    y[1] ~ Normal(m[s[1]], 0.1)

    for i = 2:N
        s[i] ~ Categorical(vec(T[s[i-1]]))
        y[i] ~ Normal(m[s[i]], 0.1)
    end
end;
```




We will use a combination of two samplers ([HMC](http://turing.ml/docs/library/#Turing.HMC) and [Particle Gibbs](http://turing.ml/docs/library/#Turing.PG)) by passing them to the [Gibbs](http://turing.ml/docs/library/#Turing.Gibbs) sampler. The Gibbs sampler allows for compositional inference, where we can utilize different samplers on different parameters.

In this case, we use HMC for `m` and `T`, representing the emission and transition matrices respectively. We use the Particle Gibbs sampler for `s`, the state sequence. You may wonder why it is that we are not assigning `s` to the HMC sampler, and why it is that we need compositional Gibbs sampling at all.

The parameter `s` is not a continuous variable. It is a vector of **integers**, and thus Hamiltonian methods like HMC and [NUTS](http://turing.ml/docs/library/#-turingnuts--type) won't work correctly. Gibbs allows us to apply the right tools to the best effect. If you are a particularly advanced user interested in higher performance, you may benefit from setting up your Gibbs sampler to use [different automatic differentiation](http://turing.ml/stable/docs/autodiff/#compositional-sampling-with-differing-ad-modes) backends for each parameter space.

Time to run our sampler.


```julia
g = Gibbs(HMC(0.01, 50, :m, :T), PG(120, :s))
chn = sample(BayesHmm(y, 3), g, 1000);
```




Let's see how well our chain performed. Ordinarily, using the `describe` function from [MCMCChain](https://github.com/TuringLang/MCMCChain.jl) would be a good first step, but we have generated a lot of parameters here (`s[1]`, `s[2]`, `m[1]`, and so on). It's a bit easier to show how our model performed graphically.

The code below generates an animation showing the graph of the data above, and the data our model generates in each sample.


```julia

# Extract our m and s parameters from the chain.
m_set = MCMCChains.group(chn, :m).value
s_set = MCMCChains.group(chn, :s).value

# Iterate through the MCMC samples.
Ns = 1:length(chn)

# Make an animation.
animation = @gif for i in Ns
    m = m_set[i, :];
    s = Int.(s_set[i,:]);
    emissions = m[s]

    p = plot(y, chn = :red,
        size = (500, 250),
        xlabel = "Time",
        ylabel = "State",
        legend = :topright, label = "True data",
        xlim = (0,30),
        ylim = (-1,5));
    plot!(emissions, color = :blue, label = "Sample $i")
end every 3
```

![](figures/04_hidden-markov-model_5_1.gif)



Looks like our model did a pretty good job, but we should also check to make sure our chain converges. A quick check is to examine whether the diagonal (representing the probability of remaining in the current state) of the transition matrix appears to be stationary. The code below extracts the diagonal and shows a traceplot of each persistence probability.


```julia
# Index the chain with the persistence probabilities.
subchain = chn[["T[1][1]", "T[2][2]", "T[3][3]"]]

plot(subchain,
     seriestype = :traceplot,
     title = "Persistence Probability",
     legend=false)
```

![](figures/04_hidden-markov-model_6_1.png)



A cursory examination of the traceplot above indicates that all three chains converged to something resembling
stationary. We can use the diagnostic functions provided by [MCMCChains](https://github.com/TuringLang/MCMCChains.jl) to engage in some more formal tests, like the Heidelberg and Welch diagnostic:

```julia

heideldiag(MCMCChains.group(chn, :T))[1]
```

```
Heidelberger and Welch Diagnostic - Chain 1
  parameters     burnin   stationarity    pvalue      mean   halfwidth     
 te ⋯
      Symbol    Float64        Float64   Float64   Float64     Float64   Fl
oat ⋯

     T[1][1]     0.0000         1.0000    0.7671    0.8733      0.0167    1
.00 ⋯
     T[1][2]     0.0000         1.0000    0.3672    0.0390      0.0177    0
.00 ⋯
     T[1][3]     0.0000         1.0000    0.2198    0.0877      0.0186    0
.00 ⋯
     T[2][1]     0.0000         1.0000    0.4008    0.0665      0.0243    0
.00 ⋯
     T[2][2]     0.0000         1.0000    0.3236    0.7719      0.0347    1
.00 ⋯
     T[2][3]     0.0000         1.0000    0.7229    0.1616      0.0273    0
.00 ⋯
     T[3][1]     0.0000         1.0000    0.0912    0.1184      0.0293    0
.00 ⋯
     T[3][2]   100.0000         1.0000    0.5894    0.1034      0.0152    0
.00 ⋯
     T[3][3]     0.0000         1.0000    0.7599    0.7673      0.0305    1
.00 ⋯
                                                                1 column om
itted
```





The p-values on the test suggest that we cannot reject the hypothesis that the observed sequence comes from a stationary distribution, so we can be reasonably confident that our transition matrix has converged to something reasonable.


## Appendix
 This tutorial is part of the TuringTutorials repository, found at: <https://github.com/TuringLang/TuringTutorials>.

To locally run this tutorial, do the following commands:
```julia, eval = false
using TuringTutorials
TuringTutorials.weave_file("04-hidden-markov-model", "04_hidden-markov-model.jmd")
```

Computer Information:
```
Julia Version 1.6.2
Commit 1b93d53fc4 (2021-07-14 15:36 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, skylake-avx512)
Environment:
  JULIA_NUM_THREADS = 2

```

Package Information:

```
      Status `~/work/TuringTutorials/TuringTutorials/tutorials/04-hidden-markov-model/Project.toml`
  [91a5bcdd] Plots v1.19.1
  [f3b207a7] StatsPlots v0.14.25
  [fce5fe82] Turing v0.16.5
  [9a3f8284] Random

```
