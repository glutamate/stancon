{-# OPTIONS_GHC -F -pgmF inlitpp #-}

# Getting more out of Stan: some ideas from the Haskell bindings

Thomas A Nielsen<sup>1</sup>, Dominic Steinitz<sup>1</sup> and Henrik Nilsson<sup>2</sup>

1. Tweag I/O
2. School of Computer Science, University of Nottingham

## Introduction

Probabilistic programming is one of the most promising developments in statistical computing in recent times.
By combining programming languages theory with statistical modelling, it empowers modellers with a flexible
framework for statistical computing in which a great variety of models can be expressed using composition of
arbitrary probability distributions within flexible control flow to express dependencies and data structure known
to, or hypothesised by, the modeller.

Probabilistic programming is distinguished from stochastic programming by the capability to condition the random
variables on observed data, that is essentially to perform the Bayesian update and move from prior to posterior
distributions. Nevertheless, probabilistic programming in its full power is not restricted only to calculating
the posterior. Computations based on this posterior may be more directly relevant to the data analyst or to the
decision maker and often require further probabilistic calculations. For instance:

* predicting outcomes for new observations
* model criticism based on residuals or posterior predictive
* forecasting
* risk analysis
* decision-making
* resource allocation

Stan has emerged as the most practical and widely used probabilistic programming language for Bayesian computation.
In its canonical form, Stan only calculates the posterior and leaves all further analysis to generic programming
languages. Some calculations can be done within Stan alone, but these are quite limited and tricky to apply. Most
often, post-posterior calculations are deferred to the host programming language such as Python or R. For
instance, to calculate residuals or to make predictions in a predictive model, one must write the model twice:
once in the Stan modelling language, for inference, and once in the host language, for simulation and prediction.
This has the disadvantage that the two models may become out of sync and some
probability distributions may be parameterised in different ways. This is not much work for simple models, such as
linear regression model; the reader may feel that the author doth protest too much. But for more complex models
it may not at all be obvious how to transfer the posterior into a simulation environment.

Here, we present the results of some experiments with creating bindings (https://github.com/diffusionkinetics/open/tree/master/stanhs)
to Stan in Haskell, a purely functional
and statically typed programming language. Rather than present “yet another Stan binding” or even worse, try to
persuade the reader to abandon their current programming language and learn Haskell, our aim here is to present
some ideas enable a richer set of probabilistic computations from Stan models. These ideas can be implemented in
other interfaces to Stan in any language. Nevertheless, we have chosen here to explore these ideas in Haskell
due to its support for embedded languages, ease of re-factoring experimental code, and its emerging data science
ecosystem.

## The Haskell programming language

Haskell is a pure, statically typed, general purpose functional
language with lazy evaluation. It is widely used in academia for
teaching and research, and increasingly for commercial software
development as Haskell is conducive to rapid development of reliable
code. Purity means that functions have no side effects, except where
explicitly reflected in the function type, and then only those effects
that are permitted by the type in question. This enforced effect
discipline, in combination with an expressive type system, make it a
lot easier to understand and maintain code, especially when it grows
large. Purity implies that equational reasoning is valid, making
Haskell a particularly good fit for mathematical applications. Lazy
evaluation means that computation is demand driven. This, to a large
extent, frees the programmer from operational concerns, allowing them
to focus on stating what to compute rather than how to do it, which is
another reason for why Haskell is a good fit for mathematical
applications.

One area where Haskell has proved particularly successful is as a host
language for domain-specific languages. There are a number of reasons
for this. One is Haskell's concise yet flexible syntax with extensive
support for overloading of operators and other notation through type
classes. Another is that all values in Haskell, including functions,
are first-class; i.e., they can be bound to variables, passed to and
returned from functions, and so on, greatly facilitating implementing
appropriate domain-specific abstractions. Haskell is also particularly
well suited for symbolic computations, like what is needed for
compiler applications. In the setting of embedded domain-specific
languages, this allows for a spectrum of implementation strategies,
from interpretation to compilation, as well as programmatic
construction of programs in the domain-specific language, often
referred to as metaprogramming. Finally, thanks to its powerful type
system, it is often possible to enforce domain-specific typing
constraints. We will see some of these features being put to good use
in the following.

For a data scientist or statistician, the Haskell language holds
several attractions. Most importantly, Haskell now has an ecosystem
and a community for data science and numerical computing that is, if
not best in class, then increasingly productive with many packages
implementing different methods, in particular for a general purpose
programming language that was not designed with numerical computing in
mind. On a spectrum of data science needs, Haskell is particularly
suited to productizing models that have been developed in languages or
environments that may be more suited for explorative data analysis. 
The inline-r project, for instance, gives Haskell programmers direct
access to all of the R programming language, facilitating moving data
science into a production environment. Haskell’s type system makes it
very simple to re-factor large code bases which may blur the boundary
between data science and software engineering.

```html_header
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
```

## Bayesian modelling in Haskell and Stan

```haskell top hide
{-# LANGUAGE FlexibleContexts           #-}
{-# LANGUAGE OverloadedStrings          #-}

import Graphics.Plotly
import Graphics.Plotly.Histogram
import Stan.AST
import Stan.Data
import Stan.Run
import Stan.Simulate
import Numeric.Datasets.BostonHousing
import Numeric.Datasets
import Lens.Micro
import Data.Monoid
import Data.Maybe
import Data.Text (unpack, pack)
import Lucid
import Lucid.Bootstrap
import Lucid.Bootstrap3
import qualified Data.Map.Strict as Map
import qualified Data.Random as R
import Data.Random.Source.PureMT
```

Unlike in other Stan interfaces, in our prototype interface the Stan model itself is described in a data structure
in the host language, here in Haskell. This has the disadvantage that the Stan file has slightly more syntactic
noise and is less familiar. However, the Stan model description is now a value in a programming language that can
be manipulated and calculated based on the circumstances; and for further advantages that will become apparent later
in this paper. In our current implementation, the Stan model value looks like this:

```haskell top
linRegression :: [Stan]
linRegression = [
  Data [ lower 0 Int ::: "n"
       , lower 0 Int ::: "p"
       , Real ::: "y"!["n"]
       , Real ::: "x"!["n","p"]
       ],
  Parameters [
       Real ::: "beta"!["p"],
       Real ::: "sigma"
       ],
  Model [
          For "i" 1 "p" ["beta"!["i"] :~ normal (0.0,1.0)]
          ,"sigma" :~ gamma (1.0,1.0)
          ,For "i" 1 "n" [
            "y"!["i"] :~ normal (("x"!["i"] `dot` "beta"), "sigma")
          ]
        ]
  ]
```
```haskell top hide
getRow b = [rooms b, crimeRate b]


postPlotRow post vnms = row_ $ rowEven MD $ flip map vnms $ \vnm -> toHtml (plotly vnm [histogram 50 $ fromJust $ Map.lookup (unpack vnm) post]
   & (layout . margin) ?~ titleMargins & (layout . height) ?~ 300 & (layout . title) ?~ (vnm))
```
Then, we have two also create the data structure holding the data input to Stan. Here is an example we will use the Boston Housing dataset found
in the datasets Haskell package. We load this into the `bh` variable which will hold a list of records describing Boston housing data (`bh :: [BostonHousing]`)

```haskell do

bh <- getDataset bostonHousing
```
Here, we plot the median value against the number of rooms in a residence:

```haskell eval
plotly "bh" [points (aes & x .~ rooms & y .~ medianValue) bh]
   & layout %~ yaxis ?~ (defAxis & axistitle ?~ "Median Value")
   & layout %~ xaxis ?~ (defAxis & axistitle ?~ "Rooms")
```
To put this into the Stan data format, we create values of the `StanEnv` type using the custom infix operator `<~`. Haskell allows library
programmers to define their own infix operators describing the model. Here, we have defined `v <~ d` to mean, create a Stan environment
where the variable named v holds the data contained in the Haskell variable d. `d` can be any type for which we have defined how to turn
values into Stan values (that is, implemented the `ToStanData` type class. We concatenate these elementary Stan environments using the
append operator `<>`.

```haskell do
let sdata = "y" <~ map medianValue bh <>
            "x" <~ map getRow bh <>
            "n" <~ length bh <>
            "p" <~ (2 :: Int)
```
Finally, we run the Stan model using the `runStan` function taking as arguments the model, the data value and a configuration value
that can specify that we are sampling or optimising the posterior. The resulting posterior will be bound to the variable `res`.

```haskell do
res <- runStan linRegression sdata sample {numSamples = 500}
```
At this point the components of the posterior can be plotted as usual.

```haskell eval
postPlotRow res ["beta.1", "beta.2", "sigma" ] :: Html ()
```

we propose that in order to a richer probabilistic programming capability based on the Bayesian update in Stan, it suffices to
add a function to simulate from a probabilistic model with fine control over the transfer of information from the posterior to
the simulation environment. By transferring no information, we are simulating from the prior (prior predictive distribution);
all information, we are simulating from the posterior (posterior predictive distribution). And by controlling the independent
variables in the dataset we can make predictions for new observations. In the case of timeseries modelling, by manipulating the
starting value we can continue a simulation from the endpoint of observed data (that is, forecast). Crucially, we are proposing
that all of these functions are possible without writing the model twice as is usual: once in Stan, and once in the host language.
Simulation operates on the same model description as that used for inference.

The type of our simulation function is:

```haskell
runSimulate
  :: Int       -- Number of simulations we would like to perform
  -> [Stan]    -- The Stan model
  -> StanEnv   -- The simulation input environment
  -> [StanEnv] -- A list of independent simulation outputs
```

Here `runSimulate n m e` will perform n independent simulations in the model m using environment e. If m contains values from a
posterior (which has been generated with `runStan`) then each of the independent simulations will use a consistent set of samples from that posterior.

We quickly demonstrate the concept of simulation by simulating a single replica dataset and plotting it:

```haskell do
seed <- seedEnv <$> newPureMT
let resEnv = seed <> Map.delete "y" sdata <> mcmcToEnv res
    simEnv = runSimulateOnce linRegression resEnv
    postPredOnce = zip (unPairDoubles $ fromJust $ Map.lookup "x" simEnv) (unDoubles $ fromJust $ Map.lookup "y" simEnv)
```

```haskell eval
plotly "bhpp" [points (aes & x .~ (fst . fst) & y .~ snd) postPredOnce]
     & layout %~ yaxis ?~ (defAxis & axistitle ?~ "Median Value")
     & layout %~ xaxis ?~ (defAxis & axistitle ?~ "Rooms")
```

This simulation facility can be used for a common operation in model criticism: calculating residuals.very often, residuals are
calculated by averaging the posterior parameters (or using a point estimate), making a single prediction and subtracting this
from the observed outcome. Here, we argue that this is incorrect. From a Bayesian point of view, it almost never makes sense
to average the parameters. Instead, the parameters, the predicted outcome and the residuals themselves are all probability
distributions. In order to achieve something plottable, we average the predicted outcome over all the Markov chain Monte Carlo
samples and subtract this average prediction from the observed outcome. This has the advantage that in case of banana shaped or
 multimodal posteriors (arising, for instance from lack of identifiability) the average prediction still has more meaning than
 the average parameter value.


```haskell do
let simEnvs = runSimulate 100 linRegression resEnv

    avgYs = avgVar simEnvs "y"
    residuals = zip (unPairDoubles $ fromJust $ Map.lookup "x" sdata)
                    $ zipWith (-) (unDoubles $ fromJust $ Map.lookup "y" sdata)
                                  (unDoubles avgYs)
```

```haskell eval
plotly "bhres" [points (aes & x .~ (fst . fst) & y .~ snd) residuals]
      & layout %~ yaxis ?~ (defAxis & axistitle ?~ "Median Value Residual")
      & layout %~ xaxis ?~ (defAxis & axistitle ?~ "Rooms")
```

## Discussion

We have shown prototype bindings to Stan in the Haskell language. Unlike other such bindings, our model definition is a data
type in the host language. This enables us to simulate from the model with fine control over the model parameters which may c
ome from a posterior following inference.

Simulation with fine control over the provenance of the parameter distribution has several advantages. We have already shown
how it can be used to define the residuals and how to simulate from the posterior predictive distribution. In many cases, such
an instruction to simulate from the posterior predictive distribution will be incompletely specified. For instance, if we are
dealing with a hierarchical regression model, should the posterior predictive distribution retain the group identities? One
posterior predictive simulation would have entirely new groups, while another could have new individuals but drawn from groups
that are specified from the data.

### Further work

The Haskell interface to Stan we have shown here is an incomplete prototype. The model definition language can be polished
to reduce the amount of syntactic noise. Ideally, we would also have the ability to parse models written in the standard Stan
language. We can also make it much easier to move data into Stan, and to parse data from the posterior samples and from the
simulation environment. This will be facilitated when the Haskell data science community converges on one of the several
proposed implementations of data frames.


## About this document

The code for this document is hosted on GitHub (https://github.com/glutamate/stancon). It is written using the inliterate
notebook format (https://github.com/diffusionkinetics/open/tree/master/inliterate) which allows a mixture of code, text
writing, and the result of running code. Compared to Jupyter notebooks,
it is less interactive (there is no caching) and it emphasises a cleaner, human- readable source format that can be checked
into version control. Plots are generated with Plotly.js.


