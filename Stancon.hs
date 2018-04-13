{-# OPTIONS_GHC -F -pgmF inlitpp #-}

# Getting more out of Stan: some ideas from the Haskell bindings

## Introduction

Probabilistic programming is one of the most promising developments in statistical computing in recent times.
By combining programming languages theory with statistical modelling, it empowers modellers with a flexible
framework for statistical computing in which a great variety of models can be expressed using composition of
arbitrary probability distributions within loop structures to express dependencies and data structure known
to or hypothesised by the modeller.

Probabilistic programming is distinguished from stochastic programming by the ability to condition the random
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
often, post-posterior calculations are deferred to the host programming language, for instance Python or R, where
the model is rewritten to simulate. This has the disadvantage that the two models may become out of sync and some
probability distributions may be parameterised in different ways.

Here, we present the results of some experiments with creating bindings (https://github.com/diffusionkinetics/open/tree/master/stanhs)
to Stan in Haskell, a purely functional
and statically typed programming language. Rather than present “yet another Stan binding” or even worse, try to
persuade the reader to abandon their current programming language and learn Haskell, our aim here is to present
some ideas enable a richer set of probabilistic computations from Stan models. These ideas can be implemented in
other interfaces to Stan in any language.

## The Haskell programming language

Haskell is a lazy statically typed purely functional programming language…

Particularly suited for compilers and domain specific languages...


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
import Data.Text (unpack)
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
   & (layout . margin) ?~ thinMargins & (layout . height) ?~ 300)
```
Then, we have two also create the data structure holding the data input to Stan. Here is an example we will use the Boston Housing dataset found
in the datasets Haskell package. We load this into the `bh` variable which will hold a list of records describing Boston housing data (`bh :: [BostonHousing]`)

```haskell do

bh <- getDataset bostonHousing
```
Here, we plot the median value against the number of rooms in a residence:

```haskell eval
plotly "bh" [points (aes & x .~ rooms & y .~ medianValue) bh]
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
```


