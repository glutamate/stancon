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

Here, we present the results of some experiments with creating bindings to Stan in Haskell, a purely functional
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
that can specify that we are sampling or optimising the posterior.

```haskell do
res <- runStan linRegression sdata sample {numSamples = 500}
```
At this point the components of the posterior can be plotted as usual.

```haskell eval
postPlotRow res ["beta.1", "beta.2", "sigma" ] :: Html ()
```

```haskell do
seed <- seedEnv <$> newPureMT
let resEnv = seed <> Map.delete "y" sdata <> mcmcToEnv res
    simEnv = runSimulateOnce linRegression resEnv
    simEnvs = runSimulate 100 linRegression resEnv
    postPredOnce = zip (unPairDoubles $ fromJust $ Map.lookup "x" simEnv) (unDoubles $ fromJust $ Map.lookup "y" simEnv)
    avgYs = avgVar simEnvs "y"
    residuals = zip (unPairDoubles $ fromJust $ Map.lookup "x" sdata)
                    $ zipWith (-) (unDoubles $ fromJust $ Map.lookup "y" sdata)
                                  (unDoubles avgYs)

```


```haskell eval
plotly "bhpp" [points (aes & x .~ (fst . fst) & y .~ snd) postPredOnce]
```


```haskell eval
plotly "bhres" [points (aes & x .~ (fst . fst) & y .~ snd) residuals]
```


