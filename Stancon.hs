{-# OPTIONS_GHC -F -pgmF inlitpp #-}

```html_header
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
```


```haskell top hide
{-# LANGUAGE FlexibleContexts           #-}
{-# LANGUAGE OverloadedStrings          #-}

import Graphics.Plotly
import Stan.AST
import Stan.Data
import Stan.Run
import Numeric.Datasets.BostonHousing
import Numeric.Datasets
import Lens.Micro
import Data.Monoid ((<>))
```


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
           "beta" :~ normal (0,1)
          ,"sigma" :~ gamma (1,1)
          ,For "i" 1 "n" [
            "y"!["i"] :~ normal (("x"!["i"] `dot` "beta"), "sigma")
          ]
        ]
  ]

getRow b = [rooms b, crimeRate b]
```

```haskell do
bh <- getDataset bostonHousing

let sdata = "y" <~ map medianValue bh <>
            "x" <~ map getRow bh<>
            "n" <~ length bh <>
            "p" <~ length (getRow $ head bh)

res <- runStan linRegression sdata sample {numSamples = 100}
```


```haskell eval
plotly "bh" [points (aes & x .~ rooms & y .~ medianValue) bh]
```


```haskell eval
show res
```