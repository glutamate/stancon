{-# OPTIONS_GHC -F -pgmF inlitpp #-}

```html_header
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
```


```haskell top hide
{-# LANGUAGE FlexibleContexts           #-}
{-# LANGUAGE OverloadedStrings          #-}

import Graphics.Plotly
import Graphics.Plotly.Histogram
import Stan.AST
import Stan.Data
import Stan.Run
import Numeric.Datasets.BostonHousing
import Numeric.Datasets
import Lens.Micro
import Data.Monoid ((<>))
import Data.Maybe
import Data.Text (unpack)
import Lucid
import Lucid.Bootstrap
import Lucid.Bootstrap3
import qualified Data.Map.Strict as Map
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


postPlotRow post vnms = row_ $ rowEven MD $ flip map vnms $ \vnm -> toHtml (plotly vnm [histogram 50 $ fromJust $ Map.lookup (unpack vnm) post]
   & (layout . margin) ?~ thinMargins & (layout . height) ?~ 300)
```

```haskell do
bh <- getDataset bostonHousing

let sdata = "y" <~ map medianValue bh <>
            "x" <~ map getRow bh<>
            "n" <~ length bh <>
            "p" <~ length (getRow $ head bh)

res <- runStan linRegression sdata sample {numSamples = 100}
let resEnv = mcmcToEnv res
```


```haskell eval
plotly "bh" [points (aes & x .~ rooms & y .~ medianValue) bh]
```

```haskell eval
postPlotRow res ["beta.1", "beta.2" ] :: Html ()
```


```haskell eval
show res
```