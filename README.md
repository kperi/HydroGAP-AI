# HydroGAP-AI

TODO: Write a high level overview of the library

## Installation

You can install the library using the following command:

```bash
pip install hydrogap-ai
```

or directly from this repo using: 

```bash
pip install git+https://github.com/kperi/HydroGAP-AI.git
```

## Predict gaps in stations

### Usage

```python

from hydroai.gap_prediction import predict_station_gaps
station_file = "./data/lib/station_11.0_cleaned.csv"

reall_combined_dfs, val_full, metrics_gaps, real_predictionst = predict_station_gaps(
    station_file, model_type="rf"
)
```

For a more detailed example of the library including outputs see this [notebook](./lib_demo.ipynb)