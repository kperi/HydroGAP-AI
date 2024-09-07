# HydroGAP-AI

TODO: Write a high level overview of the library


## Installation

You can install the library using the following command: 

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

`metrics_gaps` output: 

```json 
{'input_file_path': './data/lib/station_11.0_cleaned.csv',
 'missing_rate': 0.03592906707787201,
 'min_gap_length': 1,
 'mean_gap_length': 61.32618025751073,
 'median_gap_length': 59.0,
 'max_gap_length': 141,
 'std_gap_length': 37.822818002361764,
 'range_gap_length': 140,
 'gap_density': 0.03593460826650216,
 'nr_gap_days': 233,
 'nr_gaps': 2.0,
 'min_value': 0.0,
 'mean_value': 1679.2774152271274,
 'median_value': 94.05,
 'max_value': 11300.0,
 'std_value': 2567.0520096032656,
 'range_value': 11300.0,
 'skew_value': 1.5130792652939264,
 'kurtosis_value': 1.3201251228376147,
 'Q_lags': [1, 2, 3],
 'Q_lags_Coefficients': [0.996, -0.642, -0.169],
 'P_lags': [1,2,3],
}
```