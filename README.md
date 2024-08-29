# HydroGAP-AI

TODO: Write a high level overview of the library

 
## Feature engineering

Given a station file with values sorted by time, the file is split in contiguous blocks (pandas dataframes) of observed discharge (`obsdis`) values. For each one of the resulting blocks the corresponding features are generated using previous day lag values. All features are merged into a singe dataframe which is then used to train the correspoding station model. 


## Usage 


```bash      
SYNOPSIS
    single_station_model.py <flags>

FLAGS
    -d, --data_path=DATA_PATH
        Type: str
        Default: 'data/parquet.raw'
    -m, --model_type=MODEL_TYPE
        Default: 'RandomForestRegressor'
    -s, --station_name=STATION_NAME
        Type: Optional[str]
        Default: None
    -a, --all_stations=ALL_STATIONS
        Type: bool
        Default: False
    -p, --persist_model=PERSIST_MODEL
        Type: bool
        Default: False
```

### Example

Train models for all stations: 

```bash 
python single_station_model.py --all-stations
``` 

Train a model for a single station: 

```bash 
python single_station_model.py  --station_name=data/parquet.raw/station_518 --persist_model
``` 



## 1-day gap prediction

### Example prediction 
 

```python
import joblib

station_model_name = "station_518.pkl"
model = joblib.load( station_model_name ) 
test_df  = pd.DataFrame( [{
    "P_lag_time_1": 2.372996,
    "P_lag_time_2": 9.947697,
    "P_lag_time_3": 0.009867,
    "Q_lag_time_1": 68.500000,
    "Q_lag_time_2": 79.199997,
    "Q_lag_time_3": 76.800003,
    
}])

model.predict(test_df) # array([72.47600025])

``` 



## 2-weeks gap prediction

For each station a  RandomForest or GradientBoosting regressor model is trained, using previous days lags.
Here, the model is trained to predict the next 14 days

### Example prediction


```python

import joblib

station_model_name = "mo_station_518.pkl"
mo_regressor = joblib.load( station_model_name ) 

test_df  = pd.DataFrame( [{
    "P_lag_time_1": 2.372996,
    "P_lag_time_2": 9.947697,
    "P_lag_time_3": 0.009867,
    "Q_lag_time_1": 68.500000,
    "Q_lag_time_2": 79.199997,
    "Q_lag_time_3": 76.800003,
    
}])

mo_regressor.predict(test_df) 
# array([[ 72.47,  889.4 ,  908.11,  955.46,  973.27, 1036.01, 1041.62,
#        1069.79, 1088.86, 1111.55, 1162.48, 1178.34, 1174.13, 1195.35]])
``` 