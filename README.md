# HydroGAPML


## 1-day ahead forecast

For each station a  RandomForest or GradientBoosting regressor model is trained, using previous days lags.


### Usage


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