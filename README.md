# AutoML Time Series Forecasting Engine

A Python-based forecasting engine for time series data, built with Flask and scikit-learn. The system automates forecasting for hundreds of datasets with varying temporal patterns and seasonalities using intelligent feature engineering, model training, and dynamic inference APIs.

## Features 
- Automated Forecasting for 500+ datasets with diverse time-based patterns.

- Dynamic Lag Detection & Seasonality Analysis via PACF and FFT.

- AutoML Workflow: automated hyperparameter tuning and model selection from multiple machine learning regressors.

- RESTful API for real-time forecasting requests.

- Model Training & Inference Tracking with saved results and metadata.


## Python Version 
- 3.10.0

## Modules
- `/train_results`: Stores training MSE results for each dataset.
- `app.py`: Main Flask app file that manages API endpoints for model forecasting and loads trained models.
  - `/forecast`: POST endpoint for predicting time series data based on input and a dataset ID.
- `/meta_data`: A summary directory including the "N" value (minimum data points needed for a successful inference), and the optimal lags for each dataset.
- `train.py`: Script responsible for model training, preprocessing data, and saving trained models.
- `requirements.txt`: Contains all the required dependcies to be installed.


## Setup

### Install required dependencies

```
pip install -r requirements.txt
```

### Run the Flask app
```
python app.py
```

## API Endpoints
- `POST /forecast`: Predicts time series values based on the input data.
  - `Body`: 
    - dataset_id (str): Identifier of the dataset used to select the model.
    - values (list of dict): Time series data containing 'time' and 'value' fields.
- `Response`: JSON with predicted values or error message.




