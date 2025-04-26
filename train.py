import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.fftpack import fft
from statsmodels.tsa.stattools import pacf


def get_dfs(directory):
    
    # Get a list of all CSV files in the directory
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    
    # Extract the number from the filename and use it as the key
    return {int(file.split('_')[1].split('.')[0]): pd.read_csv(os.path.join(directory, file)) for file in csv_files}


def detect_optimal_lags(data, column, max_lag=8, threshold=0.15):
    """
    Automatically detect optimal lags using Partial Autocorrelation Function (PACF).
    
    Args:
        data (DataFrame): The time series data.
        column (str): The column name of the time series values.
        max_lag (int): Maximum number of lags to consider.
        threshold (float): The threshold for selecting significant lags.
        
    Returns:
        list: A list of optimal lags to use for feature creation.
    """
    # Calculate the PACF up to the max_lag
    pacf_values = pacf(data[column], nlags=max_lag)
    
    # Select lags where the PACF value is above the threshold
    optimal_lags = [lag for lag, value in enumerate(pacf_values) if abs(value) > threshold]
    
    # Exclude lag 0 since it refers to the current value, not a true lag
    optimal_lags = [lag for lag in optimal_lags if lag > 0]

    # If no lags are found, return a default lag (lag 1)
    if not optimal_lags:
        optimal_lags = [1]
    
    return optimal_lags



def detect_seasonality(data, column, freq_range=(3, 60)):
    """
    Detect seasonality using Fourier Transform to find the dominant frequency.
    
    Args:
        data (DataFrame): The time series data.
        column (str): The column name of the time series values.
        freq_range (tuple): The range of frequencies to search for seasonality.
        
    Returns:
        int: The detected seasonal period (number of samples in a seasonal cycle).
    """
    # Apply Fourier Transform to detect dominant frequencies
    values = data[column].values
    n = len(values)
    fft_values = fft(values)
    
    # Get frequencies and power spectrum
    freqs = np.fft.fftfreq(n)
    power_spectrum = np.abs(fft_values)
    
    # Focus on positive frequencies and within the freq_range
    valid_freqs = (freqs > 0) & (freqs >= 1/freq_range[1]) & (freqs <= 1/freq_range[0])
    
    # Find the dominant frequency
    dominant_freq = freqs[valid_freqs][np.argmax(power_spectrum[valid_freqs])]
    
    # The seasonality period (in number of samples)
    seasonality_period = int(1 / dominant_freq)
    
    return seasonality_period


def preprocess_data_extract_features(df, mode='train', optimal_lags=None):
    """
    Preprocess the input data, extracting lag features dynamically based on PACF and seasonality.
    
    Args:
        df (DataFrame): DataFrame containing 'time' and 'value' columns.
        dataset_id (str): Unique identifier for the dataset.
        mode (str): Operation mode - 'train' or 'test'.
        max_lag (int): Maximum number of lags to consider for PACF.
        pacf_threshold (float): Threshold for PACF to select optimal lags.
        optimal_lags (list): Predefined optimal lags for test mode. If None, detect lags in train mode.
        
    Returns:
        DataFrame, int: Processed data with additional features and minimum samples for inference.
    """
    # Rename and parse the timestamp column to match required request format
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'time'})

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.sort_values('time')

    # Interpolate missing values
    df['value'] = df['value'].interpolate(method='linear')

    # Remove rows marked as anomalies
    if 'anomaly' in df.columns:
        df = df[df['anomaly'] == False].drop(columns=['anomaly'])

    # Extract time-based features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month

    # Detect seasonality and add the seasonality period as a feature
    seasonality_period = detect_seasonality(df, 'value')
    df['seasonal_cycle'] = (np.arange(len(df)) % seasonality_period)

    # Interpolate missing values after seasonality extraction (if present)
    df['value'] = df['value'].interpolate(method='linear')

    if mode == 'train':
        # Detect optimal lags using PACF
        optimal_lags = detect_optimal_lags(df, 'value')
    elif optimal_lags is None:
        raise ValueError("Optimal lags must be provided in test mode.")

    # Generate lag features based on optimal lags
    for lag in optimal_lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)

    # Drop rows with missing values resulting from lag features
    df = df.dropna().reset_index(drop=True)

    # lags multiplied by 2 to compensate for samples that will be dropped after extraction
    min_samples_for_inference = max(optimal_lags)*2+seasonality_period if optimal_lags else seasonality_period

    return df, min_samples_for_inference, optimal_lags


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse


def train_models(dataframes):
    """
    Train machine learning models on the provided dataframes using TimeSeriesSplit and GridSearchCV 
    for hyperparameter tuning. Save the trained models, calculate metrics, and output results to CSV files.

    Args:
        dataframes (dict): A dictionary of DataFrames, where the keys are dataset identifiers and the 
                           values are the corresponding datasets to train on.
    """
    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "SVR": SVR()
    }

    param_grids = {
        "RandomForest": {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [5, 10],
        },
        "GradientBoosting": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 0.8],
        },
        "SVR": {
            'C': [0.1, 1],
            'kernel': ['linear', 'rbf'],
            'epsilon': [0.1, 0.5],
        }
    }

    all_metrics = []
    best_model_metrics = []
    summary = []
    lags_list = []

    for key, df in dataframes.items():
        print(f"processing df {key}")
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size]
        test_df = df[train_size:]

        # Preprocess training data
        train_preprocessed, min_samples, optimal_lags = preprocess_data_extract_features(train_df, mode='train')

        lags_list.append({"dataset_id": key, "optimal_lags": optimal_lags})
        summary.append({"dataset_id": key, "N": min_samples})

        X_train = train_preprocessed.drop(columns=['time', 'value'])
        y_train = train_preprocessed['value']

        tscv = TimeSeriesSplit(n_splits=3)
        best_model = None
        best_model_name = ""
        best_test_mse = float('inf')
        best_train_mse = float('inf')

        for model_name, model in models.items():
            param_grid = param_grids[model_name]

            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            trained_model = grid_search.best_estimator_

            # Evaluate on train set
            y_train_pred = trained_model.predict(X_train)
            train_mse = calculate_metrics(y_train, y_train_pred)

            # Preprocess testing data
            if not test_df.empty:
                test_preprocessed, _, _ = preprocess_data_extract_features(test_df, mode='test', optimal_lags=optimal_lags)

                if test_preprocessed.empty:
                    print(f"No test samples left for dataset {key}. Saving based on train results.")
                    if train_mse < best_train_mse:
                        best_train_mse = train_mse
                        best_model = trained_model
                        best_model_name = model_name
                    continue

                X_test = test_preprocessed.drop(columns=['time', 'value'])
                y_test = test_preprocessed['value']

                # Evaluate on test set
                y_test_pred = trained_model.predict(X_test)
                test_mse = calculate_metrics(y_test, y_test_pred)

                # Save metrics for all models
                all_metrics.append({
                    "dataset_id": key,
                    "model": model_name,
                    "train_mse": train_mse,
                    "test_mse": test_mse
                })

                # Track the best model based on test MSE
                if test_mse < best_test_mse:
                    best_test_mse = test_mse
                    best_train_mse = train_mse
                    best_model = trained_model
                    best_model_name = model_name
            else:
                # Save based on train MSE if no test data is available (only happened with 119)
                if train_mse < best_train_mse:
                    best_train_mse = train_mse
                    best_model = trained_model
                    best_model_name = model_name

        if best_model:
            # Save the best model
            output_model_path = f'models/{best_model_name}_{key}.pkl'
            joblib.dump(best_model, output_model_path)

            # Save metrics for the best model
            best_model_metrics.append({
                "dataset_id": key,
                "model": best_model_name,
                "train_mse": best_train_mse,
                "test_mse": best_test_mse if best_test_mse != float('inf') else None
            })

    # Save metrics for all models
    all_metrics_df = pd.DataFrame(all_metrics)
    all_metrics_df.to_csv("train_results/all_models_metrics.csv", index=False)

    # Save metrics for the best models
    best_metrics_df = pd.DataFrame(best_model_metrics)
    best_metrics_df.to_csv("train_results/best_models_metrics.csv", index=False)

    # Save N summary and lags summary for the datasets
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("meta_data/N_summary.csv", index=False)

    lags_df = pd.DataFrame(lags_list)
    lags_df.to_csv("meta_data/lags_summary.csv", index=False)

    # Calculate and save average metrics for the best models
    average_best_metrics = best_metrics_df[['train_mse', 'test_mse']].mean().reset_index()
    average_best_metrics.columns = ["metric", "average_value"]
    average_best_metrics.to_csv("train_results/average_best_models.csv", index=False)
    print(average_best_metrics)



if __name__ == '__main__':
    dataframes = get_dfs("train_splits")
    train_models(dataframes)