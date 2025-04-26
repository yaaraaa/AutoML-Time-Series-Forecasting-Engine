from flask import Flask, request, jsonify
import pandas as pd
import os
import joblib
from train import preprocess_data_extract_features

app = Flask(__name__)

MODELS_DIR = "models"

def load_model(dataset_id):
    """
    Load a pre-trained model based on the given dataset ID.
    
    Args:
        dataset_id (str): Identifier for the dataset/model.
        
    Returns:
        model: Loaded model or None if the model file does not exist.
        
    Raises:
        ValueError: If the loaded object is not a valid model.
    """

    # Construct the directory path and search for a matching model file
    model_files = os.listdir(MODELS_DIR)
    matching_files = [file for file in model_files if f"_{dataset_id}." in file]
    
    if not matching_files:
        return None
    
    model_path = os.path.join(MODELS_DIR, matching_files[0])
    
    model = joblib.load(model_path)
    
    # Ensure the loaded object has the predict method
    if not hasattr(model, 'predict'):
        raise ValueError(f"Loaded object is not a valid model, got {type(model)} instead")
    
    return model



def get_N(dataset_id): 
    """
    Retrieve the 'N' value (e.g., minimum required data points) for the specified dataset.
    
    Args:
        dataset_id (str): Identifier for the dataset.
        
    Returns:
        int: The 'N' value if found, otherwise None.
    """
    summary_df = pd.read_csv('meta_data\\N_summary.csv')

    # Look up 'N' value based on the dataset ID
    result = summary_df.loc[summary_df['dataset_id'] == int(dataset_id), 'N']

    if not result.empty:
        return result.values[0]
    return None


def get_optimal_lags(dataset_id):
    """
    Load optimal lags for a specific dataset from a CSV file.

    Args:
        dataset_id (str): Unique identifier of the dataset.

    Returns:
        list: A list of optimal lags for the dataset or None if not found.
    """
    # Load the CSV file containing optimal lags
    optimal_lags_df = pd.read_csv('meta_data\\lags_summary.csv')
    # Find the row corresponding to the dataset_id
    row = optimal_lags_df[optimal_lags_df['dataset_id'] == dataset_id]
    if row.empty:
        return None
    # Convert the 'optimal_lags' column (a string) into a list of integers
    optimal_lags = eval(row['optimal_lags'].values[0]) 
    if not optimal_lags:
        optimal_lags = [1]
    return optimal_lags
    


@app.route('/forecast', methods=['POST'])
def forecast():
    """
    Endpoint to perform time series forecasting based on input data.

    This endpoint accepts a POST request with a JSON body containing 
    a dataset ID and a list of time series data points. It preprocesses the data, 
    extracts features, and uses a pre-trained model to predict future values.

    Args:
        - dataset_id (str): A unique identifier for the dataset to load the correct model.
        - values (list): A list of dictionaries containing 'time' and 'value' keys 
          representing time series data points.

    Returns:
        - A JSON response containing the predicted value.
    """

    # Extract request data
    data = request.json
    dataset_id = data.get("dataset_id")
    N = get_N(dataset_id)
    values = data.get("values")
    
    if not dataset_id or not N or values is None:
        return jsonify({"error": "Invalid input parameters"}), 400
    
    # Create a DataFrame from the input values
    df = pd.DataFrame(values)
    if not all(key in df.columns for key in ['time', 'value']):
        return jsonify({"error": "Invalid data format in 'values'"}), 400
    
    if len(df) < N:
        return jsonify({"error": "Insufficient data entered"}), 400

    # Load the optimal lags for the dataset from the CSV
    optimal_lags = get_optimal_lags(dataset_id)
    
    try:
        # Preprocess the input data with the loaded optimal lags
        df_preprocessed, _, _ = preprocess_data_extract_features(df, mode='test', optimal_lags=optimal_lags)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 500
    
    input_features = df_preprocessed.drop(columns=['time', 'value']).values

    model = load_model(dataset_id)
    if model is None:
        return jsonify({"error": "Model not found"}), 404
    
    try:
        # Print the expected number of features by the model
        prediction = model.predict(input_features)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    return jsonify({"prediction": prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)


