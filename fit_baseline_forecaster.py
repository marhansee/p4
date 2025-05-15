import os
import glob
import time
from utils.train_utils import load_scaler_json, load_data, scale_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.multioutput import MultiOutputRegressor

def main():
       # Make folders for results and snapshots
    results_dir = "results/forecasting_results/test/LinReg_AIS_baseline"
    os.makedirs(results_dir, exist_ok=True)

    # Load scaler [FIX PATH]
    scaler_path = os.path.join(os.path.dirname(__file__),'data/norm_stats/v4/train_norm_stats.json')
    scaler = load_scaler_json(scaler_path)
    
    # Load data
    train_data_folder_path = os.path.abspath('data/petastorm/train/v4') # FIX PATH
    train_parquet_files = glob.glob(os.path.join(train_data_folder_path, '*.parquet'))
    test_data_folder_path = os.path.abspath('data/petastorm/test/v4')
    test_parquet_files = glob.glob(os.path.join(test_data_folder_path, '*.parquet'))

    test_parquet_files.sort()
    train_parquet_files.sort()

    input_features = ['MMSI', 'timestamp_epoch','Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading', 
                      'Width', 'Length', 'Draught']
    
    target_features = [f'future_lat_{i}' for i in range(6, 121, 6)] + \
           [f'future_lon_{i}' for i in range(6, 121, 6)]
    
    X_train, y_train = load_data(
        parquet_files=train_parquet_files,
        input_features=input_features,
        target_columns=target_features
    )

    X_test, y_test = load_data(
        parquet_files=test_parquet_files,
        input_features=input_features,
        target_columns=target_features
    )

    # Scale input features
    X_train_scaled = scale_data(scaler, X_train)
    X_test_scaled = scale_data(scaler, X_test)

    model = MultiOutputRegressor(LinearRegression(n_jobs=-1))
    print("Fitting model")
    model.fit(X_train_scaled, y_train)
    print("Model has been fitted!")

    start_time = time.perf_counter()
    y_pred = model.predict(X_test_scaled)
    end_time = time.perf_counter()

    inference_time_ms = (end_time - start_time) * 1000 / len(X_test_scaled)  # ms per sample
    
    # Evaluate MAE for each output dimension
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    avg_mae = mae.mean()

    print(f"Average MAE: {avg_mae:.4f}")

    # Optionally: split MAE into lat/lon parts
    num_targets = y_test.shape[1] // 2
    lat_mae = mae[:num_targets]
    lon_mae = mae[num_targets:]

    avg_mae_lat = lat_mae.mean()
    avg_mae_lon = lon_mae.mean()
    avg_mae_total = (avg_mae_lat + avg_mae_lon) / 2
    
    print(f"Lat MAE: {lat_mae:.4f} | Lon MAE: {lon_mae:.4f}")

    results_path = os.path.join(results_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"Experiment name: Linear Regression\n")
        f.write(f"Average MAE for lat: {avg_mae_lat:.4f}\n")
        f.write(f"Average MAE for lon: {avg_mae_lon:.4f}\n")
        f.write(f"Average MAE between lat/lon: {avg_mae_total:.4f}\n")
        f.write(f"MAE for lat per step: {[f'{x:.4f}' for x in lat_mae]}\n")
        f.write(f"MAE for lon per step: {[f'{x:.4f}' for x in lon_mae]}\n")
        f.write(f"Inference time per sample (ms): {inference_time_ms:.4f}\n")

    print("Converting model to ONNX")

    # Define the input type with shape (None, num_features)
    initial_type = [('input', FloatTensorType([None, X_train_scaled.shape[1]]))]

    # Convert the model
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save the ONNX model to file
    snapshot_path = 'models/forecasters'
    save_path = os.path.join(snapshot_path, "LinReg_baseline_AIS.onnx")
    with open(save_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

if __name__ == '__main__':
    main()