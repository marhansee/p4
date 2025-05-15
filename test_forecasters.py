import torch
import os
from torch.utils.data import DataLoader
import warnings
import time
import glob
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import torch.nn.functional as F

# Load utils
from utils.train_utils import load_config_file, load_scaler_json, load_data, scale_data, make_sequences2, inverse_scale_lat_lon
from utils.data_loader import Classifier_Dataloader_with_MMSI

warnings.filterwarnings('ignore')


def inference_onnx(onnx_session, device, test_loader):
    onnx_session.set_providers(['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'])

    num_samples = 0
    total_inference_time = 0
    total_mae_lat = 0
    total_mae_lon = 0

    mae_lat_list = []
    mae_lon_list = []
    mae_mean_list = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = target.size(0)
            target = target.view(batch_size, 2, 20).transpose(1, 2)  # Shape: [B, 20, 2]

            start_time = time.time()
            inputs = {onnx_session.get_inputs()[0].name: data.cpu().numpy()}
            output = onnx_session.run(None, inputs)[0]  # Shape: [B, 20, 2]
            batch_inference_time = time.time() - start_time
            total_inference_time += batch_inference_time

            lat_target = target[:, :, 0]  # [B, 20]
            lon_target = target[:, :, 1]  # [B, 20]
            lat_output = torch.from_numpy(output[:, :, 0]).to(device)  # [B, 20]
            lon_output = torch.from_numpy(output[:, :, 1]).to(device)  # [B, 20]

            # Compute MAE per prediction (per timestep)
            lat_mae_batch = torch.abs(lat_output - lat_target)  # [B, 20]
            lon_mae_batch = torch.abs(lon_output - lon_target)  # [B, 20]
            mean_mae_batch = (lat_mae_batch + lon_mae_batch) / 2  # [B, 20]

            # Flatten and accumulate
            mae_lat_list.extend(lat_mae_batch.view(-1).tolist())
            mae_lon_list.extend(lon_mae_batch.view(-1).tolist())
            mae_mean_list.extend(mean_mae_batch.view(-1).tolist())

            total_mae_lat += lat_mae_batch.sum().item()
            total_mae_lon += lon_mae_batch.sum().item()
            num_samples += lat_target.numel()

    avg_mae_lat = total_mae_lat / num_samples
    avg_mae_lon = total_mae_lon / num_samples
    avg_mae_total = (avg_mae_lat + avg_mae_lon) / 2
    avg_inference_time = (total_inference_time / num_samples) * 1000  # ms

    var_mae_lat = np.var(mae_lat_list)
    var_mae_lon = np.var(mae_lon_list)
    var_mae_mean = np.var(mae_mean_list)

    print(f'Average inference time (ms): {avg_inference_time:.4f}')
    print(f'Lat MAE variance: {var_mae_lat:.6f}')
    print(f'Lon MAE variance: {var_mae_lon:.6f}')
    print(f'Mean MAE variance: {var_mae_mean:.6f}')

    return avg_mae_lat, avg_mae_lon, avg_mae_total, avg_inference_time, \
        var_mae_lat, var_mae_lon, var_mae_mean, mae_lat_list, mae_lon_list, mae_mean_list


def plot_vessel_trajectory(mmsi, mmsi_array, X_sequences, y_labels, y_preds, seq_len, scaler, future_steps=20):
    """
    Plot input, ground truth, and predicted trajectories for a given MMSI.
    """
    import matplotlib.pyplot as plt

    # Filter the sequences for this MMSI
    indices = np.where(mmsi_array == mmsi)[0]

    if len(indices) == 0:
        print(f"No sequences found for MMSI {mmsi}")
        return

    idx = indices[-1]

    input_seq = X_sequences[idx]  # shape: [seq_len, features]
    true_future = y_labels[idx].reshape(future_steps, 2)  # shape: [20, 2]
    pred_future = y_preds[idx]  # shape: [20, 2]

    # Extract lat/lon from input (assuming lat = 0, lon = 1)
    lat_input_scaled = input_seq[:, 0]
    lon_input_scaled = input_seq[:, 1]

    lat_input, lon_input = inverse_scale_lat_lon(lat_input_scaled, lon_input_scaled, scaler)
    lat_true, lon_true = inverse_scale_lat_lon(true_future[:, 0], true_future[:, 1], scaler)
    lat_pred, lon_pred = inverse_scale_lat_lon(pred_future[:, 0], pred_future[:, 1], scaler)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(lon_input, lat_input, 'bo-', label='Input Trajectory (Past)', alpha=0.7)
    plt.plot(lon_true, lat_true, 'g*-', label='True Future Trajectory')
    plt.plot(lon_pred, lat_pred, 'r^-', label='Predicted Future Trajectory')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Vessel Trajectory for MMSI: {mmsi}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test Forecaster')
    parser.add_argument('--snapshot_name', type=str, required=True, help='Name of model you want to test')
    parser.add_argument('--seq_length', type=int, required=True, help="Input sequence length")
    args = parser.parse_args()

    # Make folders for results and snapshots
    results_dir = f"results/forecasting_results/test/{args.snapshot_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Load scaler [FIX PATH]
    scaler_path = os.path.join(os.path.dirname(__file__),'data/norm_stats/v4/train_norm_stats.json')
    scaler = load_scaler_json(scaler_path)
    
    # Load data
    test_data_folder_path = os.path.abspath('data/petastorm/test/v4')
    test_parquet_files = glob.glob(os.path.join(test_data_folder_path, '*.parquet'))

    test_parquet_files.sort()

    input_features = ['MMSI', 'timestamp_epoch','Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading', 
                      'Width', 'Length', 'Draught']
    
    target_features = [f'future_lat_{i}' for i in range(6, 121, 6)] + \
           [f'future_lon_{i}' for i in range(6, 121, 6)]
    
    X_test, y_test = load_data(
        parquet_files=test_parquet_files,
        input_features=input_features,
        target_columns=target_features
    )

    # Scale input features
    X_test_scaled = scale_data(scaler, X_test)

    X_test, y_test, mmsi_test = make_sequences2(X_test_scaled, y_test, seq_len=args.seq_length, group_col='MMSI')

    test_dataset = Classifier_Dataloader_with_MMSI(
        X_sequences=X_test,
        y_labels=y_test,
        mmsi_array=mmsi_test
    )

    # Load dataloaders                              
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=3512,
                            shuffle=False,
                            num_workers=20,
                            pin_memory=True)
    

    # ONNX model path
    onnx_model_path = f'models/forecasters/{args.snapshot_name}.onnx'  # Replace with the actual ONNX model path
    onnx_session = ort.InferenceSession(onnx_model_path)

    # Format experiment name
    experiment_name = f"{args.snapshot_name}_test"
    print(experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define folder path to save the results and weights
    results_path = os.path.join(results_dir, f"{experiment_name}.txt")

    # Evaluate ONNX model
    print("Initializing testing session...")
    avg_mae_lat, avg_mae_lon, avg_mae_total, avg_inference_time, \
    var_mae_lat, var_mae_lon, var_mae_mean, mae_lat_list, mae_lon_list, \
    mae_mean_list = inference_onnx(onnx_session, device, test_loader)


    # Write results to file
    print("Writing results")
    with open(results_path, "w") as f:  # Overwrite file to keep only the best result
        f.write(f"Experiment name: {experiment_name}\n")
        f.write(f"Average MAE for lat: {avg_mae_lat:.4f}\n")
        f.write(f"Average MAE for lon: {avg_mae_lon:.4f}\n")
        f.write(f"Average MAE between lat/lon: {avg_mae_total:.4f}\n")
        f.write(f"Variance MAE for lat: {var_mae_lat:.4f}\n")
        f.write(f"Variance MAE for lon: {var_mae_lon:.4f}\n")
        f.write(f"Variance MAE between lat/lon: {var_mae_mean:.4f}\n")
        f.write(f"MAE for lat per step: {mae_lat_list:.4f}\n")
        f.write(f"MAE for lon per step: {mae_lon_list:.4f}\n")
        f.write(f"MAE for mean lon/lon per step: {mae_mean_list:.4f}\n")
        f.write(f"Inference time (ms): {avg_inference_time}\n")

    # Plot confusion matrix and PR-curve
    save_image_path = os.path.join(results_dir, f"{experiment_name}_visualization.png")


    


if __name__ == '__main__':
    main()
