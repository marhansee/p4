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
import sys
from tqdm import tqdm

# Load utils
from utils.train_utils import load_config_file, load_scaler_json, load_data, scale_data, make_sequences, inverse_scale_lat_lon
from utils.data_loader import System_Dataloader

warnings.filterwarnings('ignore')



def inference_onnx(model_path, dataloader, args, device='cpu'):
    """
    Runs inference on a dataset using an ONNX model session.

    Args:
        onnx_session (onnxruntime.InferenceSession): Initialized ONNX runtime session.
        device (torch.device): Device on which tensors are moved for consistency (e.g., 'cuda' or 'cpu').
        test_loader (torch.utils.data.DataLoader): DataLoader providing test data batches.

    Returns:
        all_preds (list): List of predicted labels for the entire test set.
        all_labels (list): List of true labels for the entire test set.
        avg_inference_time (float): Average inference time per sample in milliseconds.
    """
    
    # Load ONNX model
    providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)

    mae_lat_list, mae_lon_list, mae_mean_list = [], [], []
    inference_times = []

    for inputs, targets in tqdm(dataloader, desc="Running inference"):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = targets.size(0)
        targets = targets.view(batch_size, 2, args.output_seq_len).transpose(1, 2)

        inputs_np = inputs.cpu().numpy().astype(np.float32)
        # targets_np = targets.numpy()

        start_time = time.time()
        outputs_np = session.run(None, {"input": inputs_np})[0]
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Convert outputs to tensor
        outputs = torch.from_numpy(outputs_np).to(targets.device)
   

        lat_target = targets[:, :, 0]
        lon_target = targets[:, :, 1]
        lat_output = outputs[:, :, 0]
        lon_output = outputs[:, :, 1]
        
        lat_mae_batch = torch.abs(lat_output - lat_target)
        lon_mae_batch = torch.abs(lon_output - lon_target)
        mean_mae_batch = (lat_mae_batch + lon_mae_batch) / 2.0

        mae_lat_list.append(lat_mae_batch)
        mae_lon_list.append(lon_mae_batch)
        mae_mean_list.append(mean_mae_batch)

    # Stack all MAEs: shape [N, 20]
    lat_tensor = torch.cat(mae_lat_list, dim=0)  # [N, 20]
    lon_tensor = torch.cat(mae_lon_list, dim=0)
    mean_tensor = torch.cat(mae_mean_list, dim=0)

    # Per-step (across all samples) MAE
    stepwise_mae_lat = lat_tensor.mean(dim=0).tolist()
    stepwise_mae_lon = lon_tensor.mean(dim=0).tolist()
    stepwise_mae_mean = mean_tensor.mean(dim=0).tolist()

    # Overall average MAE
    avg_mae_lat = lat_tensor.mean().item()
    avg_mae_lon = lon_tensor.mean().item()
    avg_mae_total = mean_tensor.mean().item()

    # Variance (spread) of MAE values
    var_mae_lat = lat_tensor.var().item()
    var_mae_lon = lon_tensor.var().item()
    var_mae_mean = mean_tensor.var().item()

    avg_inference_time = np.mean(inference_times)

    return avg_mae_lat, avg_mae_lon, avg_mae_total, avg_inference_time, \
           var_mae_lat, var_mae_lon, var_mae_mean, \
           stepwise_mae_lat, stepwise_mae_lon, stepwise_mae_mean


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
    parser.add_argument('--output_seq_len', type=int, required=True, help="Output sequence length in minutes")
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

    # Define input features
    input_features = ['MMSI', 'timestamp_epoch','Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading', 
                     'Width', 'Length', 'Draught']

    
    # Define mapping for output sequence length in minutes (just for automation)
    mapping = {
        1: 6,
        5: 30,
        10: 60,
        15: 90,
        20: 120
    }

    # Define targets
    target_features = [f'future_lat_{i}' for i in range(6, mapping[args.output_seq_len]+1, 6)] + \
           [f'future_lon_{i}' for i in range(6, mapping[args.output_seq_len]+1, 6)]
    
    # Load data
    X_test, y_test = load_data(
        parquet_files=test_parquet_files,
        input_features=input_features,
        target_columns=target_features
    )

    # Scale input features
    X_test_scaled = scale_data(scaler, X_test)

    X_test, y_test = make_sequences(X_test_scaled, y_test, seq_len=args.seq_length, group_col='MMSI')

    test_dataset = System_Dataloader(
        X_sequences=X_test,
        y_labels=y_test,
    )

    # Load dataloaders                              
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=3512, # True 3512, debugging=1
                            shuffle=False,
                            num_workers=20,
                            pin_memory=True)
    

    # ONNX model path
    onnx_model_path = f'models/forecasters/{args.snapshot_name}.onnx' 

    # Format experiment name 
    experiment_name = f"{args.snapshot_name}_test"
    print(experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define folder path to save the results and weights
    results_path = os.path.join(results_dir, f"{experiment_name}.txt")

    # Evaluate ONNX model
    print("Initializing testing session...")
    avg_mae_lat, avg_mae_lon, avg_mae_total, avg_inference_time, \
    var_mae_lat, var_mae_lon, var_mae_mean, \
    stepwise_mae_lat, stepwise_mae_lon, \
    stepwise_mae_mean = inference_onnx(onnx_model_path, test_loader, args, device)


    # Write results to file
    print("Writing results")
    with open(results_path, "w") as f:
        f.write(f"Avg MAE lat: {avg_mae_lat:.6f}\n")
        f.write(f"Avg MAE lon: {avg_mae_lon:.6f}\n")
        f.write(f"Avg MAE mean: {avg_mae_total:.6f}\n")
        f.write(f"Avg inference time (s): {avg_inference_time:.6f}\n")
        f.write(f"MAE lat variance: {var_mae_lat:.6f}\n")
        f.write(f"MAE lon variance: {var_mae_lon:.6f}\n")
        f.write(f"MAE mean variance: {var_mae_mean:.6f}\n\n")
        f.write(f"Stepwise MAE lat: {stepwise_mae_lat}\n")
        f.write(f"Stepwise MAE lon: {stepwise_mae_lon}\n")
        f.write(f"Stepwise MAE mean: {stepwise_mae_mean}\n")



if __name__ == '__main__':
    main()

