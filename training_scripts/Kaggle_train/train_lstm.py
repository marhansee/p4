
import torch
import wandb
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import sys
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time

from lstm.model import LSTMModel
from lstm.dataloader import TimeSeriesDataset
from utils.preprocessing import split_data, preprocess_data
from utils.degradations import add_degradations
from utils.random_utils import load_config

warnings.filterwarnings('ignore')

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    epoch_loss = total_loss / len(train_loader)

    # Log the train loss
    wandb.log({"Epoch Train Loss": epoch_loss, "Epoch": epoch})
    

def evaluate(model, device, test_loader):
    model.eval()
    total_mae_lat = 0  # Track total MAE for latitude
    total_mae_lon = 0  # Track total MAE for longitude
    num_samples = 0  # Track total number of samples
    total_inference_time = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            start_time = time.time()
            output = model(data)

            batch_inference_time = time.time() - start_time
            total_inference_time += batch_inference_time
            # Assuming target and output have shape [batch_size, 2] (latitude, longitude)
            lat_target, lon_target = target[:, 0], target[:, 1]  # Split the target into latitude and longitude
            lat_output, lon_output = output[:, 0], output[:, 1]  # Split the output into latitude and longitude
            
            # Compute MAE for latitude and longitude
            lat_mae = F.l1_loss(lat_output, lat_target, reduction='sum').item()  # Sum over batch
            lon_mae = F.l1_loss(lon_output, lon_target, reduction='sum').item()  # Sum over batch
            
            total_mae_lat += lat_mae  # Accumulate latitude MAE
            total_mae_lon += lon_mae  # Accumulate longitude MAE
            num_samples += target.shape[0]  # Accumulate batch size
    
    # Compute the average MAE for both latitude and longitude
    avg_mae_lat = total_mae_lat / num_samples
    avg_mae_lon = total_mae_lon / num_samples
    
    avg_inference_time = (total_inference_time / num_samples) * 1000  # ms

    print(f'\nTest set: Average MAE Loss for Latitude: {avg_mae_lat:.4f}, Average MAE Loss for Longitude: {avg_mae_lon:.4f}\n')

    # Log evaluation loss to Weights & Biases
    wandb.log({"Eval MAE Loss Latitude": avg_mae_lat, "Eval MAE Loss Longitude": avg_mae_lon})

    return avg_mae_lat, avg_mae_lon, avg_inference_time

def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()  # You can also use MinMaxScaler() for different scaling


    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)  
    X_test_scaled = scaler.transform(X_test)  
    
    return X_train_scaled, X_val_scaled, X_test_scaled



def main():
    os.makedirs('lstm_results', exist_ok=True)
    os.makedirs('snapshots/lstm', exist_ok=True)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__),'train_config.yaml')
    config = load_config(config_path)

    # Make sure downsampling and resampling aren't both activated
    if config['degradations']['apply_downsampling'] and config['degradations']['apply_resampling']:
        raise AssertionError("apply_downsampling and apply_resampling cannot both be True!")
    
    if (config['degradations']["apply_downsampling"] or \
        config['degradations']["apply_resampling"]) and not \
            config['degradations']["apply_irregular_sampling"]:
        raise AssertionError("apply_irregular_sampling must be True before enabling apply_downsampling or apply_resampling!")
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__),'data/Trajectory_IDs.csv')

    # Split data
    train_df, val_df, test_df = split_data(data_path, random_state=42)

    # Add degradations (specified in YAML-file)
    train_df, val_df, test_df = add_degradations(
        config=config,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )
    
    # Specify valid sampling intervals
    valid_intervals = {30, 60, 180}

    # Horizon mapping based on horizon_time and sampling_interval
    horizon_mapping = {
        10: {30: 20, 60: 10, 180: 4},
        20: {30: 40, 60: 20, 180: 7}
    }

    # Check if sampling interval is valid
    if config['degradations']['sampling_interval'] not in valid_intervals:
        raise AssertionError("The sampling interval must be 30, 60, or 180!")

    # Determine horizon
    if config['horizon'] in horizon_mapping:
        if config['degradations']["apply_downsampling"] or \
                                    config['degradations']["apply_resampling"]:
            horizon = horizon_mapping[config['horizon']] \
                                     [config['degradations']['sampling_interval']]
        else:
            horizon = config['horizon'] * 6  # 10 → 60, 20 → 120
    else:
        raise AssertionError("Invalid horizon_time! Expected 10 or 20.")
    
    
    # Drop features
    train_df.drop(['id','label'],axis=1, inplace=True)
    val_df.drop(['id', 'label'], axis=1, inplace=True)
    test_df.drop(['id','label'], axis=1, inplace=True)

    # Define data settings and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = \
        preprocess_data(train_df, val_df, test_df, 
                        num_lags=config['lags'], 
                        horizon=horizon)

    # Scale data
    X_train_scaled, X_val_scaled, X_test_scaled = scale_data(X_train, X_val, X_test)
    

    # Load datasets
    train_dataset = TimeSeriesDataset(
        X=X_train_scaled,
        y=y_train,
        seq_length=config['train']['seq_length'],
        horizon=horizon
    )
    val_dataset = TimeSeriesDataset(
        X=X_val_scaled,
        y=y_val,
        seq_length=config['train']['seq_length'],
        horizon=horizon
    )

    # Load dataloaders
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=config['train']['batch_size'],
                              shuffle=True,
                              num_workers=config['train']['num_workers'],
                              pin_memory=True)
                              
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config['train']['batch_size'],
                            shuffle=False,
                            num_workers=config['train']['num_workers'],
                            pin_memory=True)
    
    # Initialize WandB
    wandb.login()
    wandb.init(project=config['wandb']['project'], config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['train']['seed'])

    # Initialize model
    lstm = LSTMModel(input_size=X_train.shape[1],
                     hidden_size=config['arch']['hidden_size'],
                     num_layers=config['arch']['num_layers'],
                     output_size=config['arch']['output_size']
                     ).to(device)
    
    # Define optimizer
    optimizer = Adam(lstm.parameters(), lr=config['train']['lr'])

    # Define name of saved file
    degradations = config["degradations"]
    sampling_interval = degradations["sampling_interval"]

    # Define naming rules
    naming_rules = {
        (degradations["apply_downsampling"], degradations["apply_irregular_sampling"]): f"DS{sampling_interval}",
        (degradations["apply_resampling"], degradations["apply_irregular_sampling"]): f"Linear{sampling_interval}",
        (degradations["add_missing_values"], degradations["add_duplicates"]): "MV0_Dupl",
        (degradations["add_missing_values"], False): "MV_mean",
        (False, degradations["add_duplicates"]): "Dupl",
        (degradations["apply_irregular_sampling"], False): "ir_sampling",
    }

    # Get the name based on conditions, defaulting to "Unprocessed"
    name = naming_rules.get(
        (True, True),
        naming_rules.get((True, False), naming_rules.get((False, True), "Unprocessed"))
    )

    # Format experiment name
    experiment_name = f"{name}_horizon{config['horizon']}"
    print(experiment_name)

    # Define folder path to save the results and weights
    results_path = os.path.join('lstm_results', f'{experiment_name}.txt')
    weight_path = os.path.join('snapshots/lstm', f'{experiment_name}.pth')


    # Training loop
    print("Initializing training...")

    best_val_loss = float('inf')
    for epoch in range(1, config['train']['num_epochs']+1):
        train(lstm, device, train_loader, optimizer, epoch)
        avg_mae_lat, avg_mae_lon, avg_inference_time = evaluate(lstm, device, val_loader)

        val_loss = avg_mae_lat + avg_mae_lon  

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(lstm.state_dict(), weight_path)  # Save model weights
            print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")

            print("Writing results")
            with open(results_path, "w") as f:  # Overwrite file to keep only the best result
                f.write(f"Experiment name: {experiment_name}\n")
                f.write(f"MAE - Lat: {avg_mae_lat:.4f}, Long: {avg_mae_lon:.4f}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Inference time (ms): {avg_inference_time:.2f}\n")

    print("Completed training.")



if __name__ == '__main__':
    main()