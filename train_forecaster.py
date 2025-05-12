
import torch
import wandb
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import sys
import warnings
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pyarrow.parquet as pq
import glob

# Load model architectures
from archs.lstm_forecaster import LSTMModel
from archs.bigru_forecast import BiGRUModel
from archs.cnn_forecast import CNN1DForecaster

# Load utils
from utils.train_utils import load_config_file, load_scaler_json, \
    load_data, scale_data
from utils.data_loader import Forecasting_Dataloader

warnings.filterwarnings('ignore')




def train(model, device, train_loader, optimizer, scheduler, epoch, scaler):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Implement AMP
        with torch.amp.autocast('cuda'):
            output = model(data)
            loss = F.mse_loss(output, target)

        # Scale gradients and apply backprop
        scaler.scale(loss).backward()

        # Unscale gradients and update weights
        scaler.step(optimizer)

        # Update scaler
        scaler.update()


        total_loss += loss.item()

        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    epoch_loss = total_loss / len(train_loader)

    # Log the train loss
    wandb.log({"Epoch Train Loss": epoch_loss, "Epoch": epoch})
    

def evaluate(model, device, test_loader):
    model.eval()
    total_mae_lat = 0
    total_mae_lon = 0
    num_samples = 0
    total_inference_time = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            start_time = time.time()
            output = model(data)
            batch_inference_time = time.time() - start_time
            total_inference_time += batch_inference_time

            lat_target, lon_target = target[:, :, 0], target[:, :, 1] # output shape: [batch, 20, 2]
            lat_output, lon_output = output[:, :, 0], output[:, :, 1]

            lat_mae = F.l1_loss(lat_output, lat_target, reduction='sum').item()
            lon_mae = F.l1_loss(lon_output, lon_target, reduction='sum').item()

            total_mae_lat += lat_mae
            total_mae_lon += lon_mae
            num_samples += target.shape[0] * target.shape[1] # recall output shape: [batch, 20, 2]

    avg_mae_lat = total_mae_lat / num_samples
    avg_mae_lon = total_mae_lon / num_samples
    avg_inference_time = (total_inference_time / num_samples) * 1000  # ms

    print(f'\nTest set: Average MAE Loss for Latitude: {avg_mae_lat:.4f}, Average MAE Loss for Longitude: {avg_mae_lon:.4f}\n')
    wandb.log({
        "Eval MAE Loss Latitude": avg_mae_lat,
        "Eval MAE Loss Longitude": avg_mae_lon,
        "Avg Inference Time (ms)": avg_inference_time
    })

    return avg_mae_lat, avg_mae_lon, avg_inference_time


def main():

    # Load config
    config_path = os.path.join(os.path.dirname(__file__),'train_config.yaml')
    config = load_config_file(config_path)

    # Load scaler [FIX PATH]
    scaler_path = os.path.join(os.path.dirname(__file__),'metadata.json')
    scaler = load_scaler_json(scaler_path)

    # Make folders for results and snapshots
    os.makedirs(f"forecast_results/{config['model_name']}", exist_ok=True)
    os.makedirs(f"snapshots/forecast/{config['model_name']}", exist_ok=True)

    # Load data
    train_data_folder_path = os.path.join(os.path.dirname(__file__), 'train_data')
    train_parquet_files = glob.glob(os.path.join(train_data_folder_path, '*.parquet'))
    val_data_folder_path = os.path.join(os.path.dirname(__file__), 'val_data')
    val_parquet_files = glob.glob(os.path.join(train_data_folder_path, '*.parquet'))

    input_features = ['timestamp_epoch', 'MMSI', 'Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading', 
                      'Width', 'Length', 'Draught']
    features_to_scale = [feature for feature in input_features if feature not in ['timestamp_epoch', 'MMSI']]
    target_features = [f'future_lat{i}' for i in range(1, 21)]
    
    X_train, y_train = load_data(
        parquet_files=train_parquet_files,
        input_features=input_features,
        target_columns=target_features
    )

    X_val, y_val = load_data(
        parquet_files=val_parquet_files,
        input_features=input_features,
        target_columns=target_features
    )

    # Scale input features
    X_train_scaled = scale_data(scaler, X_train, features_to_scale)
    X_val_scaled = scale_data(scaler, X_val, features_to_scale)

    # Drop timestamp and MMSI
    X_train_scaled = np.delete(X_train_scaled, ['MMSI','timestamp_epoch'], axis=1)
    X_val_scaled = np.delete(X_val_scaled, ['MMSI','timestamp_epoch'], axis=1)
   
    # Load datasets
    train_dataset = Forecasting_Dataloader(
        X=X_train_scaled,
        y=y_train,
        seq_length=config['arch_param']['seq_len'],
    )

    val_dataset = Forecasting_Dataloader(
        X=X_val_scaled,
        y=y_val,
        seq_length=config['arch_param']['seq_len'],
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

    torch.manual_seed(config['train']['seed'])

    # Initialize ranks and process groups
    torch.cuda.set_device(int(os.environ[config['ddp']['set_device']]))
    dist.init_process_group(config['ddp']['process_group'])
    rank = dist.get_rank()

    # Define device ID and load model with device
    device_id = rank % torch.cuda.device_count()

    # Select model based on configuration
    if config['model_name'].lower() == 'lstm':
        # Initialize model
        model = LSTMModel(
            n_features=config['arch_param']['n_features'],
            hidden_size=config['arch_param']['hidden_size'],
            num_layers=config['arch_param']['num_layers'],
            output_seq_len=config['arch_param']['output_seq_len'],
            output_size=config['arch_param']['output_size'],
            dropout_prop=config['train']['dropout_prop']
        ).to(device_id)
    elif config['model_name'].lower() == 'bigru':
        model = BiGRUModel(
            n_features=config['arch_param']['n_features'],
            hidden_size=config['arch_param']['hidden_size'],
            num_layers=config['arch_param']['num_layers'],
            output_seq_len=config['arch_param']['output_seq_len'],
            output_size=config['arch_param']['output_size'],
            dropout_prop=config['train']['dropout_prop']
        ).to(device_id)
    elif config['model_name'].lower() == '1dcnn':
        model = CNN1DForecaster(
            n_features=config['arch_param']['n_features'],
            seq_len=config['arch_param']['seq_len'],
            out_channels=config['arch_param']['out_channels'],
            output_size=config['arch_param']['output_size'],
            output_seq_len=config['arch_param']['output_seq_len']
        ).to(device_id)
    else:
        raise AssertionError('Model must be either "lstm", "bigru", "1dcnn"')
    
    # Wrap model in DDP
    ddp_model = DDP(model, device_ids=[device_id])

    # Define scaler for Automatic Mixed Precision
    scaler = torch.amp.GradScaler('cuda')
    
    # Define optimizer and lr scheduler
    optimizer = Adam(ddp_model.parameters(), lr=config['train']['lr'])
    scheduler = StepLR(
        optimizer=optimizer,
        step_size=config['scheduler']['step_size'],
        gamma=config['scheduler']['gamma'])

    # Format experiment name
    experiment_name = f"{config['model_name']}_{config['experiment_name']}"
    print(experiment_name)

    # Define folder path to save the results and weights
    results_path = os.path.join(f"forecast_results/{config['model_name']}", f"{experiment_name}.txt")
    weight_path = os.path.join(f"snapshots/forecast/{config['model_name']}", f"{experiment_name}.pth")


    # Training loop
    print("Initializing training...")

    best_val_loss = float('inf')
    for epoch in range(1, config['train']['num_epochs']+1):
        train(
            model=ddp_model,
            device=device_id,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            scaler=scaler,
        )

        avg_mae_lat, avg_mae_lon, avg_inference_time = evaluate(
            model=ddp_model, 
            device=device_id, 
            test_loader=val_loader
        )

        val_loss = avg_mae_lat + avg_mae_lon  

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weight_path)  # Save model weights
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