
import torch
import wandb
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import sys
import warnings
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
import glob
import argparse

# Load model architectures
from model_architectures.lstm_forecaster import LSTMModel
from model_architectures.bigru_forecast import BiGRUModel
from model_architectures.cnn_forecast import CNN1DForecaster
from model_architectures.seq2seq_lstm import Seq2SeqLSTM

# Load utils
from utils.train_utils import load_config_file, load_scaler_json, \
    load_data, scale_data, make_sequences
from utils.data_loader import System_Dataloader
from utils.early_stopping import EarlyStopping

warnings.filterwarnings('ignore')




def train(model, device, train_loader, optimizer, epoch, scaler, config):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch_size = target.size(0)
        target = target.view(batch_size, 2, config['arch_param']['output_seq_len']).transpose(1, 2)
        optimizer.zero_grad()

        # Implement AMP
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = F.mse_loss(output, target)

        # Scale gradients and apply backprop
        scaler.scale(loss).backward()

        # Unscale gradients and update weights
        scaler.step(optimizer)

        # Update scaler
        scaler.update()


        total_loss += loss.item()

        if batch_idx % 250 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    epoch_loss = total_loss / len(train_loader)

    # Log the train loss
    wandb.log({"Epoch Train Loss": epoch_loss, "Epoch": epoch})
    

def evaluate(model, device, test_loader, config):
    model.eval()
    total_mae_lat = 0
    total_mae_lon = 0
    num_samples = 0
    total_inference_time = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = target.size(0)
            target = target.view(batch_size, 2, config['arch_param']['output_seq_len']).transpose(1, 2)

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
    print(f'Average inference time (ms): {avg_inference_time}')
    wandb.log({
        "Eval MAE Loss Latitude": avg_mae_lat,
        "Eval MAE Loss Longitude": avg_mae_lon,
    })

    return avg_mae_lat, avg_mae_lon, avg_inference_time


def main():
    parser = argparse.ArgumentParser(description='Train forecaster')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()

    # Load config
    config = load_config_file(args.config)

    # Load scaler
    scaler_path = os.path.join(os.path.dirname(__file__),'data/norm_stats/v4/train_norm_stats.json')
    scaler = load_scaler_json(scaler_path)

    # Make folders for results and snapshots
    os.makedirs(f"forecast_results/{config['model_name']}", exist_ok=True)
    os.makedirs(f"snapshots/forecast/{config['model_name']}", exist_ok=True)

    # Load data
    train_data_folder_path = os.path.abspath('data/petastorm/train/v4') # FIX PATH
    train_parquet_files = glob.glob(os.path.join(train_data_folder_path, '*.parquet'))
    val_data_folder_path = os.path.abspath('data/petastorm/val/v4')  # FIX PATH
    val_parquet_files = glob.glob(os.path.join(val_data_folder_path, '*.parquet'))

    val_parquet_files.sort()
    train_parquet_files.sort()


    input_features = ['MMSI','timestamp_epoch','Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading', 
                     'Width', 'Length', 'Draught']

    # Define mapping for output sequence length in minutes
    mapping = {
        1: 6,
        5: 30,
        10: 60,
        15: 90,
        20: 120
    }

    # Define target features
    target_features = [f'future_lat_{i}' for i in range(6, mapping[config['arch_param']['output_seq_len']]+1, 6)] + \
           [f'future_lon_{i}' for i in range(6, mapping[config['arch_param']['output_seq_len']]+1, 6)]

    # Load the data
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
    X_train_scaled = scale_data(scaler, X_train)
    X_val_scaled = scale_data(scaler, X_val)

    # Make input sequences
    X_train, y_train = make_sequences(X_train_scaled, y_train, 
                                      seq_len=config['arch_param']['seq_len'], 
                                      group_col='MMSI')
    X_val, y_val = make_sequences(X_val_scaled, y_val, 
                                  seq_len=config['arch_param']['seq_len'], 
                                  group_col='MMSI')
    # Load datasets
    train_dataset = System_Dataloader(
        X_sequences=X_train,
        y_labels=y_train
    )

    val_dataset = System_Dataloader(
        X_sequences=X_val,
        y_labels=y_val
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

    # Format experiment name
    experiment_name = f"{config['model_name']}_{config['experiment_name']}"
    print(experiment_name)
    
    # Initialize WandB
    wandb.login()
    wandb.init(project=config['wandb']['project'], config=config, name=experiment_name)

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
            dropout_prop=config['train']['dropout_prob']
        ).to(device_id)
    elif config['model_name'].lower() == 'bigru':
        model = BiGRUModel(
            n_features=config['arch_param']['n_features'],
            hidden_size=config['arch_param']['hidden_size'],
            num_layers=config['arch_param']['num_layers'],
            output_seq_len=config['arch_param']['output_seq_len'],
            output_size=config['arch_param']['output_size'],
            dropout_prob=config['train']['dropout_prob']
        ).to(device_id)
    elif config['model_name'].lower() == '1dcnn':
        model = CNN1DForecaster(
            n_features=config['arch_param']['n_features'],
            seq_len=config['arch_param']['seq_len'],
            out_channels=config['arch_param']['out_channels'],
            output_size=config['arch_param']['output_size'],
            output_seq_len=config['arch_param']['output_seq_len']
        ).to(device_id)
    elif config['model_name'].lower() == 's2s_lstm':
        model = Seq2SeqLSTM(
            n_features=config['arch_param']['n_features'],
            hidden_size=config['arch_param']['hidden_size'],
            num_layers=config['arch_param']['num_layers'],
            output_seq_len=config['arch_param']['output_seq_len'],
            output_size=config['arch_param']['output_size'],
            dropout=config['train']['dropout_prob']
        ).to(device_id)
    else:
        raise AssertionError('Model must be either "lstm", "bigru", "1dcnn"')
    
    # Wrap model in DDP
    ddp_model = DDP(model, device_ids=[device_id])

    # Define scaler for Automatic Mixed Precision
    scaler = torch.cuda.amp.GradScaler()
    
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

    # Initialize early stopping
    early_stopping = EarlyStopping(10, min_delta=0.01)

    # Training loop
    print("Initializing training...")

    best_val_loss = float('inf')
    for epoch in range(1, config['train']['num_epochs']+1):
        train(
            model=ddp_model,
            device=device_id,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            scaler=scaler,
            config=config
        )

        avg_mae_lat, avg_mae_lon, avg_inference_time = evaluate(
            model=ddp_model, 
            device=device_id, 
            test_loader=val_loader,
            config=config
        )

        val_loss = (avg_mae_lat + avg_mae_lon) / 2  

        scheduler.step()

        print(experiment_name)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weight_path)  # Save model weights
            print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")

            print("Writing results")
            with open(results_path, "w") as f:  # Overwrite file to keep only the best result
                f.write(f"Experiment name: {experiment_name}\n")
                f.write(f"MAE - Lat: {avg_mae_lat:.4f}, Long: {avg_mae_lon:.4f}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Inference time (ms): {avg_inference_time}\n")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch} with best validation loss {early_stopping.best_loss:.4f}")
            break
    print("Completed training.")



if __name__ == '__main__':
    main()
