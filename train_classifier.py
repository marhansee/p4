
import torch
import wandb
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import warnings
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
import glob
import numpy as np

# Load model architectures
from archs.cnn1d_classifier import CNN1DClassifier
from archs.cnn_lstm_classifier import CNN_LSTM
from archs.lstm_classifier import LSTMClassifier 

# Load utils
from utils.train_utils import load_config_file, load_scaler_json, load_data, scale_data
from utils.data_loader import Classifier_Dataloader

warnings.filterwarnings('ignore')

def train(model, device, train_loader, optimizer, scheduler, epoch, scaler):
    model.train()
    total_loss = 0
    correct = 0
    num_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Implement AMP (Automatic Mixed Precision)
        with torch.amp.autocast('cuda'):
            output = model(data)  # Forward pass through the model
            # Binary Cross Entropy loss (with logits)
            loss = F.binary_cross_entropy_with_logits(output.squeeze(), target.float())

        # Scale gradients and apply backpropagation
        scaler.scale(loss).backward()

        # Unscale gradients and update weights
        scaler.step(optimizer)

        # Update scaler for mixed precision
        scaler.update()

        total_loss += loss.item()

        # Track accuracy (for monitoring)
        pred = torch.round(torch.sigmoid(output))  # Apply sigmoid to get probability, then round to 0 or 1
        correct += pred.eq(target.view_as(pred)).sum().item()
        num_samples += len(target)

        if batch_idx % 500 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    epoch_loss = total_loss / len(train_loader)  # Average loss over all batches

    # Log the train loss and accuracy to WandB
    wandb.log({"Epoch Train Loss": epoch_loss})


def evaluate(model, device, test_loader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    num_samples = 0
    total_inference_time = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Turn off gradients for evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            start_time = time.time()
            output = model(data)  # Forward pass
            batch_inference_time = time.time() - start_time
            total_inference_time += batch_inference_time

            # Binary cross-entropy loss for binary classification
            test_loss += F.binary_cross_entropy_with_logits(output.squeeze(), target.float(), reduction='sum').item()
            
            # Predictions: output is logits, so we apply a sigmoid to get probabilities
            pred = torch.round(torch.sigmoid(output))  # Sigmoid to get probability, then round to 0 or 1
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_samples += len(target)

            # Store predictions and labels for F1 score calculation
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    test_loss /= num_samples  # Average loss
    accuracy = 100. * correct / num_samples  # Accuracy as a percentage

    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds)

    avg_inference_time = (total_inference_time / num_samples) * 1000  # ms

    print(f'\nTest set: Average Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}')
    wandb.log({
        "Eval Loss": test_loss,
        "Eval Accuracy": accuracy,
        "Eval F1 Score": f1,
        "Avg Inference Time (ms)": avg_inference_time
    })

    return test_loss, accuracy, f1, avg_inference_time




def main():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__),'train_config.yaml')
    config = load_config_file(config_path)

    # Make folders for results and snapshots
    os.makedirs(f"classification_results/{config['model_name']}", exist_ok=True)
    os.makedirs(f"snapshots/classification/{config['model_name']}", exist_ok=True)

    # Load scaler [FIX PATH]
    scaler_path = os.path.join(os.path.dirname(__file__),'metadata.json')
    scaler = load_scaler_json(scaler_path)
    
    # Load data
    train_data_folder_path = os.path.join(os.path.dirname(__file__), 'train_data')
    train_parquet_files = glob.glob(os.path.join(train_data_folder_path, '*.parquet'))
    val_data_folder_path = os.path.join(os.path.dirname(__file__), 'val_data')
    val_parquet_files = glob.glob(os.path.join(val_data_folder_path, '*.parquet'))

    input_features = ['timestamp_epoch', 'MMSI', 'Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading', 
                      'Width', 'Length', 'Draught']
    features_to_scale = [feature for feature in input_features if feature not in ['timestamp_epoch', 'MMSI']]
    target_feature = ['trawling']
    
    X_train, y_train = load_data(
        parquet_files=train_parquet_files,
        input_features=input_features,
        target_columns=target_feature
    )

    X_val, y_val = load_data(
        parquet_files=val_parquet_files,
        input_features=input_features,
        target_columns=target_feature
    )

    # Scale input features
    X_train_scaled = scale_data(scaler, X_train, features_to_scale)
    X_val_scaled = scale_data(scaler, X_val, features_to_scale)

    # Drop timestamp and MMSI
    X_train_scaled = np.delete(X_train_scaled, ['MMSI','timestamp_epoch','trawling'], axis=1)
    X_val_scaled = np.delete(X_val_scaled, ['MMSI','timestamp_epoch','trawling'], axis=1)

    train_dataset = Classifier_Dataloader(
        X=X_train_scaled,
        y=y_train,
        seq_length=config['arch_param']['seq_len']
    )

    val_dataset = Classifier_Dataloader(
        X=X_val_scaled,
        y=y_val,
        seq_length=config['arch_param']['seq_len']
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
    if config['model_name'].lower() == 'hybrid':
        model = CNN_LSTM(
            n_features=config['arch_param']['n_features'],
            out_channels=config['arch_param']['out_channels'],
            hidden_size=config['arch_param']['hidden_size'],
            num_layers=config['arch_param']['num_layers'],
            num_classes=config['arch_param']['output_size']
        )
    elif config['model_name'].lower() == '1dcnn':
        model = CNN1DClassifier(
            n_features=config['arch_param']['n_features'],
            seq_len=config['arch_param']['seq_len'],
            out_channels=config['arch_param']['out_channels'],
            num_classes=config['arch_param']['output_size']
        )
    elif config['model_name'].lower() == 'lstm':
        model = LSTMClassifier(
            n_features=config['arch_param']['n_features'],
            hidden_size=config['arch_param']['hidden_size'],
            num_layers=config['arch_param']['num_layers'],
            dropout_prop=config['train']['dropout_prop'],
            num_classes=config['arch_param']['output_size']
        )
    else:
        raise AssertionError('Model must be either "hybrid", "1dcnn", "lstm"')
    
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

        val_loss, accuracy, f1, avg_inference_time = evaluate(
            model=ddp_model, 
            device=device_id, 
            test_loader=val_loader
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weight_path)  # Save model weights
            print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")

            print("Writing results")
            with open(results_path, "w") as f:  # Overwrite file to keep only the best result
                f.write(f"Experiment name: {experiment_name}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"F1-score: {f1:.4f}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Inference time (ms): {avg_inference_time:.2f}\n")

    print("Completed training.")



if __name__ == '__main__':
    main()