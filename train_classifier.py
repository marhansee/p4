
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
import argparse

# Load model architectures
from archs.cnn1d_classifier import CNN1DClassifier
from archs.cnn_lstm_classifier import CNN_LSTM
from archs.lstm_classifier import LSTMClassifier 

# Load utils
from utils.train_utils import load_config_file, load_scaler_json, load_data, scale_data, make_sequences
from utils.data_loader import Classifier_Dataloader, Classifier_Dataloader2
from utils.early_stopping import EarlyStoppingF1

warnings.filterwarnings('ignore')

def train(model, device, train_loader, optimizer, epoch, scaler):
    model.train()
    total_loss = 0
    correct = 0
    num_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # AMP (Automatic Mixed Precision)
        with torch.cuda.amp.autocast():
            output = model(data)  # Forward pass (logits)
            loss = F.binary_cross_entropy_with_logits(output, target.float())
        # Scale loss, backprop, and update
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Accuracy calculation (sigmoid + thresholding)
        pred = torch.sigmoid(output)  
        pred = (pred > 0.5).float()  # Better than torch.round()
        correct += pred.eq(target.view_as(pred)).sum().item()
        num_samples += len(target)

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # Average loss & accuracy
    epoch_loss = total_loss / num_samples  # More precise than len(train_loader)
    train_accuracy = 100. * correct / num_samples

    # Log to wandb
    wandb.log({
        "Epoch Train Loss": epoch_loss,
        "Epoch Train Accuracy": train_accuracy,  # Added missing metric
        "Epoch": epoch
    })

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
            test_loss += F.binary_cross_entropy_with_logits(output, target.float(), reduction='sum').item()
            
            # Predictions: output is logits, so we apply a sigmoid to get probabilities
            pred = torch.sigmoid(output)  # Round to 0 or 1
            pred = torch.round(pred)
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
    print(f'Average inference time (ms): {avg_inference_time}')
    wandb.log({
        "Eval Loss": test_loss,
        "Eval Accuracy": accuracy,
        "Eval F1 Score": f1
    })

    return test_loss, accuracy, f1, avg_inference_time




def main():
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()

    # Load config
    config = load_config_file(args.config)

    # Make folders for results and snapshots
    os.makedirs(f"classification_results/{config['model_name']}", exist_ok=True)
    os.makedirs(f"snapshots/classification/{config['model_name']}", exist_ok=True)

    # Load scaler [FIX PATH]
    scaler_path = os.path.join(os.path.dirname(__file__),'data/norm_stats/v4/train_norm_stats.json')
    scaler = load_scaler_json(scaler_path)
    
    # Load data
    train_data_folder_path = os.path.abspath('data/petastorm/train/v4') # FIX PATH
    train_parquet_files = glob.glob(os.path.join(train_data_folder_path, '*.parquet'))
    val_data_folder_path = os.path.abspath('data/petastorm/val/v4')  # FIX PATH
    val_parquet_files = glob.glob(os.path.join(val_data_folder_path, '*.parquet'))
    
    val_parquet_files.sort()
    train_parquet_files.sort()


    input_features = ['MMSI', 'timestamp_epoch','Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading', 
                      'Width', 'Length', 'Draught']
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


    X_train_scaled = scale_data(scaler, X_train)
    X_val_scaled = scale_data(scaler, X_val) 
  
    X_train, y_train = make_sequences(X_train_scaled, y_train, seq_len=config['arch_param']['seq_len'], group_col='MMSI')
    X_val, y_val = make_sequences(X_val_scaled, y_val, seq_len=config['arch_param']['seq_len'], group_col='MMSI')


    train_dataset = Classifier_Dataloader2(
        X_sequences=X_train,
        y_labels=y_train
    )

    val_dataset = Classifier_Dataloader2(
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
    if config['model_name'].lower() == 'hybrid':
        model = CNN_LSTM(
            n_features=config['arch_param']['n_features'],
            out_channels=config['arch_param']['out_channels'],
            hidden_size=config['arch_param']['hidden_size'],
            num_layers=config['arch_param']['num_layers'],
            num_classes=config['arch_param']['n_classes']
        )
    elif config['model_name'].lower() == '1dcnn':
        model = CNN1DClassifier(
            n_features=config['arch_param']['n_features'],
            seq_len=config['arch_param']['seq_len'],
            out_channels=config['arch_param']['out_channels'],
            num_classes=config['arch_param']['n_classes']
        )
    elif config['model_name'].lower() == 'lstm':
        model = LSTMClassifier(
            n_features=config['arch_param']['n_features'],
            hidden_size=config['arch_param']['hidden_size'],
            num_layers=config['arch_param']['num_layers'],
            dropout_prob=config['train']['dropout_prob'],
            num_classes=config['arch_param']['n_classes']
        )
    else:
        raise AssertionError('Model must be either "hybrid", "1dcnn", "lstm"')


    model = model.to(torch.device("cuda", device_id))
    
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


    # Define folder path to save the results and weights
    results_path = os.path.join(f"classification_results/{config['model_name']}", f"{experiment_name}.txt")
    weight_path = os.path.join(f"snapshots/classification/{config['model_name']}", f"{experiment_name}.pth")

    early_stopping = EarlyStoppingF1(patience=7, min_delta=0.01)
    # Training loop
    print("Initializing training...")

    best_f1 = 0.0
    for epoch in range(1, config['train']['num_epochs']+1):
        train(
            model=ddp_model,
            device=device_id,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            scaler=scaler,
        )

        val_loss, accuracy, f1, avg_inference_time = evaluate(
            model=ddp_model, 
            device=device_id, 
            test_loader=val_loader
        )

        scheduler.step()
        print(f"Experiment: {experiment_name}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), weight_path)  # Save model weights
            print(f"New best model saved with F1-score: {best_f1:.4f}")
            print("Writing results")
            with open(results_path, "w") as f:  # Overwrite file to keep only the best result
                f.write(f"Experiment name: {experiment_name}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"F1-score: {f1:.4f}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Inference time (ms): {avg_inference_time}\n")

        early_stopping(f1)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch} with best F1-score {early_stopping.best_score:.4f}")
            break

    print("Completed training.")



if __name__ == '__main__':
    main()
