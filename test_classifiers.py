
import torch
import os
from torch.utils.data import DataLoader
import warnings
import time
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
import glob
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

# Load utils
from utils.train_utils import load_config_file, load_scaler_json, load_data, scale_data, make_sequences
from utils.data_loader import Classifier_Dataloader

warnings.filterwarnings('ignore')


def inference_onnx(onnx_session, device, test_loader):
    onnx_session.set_providers(['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'])
    correct = 0
    num_samples = 0
    total_inference_time = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            start_time = time.time()
            # ONNX model inference
            inputs = {onnx_session.get_inputs()[0].name: data.cpu().numpy()}
            output = onnx_session.run(None, inputs)[0]
            batch_inference_time = time.time() - start_time
            total_inference_time += batch_inference_time

            # Apply sigmoid and round for binary classification
            pred = torch.sigmoid(torch.tensor(output)).round()
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_samples += len(target)

            # Store predictions and labels for F1 score calculation
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    avg_inference_time = (total_inference_time / num_samples) * 1000  # ms

    print(f'Average inference time (ms): {avg_inference_time}')

    return all_preds, all_labels, avg_inference_time

def compute_performance_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    
    print(f'\nF1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    return f1, precision, recall 

def plot_confusion_matrix(y_true, y_pred, save_img_path):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100 # Normalize

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['No Trawling', 'Trawling'],
                yticklabels=['No Trawling', 'Trawling'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    if save_img_path:
        plt.savefig(save_img_path, dpi=300)  
    plt.close() 

def plot_pr_curve(y_true, y_pred, save_img_path):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b',
             label='PR curve (area = {:.2f})'.format(np.trapz(precision, recall)))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')

    plt.tight_layout()
    if save_img_path:
        plt.savefig(save_img_path, dpi=300)  
    plt.close()  


def main():
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument('--snapshot_name', type=str, required=True, help='Name of model you want to test')
    parser.add_argument('--seq_length', type=int, required=True, help="Input sequence length")
    args = parser.parse_args()
    # Load config
    config_path = os.path.join(os.path.dirname(__file__),'train_config.yaml')
    config = load_config_file(config_path)

    # Make folders for results and snapshots
    results_dir = f"classification_results/test/{config['model_name']}"
    os.makedirs(results_dir, exist_ok=True)

    # Load scaler [FIX PATH]
    # scaler_path = os.path.join(os.path.dirname(__file__),'metadata.json')
    # scaler = load_scaler_json(scaler_path)
    
    # Load data
    val_data_folder_path = os.path.abspath('data/parquet')  # FIX PATH
    val_parquet_files = glob.glob(os.path.join(val_data_folder_path, '*.parquet'))

    val_parquet_files.sort()


    input_features = ['MMSI','timestamp_epoch','Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading', 
                      'Width', 'Length', 'Draught']
    target_feature = ['trawling']
    
    X_test, y_test = load_data(
        parquet_files=val_parquet_files,
        input_features=input_features,
        target_columns=target_feature
    )

    X_test, y_test = make_sequences(X_test_scaled, y_test, seq_len=args.seq_length, group_col='MMSI')
    # Scale input features
    # X_val_scaled = scale_data(scaler, X_val, features_to_scale)


    test_dataset = Classifier_Dataloader(
        X=X_test,
        y=y_test,
        seq_length=config['arch_param']['seq_len']
    )

    # Load dataloaders                              
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=config['train']['batch_size'],
                            shuffle=False,
                            num_workers=config['train']['num_workers'],
                            pin_memory=True)
    

    # ONNX model path
    onnx_model_path = 'path.onnx'  # Replace with the actual ONNX model path
    onnx_session = ort.InferenceSession(onnx_model_path)

    # Format experiment name
    experiment_name = f"{config['model_name']}_test"
    print(experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define folder path to save the results and weights
    results_path = os.path.join(results_dir, f"{experiment_name}.txt")

    # Evaluate ONNX model
    print("Initializing testing session...")
    y_true, y_pred, avg_inference_time = inference_onnx(onnx_session, device, test_loader)
    f1, precision, recall = compute_performance_metrics(y_true, y_pred)

    # Write results to file
    print("Writing results")
    with open(results_path, "w") as f:  # Overwrite file to keep only the best result
        f.write(f"Experiment name: {experiment_name}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Inference time (ms): {avg_inference_time:.2f}\n")

    # Plot confusion matrix and PR-curve
    cm_path = os.path.join(results_dir, f"{experiment_name}_CM.png")
    pr_path = os.path.join(results_dir, f"{experiment_name}_PR.png")
    plot_confusion_matrix(y_true, y_pred, save_img_path=cm_path)
    plot_pr_curve(y_true, y_pred, save_img_path=pr_path)
    


if __name__ == '__main__':
    main()