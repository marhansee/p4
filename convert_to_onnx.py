import torch
import torch.onnx
import os
import sys


from archs.cnn_lstm_classifier import CNN_LSTM
from archs.cnn1d_classifier import CNN1DClassifier
from archs.lstm_classifier import LSTMClassifier
from archs.cnn_forecast import CNN1DForecaster
from archs.lstm_forecaster import LSTMModel
from archs.bigru_forecast import BiGRUModel


"""
Script has been inspired by the official PyTorch guidelines for exporting
PyTorch models into ONNX-models.

Source:
https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html

"""


classification_models = {
    'hybrid': {
        'config': CNN_LSTM(
            n_features=9,
            out_channels=32,
            hidden_size=32,
            num_layers=2,
            num_classes=1
        ),
        'weight_path': 'models/classifiers/snapshots/hybrid_finalv3.pth',
        'onnx_path': 'models/classifiers/onnx/hybrid_finalv3.onnx'
    },

    'cnn1d': {
        'config': CNN1DClassifier(
            n_features=9,
            seq_len=60,
            out_channels=32,
            num_classes=1
        ),
        'weight_path': 'models/classifiers/snapshots/1dcnn_finalv3.pth',
        'onnx_path': 'models/classifiers/onnx/1dcnn_finalv3.onnx'
    },

    'lstm': {
        'config': LSTMClassifier(
            n_features=9,
            hidden_size=64,
            num_classes=1,
            num_layers=2,
            dropout_prob=0.2
        ),
        'weight_path': 'models/classifiers/snapshots/lstm_finalv3.pth',
        'onnx_path': 'models/classifiers/onnx/lstm_finalv3.onnx'
    }
}

forecasting_models = {
    'lstm': {
        'config': LSTMModel(
            n_features=9,
            hidden_size=32,
            num_layers=2,
            dropout_prop=0.2
        ),
        'weight_path': 'models/forecasters/lstm_mainv1.pth',
        'onnx_path': 'models/forecasters/onnx/lstm_mainv1.onnx'
    },

    '1dcnn': {
        'config': CNN1DForecaster(
            n_features=9,
            seq_len=60,
            out_channels=32
        ),
        'weight_path': 'models/forecasters/1dcnn_mainv1.pth',
        'onnx_path': 'models/forecasters/onnx/1dcnn_mainv1.onnx'
    },

    'bigru': {
        'config': BiGRUModel(
            n_features=9,
            hidden_size=64,
            num_layers=1,          
        ),
        'weight_path': 'models/forecasters/bigru_mainv2.pth',
        'onnx_path': 'models/forecasters/onnx/bigru_mainv2.onnx'
    }
}

def main():
    # os.makedirs('snapshots/classification/trained_models', exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # models = classification_models
    models = forecasting_models

    for model_name, model_info in models.items():
        model = model_info['config'].to(device)
        model.load_state_dict(torch.load(model_info['weight_path'], map_location=device))
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 60, 9).to(device) # Batch size 1, 60 sequence length, 9 features


        # Export the model to ONNX
        torch.onnx.export(
            model,                        
            dummy_input,             
            model_info['onnx_path'],      # Path to save the ONNX model
            export_params=True,          
            opset_version=12,             
            do_constant_folding=True,    
            input_names=['input'],       # Name for the input tensor
            output_names=['output'],     # Name for the output tensor
            dynamic_axes={
                'input': {0: 'batch_size'},  # Allow dynamic batch size
                'output': {0: 'batch_size'}
            }
        )

        print(f"Model {model_name} exported to ONNX format at {model_info['onnx_path']}")

if __name__ == '__main__':
    main()
