import torch
import torch.onnx



from model_architectures.cnn_lstm_classifier import CNN_LSTM
from model_architectures.cnn1d_classifier import CNN1DClassifier
from model_architectures.lstm_classifier import LSTMClassifier
from model_architectures.cnn_forecast import CNN1DForecaster
from model_architectures.bigru_forecast import BiGRUModel
from model_architectures.seq2seq_lstm import Seq2SeqLSTM


"""
Script has been inspired by the official PyTorch guidelines for exporting
PyTorch models into ONNX-models.

Source:
https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html

"""


classification_models = {
    'hybrid': {
        'config': CNN_LSTM(
            n_features=2,
            out_channels=32,
            hidden_size=32,
            num_layers=2,
            num_classes=1
        ),
        'weight_path': 'snapshots/classification/hybrid/hybrid_seq5_latlon_clf.pth',
        'onnx_path': 'models/classifiers/hybrid_seq5_latlon_clf.onnx'
    },

   'cnn1d': {
       'config': CNN1DClassifier(
           n_features=2,
           seq_len=60,
           out_channels=32,
           num_classes=1
       ),
       'weight_path': 'snapshots/classification/1dcnn/1dcnn_latlon.pth',
       'onnx_path': 'models/classifiers/1dcnn_latlon_clf.onnx'
   },

   'lstm': {
       'config': LSTMClassifier(
           n_features=2,
           hidden_size=64,
           num_classes=1,
           num_layers=2,
           dropout_prob=0.2
       ),
       'weight_path': 'snapshots/classification/lstm/lstm_latlon.pth',
       'onnx_path': 'models/classifiers/lstm_latlon_clf.onnx'
   }
}

forecasting_models = {
    'lstm': {
        'config': Seq2SeqLSTM(
            n_features=9,
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            output_seq_len=20
        ),
        'weight_path': 'snapshots/forecast/s2s_lstm/s2s_lstm_w_RELU.pth',
        'onnx_path': 'models/forecasters/s2s_lstm_w_RELU.onnx'
    },

   '1dcnn': {
       'config': CNN1DForecaster(
           n_features=2,
           seq_len=60,
           out_channels=32
       ),
       'weight_path': 'snapshots/forecast/1dcnn/1dcnn_latlon.pth',
       'onnx_path': 'models/forecasters/1dcnn_latlon_fc.onnx'
   },

   'bigru': {
       'config': BiGRUModel(
           n_features=2,
           hidden_size=64,
           num_layers=1,          
       ),
       'weight_path': 'snapshots/forecast/bigru/bigru_latlon.pth',
       'onnx_path': 'models/forecasters/bigru_latlon_fc.onnx'
   }
}

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
   # models = classification_models
    models = forecasting_models

    for model_name, model_info in models.items():
        model = model_info['config'].to(device)
        model.load_state_dict(torch.load(model_info['weight_path'], map_location=device))
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 30, 9).to(device) # Batch size 1, 30 sequence length, 9 features


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
