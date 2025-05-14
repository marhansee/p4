
import torch
import torch.onnx
import os
import sys


from archs.lstm_forecaster import LSTMModel


"""
Script has been inspired by the official PyTorch guidelines for exporting
PyTorch models into ONNX-models.

Source:
https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html

"""

models = {
    'lstm': {
        'config': LSTMModel(
            n_features=9,
            hidden_size=32,
            num_layers=2,
            dropout_prop=0.2
        ),
        'weight_path': 'models/forecasters/lstm_mainv2.pth',
        'onnx_path': 'models/forecasters/onnx/lstm_dummy.onnx'
    }
}

def main():
    os.makedirs('models/forecasters/onnx', exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for model_name, model_info in models.items():
        model = model_info['config'].to(device)
        model.load_state_dict(torch.load(model_info['weight_path'], map_location=device))
        model.eval()

        dummy_input = torch.randn(1, 60, 9).to(device)  # Batch size 1, 60 sequence length, 9 features
        
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
