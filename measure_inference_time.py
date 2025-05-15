import onnxruntime as ort
import numpy as np
import time
import torch
import argparse

def measure_onnx_inference_time(model_path, input_shape, n_runs=100, device='cpu'):
    """
    Measures average per-sample inference time for an ONNX model using dummy data.

    Args:
        model_path (str): Path to the ONNX model file.
        input_shape (tuple): Shape of the model input (e.g., (1, 10, 4)).
        n_runs (int): Number of times to run inference.
        device (str): 'cpu' or 'cuda'.

    Returns:
        float: Average inference time per sample in milliseconds.
    """
    providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)

    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    input_name = session.get_inputs()[0].name
    total_time = 0.0

    for _ in range(n_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        total_time += (end - start)

    avg_time_ms = (total_time / n_runs) * 1000
    print(f"Average per-sample inference time over {n_runs} runs: {avg_time_ms:.4f} ms")
    return avg_time_ms

# Example usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument('--model_type', type=str, required=True, help="Model type: [fc / clf] for forecaster and classifier, respectively")
    parser.add_argument('--snapshot_name', type=str, required=True, help='Name of model you want to test', default='hybrid_finalv3')
    parser.add_argument('--seq_length', type=int, required=True, help="Input sequence length", default=60)
    args = parser.parse_args()

    if args.model_type.lower() == 'fc':
        model_type = 'forecasters'
    else:
        model_type = 'classifiers'

    model_path = f"models/{model_type}/{args.snapshot_name}.onnx"  # Replace with your ONNX model path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.seq_length is None:
        input_shape = (1, args.seq_length, 9)        # [batch, seq_len, features]
    else:
        input_shape = (1, 9)

    measure_onnx_inference_time(model_path, input_shape, n_runs=100, device=device)