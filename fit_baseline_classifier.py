import os
import glob
import time
from utils.train_utils import load_scaler_json, load_data, scale_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def main():
       # Make folders for results and snapshots
    results_dir = "results/classification_results/test/RF_AIS_baseline"
    os.makedirs(results_dir, exist_ok=True)

    # Load scaler [FIX PATH]
    scaler_path = os.path.join(os.path.dirname(__file__),'data/norm_stats/v4/train_norm_stats.json')
    scaler = load_scaler_json(scaler_path)
    
    # Load data
    train_data_folder_path = os.path.abspath('data/petastorm/train/v4') # FIX PATH
    train_parquet_files = glob.glob(os.path.join(train_data_folder_path, '*.parquet'))
    test_data_folder_path = os.path.abspath('data/petastorm/test/v4')
    test_parquet_files = glob.glob(os.path.join(test_data_folder_path, '*.parquet'))

    test_parquet_files.sort()
    train_parquet_files.sort()

    input_features = ['MMSI', 'timestamp_epoch','Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading', 
                      'Width', 'Length', 'Draught']
    
    target_features = ['trawling']
    
    X_train, y_train = load_data(
        parquet_files=train_parquet_files,
        input_features=input_features,
        target_columns=target_features
    )

    X_test, y_test = load_data(
        parquet_files=test_parquet_files,
        input_features=input_features,
        target_columns=target_features
    )

    # Scale input features
    X_train_scaled = scale_data(scaler, X_train)
    X_test_scaled = scale_data(scaler, X_test)

    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    print("Fitting model")
    model.fit(X_train_scaled, y_train)
    print("Model has been fitted!")

    start_time = time.perf_counter()
    y_pred = model.predict(X_test_scaled)
    end_time = time.perf_counter()

    inference_time_ms = (end_time - start_time) * 1000 / len(X_test_scaled)  # ms per sample
    
    # Evaluate MAE for each output dimension
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"F1 score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


    results_path = os.path.join(results_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"Experiment name: Random Forest AIS Baseline\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}")
        f.write(f"Inference time per sample (ms): {inference_time_ms:.4f}\n")

    print("Converting model to ONNX")

    # Define the input type with shape (None, num_features)
    initial_type = [('input', FloatTensorType([None, X_train_scaled.shape[1]]))]

    # Convert the model
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save the ONNX model to file
    snapshot_path = 'models/classifiers'
    save_path = os.path.join(snapshot_path, "RF_baseline_AIS.onnx")
    with open(save_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
if __name__ == '__main__':
    main()