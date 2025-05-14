
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
import pandas as pd
import folium
from folium.plugins import BoatMarker, MousePosition
from folium import Element
import webbrowser
from shapely.geometry import LineString, Polygon, Point


from shapely.ops import transform
import pyproj



import sys

# Load utils
from utils.train_utils import load_config_file, load_scaler_json, load_data, scale_data, make_sequences2
from utils.data_loader import Classifier_Dataloader2

warnings.filterwarnings('ignore')


def load_cable_position_data(cable_data_path):
    try:
        df = pd.read_csv(cable_data_path, sep=";")

        # Convert "," to "." (decimal separator)
        for col in df.columns[2:]:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
        
        # Create dictionary of cable paths
        cable_paths = {}
        for i, row in df.iterrows():
            if row['Cable'] not in cable_paths:
                cable_paths[row['Cable']] = []
            cable_paths[row['Cable']].append((row['Latitude'], row['Longitude']))

        
        return cable_paths
    
    except Exception as e:
        print(f"Error occurred: \n {e}")
        sys.exit()

def add_critical_zone(cable_coords, radius_meters=1602):
    """
    Function that adds the critical zone around cables coordinates.

    Note,
    This function is based of the official documentation for shapely and pyproj.
    Sources: 
        https://shapely.readthedocs.io/en/stable/reference/shapely.LineString.html
        https://pyproj4.github.io/pyproj/stable/examples.html

    Args:
    - cable_coords: cable coordinates in the form (lon, lat)

    Returns:
    - buffered_latlon: the critical zone around each cable
    """

    # Convert lat/lon to LineString
    line = LineString([(lat, lon) for lat, lon in cable_coords])

    # Define projection: WGS84 (lat/lon) to UTM
    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    line_projected = transform(project, line)

    buffered = line_projected.buffer(radius_meters)

    # Project back to lat/lon
    project_back = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    buffered_latlon = transform(project_back, buffered)

    return buffered_latlon

def generate_all_buffers(cable_dict, radius_meters=1600):
    """
    Generate a dictionary of buffer zones for all cables.

    Args:
        cable_dict: Dictionary of {cable_id: [(lat, lon), ...]}
        radius_meters: Buffer radius around cable in meters

    Returns:
        Dictionary of {cable_id: buffered_polygon}
    """
    buffers = {}
    for cable_id, coords in cable_dict.items():
        buffer = add_critical_zone(coords, radius_meters)
        buffers[cable_id] = buffer
    return buffers

def predict_future_positions(onnx_session, device, data):
    onnx_session.set_providers(['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'])
    with torch.no_grad():  
        data = data.to(device)
        # ONNX model inference
        inputs = {onnx_session.get_inputs()[0].name: data.cpu().numpy()}
        output = onnx_session.run(None, inputs)[0]

        return output

def is_vessel_in_any_buffer(vessel_lat, vessel_lon, buffers):
    point = Point(vessel_lat, vessel_lon)
    return any(buffer.contains(point) for buffer in buffers.values())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cable_data_path = os.path.join(os.path.dirname(__file__),'data/cable_positions.csv')
    cable_dict = load_cable_position_data(cable_data_path)

    data_path = 'data/aisdk-2025-01-04_fishing_labeled.csv'
    df = pd.read_csv(data_path)
    # print(df.shape)
    df = df[df['trawling'] == 1]

    # print(df.shape)
    df = df.sort_values(by=['MMSI', '# Timestamp'])

    zones = generate_all_buffers(cable_dict, radius_meters=1602)

    hit_detected = False  # Track if any vessel is in the zone

    sus_vessels = {}

    for mmsi, group in df.groupby('MMSI'):
        for _, row in group.iterrows():
            if is_vessel_in_any_buffer(row['Latitude'], row['Longitude'], buffers=zones):
                if mmsi not in sus_vessels.keys():
                    sus_vessels[mmsi] = row['# Timestamp']
                    print(f"Vessel {mmsi} was trawling inside the critical zone at time {row['# Timestamp']}")
            # else:
                # print("No trawling vessel was detected inside the critical zone.")
        
    print(sus_vessels) # {219013178: '04/01/2025 20:08:06', 220127000: '04/01/2025 02:16:42', 220278000: '04/01/2025 05:19:53'}

    





    sys.exit()

    # Load data
    test_data_folder_path = os.path.abspath('data/test')
    test_parquet_files = glob.glob(os.path.join(test_data_folder_path, '*.parquet'))

    test_parquet_files.sort()
    test_parquet_files = test_parquet_files[10:12]

    input_features = ['MMSI', 'timestamp_epoch','Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading', 
                    'Width', 'Length', 'Draught']
    target_feature = ['trawling']
    
    X_test, y_test = load_data(
        parquet_files=test_parquet_files,
        input_features=input_features,
        target_columns=target_feature
    )

        # Combine X_test and y_test for easy filtering
    test_data = pd.concat([X_test, y_test], axis=1)

    # Filter vessels with trawling == 1
    trawling_vessels = test_data[test_data['trawling'] == 1]

    vessels_with_trawling_in_buffer = []

    dummy_forecaster_path = 'models/forecasters/onnx/lstm_dummy.onnx'
    dummy_forecaster = ort.InferenceSession(dummy_forecaster_path)

        # # Load scaler [FIX PATH]
    scaler_path = os.path.join(os.path.dirname(__file__),'data/norm_stats/v4/train_norm_stats.json')
    scaler = load_scaler_json(scaler_path)
    
    X_test_scaled = scale_data(scaler, X_test)
    X_test, y_test = make_sequences2(X_test_scaled, y_test, seq_len=60, group_col='MMSI')

    test_dataset = Classifier_Dataloader2(
        X_sequences=X_test,
        y_labels=y_test
    )

    # Load dataloaders                              
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=3512,
                            shuffle=False,
                            num_workers=20,
                            pin_memory=True)

    for data, _ in test_loader:
        predicted_trajectory = predict_future_positions(dummy_forecaster, device, data)



        polygon = Polygon(buffer)

            # Iterate over trawling vessels and check if they are within the buffer
        for _, row in trawling_vessels.iterrows():
            vessel_coords = {'Latitude': row['Latitude'], 'Longitude': row['Longitude']}
            if is_vessel_in_buffer(vessel_coords, polygon):
                vessels_with_trawling_in_buffer.append(row['MMSI'])

    # Now you have a list of MMSIs of vessels that are trawling and within the buffer
    print("Vessels within the buffer:", vessels_with_trawling_in_buffer)


    sys.exit()


    

    # ONNX model path
    classifier_onnx_model_path = f'models/classifiers/onnx/{args.classifier_name}_finalv3.onnx'  
    forecaster_onnx_model_path = f'models/forecasters/onnx/'
    classifier = ort.InferenceSession(classifier_onnx_model_path)
    forecaster = ort.InferenceSession(forecaster_onnx_model_path)

    # Format experiment name
    experiment_name = f"{args.snapshot_name}_test"
    print(experiment_name)


    # Define folder path to save the results and weights
    results_path = os.path.join(results_dir, f"{experiment_name}.txt")



if __name__ == '__main__':
    main()

    # parser = argparse.ArgumentParser(description='Train classifier')
    # parser.add_argument('--experiment_name', type=str, required=True, help="Name of experiment")
    # parser.add_argument('--classifier_name', type=str, required=True, help='Name of classifier model: lstm, 1dcnn, hybrid', default='hybrid')
    # parser.add_argument('--forecaster_name', type=str, required=True, help='Name of forecaster model: lstm, 1dcnn, bigru', default='lstm')
    # parser.add_argument('--seq_length', type=int, required=True, help="Input sequence length", default=60)
    # args = parser.parse_args()

    # # Make folders for results and snapshots
    # results_dir = f"results/end_to_end/test/{args.experiment_name}"
    # os.makedirs(results_dir, exist_ok=True)
