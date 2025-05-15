import os
import torch
from utils.train_utils import load_scaler_json, load_data, scale_data, make_sequences
import glob
from torch.utils.data import DataLoader
import sys
from utils.data_loader import Classifier_Dataloader2
from archs.cnn_forecast import CNN1DForecaster



def main():
    weights_path = 'snapshots/forecast/1dcnn/1dcnn_mainv1.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1DForecaster().to(device)
    model.load_state_dict(torch.load(weights_path), map_location=device)
    model.eval()

        # Load scaler [FIX PATH]
    scaler_path = os.path.join(os.path.dirname(__file__),'data/norm_stats/v4/train_norm_stats.json')
    scaler = load_scaler_json(scaler_path)
    
    # Load data
    test_data_folder_path = os.path.abspath('data/petastorm/test/v4')
    test_parquet_files = glob.glob(os.path.join(test_data_folder_path, '*.parquet'))

    test_parquet_files.sort()

    input_features = ['MMSI', 'timestamp_epoch','Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading', 
                      'Width', 'Length', 'Draught']
    
    target_features = [f'future_lat_{i}' for i in range(6, 121, 6)] + \
           [f'future_lon_{i}' for i in range(6, 121, 6)]
    
    X_test, y_test = load_data(
        parquet_files=test_parquet_files,
        input_features=input_features,
        target_columns=target_features
    )

    # Scale input features
    X_test_scaled = scale_data(scaler, X_test)

    X_test, y_test = make_sequences(X_test_scaled, y_test, seq_len=60, group_col='MMSI')

    test_dataset = Classifier_Dataloader2(
        X_sequences=X_test,
        y_labels=y_test,
    )

    test_loader = DataLoader(dataset=test_dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=10,
                    pin_memory=True)
    
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            output = model(data)  # shape [B, 40]
            print(output.shape)
            print("Raw output:", output[0])
            
            output = output.view(-1, 20, 2)
            print("Latitudes:", output[0, :, 0])
            print("Longitudes:", output[0, :, 1])
        break

    sys.exit()


if __name__ == '__main__':
    main()