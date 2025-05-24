import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
# from xgboost import XGBRegressor
import argparse
import warnings
import matplotlib.pyplot as plt
# import missingno as msno
import sys
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

from utils.preprocessing import split_data, add_lagged_value, add_future_value, \
        apply_downsampling, apply_resampling, sliding_window_features, preprocess_data
from utils.degradations import add_duplicates, add_missing_values, synthesize_irregular_sampling
from utils.train_eval import train


warnings.filterwarnings("ignore")

def plot_feature_importance(model, X_train):
    feature_names = X_train.columns
    estimators = model.estimators_


    for i, est in enumerate(estimators):
        coef = est.coef_
        importance = pd.Series(coef, index=feature_names).sort_values(key=abs, ascending=False)
        print(f"\nFeature importance for target #{i + 1}:")
        # print(importance)

    plt.figure(figsize=(10,10))

    importance_abs = importance.abs()
    importance_abs.plot(kind='bar')
    plt.title("Feature Importance (Coefficient Magnitude)")
    plt.xticks(fontsize=15)
    plt.tight_layout()
    plt.show()

def inference_plot(model_path, X_val, X_val_scaled, y_val):
    model = joblib.load(model_path)

    # Use the first 10 steps as the true trajectory
    X_gt = X_val.iloc[0:10]

    # Get initial true position
    start_lat = X_gt.iloc[0]['latitude']
    start_lon = X_gt.iloc[0]['longitude']

    # Predict final future point (assumes [lat, lon] order in model output)
    pred = model.predict(X_val_scaled.iloc[0:1])[0]
    pred_lat = pred[0]
    pred_lon = pred[1]

    # Plot actual trajectory
    plt.figure(figsize=(8, 6))
    plt.scatter(start_lon, start_lat, color='black', s=100, label='Start') # Label for start point
    plt.plot(X_gt['longitude'], X_gt['latitude'], marker='o', color='green', label='True Trajectory')

    # Plot straight line prediction
    plt.plot([start_lon, pred_lon], [start_lat, pred_lat], color='blue', linestyle='--', marker='x', label='Predicted Destination')



    plt.xlabel('Longitude', fontsize=20)
    plt.ylabel('Latitude', fontsize=20)
    plt.title('Predicted vs True Trajectory', fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()



def main():
    os.makedirs('results', exist_ok=True)

    parser = argparse.ArgumentParser(description='Baseline Forecaster')
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 5])
    parser.add_argument('--horizon', type=int, default=10,
                        help="Forecasting horizon in num. samples")
    parser.add_argument('--experiment_name', type=str, default="Forecasting",
                        help="Name of the experiment")
    parser.add_argument('--add_duplicates', action='store_true', default=False, # Add duplicates
                        help='If True, duplicates are added')
    parser.add_argument('--add_missing_values', action='store_true', default=True, # Add missing values
                        help="If True then missing values are added")
    parser.add_argument('--MV_impute_strategy', type=str, default='linear',
                        help='"zero": 0, "linear": linear interpolation, "mean": Mean')
    parser.add_argument('--apply_irregular_sampling', action='store_true', default=False, # Add irregular sampling
                        help='If True then rows are removed from trajectories to simulate irregular sampling')
    parser.add_argument('--apply_downsampling', action='store_true', default=False, # Apply downsampling
                        help='If True then downsampling is enabled')
    parser.add_argument('--apply_resampling', action='store_true', default=True, # Apply resampling
                        help='If True then resampling is enabled')
    parser.add_argument('--sampling_interval', type=int, default=60,
                        help='Sampling interval in seconds if resample or downsample is enabled')

    
    args = parser.parse_args()

    # Join data
    data_path = os.path.join(os.path.dirname(__file__),'data/Trajectory_IDs.csv')

    # Split data
    train_df, val_df, test_df = split_data(data_path, random_state=42)



    # Define missing value percentages
    missing_values_features = {
        'signed_turn': 29,
        'bearing': 21,
        'euc_speed': 1,
        'distanceToShore': 70
    }


    if args.apply_irregular_sampling:
        train_df = synthesize_irregular_sampling(train_df,
                    min_percentage=1, max_percentage=10, random_state=42)
        val_df = synthesize_irregular_sampling(val_df,
                    min_percentage=1, max_percentage=10, random_state=42)
        test_df = synthesize_irregular_sampling(test_df,
                    min_percentage=1, max_percentage=10, random_state=42)


    if args.apply_downsampling:
        train_df = apply_downsampling(train_df, 
                   time_interval=args.sampling_interval)
        val_df = apply_downsampling(val_df,
                   time_interval=args.sampling_interval)
        test_df = apply_downsampling(test_df,
                   time_interval=args.sampling_interval)
        
    if args.apply_resampling:
        train_df = apply_resampling(train_df,
                   time_interval=args.sampling_interval)
        val_df = apply_resampling(val_df,
                   time_interval=args.sampling_interval)
        test_df = apply_resampling(test_df,
                   time_interval=args.sampling_interval)


    # Add degradations
    if args.add_missing_values:
        train_df = add_missing_values(train_df, 
                   missing_dict=missing_values_features)
        val_df = add_missing_values(val_df,
                   missing_dict=missing_values_features)
        test_df = add_missing_values(test_df,
                   missing_dict=missing_values_features)

        if args.MV_impute_strategy == 'zero':
            train_df.fillna(0, inplace=True)
            val_df.fillna(0, inplace=True)
            test_df.fillna(0, inplace=True)
        if args.MV_impute_strategy == 'linear':
            train_df.interpolate(method='linear', inplace=True, limit_direction='both')
            val_df.interpolate(method='linear', inplace=True, limit_direction='both')
            test_df.interpolate(method='linear', inplace=True, limit_direction='both')
        if args.MV_impute_strategy == 'mean':
            for feature in missing_values_features.keys():
                train_df[feature].fillna(train_df[feature].mean(), inplace=True)
                val_df[feature].fillna(val_df[feature].mean(), inplace=True)
                test_df[feature].fillna(test_df[feature].mean(), inplace=True)

    if args.add_duplicates:
        train_df = add_duplicates(train_df, duplicate_percentage=30)
        val_df = add_duplicates(val_df, duplicate_percentage=30)
        test_df = add_duplicates(test_df, duplicate_percentage=30)

    print("Added degradations!")

    # Drop features
    train_df.drop(['id','label'],axis=1, inplace=True)
    val_df.drop(['id', 'label'], axis=1, inplace=True)
    test_df.drop(['id','label'], axis=1, inplace=True)


    # Define data settings and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = \
        preprocess_data(train_df, val_df, test_df, num_lags=args.lags, horizon=args.horizon)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print("Preprocessing complete!")

    
    features_to_keep = ['longitude','longitude_lag1','x_lag1','x',
                    'latitude_lag1','y_lag1','latitude','y','latitude_lag5',
                    'y_lag5','x_lag5','longitude_lag5']
    
    X_train_scaled = X_train_scaled[features_to_keep]
    X_val_scaled = X_val_scaled[features_to_keep]


    # Define models to test
    models = {
        'Linear Regression': LinearRegression(n_jobs=-1),
        # 'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        # 'XGB': XGBRegressor(learning_rate=0.3, max_depth=10, subsample=0.5)
    }

    train = False
    if train:
        results, trained_models = train(models_dict=models,
                        X_train=X_train_scaled,
                        y_train=y_train,
                        X_val=X_val_scaled,
                        y_val=y_val,
                        target_latitude_feature=f'latitude_future{args.horizon}',
                        target_longitude_feature=f'longitude_future{args.horizon}'
                        )    
        
        print(results)

        lr_model = trained_models['Linear Regression']

        # Save model
        weight_path = os.path.join('snapshots/forecast_baseline')
        joblib.dump(lr_model, weight_path+"/LinReg_horizon10")
        print("Saved model")


    # Load model
    model_path = os.path.join(os.path.dirname(__file__),'snapshots/forecast_baseline/LinReg_horizon10')
    inference_plot(model_path, X_val=X_val, X_val_scaled=X_val_scaled, y_val=y_val)

    # Visualize feature importance
    # plot_feature_importance(lr_model, X_train)

    # results_path = os.path.join('results', f'{args.experiment_name}.txt')
    # with open(results_path, 'w') as f:
    #     f.write(f"Experiment name: {args.experiment_name} \n")
    #     for result in results:
    #         f.write(f"Model: {result['model']}\n")
    #         f.write(f"MAE - Lat: {result['MAE']['Lat']}, Long: {result['MAE']['Long']}\n")
    #         f.write(f"Inference Time (ms): {result['Inference Time (ms)']}\n")
    #         f.write("\n") 


if __name__ == '__main__':
    main()