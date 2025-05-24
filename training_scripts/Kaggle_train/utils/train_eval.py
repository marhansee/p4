from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import time
import numpy as np

def get_model_name(model):
    if hasattr(model, 'steps'):
        return model.steps[-1][1].__class__.__name__
    return model.__class__.__name__

def measure_inference_time(model, X_val, n_runs=100):
    """
    Measure average inference time of a model.

    Args:
        model: Trained classifier
        X_val: Validation features (pandas DataFrame or ndarray)
        n_runs: Number of samples to time

    Returns:
        avg_time: Average prediction time per sample (in milliseconds)
    """
    sample_indices = np.random.choice(len(X_val), size=min(n_runs, len(X_val)), replace=False)
    samples = X_val.iloc[sample_indices] if hasattr(X_val, 'iloc') else X_val[sample_indices]

    start = time.time()
    for i in range(len(samples)):
        _ = model.predict(samples[i:i+1])
    end = time.time()

    total_time = end - start
    avg_time_ms = (total_time / len(samples)) * 1000
    return avg_time_ms


def evaluate(model, model_name, y_pred, y_val, X_val, 
             target_latitude_feature, target_longitude_feature):
    mae_lat = mean_absolute_error(y_val[target_latitude_feature], y_pred[:, 0])
    mae_lon = mean_absolute_error(y_val[target_longitude_feature], y_pred[:, 1])


    inference_time = measure_inference_time(model, X_val)

    return {
        "model": model_name,
        "MAE": {
            'Lat': mae_lat,
            'Long': mae_lon
        },
        "Inference Time (ms)": inference_time
    }


def train(models_dict, X_train, y_train, X_val, y_val, target_latitude_feature, target_longitude_feature):
    assert set(X_train.columns) == set(X_val.columns), "Mismatch in training and validation features!"

    if not isinstance(models_dict, dict):
        raise AssertionError("models_dict must be a dictionary!")
    
    if not models_dict:
        raise AssertionError("models_dict must not be empty!")

    results = []
    trained_models = {}

    for name, model in models_dict.items():
        print(f"Training {name}")
        # model.fit(X_train, y_train)
        multi_model = MultiOutputRegressor(model, n_jobs=-1)
        multi_model.fit(X_train, y_train)
        print("Finished training...")

        print("Making predictions...")
        pred_val = multi_model.predict(X_val)
        result = evaluate(
            model=multi_model,
            model_name=name,
            y_pred=pred_val,
            y_val=y_val,
            X_val=X_val,
            target_latitude_feature=target_latitude_feature,
            target_longitude_feature=target_longitude_feature
        )
        results.append(result)
        trained_models[name] = multi_model

    return results, trained_models




