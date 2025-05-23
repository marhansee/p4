import pandas as pd
import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier
import os
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from function_house import split_data, measure_inference_time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

def get_model_name(model):
    if hasattr(model, 'steps'):
        return model.steps[-1][1].__class__.__name__
    return model.__class__.__name__

def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    accuracy = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    avg_inf_time = measure_inference_time(model, X_val)

    return {
        "model": get_model_name(model),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "avg_inference_time_ms": avg_inf_time
    }

def train_all_classifiers(train_csv, val_csv, feature_columns):

    # Load data
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Separate inputs and labels
    X_train = train_df[feature_columns]
    y_train = train_df["label"]
    X_val = val_df[feature_columns]
    y_val = val_df["label"]

    models = [
        make_pipeline(SimpleImputer(strategy="constant", fill_value=0), LogisticRegression(max_iter=1000, C=0.5)),
        make_pipeline(SimpleImputer(strategy="constant", fill_value=0), KNeighborsClassifier(n_neighbors=5)),
        make_pipeline(SimpleImputer(strategy="constant", fill_value=0), DecisionTreeClassifier(max_depth=10, min_samples_split=5)),
        make_pipeline(SimpleImputer(strategy="constant", fill_value=0), RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        make_pipeline(SimpleImputer(strategy="constant", fill_value=0), XGBClassifier(use_label_encoder=False, eval_metric='logloss', learning_rate=0.3, max_depth=10, subsample=0.5, n_jobs=-1, verbosity=0, random_state=42)),
    ]

    results = []

    for model in tqdm(models, desc=f"Models on {os.path.basename(train_csv)}", leave=False):
        metrics = evaluate_model(model, X_train, y_train, X_val, y_val)
        metrics["train_file"] = os.path.basename(train_csv)
        metrics["corruption"] = classify_corruption(train_csv)
        results.append(metrics)

    return results

def train_all_classifiers_mean(train_csv, val_csv, feature_columns):

    # Load data
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Separate inputs and labels
    X_train = train_df[feature_columns]
    y_train = train_df["label"]
    X_val = val_df[feature_columns]
    y_val = val_df["label"]

    models = [
        make_pipeline(SimpleImputer(strategy="mean"), LogisticRegression(max_iter=1000, C=0.5)),
        make_pipeline(SimpleImputer(strategy="mean"), KNeighborsClassifier(n_neighbors=5)),
        make_pipeline(SimpleImputer(strategy="mean"), DecisionTreeClassifier(max_depth=10, min_samples_split=5)),
        make_pipeline(SimpleImputer(strategy="mean"), RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        make_pipeline(SimpleImputer(strategy="mean"), XGBClassifier(use_label_encoder=False, eval_metric='logloss', learning_rate=0.3, max_depth=10, subsample=0.5, n_jobs=-1, verbosity=0, random_state=42)),
    ]

    results = []

    for model in tqdm(models, desc=f"Models on {os.path.basename(train_csv)}", leave=False):
        metrics = evaluate_model(model, X_train, y_train, X_val, y_val)
        metrics["train_file"] = os.path.basename(train_csv)
        metrics["corruption"] = classify_corruption(train_csv)
        results.append(metrics)

    return results


def classify_corruption(csv_path):
    filename = os.path.basename(csv_path)
    if "missing" in filename and "dupes" in filename and "labelnoise" in filename:
        return "missing+dupes+labelnoise"
    elif "missing" in filename and "dupes" in filename:
        return "missing+dupes"
    elif "missing" in filename and "labelnoise" in filename:
        return "missing+labelnoise"
    elif "dupes" in filename and "labelnoise" in filename:
        return "dupes+labelnoise"
    elif "missing" in filename:
        return "missing"
    elif "dupes" in filename:
        return "dupes"
    elif "labelnoise" in filename:
        return "labelnoise"
    else:
        return "clean"

def create_sequence_dataset(coordinates, lookback=5):
    """
    Create sequence dataset from coordinate data for trajectory prediction.
    
    Args:
        coordinates: Array of shape (n_samples, 2) containing (x, y) coordinates
        lookback: Number of previous positions to use for prediction
        
    Returns:
        X: Features array of shape (n_samples - lookback, lookback * 2)
        y: Targets array of shape (n_samples - lookback, 2)
    """
    X, y = [], []

    for i in range(lookback, len(coordinates)-1):
        past_positions = coordinates[i-lookback:i].flatten()
        target = coordinates[i+1]

        X.append(past_positions)
        y.append(target)
    
    return np.array(X), np.array(y)

def train_svr_trajectory_predictor(X_train, y_train, kernel='rbf', C=1.0, epsilon=0.1):
    """
    Train a Support Vector Regression model for trajectory prediction.
    
    Args:
        X_train: Training features (past positions)
        y_train: Training targets (next positions)
        kernel: SVR kernel type ('rbf', 'linear', 'poly', etc.)
        C: Regularization parameter
        epsilon: Epsilon in the epsilon-SVR model
        
    Returns:
        model: Trained SVR model
        scaler: Fitted scaler for preprocessing new data
    """
    # Scale features (important for SVR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create and train SVR model
    svr = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model = MultiOutputRegressor(svr)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def train_rfr_trajectory_predictor(X_train, y_train, n_estimators=100, random_state=None):
    """
    Train a Random Forest Regression model for trajectory prediction.
    
    Args:
        X_train: Training features (past positions)
        y_train: Training targets (next positions)
        n_estimators: Number of trees in the forest
        random_state: Random seed for reproducibility
        
    Returns:
        model: Trained RFR model
    """
    # Create and train RFR model
    rfr = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model = MultiOutputRegressor(rfr)
    model.fit(X_train, y_train)
    
    return model

def evaluate_trajectory_predictor(model, X_test, y_test, scaler=None):
    """
    Evaluate a trajectory prediction model.
    
    Args:
        model: Trained model to evaluate
        X_test: Test features
        y_test: Test targets
        scaler: Scaler object if needed (for SVR)
        
    Returns:
        mae: Mean Absolute Error
        mse: Mean Squared Error
    """
    # Preprocess if scaler is provided (for SVR)
    if scaler:
        X_test = scaler.transform(X_test)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    return mae, mse

def predict_next_positions(model, current_sequence, lookback, scaler=None):
    """
    Predict the next position given a sequence of recent positions.
    
    Args:
        model: Trained prediction model
        current_sequence: Array of recent (x,y) positions (shape: [lookback, 2])
        lookback: Number of positions in the sequence
        scaler: Scaler object if needed (for SVR)
        
    Returns:
        Array containing predicted (x,y) position
    """
    # Prepare input
    input_data = current_sequence.flatten().reshape(1, -1)
    
    # Scale if needed
    if scaler:
        input_data = scaler.transform(input_data)