import onnxruntime as ort
import numpy as np
import json
import os

class AISInferenceModel:

    def __init__(self, classifier_path: str, forecaster_path: str, verbose=True):
        self.classifier = ort.InferenceSession(classifier_path)
        self.classifier_input_name = self.classifier.get_inputs()[0].name
        self.forecaster = ort.InferenceSession(forecaster_path)
        self.forecaster_input_name = self.forecaster.get_inputs()[0].name
        self.verbose = verbose

        stats_path = os.path.join(os.path.dirname(__file__), "../data/train_norm_stats.json")
        with open(stats_path, "r") as f:
            stats = json.load(f)

        self.lat_mean = stats["Latitude"]["mean"]
        self.lat_std = stats["Latitude"]["std"]
        self.lon_mean = stats["Longitude"]["mean"]
        self.lon_std = stats["Longitude"]["std"]

        if self.verbose:
            print("Classifier:")
            print("Input name:", self.classifier_input_name)
            print("Input shape:", self.classifier.get_inputs()[0].shape)
            print("Input type:", self.classifier.get_inputs()[0].type)

            print("Forecaster:")
            print("Input name:", self.forecaster_input_name)
            print("Input shape:", self.forecaster.get_inputs()[0].shape)
            print("Input type:", self.forecaster.get_inputs()[0].type)

            print("Loaded normalization stats:")
            print(f"Latitude mean: {self.lat_mean}, std: {self.lat_std}")
            print(f"Longitude mean: {self.lon_mean}, std: {self.lon_std}")


    def sigmoid(self, x:float):
        return 1 / (1 + np.exp(-x))

    def predict(self, input_tensor: np.ndarray):

        classifier_output = self.classifier.run(None, {self.classifier_input_name: input_tensor})
        logit = classifier_output[0][0][0]
        probability = self.sigmoid(logit)
        label = int(probability >= 0.5)
        if self.verbose:
            print(f"Logit: {logit:.4f}, Probability: {probability:.4f}, Label: {label}")
        forecast = None
        if label == 1:
            forecaster_output = self.forecaster.run(None, {self.forecaster_input_name: input_tensor})
            forecast = forecaster_output[0][0]
        if forecast is not None and self.verbose:
            print("Forecast coordinates at future steps:")
            key_steps = [0, 1, 2, 4, 9, 19]
            for step in key_steps:
                if step < len(forecast):
                    lat, lon = forecast[step]
                    print(f"Step {step + 1:>2}: Latitude = {lat:.5f}, Longitude = {lon:.5f}")
                else:
                    print(f"Step {step + 1:>2}: Not available (only {len(forecast)}) steps predicted.")

        return label, probability, logit, forecast

