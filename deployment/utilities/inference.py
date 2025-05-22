import onnxruntime as ort
import numpy as np
import json
import os

class AISInferenceModel:

    def __init__(self, classifier_path, forecaster_path,
                 fallback_classifier_path=None, fallback_forecaster_path=None,
                 verbose=True):
        self.classifier = ort.InferenceSession(classifier_path)
        self.classifier_input_name = self.classifier.get_inputs()[0].name

        self.forecaster = ort.InferenceSession(forecaster_path)
        self.forecaster_input_name = self.forecaster.get_inputs()[0].name

        self.fallback_classifier = None
        self.fallback_classifier_input_name = None
        if fallback_classifier_path:
            self.fallback_classifier = ort.InferenceSession(fallback_classifier_path)
            self.fallback_classifier_input_name = self.fallback_classifier.get_inputs()[0].name

        self.fallback_forecaster = None
        self.fallback_forecaster_input_name = None
        if fallback_forecaster_path:
            self.fallback_forecaster = ort.InferenceSession(fallback_forecaster_path)
            self.fallback_forecaster_input_name = self.fallback_forecaster.get_inputs()[0].name

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

    def predict(self, input_tensor: np.ndarray, use_fallback: bool = False):
        if use_fallback and self.fallback_classifier:
            clf_session = self.fallback_classifier
            clf_input_name = self.fallback_classifier_input_name
        else:
            clf_session = self.classifier
            clf_input_name = self.classifier_input_name

        classifier_output = clf_session.run(None, {clf_input_name: input_tensor})
        logit = classifier_output[0][0][0]
        probability = self.sigmoid(logit)
        label = int(probability >= 0.5)

        forecast = None
        if label == 1:
            if use_fallback and self.fallback_forecaster:
                fc_session = self.fallback_forecaster
                fc_input_name = self.fallback_forecaster_input_name
            else:
                fc_session = self.forecaster
                fc_input_name = self.forecaster_input_name

            forecast_output = fc_session.run(None, {fc_input_name: input_tensor})
            forecast = forecast_output[0][0]

        return label, probability, logit, forecast




