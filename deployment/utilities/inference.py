import onnxruntime as ort
import numpy as np

class AISInferenceModel:

    def __init__(self, classifier_path: str, forecaster_path: str, verbose=True):
        self.classifier = ort.InferenceSession(classifier_path)
        self.classifier_input_name = self.classifier.get_inputs()[0].name
        self.forecaster = ort.InferenceSession(forecaster_path)
        self.forecaster_input_name = self.forecaster.get_inputs()[0].name
        self.verbose = verbose

        if self.verbose:
            print("Classifier:")
            print("Input name:", self.classifier_input_name)
            print("Input shape:", self.classifier.get_inputs()[0].shape)
            print("Input type:", self.classifier.get_inputs()[0].type)

            print("Forecaster:")
            print("Input name:", self.forecaster_input_name)
            print("Input shape:", self.forecaster.get_inputs()[0].shape)
            print("Input type:", self.forecaster.get_inputs()[0].type)


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
        if forecast is not None:
            print("Forecast coordinates at future steps:")
            for step in [1, 5, 10, 20]:
                if step <= len(forecast):
                    lat, lon = forecast[step - 1]
                    print(f"Step {step:>2}: Latitude = {lat:.5f}, Longitude = {lon:.5}")
                else:
                    print(f"Step {step:>2}: Not available (Only {len(forecast)}) steps predicted.")

        return label, probability, logit, forecast

