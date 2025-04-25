import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


class DummyForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 2)  # output: future lat, lon
        )

    def forward(self, x):
        return self.fc(x)

# Dummy forecaster
model = DummyForecaster()
dummy_input = torch.randn(1,5)
torch.onnx.export(model, dummy_input, "models/dummy_forecaster.onnx", 
                  input_names=['input'], output_names=['output'], 
                  opset_version=11)


# Dummy classifier
X, y = make_classification(n_samples=100, n_features=5)
clf = LogisticRegression()
clf.fit(X, y)
initial_type = [('input', FloatTensorType([None, 5]))]
onnx_model = convert_sklearn(clf, initial_types=initial_type)
with open("models/dummy_classifier.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())