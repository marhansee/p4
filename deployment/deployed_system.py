from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import logging
import uvicorn
from typing import List

app = FastAPI()

# Loading the ONNX models
forecaster_path = 'models_for_deployment/FORECASTER' # Fix path
classifier_path = 'models_for_deployment/CLASSIFIER'

try:
    classifier_model = ort.InferenceSession(classifier_path)
    forecaster_model = ort.InferenceSession(forecaster_path)
except Exception as e:
    logging.error(f"Error loading ONNX model: {e}")
    raise e

# Define input data
class AISDataPoint(BaseModel):
    Latitude: float
    Longitude: float
    ROT: float
    SOG: float
    COG: float
    Heading: float
    Width: float
    Length: float
    Draught: float


class AIS_Sequence(BaseModel):
    data: List[AISDataPoint]

# Insert Preprocess logic blow:


@app.get("/", response_class=RedirectResponse)
def index():
    url = "/docs"
    return RedirectResponse(url)

def preprocess_raw_data(data_request: AIS_Sequence):
    # Prepare the input data as a numpy array

    input_data = np.array([[
        data_request.Latitude,
        data_request.Longitude,
        data_request.ROT,
        data_request.SOG,
        data_request.COG,
        data_request.Heading,
        data_request.Width,
        data_request.Length,
        data_request.Draught
        ]], dtype=np.float32
    )


    # Preprocess logic here.




    # Preprocess logic end here.

    print(f"Original input data for prediction: {input_data}")

    return input_data

def get_classifier_prediction(data):
    inputs = {classifier_model.get_inputs()[0].name: data}
    prediction = classifier_model.run(None, inputs)
    return prediction[0]

def get_forecaster_prediction(data):
    inputs = {forecaster_model.get_inputs()[0].name: data}
    prediction = forecaster_model.run(None, inputs)
    return prediction[0]


@app.post("/predict")
def predict_trawling(data_request: AIS_Sequence):

    preprocessed_df = preprocess_raw_data(data_request.data)

    # Convert processed data to the format needed for the models
    input_data = preprocessed_df[['Latitude', 'Longitude', 'ROT', 'SOG',
                                  'COG','Heading','Width','Length','Draught']
                                  ].values.astype(np.float32)


    try:
        # First, use the classifier model to detect if trawling is occurring
        classifier_prediction = get_classifier_prediction(input_data)

        print("Prediction probabilities:")
        print(classifier_prediction[0])

        # Activate forecaster if trawling is detected
        if np.round(classifier_prediction[0])==1:
            forecast_trajectory = get_forecaster_prediction(input_data)

            # MANGLER LOGIC FOR CRITICAL ZONE INTERSECTION


            ######
            return {
                "classifier_prediction": classifier_prediction[0].tolist(),
                "forecaster_prediction": forecast_trajectory[0].tolist()
            }
        else:
            # If trawling is not detected, return only the classifier prediction
            return {
                "classifier_prediction": classifier_prediction[0].tolist(),
                "message": "No trawling detected"
            }
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)