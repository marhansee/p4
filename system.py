import numpy as np
import time
import onnxruntime as ort
import os
import sys
import pandas as pd
import folium
from folium.plugins import BoatMarker
import webbrowser
from shapely.geometry import LineString
import geopandas as gpd
from shapely.geometry import mapping

from shapely.ops import transform
import pyproj

from numpy.linalg import norm
from geopy.distance import geodesic


"""
Script requires following packages:
pip install folium shapely geopandas pyproj geopy

"""

# Simulate data [timestamp, lat, lon, speed, heading]
data_stream =     [
    np.array([0, 59.0, 10.0, 6.0, 180.0]),
    np.array([1, 59.0005, 10.0005, 5.5, 182.0]),
    np.array([2, 59.0010, 10.0010, 5.0, 185.0]),
    np.array([3, 59.0015, 10.0015, 4.5, 190.0]), # Approaching cable zone
]

def load_model(model_path):
    try:
        model = ort.InferenceSession(model_path)
        print("Model has been loaded")
        return model
    
    except Exception as e:
        print(f"Error occurred: \n {e}")
        sys.exit()


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

def add_critical_zone(cable_coords, radius_meters=1000):
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
    line = LineString([(lon, lat) for lat, lon in cable_coords])

    # Define projection: WGS84 (lat/lon) to UTM
    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    line_projected = transform(project, line)

    buffered = line_projected.buffer(radius_meters)

    # Project back to lat/lon
    project_back = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    buffered_latlon = transform(project_back, buffered)

    return buffered_latlon

def interactive_map(cable_dict, vessel_position):
    """
    Function to create an interactive map with multiple cables. The interactive map is opened in a webbrowser.
    Each cable will be plotted as a separate polyline on the map.

    Note,
    This function was created on the basis of multiple sources.
    Sources: 
        https://python-visualization.github.io/folium/latest/getting_started.html
        https://python-visualization.github.io/folium/latest/user_guide/plugins/boat_marker.html'
        https://stackoverflow.com/questions/51486454/convert-geopandas-shapely-polygon-to-geojson

    Args:
    - cable_dict: A dictionary containing cables with lists of coordinates.
                  Example: {'Cable_1': [(lat1, lon1), (lat2, lon2), ...], 'Cable_2': [(lat3, lon3), (lat4, lon4), ...]}

    Returns:
    - Nothing. 
    """
    
    map_dir = os.path.join(os.path.dirname(__file__), 'cable_map')
    map_file = os.path.join(map_dir, 'interactive_cable_map.html')

    if not os.path.exists(map_file):
        os.makedirs(map_dir, exist_ok=True)

    # Define map
    m = folium.Map(location=[56.573213, 10.687685], 
                zoom_start=7, 
                tiles='Cartodb Positron'
    )

    # Draw cable lines
    for cable, coords in cable_dict.items():
        folium.PolyLine(coords, tooltip=cable).add_to(m)

        buffered_polygon = add_critical_zone(coords, radius_meters=1600)

        # Convert Shapely Polygon to GeoJSON FeatureCollection
        feature = {
            "type": "Feature",
            "properties": {"name": f"{cable} critical zone"},
            "geometry": mapping(buffered_polygon)
        }
        feature_collection = {
            "type": "FeatureCollection",
            "features": [feature]
        }

        folium.GeoJson(
            feature_collection,
            name=f"{cable} Buffer",
            tooltip=f"{cable} 1km zone",
            style_function=lambda x: {
                'fillColor': 'red',
                'color': 'red',
                'fillOpacity': 0.2,
                'weight': 1
            }
        ).add_to(m)


        BoatMarker(
            location=vessel_position, heading=30, color="#8f8"
        ).add_to(m)

    
    # Save html file
    m.save(map_file)

    # Open the html file in browser
    webbrowser.open(f'file://{map_file}')


def compute_distance_from_point_to_cable(current_position, seg_start, seg_end):
    """
    Compute the shortest distance between a point and a line segment.
    
    Args:
    - point: The point coordinates (numpy array).
    - seg_start: The start coordinates of the segment (numpy array).
    - seg_end: The end coordinates of the segment (numpy array).
    
    Returns:
    - The shortest distance from the point to the segment.
    """

    # Define vectors
    seg_vector = seg_end - seg_start
    current_pos_vector = current_position - seg_start

    # Project current_pos_vector onto seg_vector
    seg_length = norm(seg_vector)
    if seg_length != 0:
        seg_unit_vector = seg_vector / seg_length
    else:
        seg_unit_vector = seg_vector 
    projection = np.dot(current_pos_vector, seg_unit_vector)

    # Find closest point on the segment
    closest_point = seg_start + projection * seg_unit_vector


    # If the projection is outside the segment, use the closest endpoint
    if projection < 0:
        closest_point = seg_start
    elif projection > seg_length:
        closest_point = seg_end

    # Compute distance in km using geopy
    dist = geodesic(current_position, closest_point).kilometers


    return dist

def unit_vector(v):
    # Function computes the unit vector
    return v / norm(v)

def compute_angle(vector1, vector2):
    """
    Function that computes the angle between two vectors

    Note,
    This function has been inspired by a Stackoverflow post.
    Source: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    
    Interpretation:
    0°	Vessel is heading exactly along the segment's direction
    < 30°	Vessel is heading roughly in the same direction — likely approaching
    90°	Vessel is moving perpendicular to the cable — not approaching
    > 150°	Vessel is moving away from the cable segment

    Args:
    - vector1: The first vector
    - vector2: The second vector

    Returns:
    - The angle in degrees.
    """


    v1_unit, v2_unit = unit_vector(vector1), unit_vector(vector2)
    angle_radian = np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0))
    return np.degrees(angle_radian)

def is_heading_toward_cable(cables, current_pos, predicted_pos, angle_threshold=30, proximity_threshold=10):
    """
    Function checks whether a vessel is headed toward a subsea cable

    Args:
    - cables: Dictionary of cable coordinates
    - current_position: Current position of the vessel at time t=0 (used to compute the distance to the cable from current position)
    - predicted_pos: Forecasting predictions
    - angle_threshold: An angle below 30 degrees signifies that a vessel is headed toward cable
    - proximity_threshold: Threshold distance for determining "close" proximity to a cable. Measured in km

    Returns:
    - Bool: True | False
        True: A vessel is headed toward one of the cables
        False: A vessel is NOT headed toward one of the cables.
    """
    vessel_vector = np.array(predicted_pos[-1]) - np.array(predicted_pos[-2])
    for cable_name, coord in cables.items():
        for i in range(len(coord)-1):
            # Take into account if the cable bends by dividing into multiple segments
                # For example with 1 bend:
                # Segment 1: from start to bend.
                # Segment 2: from bend to end
            seg_start = np.array(coord[i])
            seg_end = np.array(coord[i+1])

            # Compute the shortest distance from current position to cable
            distance = compute_distance_from_point_to_cable(
                current_position=current_pos, 
                seg_start=seg_start, 
                seg_end=seg_end
            )

            if distance <= proximity_threshold:
                print(f"ALARM! Vessel is {round(distance,2)}km from {cable_name}")
                cable_vector = seg_end - seg_start

                # Compute angle
                angle = compute_angle(vessel_vector, cable_vector)
                if angle.any() < angle_threshold:
                    print(f"{cable_name} is likely approached! Angle: {angle}")
                    return True
        
    return False

def main():
<<<<<<< HEAD
    classifier_path = os.path.join(os.path.dirname(__file__),'models/dummy_classifier.onnx')
    forecaster_path = os.path.join(os.path.dirname(__file__),'models/dummy_forecaster.onnx')

    classifier = load_model(classifier_path)
    forecaster = load_model(forecaster)
=======
    # model_path = os.path.join(os.path.dirname(__file__),'models/dummy_classifier.onnx')
    # model = load_model(model_path)
>>>>>>> bd9477701c794a870cdd7eeaa0b0aa07bf73348c

    cable_data_path = os.path.join(os.path.dirname(__file__),'data/cable_positions.csv')
    cable_dict = load_cable_position_data(cable_data_path)

    

    # TESTING WITH DUMMY
    current_position = (57.3569, 10.7360)
    dummy_predicted_pos = [
        (57.30000000, 10.40000000),
        (57.34000000, 10.50000000),
        (57.38000000, 10.60000000)
    ]

<<<<<<< HEAD
    heading_toward = is_heading_toward_cable(
        cables=cable_dict,
        current_pos=current_position,
        predicted_pos=dummy_predicted_pos,
        angle_threshold=30
    )
=======
    # is_heading_toward_cable(
    #     cables=cable_dict,
    #     current_pos=current_position,
    #     predicted_pos=dummy_predicted_pos,
    #     angle_threshold=30
    # )
>>>>>>> bd9477701c794a870cdd7eeaa0b0aa07bf73348c

    if heading_toward:
        print("Vessel is heading toward a cable. Running trawling classifier...")

        data = preprocess(trawling=True) # En eller anden preprocess pipeline

        prediction = classifier.predict(data)
        
        if prediction == 1:
            print("Trawling activity detected! Take action!")
        else:
            print("Vessel is approaching, but no trawling detected.")

    else:
        print("Vessel is not heading toward any cable.")


    # interactive_map(cable_dict, vessel_position=current_position)

if __name__ == '__main__':
    main()