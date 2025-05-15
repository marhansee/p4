from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from geopy.distance import geodesic
import pandas as pd
import numpy as np

def load_cable_lines(csv_path: str):
    df = pd.read_csv(csv_path)
    cable_lines = []

    for cable_id, group in df.groupby("Cable"):
        coords = [(lon, lat) for lat, lon in zip(group["Latitude"], group["Longitude"])]
        line = LineString(coords)
        cable_lines.append(line)

    return cable_lines

def build_buffered_zone(cable_lines, buffer_meters=1602):

    buffer_deg = buffer_meters / 111320  # approx meters to degrees (latitude-based)
    buffered = [line.buffer(buffer_deg) for line in cable_lines]
    return unary_union(buffered)

def point_near_cables(lat, lon, cable_lines, radius_m=1602):

    for line in cable_lines:
        for i in range(len(line.coords) - 1):
            seg_start = line.coords[i]
            seg_end = line.coords[i + 1]

            segment = LineString([seg_start, seg_end])
            closest = segment.interpolate(segment.project(Point(lon, lat)))  # note: Point(lon, lat)

            # Compute distance using geodesic (lat, lon) = (y, x)
            dist = geodesic((lat, lon), (closest.y, closest.x)).meters

            if dist <= radius_m:
                return True
    return False

def any_forecast_in_zone(forecast: list | np.ndarray, cable_lines, radius_m=1602):

    for lat, lon in forecast:
        if point_near_cables(lat, lon, cable_lines, radius_m):
            return True
    return False
