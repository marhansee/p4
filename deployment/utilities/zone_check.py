from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from geopy.distance import geodesic
import pandas as pd


def load_cable_lines(csv_path: str):
    df = pd.read_csv(csv_path)
    cable_lines = []

    for cable_id, group in df.groupby("Cable"):
        coords = [(lon, lat) for lat, lon in zip(group["Latitude"], group["Longitude"])]
        line = LineString(coords)
        cable_lines.append(line)

    return cable_lines

def build_buffered_zone(cable_lines, buffer_meters=1602):
    buffer_deg = buffer_meters / 111320  # very rough lat/deg approximation
    buffered = [line.buffer(buffer_deg) for line in cable_lines]
    return unary_union(buffered)  # merges all buffer polygons into one

def any_forecast_in_zone(forecast, buffered_zone):
    for lat, lon in forecast:
        point = Point(lon, lat)
        if buffered_zone.contains(point):
            return True
    return False

def first_forecast_in_zone(forecast, buffered_zone):
    for i, (lat, lon) in enumerate(forecast):
        point = Point(lon, lat)
        if buffered_zone.contains(point):
            return i
    return None

def all_forecast_steps_in_zone(forecast, buffered_zone):
    return [
        i for i, (lat, lon) in enumerate(forecast)
        if buffered_zone.contains(Point(lon, lat))
    ]



def vessel_near_any_cable(current_lat, current_lon, cable_lines, radius_m=2200):
    point = Point(current_lon, current_lat)  # note: Point(long, lat)
    for line in cable_lines:
        for i in range(len(line.coords) - 1):
            segment = LineString([line.coords[i], line.coords[i + 1]])
            closest = segment.interpolate(segment.project(point))
            dist = geodesic((current_lat, current_lon), (closest.y, closest.x)).meters
            if dist <= radius_m:
                return True
    return False