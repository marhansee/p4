import folium
import json
from shapely.geometry import mapping
from utilities.zone_check import load_cable_lines, build_buffered_zone
import yaml
import pandas as pd
from branca.element import Template, MacroElement

# --- Load forecast results ---
with open("data/results.json") as f:
    data = json.load(f)

results = data["results"]


# --- Load YAML config and cable lines ---
with open("utilities/inference_config.yaml", "r") as f:
    config = yaml.safe_load(f)

cable_path = config["cable_coordinates_path"]
cable_lines = load_cable_lines(cable_path)

cable_lines = load_cable_lines(cable_path)
buffered_zone_1602 = build_buffered_zone(cable_lines, buffer_meters=1602)
buffered_zone_2136 = build_buffered_zone(cable_lines, buffer_meters=2136)

# --- Load true (unnormalized) vessel trajectory ---
true_path_df = pd.read_csv("data/last_input.csv")
true_track = true_path_df[["Latitude", "Longitude"]].dropna().values.tolist()

# --- Initialize map ---
m = folium.Map(location=[57, 11], zoom_start=8)

# --- Draw critical zone ---
folium.GeoJson(
    mapping(buffered_zone_2136),
    name="Pre-alert Zone",
    style_function=lambda x: {
        "fillColor": "yellow",
        "color": "orange",
        "fillOpacity": 0.1,
        "weight": 1
    }
).add_to(m)

# Draw 1602m critical zone (opaque red)
folium.GeoJson(
    mapping(buffered_zone_1602),
    name="Critical Zone",
    style_function=lambda x: {
        "fillColor": "#ff4d4d",
        "color": "#ff4d4d",
        "fillOpacity": 0.3,
        "weight": 1
    }
).add_to(m)

# --- Draw true AIS track (dotted gray line) ---
if true_track:
    folium.PolyLine(
        locations=true_track,
        color="gray",
        weight=2,
        dash_array="5, 5",
        popup="True AIS trajectory"
    ).add_to(m)

for i, result in enumerate(results[::-1][::20]):
    forecast = result["forecast"]

    entry_step = result.get("zone_entry_step")

    risk_level = result.get("risk_level", 0)
    color = {
        3: "red",
        2: "yellow",
        1: "green",
    }.get(risk_level, "gray")

    # Plot forecast path
    folium.PolyLine(
        locations=[(lat, lon) for lat, lon in forecast],
        color=color,
        weight=4,
        popup=f"Forecast window {i + 1}"
    ).add_to(m)

    # Mark starting point of forecast
    start_lat, start_lon = forecast[0]
    folium.CircleMarker(
        location=(start_lat, start_lon),
        radius=5,
        color="black",
        fill=True,
        fill_color="blue",
        fill_opacity=1,
        tooltip=f"Start of forecast {i + 1}"
    ).add_to(m)

    # Dots for each forecast step

    for step, (lat, lon) in enumerate(forecast):
        folium.CircleMarker(
            location=(lat, lon),
            radius=1,
            color="black",
            fill=True,
            fill_color="white",
            fill_opacity=0.9,
            tooltip=f"Window {i + 1}, Step {step}"
        ).add_to(m)

    # # Critical zone entry
    # if entry_step is not None:
    #     entry_point = forecast[entry_step]
    #     folium.CircleMarker(
    #         location=(entry_point[0], entry_point[1]),
    #         radius=6,
    #         color="yellow",
    #         fill=True,
    #         fill_color="yellow",
    #         popup=f"Zone entry in window {i + 1}"
    #     ).add_to(m)

    # # Plot input path if available
    # if "input" in result:
    #     input_coords = result["input"]
    #     folium.PolyLine(
    #         input_coords,
    #         color="blue",
    #         weight=2,
    #         opacity=0.7,
    #         dash_array="4",
    #     ).add_to(m)
    #
    #     for lat, lon in input_coords:
    #         folium.CircleMarker(
    #             location=(lat, lon),
    #             radius=2,
    #             color="blue",
    #             fill=True,
    #             fill_opacity=1
    #         ).add_to(m)



legend_html = """
{% macro html(this, kwargs) %}
<div style="
    position: fixed;
    bottom: 40px;
    left: 40px;
    width: 220px;
    background-color: white;
    border:2px solid grey;
    z-index:9999;
    font-size:14px;
    padding: 10px;
    border-radius: 5px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
">
<b>Legend</b><br>
<span style="color: red;">■</span> Forecast path (zone alert)<br>
<span style="color: green;">■</span> Forecast path (no alert)<br>
<span style="color: blue;">●</span> Start of forecast<br>
<span style="color: yellow;">●</span> Zone entry point<br>
<span style="color: gray;">⋯</span> True AIS trajectory<br>
<span style="color: orange;">⬛</span> Pre-alert zone (2136m)<br>
<span style="color: #ff4d4d;">⬛</span> Critical zone (1602m)<br>
</div>
{% endmacro %}
"""

legend = MacroElement()
legend._template = Template(legend_html)
m.get_root().add_child(legend)


# --- Save map to file ---
m.save("forecast_map.html")
print("Map saved to forecast_map.html")
