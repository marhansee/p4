import folium
import json
from shapely.geometry import mapping
from utilities.zone_check import load_cable_lines, build_buffered_zone
import yaml
import pandas as pd
from branca.element import Template, MacroElement, Html

# --- Load forecast results ---
with open("data/results.json") as f:
    data = json.load(f)

results = data["results"]

fallback_count = sum(1 for r in results if r.get("used_fallback"))
total_count = len(results)
fallback_ratio = fallback_count / max(total_count, 1)

# --- Load YAML config and cable lines ---
with open("utilities/inference_config.yaml", "r") as f:
    config = yaml.safe_load(f)

cable_path = config["cable_coordinates_path"]
cable_lines = load_cable_lines(cable_path)

cable_lines = load_cable_lines(cable_path)
buffered_zone_1602 = build_buffered_zone(cable_lines, buffer_meters=1602)
buffered_zone_2136 = build_buffered_zone(cable_lines, buffer_meters=3738)

# --- Load true (unnormalized) vessel trajectory ---
true_path_df = pd.read_csv("data/latest_forecast_input.csv")
true_track = true_path_df[["Latitude", "Longitude"]].dropna().values.tolist()

# --- Initialize map ---
m = folium.Map(location=[57, 11], zoom_start=8, tiles="OpenStreetMap")

for cable in cable_lines:
    coords = list(cable.coords)
    folium.PolyLine(
        locations=[(lat, lon) for lon, lat in coords],  # flip (lon, lat) -> (lat, lon)
        color="grey",
        weight=3,
        tooltip="Subsea cable"
    ).add_to(m)

# --- Draw critical zone ---
folium.GeoJson(
    mapping(buffered_zone_2136),
    name="Pre-alert Zone",
    style_function=lambda x: {
        "fillColor": "yellow",
        "color": "orange",
        "fillOpacity": 0.2,
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
        "fillOpacity": 0.35,
        "weight": 1
    }
).add_to(m)

# --- Draw true AIS track (dotted gray line) ---
if true_track:
    folium.PolyLine(
        locations=true_track,
        color="black",
        weight=3,
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
    confidence = result.get("fishing_confidence", None)
    tooltip = f"Forecast window {i + 1}"
    if confidence is not None:
        tooltip += f" | Fishing confidence: {round(confidence * 100, 2)}%"

    folium.PolyLine(
        locations=[(lat, lon) for lat, lon in forecast],
        color=color,
        weight=4,
        tooltip=tooltip
    ).add_to(m)

    # Mark starting point of forecast
    start_lat, start_lon = forecast[0]
    fallback_used = result.get("used_fallback", False)

    folium.CircleMarker(
        location=(start_lat, start_lon),
        radius=5,
        color="black",
        fill=True,
        fill_color="blue",
        fill_opacity=1,
        tooltip=f"Start of forecast {i + 1}" + (" ⚠️ Fallback used" if fallback_used else "")
    ).add_to(m)

    if fallback_used:
        folium.Marker(
            location=(start_lat + 0.002, start_lon + 0.002),  # slightly offset
            icon=folium.DivIcon(html="""
                <div style="font-size: 18px; color: red;">⚠️</div>
            """)
        ).add_to(m)

    # Dots for each forecast step

    for step, (lat, lon) in enumerate(forecast):
        folium.CircleMarker(
            location=(lat, lon),
            radius=0.1,
            color="black",
            fill=True,
            fill_color="white",
            fill_opacity=0.4,
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
    width: 280px;
    background-color: white;
    border:2px solid grey;
    z-index:9999;
    font-size:14px;
    padding: 10px;
    border-radius: 5px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
">
<b>Legend</b><br>
<span style="color: red;">■</span> Risk level 3 forecast path (trawling inside critical zone)<br>
<span style="color: yellow;">■</span> Risk level 2 forecast path (trawling near critical zone)<br>
<span style="color: green;">■</span> Risk level 1 forecast path (trawling outside zones)<br>
<span style="color: blue;">●</span> Start of forecast<br>
<span style="font-size: 16px; color: red;">⚠️</span> Fallback model marker (offset)<br>
<span style="color: gray;">⋯</span> True AIS trajectory (historical)<br>
<span style="color: grey;">━━</span> Subsea cable line<br>
<span style="display:inline-block; width:16px; height:16px; background-color: yellow; border:1px solid black; opacity:0.2; margin-right:4px;"></span> Pre-alert zone (2136m buffer)<br>
<span style="display:inline-block; width:16px; height:16px; background-color: #ff4d4d; border:1px solid black; opacity:0.35; margin-right:4px;"></span> Critical zone (1602m buffer)<br>
</div>
{% endmacro %}
"""



model_info_html = f"""
<div style="
    position: fixed;
    top: 20px;
    left: 40px;
    background-color: white;
    border: 2px solid grey;
    padding: 10px 14px;
    z-index: 9999;
    font-size: 14px;
    font-weight: normal;
    border-radius: 5px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
">
<b>Classification Model Performance (F₁ Score)</b><br>
Primary model: <b>90.72%</b><br>
Fallback model: <b>65.62%</b><br>
<br>
<b>Fallback usage:</b> {fallback_count} of {total_count} windows
</div>
"""
model_info = Html(model_info_html, script=True)
m.get_root().html.add_child(model_info)

model_info = Html(model_info_html, script=True)
m.get_root().html.add_child(model_info)

legend = MacroElement()
legend._template = Template(legend_html)
m.get_root().add_child(legend)


# --- Save map to file ---
m.save("forecast_map.html")
print("Map saved to forecast_map.html")
