import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import numpy as np

from utils.utils import *

"""
This script is an example usecase of the trajectory plot function

"""



def plot_raw_trajectory(df, mmsi, id_feature='MMSI', speed_feature='SOG',
                        time_feature='# Timestamp', lat_feature='Latitude', 
                        long_feature='Longitude',
                        num_hours='full', scale_factor=1.5, color_speed=True, 
                        norm='ALL', target_feature=None,
                        label=1):
    """
    Purpose: Visualize the trajectory of a vessel identified by MMSI from the start of logging to a specified time frame.

    Functionalities:
        - Is able to plot the trajectory of trawling activities
        - Is able to plot the entire trajectory of a vessel (regardless of trawling or not)
        - Is able to plot the trajectory of a vessel based on time duration (e.g., 10 min trajectory)
        - Is able to color the trajectory based on a vessel's speed

    Args:
        - df (DataFrame): The input data containing vessel tracking information.
        - mmsi: The unique MMSI identifier for the vessel to be plotted.
        - id_feature: Column name representing the vessel's MMSI. Default is 'MMSI'.
        - speed_feature: Column name representing the vessel's speed over ground (SOG). Default is 'SOG'.
        - time_feature: Column name for timestamps. Default is '# Timestamp'.
        - lat_feature: Column name representing latitude. Default is 'Latitude'.
        - long_feature: Column name representing longitude. Default is 'Longitude'.
        - num_hours (int, float, or 'full', optional): The duration (in hours) for which the trajectory should be plotted.
            - If 'full', the entire available trajectory is plotted. Default is 'full'.
        - scale_factor (float, optional): Scaling factor for adjusting the plotted map's boundaries (zoom in/out). Default is 1.5.
        - color_speed (bool, optional): If True, colors the trajectory based on speed (SOG). Default is True.
        - norm (str, optional): Normalization that determines how speed values are normalized for coloring.
            - 'ALL': Uses speed values across all vessels.
            - 'MMSI': Uses speed values of only the selected vessel.
            - Default is 'ALL'.
        - target_feature (str, optional): Column name for the target feature
            - Used for plotting the trajectory only based on the target-feature (e.g. trawling trajectory)
            - Default is None.
        - label (int, optional): The label value used for filtering if target_feature is provided. Default is 1.

    Returns:
        - None. Only used for plotting

    Notes:
        - The function assumes that the input DataFrame (pandas) contains valid AIS data.
        - If there is no available data after filtering, it prints a message and does not generate a plot.
        - Requires the `matplotlib`, `numpy`, `pandas`, and `mpl_toolkits.basemap` packages.
    
    """
    
    print(f'Plotting for {mmsi}')
    # SOG values for EVERY vessel
    sog_value = df[speed_feature].values
    custom_max_sog = 20

    # Only keep rows with specified MMSI number
    df = df[df[id_feature] == mmsi]

    # If a trawling feature is specified, filter by the label
    if target_feature is not None:
        df = df[df[target_feature] == label]

    # SOG values for specified vessel (MMSI)
    mmsi_sog = df[speed_feature].values

    # Define normalization for colors in plot
    if norm.lower() == 'all':
        norm = plt.Normalize(min(sog_value), custom_max_sog) 
    elif norm.lower() == 'mmsi':
        norm = plt.Normalize(min(mmsi_sog), max(mmsi_sog))
    else:
        raise AssertionError("You must define either 'all' or 'mmsi'")
    
    # Convert time-feature to datetime
    df[time_feature] = pd.to_datetime(df[time_feature]) 

    # Convert start_time to datetime
    start_time = df[time_feature].min()

    if num_hours == 'full':
        end_time = df[time_feature].max() # Full trajectory
    else:
        end_time = start_time + pd.Timedelta(hours=num_hours) # Part of trajectory

    # Filter by time range
    df = df[(df[time_feature] >= start_time) & (df[time_feature] <= end_time)].copy()

    # Check if df is empty after filtering
    if df.empty:
        print(f"No data available for MMSI {mmsi} between {start_time} and {end_time}.")
        return

    # Extract latitude & longitude
    lats = df[lat_feature].values
    longs = df[long_feature].values
    min_lat, max_lat = df[lat_feature].min(), df[lat_feature].max()
    min_long, max_long = df[long_feature].min(), df[long_feature].max()

    # Calculate center
    center_lat = (min_lat + max_lat) / 2
    center_long = (min_long + max_long) / 2

    # Create a temporary Basemap to project coordinates
    temp_map = Basemap(projection='merc', lat_0=center_lat, lon_0=center_long, resolution='l')
    
    # Convert min/max lat/lon to projected x, y coordinates
    x_min, y_min = temp_map(min_long, min_lat)
    x_max, y_max = temp_map(max_long, max_lat)

    # Calculate width and height of the bounding box
    width = x_max - x_min
    height = y_max - y_min

    # Ddjusting the smaller dimension to ensure square format
    if width > height:
        delta = width - height
        y_min -= delta / 2
        y_max += delta / 2
    else:
        delta = height - width
        x_min -= delta / 2
        x_max += delta / 2
    
    # Determine the maximum range in projected coordinates
    max_range = max(x_max - x_min, y_max - y_min) * scale_factor

    # Convert projected square range back to lat/lon
    new_min_long, new_min_lat = temp_map(x_min - max_range, y_min - max_range, inverse=True)
    new_max_long, new_max_lat = temp_map(x_max + max_range, y_max + max_range, inverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define Basemap with proper bounds
    m = Basemap(llcrnrlon=new_min_long, llcrnrlat=new_min_lat,
                urcrnrlon=new_max_long, urcrnrlat=new_max_lat,
                resolution='l', projection='merc',
                lat_0=np.mean(lats), lon_0=np.mean(longs))
    
    m.drawcoastlines()
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    m.drawparallels(np.arange(-90., 91., 5.), labels=[1,0,0,0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 5.), labels=[0,0,0,1], fontsize=10)
    m.drawmapboundary(fill_color='lightblue')

    # Convert lat/lon to map-coordinates
    x, y = m(longs, lats)

    if color_speed:
        cmap = plt.get_cmap('coolwarm')  # Colormap (blue to red)

        # Plot line segments with color based on SOG values
        for i in range(len(x) - 1):
            color = cmap(norm(mmsi_sog[i]))  # Map SOG value to a color
            m.plot([x[i], x[i+1]], [y[i], y[i+1]], color=color, linewidth=2)
        
        # Create a ScalarMappable to display the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Empty array needed for colorbar
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label="SOG (Speed Over Ground)")  
    else:
        m.plot(x, y, marker='o', linestyle='-', color='b', markersize=3, linewidth=1)

    # Mark the start and end points with annotations
    start_point = (x[0], y[0])
    end_point = (x[-1], y[-1])

    m.scatter(*start_point, color='green', marker='o', s=100, label="Start")
    m.scatter(*end_point, color='red', marker='o', s=100, label="End")

    # Define offsets so it accounts for varying plot sizes.
    x_offset = (max(x) - min(x)) * 0.05  # 5% of the x-axis range
    y_offset = (max(y) - min(y)) * 0.05  # 5% of the y-axis range

    # Annotate "Start" and "End"
    ax.text(start_point[0] + x_offset, start_point[1] + y_offset, "Start", 
            color='green', fontsize=12, fontweight='bold')
    ax.text(end_point[0] + x_offset, end_point[1] + y_offset, "End", 
            color='red', fontsize=12, fontweight='bold')


    ax.set_title(f"Vessel Trajectory for MMSI {mmsi} ({start_time} + {num_hours} hours)")
    plt.tight_layout()
    plt.show()


def main():
    # Load data
    data_path = 'Alan Turings baghave i tordenvejr'
    df = load_data_pandas(data_path)

    # Get a list of MMSI numbers that you can plot
        # The following function returns a list of MMSI-numbers where the vessels are trawling=1
    vessels = print_unique_values(
        df=df,
        target_feature='trawling', # Column name of the target feature
        label=1, # Label-value from the target feature of intereset
        unique_feature='MMSI' # Column name of the feature you want unique values for
    )

    # Plot the trajectories
    for vessel in vessels:
        plot_raw_trajectory(
            df=df, 
            mmsi=vessel, 
            num_hours=10/60, # 10 minutes
            scale_factor=0.1, 
            color_speed=False,
            norm='mmsi', # Normalize speed based on the specific vessel
            target_feature='trawling', # Only show the trajectory of trawling activity
            label=1
            ) 



if __name__ == '__main__':
    main()