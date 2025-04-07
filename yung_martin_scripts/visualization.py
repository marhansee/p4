import pyspark
import warnings
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, to_timestamp, unix_timestamp, lag, when
from pyspark.sql.window import Window
import logging
import findspark
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
import seaborn as sns


from utils.utils import load_data_pyspark, load_data_pandas, \
    standardize_dateformat, print_unique_values



logging.getLogger("py4j").setLevel(logging.ERROR)
findspark.init()
spark = SparkSession.builder \
    .appName("P4_project") \
    .config("spark.driver.memory", "15g") \
    .getOrCreate()

def add_time_difference_feature(df: DataFrame) -> DataFrame:
    # Drop duplicates
    df = df.drop_duplicates()

    # Ensure correct format
    df = df.withColumn("# Timestamp", to_timestamp(col("# Timestamp"), "dd-MM-yyyy HH:mm:ss"))
    
    # Sort the df
    df = df.orderBy(col("MMSI").asc(), col("# Timestamp").asc())

    # Define a window partitioned by MMSI and ordered by timestamp
    window_spec = Window.partitionBy("MMSI").orderBy(col("# Timestamp").asc())

    # Get the lagged value
    df = df.withColumn("prev_timestamp", lag(col("# Timestamp")).over(window_spec))

    # Calculate the difference in seconds
    df = df.withColumn("DiffInSeconds", 
                       (unix_timestamp(col("# Timestamp")) - unix_timestamp(col("prev_timestamp")))) 
    
    # Set threshold (assume time gap of +0.5 hour is outlier -- MMSI sailed twice)
    threshold = 1800
    df = df.withColumn("DiffInSeconds", 
                    when(col("DiffInSeconds") > threshold, 0).otherwise(col("DiffInSeconds"))) # Restart DiffInSeconds
    
    df = df.fillna({"DiffInSeconds": 0}) # Fill NULLs with 0

    df = df.drop("prev_timestamp")  # Drop intermediate column if not needed

    return df

def plot_histogram(df, feature: str, num_bins: int, max_value: str, with_target: bool):
    if isinstance(df, DataFrame):
        raise AssertionError("The DataFrame must be a Pandas DataFrame")
    
    if with_target == True:
        # Filter rows based on target column
        df_trawler_0 = df[df['trawling'] == 0]
        df_trawler_1 = df[df['trawling'] == 1]
        plt.hist(
            x=df_trawler_0[feature], 
            bins=num_bins, 
            range=[min(df[feature]), max_value], 
            color='blue', 
            alpha=1, 
            label="Trawling = 0")
        plt.hist(
            x=df_trawler_1[feature], 
            bins=num_bins, 
            range=[min(df[feature]), max_value], 
            color='red', 
            alpha=1, 
            label="Trawling = 1")
    
    else:
        plt.hist(df[feature], bins=num_bins, range=[min(df[feature]), max_value])

    plt.title(f"Histogram of {feature}")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    
def plot_raw_trajectory(df, mmsi, id_feature='MMSI', speed_feature='SOG',
                        time_feature='# Timestamp', lat_feature='Latitude', 
                        long_feature='Longitude',
                        num_hours='full', scale_factor=1.5, color_speed=True, 
                        norm='ALL', target_feature=None,
                        label=1):
    # if not isinstance(mmsi, int):
    #     raise AssertionError("MMSI must be an integer!")
    # if not isinstance(num_hours, (int, float)) or num_hours <= 0:
    #     raise AssertionError("num_hours must be a positive number!")
    
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

    # Convert lat/lon to map coordinates
    x, y = m(longs, lats)

    if color_speed:
        cmap = plt.get_cmap('coolwarm')  # Colormap (blue to red)

        # Plot line segments with color based on SOG values
        for i in range(len(x) - 1):
            # Color of the current line segment based on SOG
            color = cmap(norm(mmsi_sog[i]))  # Map SOG value to a color
            m.plot([x[i], x[i+1]], [y[i], y[i+1]], color=color, linewidth=2)
        
        # Create a ScalarMappable to display the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Empty array needed for colorbar
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label="SOG (Speed Over Ground)")  # Colorbar to show the SOG scale
    else:
        # If no color_speed, plot as a simple line with blue color
        m.plot(x, y, marker='o', linestyle='-', color='b', markersize=3, linewidth=1)

    # Mark the start and end points with annotations
    start_point = (x[0], y[0])
    end_point = (x[-1], y[-1])

    m.scatter(*start_point, color='green', marker='o', s=100, label="Start")
    m.scatter(*end_point, color='red', marker='o', s=100, label="End")

    # Define offsets so it accounts for varying plot sizes.
    x_offset = (max(x) - min(x)) * 0.05  # 10% of the x-axis range
    y_offset = (max(y) - min(y)) * 0.05  # 10% of the y-axis range

    # Annotate "Start" and "End"
    ax.text(start_point[0] + x_offset, start_point[1] + y_offset, "Start", 
            color='green', fontsize=12, fontweight='bold')
    ax.text(end_point[0] + x_offset, end_point[1] + y_offset, "End", 
            color='red', fontsize=12, fontweight='bold')


    ax.set_title(f"Vessel Trajectory for MMSI {mmsi} ({start_time} + {num_hours} hours)")
    plt.tight_layout()
    plt.show()

def plot_corr_matrix(df):
    
    # Drop non-numerical features
    for feature in df.columns:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            df = df.drop(feature, axis=1)
        
    matrix = df.corr()
    sns.heatmap(matrix, cmap="Greens", annot=True)
    plt.show()


def plot_rate_of_turn(df, mmsi, window_minutes=10):
    if not isinstance(mmsi, int):
        raise AssertionError("MMSI must be an integer!")
    
    # Filter the data for the specified MMSI
    df_vessel = df[df['MMSI'] == mmsi].copy()
    
    # Convert timestamp to datetime
    df_vessel['# Timestamp'] = pd.to_datetime(df_vessel['# Timestamp'])
    
    # # Calculate the heading change (in degrees)
    df_vessel['Heading Change'] = df_vessel['COG'].diff()

    # Calculate time difference in seconds
    df_vessel['Time Diff'] = df_vessel['# Timestamp'].diff().dt.total_seconds()
    
    # Calculate Rate of Turn (ROT) in degrees per second
    df_vessel['ROT'] = df_vessel['Heading Change'] / df_vessel['Time Diff']
    
    # Apply rolling window to smooth the ROT (using a window of 10 minutes)
    window_size = window_minutes * 60  # convert minutes to seconds
    df_vessel['Smoothed ROT'] = df_vessel['ROT'].rolling(window=window_size, min_periods=1).mean()
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df_vessel['# Timestamp'], df_vessel['Smoothed ROT'], label='Smoothed ROT', color='blue')
    
    # Highlight the turning rate
    plt.axhline(0, color='black', linestyle='--', linewidth=1)  # Zero line
    plt.title(f"Rate of Turn (ROT) for Vessel MMSI {mmsi} (Window: {window_minutes} minutes)")
    plt.xlabel('Time')
    plt.ylabel('Rate of Turn (Degrees per second)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_heading_change(df, mmsi):
    if not isinstance(mmsi, int):
        raise AssertionError("MMSI must be an integer!")
    
    # Filter the data for the specified MMSI
    df_vessel = df[df['MMSI'] == mmsi].copy()
    
    # Drop MVs row
    df_vessel['COG'] = df_vessel['COG'].dropna()

    # Convert timestamp to datetime
    df_vessel['# Timestamp'] = pd.to_datetime(df_vessel['# Timestamp'])
    
    # Calculate the heading change (in degrees)
    df_vessel['Heading Change'] = df_vessel['COG'].diff()

    # Drop NaNs (diff introduces MV's)
    df_vessel = df_vessel.dropna(subset=['COG', 'Heading Change'])
    
    plot_histogram(
        df=df_vessel,
        feature='Heading Change',
        num_bins=10,
        max_value=df_vessel['Heading Change'].max(),
        with_target=False
    )

def plot_scatterplot(df):
    # Remove non-numerical features and static features
    features_to_drop = ['Unnamed: 0', 'MMSI','Cargo type','Width','Length',
                        'A','B','C','D','DiffInSeconds','Latitude','Longitude']
    
    for feature in df.columns:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            df = df.drop(feature, axis=1)
        if feature in features_to_drop:
            df = df.drop(feature, axis=1)
    

    # print(df['Draught'])
    features = list(df.columns)

    # Loop through all pairs of features
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            plt.figure(figsize=(6, 6))
            
            # Scatter plot for each combination of features
            feature_1 = features[i]
            feature_2 = features[j]
            
            # Set the color based on 'trawler' column (red if 'trawler' == 1, else blue)
            colors = np.where(df['trawling'] == 1, 'red', 'blue')
            
            # Plot
            plt.scatter(df[feature_1], df[feature_2], c=colors, alpha=0.7)
            
            # Labels and title
            plt.xlabel(feature_1)
            plt.ylabel(feature_2)
            plt.title(f'Scatter plot of {feature_1} vs {feature_2}')
            
            # Show plot
            plt.show()

def print_duplicates(df):
    if not isinstance(df, pd.DataFrame):
        raise AssertionError("DataFrame must be a Pandas DataFrame!")
    
    num_duplicates = df.duplicated(keep=False).sum()
    print(f'Total number of rows in Df: {df.shape[0]}')
    print(f'Total number of duplicates: {num_duplicates}')
    print(f'Rows after removal of duplicates: {df.shape[0]-num_duplicates}')




def main():
    # load data
    # data_path = os.path.join(os.path.dirname(__file__), 'preprocessed_data.csv')
    # df = load_data_pyspark(spark=spark, file_name=data_path)
    # df = standardize_dateformat(df, '# Timestamp')
    # df = add_time_difference_feature(df)
    # df.toPandas().to_csv('processed_data_4.csv')

    data_path = os.path.join(os.path.dirname(__file__), 'data/aisdk-2025-01-01_fishing_labeled.csv')
    df_pd = load_data_pandas(data_path)
    print(df_pd.head(5))

    if 'Unnamed: 0' in df_pd.columns:
        df_pd = df_pd.drop(['Unnamed: 0'], axis=1)

    """Print number of duplicates"""
    # print_duplicates(df_pd)
    # print_duplicates(df_pd)
    # print(df_pd[df_pd.duplicated(keep=False)])


    """Plot irregular time intervals histogram"""
    # plot_histogram(df_pd, 'DiffInSeconds', num_bins=15, max_value=60, with_target=False)
    # print(df_pd[df_pd['Gear Type']=='FISHING'])

    """Get unique MMSI's of trawling vessels"""
    # trawlers = print_unique_values(df=df_pd, target_feature='trawling',unique_feature='MMSI')

    """Plot raw trajectories"""
    # TRAWLER:
    # trawler_list = [219001039, 219001125, 219793000, 220012000, 219009229, 219959000, 244880000,
    # 219005583, 219005867, 219024923, 220072000, 220329000, 220336000, 219002005, 
    # 219006219, 219024715, 219862000, 220043000, 219006971, 219948000, 220138000, 
    # 219007034, 219002585, 219006835, 219904000, 220279000, 220359000, 219004473, 
    # 219017895, 220334000, 219001149, 219006113, 219024000, 220225000, 220323000, 
    # 220343000, 219005956, 219001604, 219005954, 219004242, 219005732]

    # for mmsi in trawlers:
    #     plot_raw_trajectory(
    #         df=df_pd, 
    #         mmsi=mmsi, 
    #         num_hours=10/60, 
    #         scale_factor=0.1, 
    #         color_speed=False,
    #         norm='mmsi',
    #         trawling_feature='trawling'
    #         ) # Trawler!

    # # Gillnets
    # plot_raw_trajectory(
    #     df=df_pd, 
    #     mmsi=219005929, 
    #     num_hours=10/60, 
    #     scale_factor=0.1,
    #     color_speed=False,
    #     norm='mmsi') 

    # # # Fishing
    # plot_raw_trajectory(
    #     df=df_pd, 
    #     mmsi=219005671, 
    #     num_hours=10/60, 
    #     scale_factor=0.1,
    #     color_speed=False,
    #     norm='mmsi') 

    # # # Dredging
    # plot_raw_trajectory(
    #     df=df_pd, 
    #     mmsi=219025739, 
    #     num_hours=10/60, 
    #     scale_factor=0.1,
    #     color_speed=False,
    #     norm='mmsi') 


    # # JUST CHILLING:
    # plot_raw_trajectory(
    #     df=df_pd, 
    #     mmsi=220088000, 
    #     num_hours=10/60, 
    #     scale_factor=0.1, 
    #     color_speed=False,
    #     norm='mmsi'
    #     ) # Just chilling



    # duration = [10, 5, 1, 0.5, 1/3, 1/6]
    # for i in duration:
    #     plot_raw_trajectory(
    #         df=df_pd,
    #         mmsi=219025739,
    #         num_hours=i,
    #     )

    """Plot histogram of vessel speeds when fishing and not (no outlier removal) """
    # plot_histogram(df_pd, 'SOG', num_bins=8, max_value=20, with_target=True)

    """Plot correlation matrix"""
    # plot_corr_matrix(df_pd)

    """Plot rate of turns within a window of x minutes"""
    # print(df_pd[df_pd['MMSI']==219009229]['Heading'])

    # plot_rate_of_turn(
    #     df=df_pd,
    #     mmsi=219009229,
    #     window_minutes=1
    #     )
    
    # plot_rate_of_turn(
    #     df=df_pd,
    #     mmsi=220088000,
    #     window_minutes=1
    #     )

    """Plot COG histogram"""
    # df_pd['COG'] = df_pd['COG'].fillna(0)
    # df_pd[df_pd['MMSI']==219009229]['Heading Change'] = df_pd[df_pd['MMSI']==219009229]['COG'].diff()

    # plot_histogram(
    #     df=df_pd[df_pd['MMSI']==219009229], 
    #     feature='Heading Change', 
    #     with_target=True,
    #     num_bins=15,
    #     max_value=df_pd['Heading Change'].max()
    #     )

    """Plot Heading change of a vessel"""
    # plot_heading_change(df=df_pd, mmsi=219009229)
    # plot_heading_change(df=df_pd, mmsi=220088000)

    """Plot scatter plot"""
    # plot_scatterplot(df_pd)

    """Synthetic data:"""
    s_data_path = os.path.join(os.path.dirname(__file__), 'data/128_fishing_trajs.csv')
    s_df = load_data_pandas(s_data_path)

    # print_duplicates(s_df)

    print_unique_values(df=s_df, target_feature='label', 
                        label='02-fishing', unique_feature='id')
    # print(s_df.head())
    # print(s_df.isna().sum())
    # print(s_df.info())
    # print(s_df[s_df['id']=='235008380-3'])

    # # Synthetic fishing MMSI: 235008380-3
    # # Synthetic non-fishing MMSI: 211477000-2
    # # min_in_hours = 10 / 60

    plot_raw_trajectory(
        df=s_df,
        mmsi='220341000-2',
        id_feature='id',
        speed_feature='euc_speed',
        time_feature='t',
        lat_feature='latitude',
        long_feature='longitude',
        num_hours='full',
        scale_factor=1.5,
        color_speed=True,
        norm='All',
        target_feature='label',
        label='02-fishing'
    )
    # plot_histogram(
    #     df=s_df,
    #     feature='euc_speed',
    #     num_bins=15,
    #     max_value=20,
    #     with_target=False
    # )


if __name__ == '__main__':
    main()