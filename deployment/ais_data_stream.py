from utilities.preprocessing import preprocess_vessel_df

input_path = "aisdk-2024-06-04_fishing_labeled.csv"
MMSI = 245265000

df = preprocess_vessel_df(input_path, MMSI)
print(df.head(5))

