
def main():
    input_features = ['timestamp_epoch', 'MMSI', 'Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading', 
                      'Width', 'Length', 'Draught']
    features_to_scale = [feature for feature in input_features if feature not in ['timestamp_epoch', 'MMSI']]

    print(features_to_scale)

if __name__ == '__main__':
    main()