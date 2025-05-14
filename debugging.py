target_features = [f'future_lat_{i}' for i in range(6, 121, 6)] + \
           [f'future_lon_{i}' for i in range(6, 121, 6)]

print(len(target_features))