import os
import shutil
from datetime import datetime

# Paths
src_dir = "/ceph/project/gatehousep4/data"
train_dir = "/ceph/project/gatehousep4/data/train"
test_dir = "/ceph/project/gatehousep4/data/test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Go through each file
for filename in os.listdir(src_dir):
    if filename.endswith(".csv") and filename.startswith("aisdk"):
        try:
            # Extract date part: '2024-03-01' from 'aisdk-2024-03-01_fishing_labeled.csv'
            date_str = filename.split("-")[1] + "-" + filename.split("-")[2] + "-" + filename.split("-")[3][:2]
            date = datetime.strptime(date_str, "%Y-%m-%d")

            if date.month in [3,4,5,6,7,8,9,10,11]:  # March to November
                shutil.move(os.path.join(src_dir, filename), os.path.join(train_dir, filename))
            else:  # December to February
                shutil.move(os.path.join(src_dir, filename), os.path.join(test_dir, filename))
        except Exception as e:
            print(f"Skipping file {filename} due to error: {e}")
