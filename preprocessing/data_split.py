import os
import shutil
import json
import re
from datetime import datetime

# Load config file
with open("/ceph/project/gatehousep4/data/configs/data_split_config.json") as f:
    config = json.load(f)

train_years = set(config["train_years"])
train_months = set(config["train_months"])
test_years = set(config["test_years"])
test_months = set(config["test_months"])
source_dir = config["source_dir"]
train_base = config["train_base"]
test_base = config["test_base"]

# Determine next version number
def get_next_version(base_path):
    versions = [d for d in os.listdir(base_path) if re.match(r"v\d+$", d)]
    version_nums = [int(v[1:]) for v in versions]
    return f"v{max(version_nums, default=0) + 1}"

version = get_next_version(train_base)
print(f"Auto-selected version: {version}")

# Set paths
train_dir = os.path.join(train_base, version)
test_dir = os.path.join(test_base, version)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Copy the defined filesS
for filename in os.listdir(source_dir):
    if filename.endswith(".csv") and filename.startswith("aisdk"):
        try:
            parts = filename.split("-")
            if len(parts) < 4:
                continue
            date_str = f"{parts[1]}-{parts[2]}-{parts[3][:2]}"
            date = datetime.strptime(date_str, "%Y-%m-%d")
            src_path = os.path.join(source_dir, filename)

            if (date.year in train_years) and (date.month in train_months):
                shutil.copy2(src_path, os.path.join(train_dir, filename))
            elif (date.year in test_years) and (date.month in test_months):
                shutil.copy2(src_path, os.path.join(test_dir, filename))

        except Exception as e:
            print(f"Skipping {filename}: {e}")
