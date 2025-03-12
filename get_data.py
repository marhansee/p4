import os
import requests
import zipfile

import os
import requests
import zipfile
import datetime

# Define the date range you want to process
start_date = datetime.date(2025, 2, 1)  # starting date
end_date = datetime.date(2025, 2, 1)  # ending date (inclusive)

# Create a directory for extracted files if it doesn't exist
extracted_root = "data"
os.makedirs(extracted_root, exist_ok=True)

# Iterate over each date in the range
current_date = start_date
while current_date <= end_date:
    # Format the date as YYYY-MM-DD
    date_str = current_date.strftime("%Y-%m-%d")
    # Construct the URL using the formatted date
    url = f"https://web.ais.dk/aisdata/aisdk-{date_str}.zip"
    zip_path = f"aisdk-{date_str}.zip"

    try:
        print(f"Processing {url}...")
        # Download the zip file
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        # Save the zip file locally
        with open(zip_path, "wb") as file:
            file.write(response.content)

        # Define a directory to extract the files for this specific date
        extract_dir = os.path.join(extracted_root, date_str)
        os.makedirs(extract_dir, exist_ok=True)

        # Extract the zip file contents
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted files for {date_str} to {extract_dir}.")

        # Remove the zip file after extraction
        os.remove(zip_path)
        print(f"Removed {zip_path}.\n")

    except requests.exceptions.RequestException as e:
        print(f"Download error for {url}: {e}\n")
    except zipfile.BadZipFile as e:
        print(f"Extraction error for {zip_path}: {e}\n")
    except Exception as e:
        print(f"Unexpected error for {url}: {e}\n")

    # Move to the next date
    current_date += datetime.timedelta(days=1)
