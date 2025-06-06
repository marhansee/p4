import datetime
import os
import requests
import zipfile
from pyspark.sql.functions import *

def get_data():
    """
    Prompts the user for a start date and end date in YYYY-MM-DD format,
    then downloads from Danish Maritime Authority (https://web.ais.dk/aisdata) and
    extracts AIS data files for each date in the range if the corresponding CSV file is not already present.
    """
    # Prompt the user for start/end dates
    start_date_str = input("Enter the start date (YYYY-MM-DD): ")
    end_date_str = input("Enter the end date (YYYY-MM-DD): ")

    # Convert the input strings to datetime.date objects
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()

    # Destination folder for AIS data
    data_destination = "data"
    os.makedirs(data_destination, exist_ok=True)

    # Iterate over each date in the range
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        # Check if the CSV file exists. Adjust the file name if needed based on actual CSV file naming.
        csv_file = os.path.join(data_destination, f"aisdk-{date_str}.csv")

        if os.path.exists(csv_file):
            print(f"CSV file for {date_str} already exists. Skipping.")
        else:
            url = f"https://web.ais.dk/aisdata/aisdk-{date_str}.zip"
            zip_file = f"aisdk-{date_str}.zip"

            try:
                print(f"Processing {url}...")
                # Download the zip file
                response = requests.get(url)
                response.raise_for_status()  # Raise an error for bad responses

                # Save the zip file locally
                with open(zip_file, "wb") as file:
                    file.write(response.content)

                # Extract the zip file contents into the "data" folder
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(data_destination)
                print(f"Extracted files for {date_str} into {data_destination}.")

                # Remove the zip file after extraction
                os.remove(zip_file)
                print(f"Removed {zip_file}.\n")

            except requests.exceptions.RequestException as e:
                print(f"Download error for {url}: {e}\n")
            except zipfile.BadZipFile as e:
                print(f"Extraction error for {zip_file}: {e}\n")
            except Exception as e:
                print(f"Unexpected error for {url}: {e}\n")

        # Move to the next date
        current_date += datetime.timedelta(days=1)

get_data()
