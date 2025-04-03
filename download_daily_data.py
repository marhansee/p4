import os
import requests
import zipfile
from datetime import date, timedelta, datetime

def download_daily_data():
    """
    Prompts the user for a start date and end date in YYYY-MM-DD format,
    then downloads from Danish Maritime Authority (https://web.ais.dk/aisdata) and
    extracts AIS data files for each date in the range if the corresponding CSV file is not already present.
    """
    # Getting todays date minus 3 days
    date = (datetime.today() - timedelta(days=3)).strftime('%Y-%m-%d')

    # Destination folder for AIS data
    data_destination = "data"
    os.makedirs(data_destination, exist_ok=True)


    # Check if the CSV file exists. Adjust the file name if needed based on actual CSV file naming.
    csv_file = os.path.join(data_destination, f"aisdk-{date}.csv")

    if os.path.exists(csv_file):
        print(f"CSV file for {date} already exists. Skipping.")
    else:
        url = f"https://web.ais.dk/aisdata/aisdk-{date}.zip"
        zip_file = f"aisdk-{date}.zip"

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
            print(f"Extracted files for {date} into {data_destination}.")

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


download_daily_data()
