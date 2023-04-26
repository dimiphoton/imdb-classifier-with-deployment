import json
import kaggle
import os
def download_kaggle_dataset(dataset_url):
    # Load the Kaggle API credentials from the kaggle.json file
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "r") as f:
        kaggle_credentials = json.load(f)

    # Authenticate with the Kaggle API using the loaded credentials
    kaggle.api.authenticate()

    # Extract the dataset information from the URL
    dataset_owner, dataset_name = dataset_url.split("/")[-2:]

    # Download the dataset using the Kaggle API
    kaggle.api.dataset_download_files(f"{dataset_owner}/{dataset_name}", path='./', unzip=True)

# Set the dataset URL
dataset_url = "https://www.kaggle.com/rounakbanik/the-movies-dataset"

# Download the dataset
download_kaggle_dataset(dataset_url)
