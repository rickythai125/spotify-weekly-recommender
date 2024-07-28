import boto3
import pandas as pd
import os
import datetime

def download_from_s3(bucket_name, file_key, download_path):
    """
    Download a file from an S3 bucket.

    Parameters:
    - bucket_name (str): Name of the S3 bucket
    - file_key (str): Key of the file in the S3 bucket
    - download_path (str): Local path where the file will be saved
    """
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, file_key, download_path)
    print(f"File downloaded from S3 bucket '{bucket_name}' and saved to '{download_path}'.")

def read_json_to_dataframe(file_path):
    """
    Read a JSON file into a pandas DataFrame.

    Parameters:
    - file_path (str): Path to the JSON file

    Returns:
    - df (pandas.DataFrame): DataFrame containing the JSON data
    """
    df = pd.read_json(file_path)
    return df

# Example usage
bucket_name = os.getenv('S3_BUCKET_NAME')

current_date = datetime.datetime.now().strftime('%Y-%m-%d')
filename = f'spotify_top_tracks_{current_date}.json'

file_key = filename
download_path = 'spotify_top_tracks/' + filename

# Download the file from S3
download_from_s3(bucket_name, file_key, download_path)

# Read the JSON file into a pandas DataFrame
df = read_json_to_dataframe(download_path)
print(df.head())
