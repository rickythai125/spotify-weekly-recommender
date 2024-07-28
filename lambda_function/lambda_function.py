import json
import datetime
import boto3
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logger.info('Lambda function started')

    try:
        # Spotify API credentials
        SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
        SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
        SPOTIPY_REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URI')
        SPOTIPY_REFRESH_TOKEN = os.getenv('SPOTIPY_REFRESH_TOKEN')

        # AWS S3 credentials
        S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

        logger.info('Initializing Spotify client')
        # Initialize the Spotify client
        sp_oauth = SpotifyOAuth(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET,
            redirect_uri=SPOTIPY_REDIRECT_URI
        )
        token_info = sp_oauth.refresh_access_token(SPOTIPY_REFRESH_TOKEN)
        sp = spotipy.Spotify(auth=token_info['access_token'])

        logger.info('Fetching Spotify top tracks')
        # Fetch Spotify top tracks
        results = sp.current_user_top_tracks(limit=50, time_range='short_term')

        logger.info('Formatting tracks data')
        tracks = []
        for item in results['items']:
            try:
                track_info = {
                    'name': item['name'],
                    'artist': item['artists'][0]['name'],
                    'id': item['id'],
                    'album': item['album']['name'],
                    'date_added': item.get('added_at', 'N/A')  # Use .get to handle missing keys
                }
                tracks.append(track_info)
            except KeyError as e:
                logger.error(f"KeyError: {e}")
                continue

        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        filename = f'spotify_top_tracks_{current_date}.json'
        logger.info('Uploading to S3')

        # Save the results to S3
        s3 = boto3.client('s3')
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            Body=json.dumps(tracks),
            ContentType='application/json'
        )

        logger.info('Top tracks saved successfully')
        return {
            'statusCode': 200,
            'body': json.dumps(f'Top tracks saved successfully as {filename}')
        }
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }