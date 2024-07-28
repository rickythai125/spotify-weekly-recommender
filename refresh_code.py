import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import os

SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = 'http://localhost:5000/callback'

sp_oauth = SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope='user-top-read'
)

token_info = sp_oauth.get_access_token(as_dict=True)
print("Access Token: ", token_info['access_token'])
print("Refresh Token: ", token_info['refresh_token'])
print("Expires in: ", token_info['expires_in'])

with open("token_info.json", "w") as file:
    json.dump(token_info, file)
