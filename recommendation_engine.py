import pandas as pd
import numpy as np
import json
import re 
import sys
import itertools
import os
import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

import warnings
warnings.filterwarnings("ignore")

#another useful command to make data exploration easier
# NOTE: if you are using a massive dataset, this could slow down your code. 
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)

# spotify_df = pd.read_csv('data/data.csv')
# spotify_df['artists_upd_v1'] = spotify_df['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))

# spotify_df['artists_upd_v2'] = spotify_df['artists'].apply(lambda x: re.findall('\"(.*?)\"',x))
# spotify_df['artists_upd'] = np.where(spotify_df['artists_upd_v1'].apply(lambda x: not x), spotify_df['artists_upd_v2'], spotify_df['artists_upd_v1'] )
# spotify_df['artists_song'] = spotify_df.apply(lambda row: row['artists_upd'][0]+row['name'],axis = 1)
# spotify_df.sort_values(['artists_song','release_date'], ascending = False, inplace = True)
# spotify_df.drop_duplicates('artists_song',inplace = True)

# data_w_genre = pd.read_csv('data/data_w_genres.csv')
# data_w_genre['genres_upd'] = data_w_genre['genres'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])

# artists_exploded = spotify_df[['artists_upd','id']].explode('artists_upd')
# artists_exploded_enriched = artists_exploded.merge(data_w_genre, how = 'left', left_on = 'artists_upd',right_on = 'artists')
# artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull()]

# artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id')['genres_upd'].apply(list).reset_index()
# artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))

# spotify_df = spotify_df.merge(artists_genres_consolidated[['id','consolidates_genre_lists']], on = 'id',how = 'left')
# spotify_df['year'] = spotify_df['release_date'].apply(lambda x: x.split('-')[0])
# float_cols = spotify_df.dtypes[spotify_df.dtypes == 'float64'].index.values

# ohe_cols = 'popularity'
# # create 5 point buckets for popularity 
# spotify_df['popularity_red'] = spotify_df['popularity'].apply(lambda x: int(x/5))
# spotify_df['consolidates_genre_lists'] = spotify_df['consolidates_genre_lists'].apply(lambda d: d if isinstance(d, list) else [])

# spotify_df.to_pickle('spotify_df.pkl')

spotify_df = pd.read_pickle('spotify_df.pkl')
float_cols = spotify_df.dtypes[spotify_df.dtypes == 'float64'].index.values

def ohe_prep(df, column, new_name): 
    """ 
    Create One Hot Encoded features of a specific column

    Parameters: 
        df (pandas dataframe): Spotify Dataframe
        column (str): Column to be processed
        new_name (str): new column name to be used
        
    Returns: 
        tf_df: One hot encoded features 
    """
    
    tf_df = pd.get_dummies(df[column])
    feature_names = tf_df.columns
    tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop = True, inplace = True)    
    return tf_df

#function to build entire feature set
def create_feature_set(df, float_cols):
    """ 
    Process spotify df to create a final set of features that will be used to generate recommendations

    Parameters: 
        df (pandas dataframe): Spotify Dataframe
        float_cols (list(str)): List of float columns that will be scaled 
        
    Returns: 
        final: final set of features 
    """
    
    #tfidf genre lists
    tfidf = TfidfVectorizer()
    tfidf_matrix =  tfidf.fit_transform(df['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
    genre_df.reset_index(drop = True, inplace=True)

    #explicity_ohe = ohe_prep(df, 'explicit','exp')    
    year_ohe = ohe_prep(df, 'year','year') * 0.5
    popularity_ohe = ohe_prep(df, 'popularity_red','pop') * 0.15

    #scale float columns
    floats = df[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2

    #concanenate all features
    final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis = 1)
     
    #add song id
    final['id']=df['id'].values
    
    return final

complete_feature_set = create_feature_set(spotify_df, float_cols=float_cols)

client_id = os.getenv('SPOTIPY_CLIENT_ID') #'39515909d60a41839ec92e15d3cb478b'
client_secret= os.getenv('SPOTIPY_CLIENT_SECRET') #'1c08ef8f9f04441896d710a2ed023ad6'
refresh_token = os.getenv('SPOTIPY_REFRESH_TOKEN')
scope = 'user-library-read,playlist-read-private,playlist-modify-public,playlist-modify-private,playlist-modify-public'

sp_oauth = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=os.getenv('SPOTIPY_REDIRECT_URI')
)
token_info = sp_oauth.refresh_access_token(refresh_token)
sp = spotipy.Spotify(auth=token_info['access_token'])

current_date = datetime.datetime.now().strftime('%Y-%m-%d')
filename = f'spotify_top_tracks_{current_date}.json'
#file_path = 'spotify_top_tracks/' + filename
file_path = 'spotify_top_tracks/spotify_top_tracks_2024-07-27.json'

# Open and read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

weekly_df = pd.json_normalize(data)

def generate_playlist_feature(complete_feature_set, weekly_df, weight_factor):
    """ 
    Summarize a user's playlist into a single vector weighted by the position in the sorted list of most played songs.

    Parameters: 
        complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
        weekly_df (pandas dataframe): weekly dataframe sorted by most listened to
        weight_factor (float): float value that represents the play count bias. The larger the bias, the more priority frequently played songs get. Value should be close to 1. 
        
    Returns: 
        weekly_feature_set_weighted_final (pandas series): single feature that summarizes the playlist
        complete_feature_set_nonweekly(pandas dataframe): 
    """
    
    # Ensure the playlist_df is sorted by most listened to
    weekly_df = weekly_df.reset_index(drop=True)
    
    # Calculate the weight based on the position in the list
    weekly_df['weight'] = weekly_df.index.to_series().apply(lambda x: weight_factor ** (-x))
    
    # Merge with complete_feature_set
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(weekly_df['id'].values)]
    complete_feature_set_playlist = complete_feature_set_playlist.merge(weekly_df[['id', 'weight']], on='id', how='inner')
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(weekly_df['id'].values)]
    
    # Apply the weights
    playlist_feature_set_weighted = complete_feature_set_playlist.copy()
    playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:, :-2].mul(playlist_feature_set_weighted.weight, axis=0))
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-2]
    
    return playlist_feature_set_weighted_final.sum(axis=0), complete_feature_set_nonplaylist

complete_feature_set_weekly, complete_feature_set_nonweekly = generate_playlist_feature(complete_feature_set, weekly_df, 1.09)

def generate_playlist_recos(df, features, nonweekly_features):
    """ 
    Pull songs from a specific playlist.

    Parameters: 
        df (pandas dataframe): spotify dataframe
        features (pandas series): summarized playlist feature
        nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist
        
    Returns: 
        non_playlist_df_top_40: Top 40 recommendations for that playlist
    """
    
    non_weekly_df = df[df['id'].isin(nonweekly_features['id'].values)]
    non_weekly_df['sim'] = cosine_similarity(nonweekly_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]
    non_weekly_df_top_40 = non_weekly_df.sort_values('sim',ascending = False).head(40)
    non_weekly_df_top_40['url'] = non_weekly_df_top_40['id'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])
    
    return non_weekly_df_top_40

weekly_recs = generate_playlist_recos(spotify_df, complete_feature_set_weekly, complete_feature_set_nonweekly)

# Create a new playlist
playlist_name=datetime.datetime.now().strftime('%Y-%m-%d')
playlist = sp.user_playlist_create(user=12130351891, name=playlist_name, public=True, description='Playlist created with spotipy')
playlist_id = playlist['id']
print(f"Playlist '{playlist_name}' created successfully.")

# Add tracks to the playlist
track_ids = weekly_recs['id'].tolist()
sp.playlist_add_items(playlist_id, track_ids)
print(f"Added {len(track_ids)} tracks to the playlist.")

