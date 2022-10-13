import json
import numpy as np
from time import sleep
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

############ APIs ############
import config as cfg

def get_spotipy_api():
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    client_credentials_manager = SpotifyClientCredentials(client_id=cfg.spotipy['client_id'], 
                                            client_secret=cfg.spotipy['client_secret'])
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return sp

def get_lyricsgenius_api():
    from lyricsgenius import Genius
    genius = Genius(cfg.genius['token'])
    genius.verbose = False
    return genius

import timeout_decorator
@timeout_decorator.timeout(5)
def get_genius_song(trackName, artistName):
    return genius.search_song(trackName, artistName)

import requests
def get_tracks_from_spotify(limit, offset):
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + cfg.spotify['token'],
    }
    
    params = {
        'limit': str(limit),
        'offset': str(offset)
    }
    
    response = requests.get('https://api.spotify.com/v1/me/tracks', params=params, headers=headers)
    tracks_dict = json.loads(response.text)
    return tracks_dict
    

############ Curve Fitting ############
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def const_func(x, k):
    return 0 * x + k

def lin_func(lx, k, alpha):
    return k + alpha * lx

############ Date and Time Formatting ############
from datetime import date, timedelta
import datetime
import pytz

def convert_ms_to_s(df, old_col, new_col):
    if old_col in df.columns:
        df[new_col] = df[old_col]/1000
        df = df.drop(old_col, axis = 1)
    return df

def format_date(date_str):
    ymd = [int(x) for x in date_str.split('-')]
    return date(*ymd)

def convert_UTC_to_PT(utc_dateTime):
    utc_timezone = pytz.timezone("UTC")
    pt_timezone = pytz.timezone("US/Pacific")
    utc_time = utc_timezone.localize(utc_dateTime)
    pt_time = utc_time.astimezone(pt_timezone)
    return datetime.datetime(pt_time.year, pt_time.month, pt_time.day, pt_time.hour, pt_time.minute, pt_time.second)

def get_date_from_datetime_obj(dt):
    '''For extracting the start date'''
    return datetime.datetime(dt.year, dt.month, dt.day)

def get_absolute_date(relative_day_number, start_date):
    return start_date + timedelta(int(relative_day_number))

def format_absolute_date(relative_day_number, start_date):
    return get_absolute_date(relative_day_number, start_date).strftime('%b %d, %Y')

def format_timeofday(h):
    mn_str = 'AM' if (h)//12 == 0 else 'PM'
    hour_str = str(12) if h % 12 == 0 else str((h-1)%12+1)
    return str((h-1)%12+1) + mn_str

def format_month(m):
    month_num = str(m[1])
    datetime_object = datetime.datetime.strptime(month_num, "%m")
    month_name = datetime_object.strftime("%b")
    year = str(m[0])
    return month_name + " " + year

def str_to_datetime_no_seconds(s):
    return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M')

def str_to_datetime(s):
    return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

def convert_to_datetime(df, columnName):
    if type(df[columnName][0]) == str:
        df[columnName] = df.apply(lambda x: str_to_datetime(x[columnName]), axis = 1)
    return df

############ Lyrics ############
import re

def remove_embed_message(lyrics):
    '''Replace the random number followed by embed'''
    lyrics_rev = lyrics[::-1]
    embed_rev = 'Embed'[::-1]
    return re.sub(embed_rev + '*\d+', '', lyrics_rev, 1)[::-1]

def format_lyrics(lyrics):
    '''Standardize the formatting of lyrics 
        from lyricsgenius API so we can feed them into
        a zero-shot language classification model'''
    lyrics = re.sub('\[+.+?\]+', '', lyrics.replace("\n", " " ))
    lyrics = remove_embed_message(lyrics)
    lyrics = re.sub('.+? Lyrics*', '', lyrics)
    lyrics = lyrics.replace('/\s\s+/g', ' ');
    return lyrics




