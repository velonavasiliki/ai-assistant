
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta

load_dotenv()
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')


def ytretriever(query: str, order: str = 'viewCount', duration='medium', num_results: int = 10, before: str = None, after: str = None) -> list[dict]:
    """
    Searches YouTube for videos related to query, inside a timespan.
    Returns list with title, channel, date, ID.

    Parameters:
    -----------
        query : str 
            Search term.
        order : str
            Search results sorted by one of the following: 'date', 'rating', 'relevance', 'viewCount'.
        duration : str
            Duration of search results: 'any', 'long' (minutes 20+), 'medium' (4-20), 'short' (<4).
        num_results : int 
            Maximum number of videos to retrieve.
        before : str
            Date upper limit in form %m/%d/%Y.
        after : str
            Date lower limit in form %m/%d/%Y.

    Returns:
    --------
        info list[dict]: 
            title, channel, date, ID of retrieved videos
    """
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    now_time = datetime.now(timezone.utc)

    if before:
        before_time = datetime.strptime(before, "%m/%d/%Y").replace(
            hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc
        )
    else:
        before_time = datetime.now(timezone.utc)

    if after:
        after_time = datetime.strptime(after, "%m/%d/%Y").replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )
    else:
        after_time = (now_time - relativedelta(years=15)).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )

    try:
        search_response = youtube.search().list(
            part='id,snippet',
            q=query,
            type='video',
            order=order,
            relevanceLanguage='en',  # prefer videos relevant to english language
            safeSearch='strict',
            videoDuration=duration,
            videoCaption='closedCaption',
            maxResults=num_results,
            publishedAfter=after_time.isoformat(),
            publishedBefore=before_time.isoformat()
        ).execute()

        info = []

        for item in search_response['items']:
            # eliminate videos set to premier later
            if item['snippet']['liveBroadcastContent'] == 'none' and item['id']['kind'] == 'youtube#video':
                info.append(
                    {
                        'title': item['snippet']['title'].replace('&amp;', '&'),
                        'channel': item['snippet']['channelTitle'],
                        'date': item['snippet']['publishedAt'],
                        'id': item['id']['videoId']
                    }
                )

        if not info:
            raise ValueError("Could not recover videos of the right format.")

        return info

    except Exception as e:
        print(f"An error occurred: {e}")
        return []
