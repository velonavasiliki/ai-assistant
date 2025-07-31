import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
import html
import time
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')


class ytinteraction:

    def __init__(self) -> None:
        self.info = {}

    def ytretriever(self, query: str, order: str = 'viewCount', duration='medium', num_results: int = 1, before: str = None, after: str = None):
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
            self.info dict[dict]:
                Populates self.info with {ID: title, channel, date, transcript, transcript_summary} 
                of retrieved videos. transcript, transcript_summary are set to None.
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
            after_time = (now_time - relativedelta(years=5)).replace(
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

            for item in search_response['items']:
                # eliminate videos set to premier later
                if item['snippet']['liveBroadcastContent'] == 'none' and item['id']['kind'] == 'youtube#video':
                    self.info['id'] = {
                        'title': html.unescape(item['snippet']['title']),
                        'channel': item['snippet']['channelTitle'],
                        'date': item['snippet']['publishedAt'],
                        'id': item['id']['videoId'],
                        'transcript': None,
                        'transcript_summary': None
                    }

            if not self.info:
                raise ValueError(
                    "Could not recover videos of the right format.")

            return self.info

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def yttranscript(self, ids: list[str]):
        """
        Searches and assembles transcripts of YouTube videos, updates self.info.

        Parameters:
        -----------
            id : list(str)
            List of video IDs to get transcript.

        Returns:
        --------
            self.info : dict
            Dictionary with video IDs as keys and title, channel, date, transcript, 
            transcript_summary as values.
        """
        ytt_api = YouTubeTranscriptApi()

        for id in ids:
            try:
                fetched_transcript = ytt_api.fetch(
                    id, languages=['en'])
                transcript_snippets = [
                    snippet.text for snippet in fetched_transcript]
                transcript = 'TRANSCRIPT: ' + ' '.join(transcript_snippets)
                if 'id' in self.info:
                    self.info[id]['transcript'] = transcript
                else:
                    self.info[id] = {
                        'title': None,
                        'channel': None,
                        'date': None,
                        'id': id,
                        'transcript': transcript,
                        'transcript_summary': None
                    }
                time.sleep(3)
            except Exception as e:
                print(e)
                continue

        return self.info
